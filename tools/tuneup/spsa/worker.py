#!/usr/bin/env python3
"""
SPSA Tuning Worker.

Polls the coordinator for work, runs cutechess-cli games, reports results.
Zero external dependencies — uses only Python stdlib.

Usage:
    python worker.py -c worker.json
"""

import argparse
import json
import logging
import logging.handlers
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from config import WorkerConfig, WorkItem

logger = logging.getLogger("worker")

class RetryableError(Exception):
    """Recoverable error during a single chunk — worker retries."""
    pass

# Subprocess timeouts and poll cadence
GAME_TIMEOUT = 3600        # max seconds for a cutechess-cli run
PIPE_DRAIN_TIMEOUT = 30    # seconds to wait for pipe readers after exit
POLL_INTERVAL = 0.5        # seconds between process-alive checks
HTTP_TIMEOUT = 30          # seconds for general HTTP requests
VALIDATE_TIMEOUT = 3       # seconds for chunk-validity HTTP checks

# Graceful-shutdown state
_current_process = None       # cutechess-cli Popen while games run
_shutdown_requested = False   # set True after operator chooses "wait"
_cutechess_debug = False      # set True via --cutechess-debug flag


# RAM disk helpers

IMDISK_URL = "https://sourceforge.net/projects/imdisk-toolkit/files/20250206/ImDiskTk-x64.zip/download"
RAMDISK_MARKER = "spsa_ramdisk"  # marker file to identify our RAM disks
_ramdisk_mount = ""  # resolved mount point for the active RAM disk
_ramdisk_owned = False  # True only if we created the drive this session


def _is_script(path: str) -> bool:
    """True if path is a Python script (no PyInstaller extraction needed)."""
    return path.endswith(".py")


def _needs_ramdisk(worker_config) -> bool:
    """Check if any engine is a non-script binary that benefits from a RAM disk."""
    if not worker_config.ramdisk:
        return False
    engines = [worker_config.engine]
    if worker_config.reference_engine:
        engines.append(worker_config.reference_engine)
    return any(not _is_script(e) for e in engines)


def _estimate_ramdisk_mb(worker_config) -> int:
    """Estimate RAM disk size in MB from engine binary sizes and concurrency.

    Each concurrent engine process extracts its own copy.  Gauntlet mode
    uses reference + theta_plus + theta_minus, all × concurrency.
    PyInstaller decompresses ~2x the binary size; we add a flat headroom.
    If ramdisk_size is set in config, use that directly.
    """
    if worker_config.ramdisk_size > 0:
        return worker_config.ramdisk_size
    ref_size = 0
    eng_size = 0
    if worker_config.reference_engine and not _is_script(worker_config.reference_engine):
        try:
            ref_size = os.path.getsize(worker_config.reference_engine)
        except OSError as e:
            logger.warning("Cannot stat reference engine %s: %s", worker_config.reference_engine, e)
    if not _is_script(worker_config.engine):
        try:
            eng_size = os.path.getsize(worker_config.engine)
        except OSError as e:
            logger.warning("Cannot stat engine %s: %s", worker_config.engine, e)
    conc = max(1, worker_config.concurrency)
    # reference × concurrency + engine × 2 (theta+, theta-) × concurrency, × 2 decompression ratio
    raw = (ref_size * conc + eng_size * 2 * conc) * 2
    mb = max(256, int(raw / (1024 * 1024)) + 128)
    return mb


def _has_imdisk() -> bool:
    """Check if imdisk is available on the system."""
    return shutil.which("imdisk") is not None


def _install_imdisk() -> bool:
    """Download and silently install ImDisk Toolkit. Returns True on success."""
    logger.info("Downloading ImDisk Toolkit from SourceForge...")
    tmp = None
    try:
        tmp = tempfile.mkdtemp(prefix="imdisk_")
        zip_path = os.path.join(tmp, "ImDiskTk-x64.zip")
        import ssl
        ctx = ssl.create_default_context()
        try:
            certifi = __import__("certifi")
            ctx.load_verify_locations(certifi.where())
        except (ImportError, ssl.SSLError):
            pass
        opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
        try:
            with opener.open(IMDISK_URL) as resp, open(zip_path, "wb") as out:
                out.write(resp.read())
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                raise
            logger.warning("SSL verification failed, retrying without verification")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            with opener.open(IMDISK_URL) as resp, open(zip_path, "wb") as out:
                out.write(resp.read())
        logger.info("Downloaded ImDisk to %s", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        logger.info("Extracted ImDisk to %s", tmp)
        install_bat = os.path.join(tmp, "install.bat")
        if not os.path.exists(install_bat):
            for root, dirs, files in os.walk(tmp):
                if "install.bat" in files:
                    install_bat = os.path.join(root, "install.bat")
                    break
        if not os.path.exists(install_bat):
            logger.error("install.bat not found in ImDisk archive")
            return False
        logger.info("Running ImDisk installer (UAC prompt may appear)...")
        result = subprocess.run([install_bat, "/fullsilent"], cwd=os.path.dirname(install_bat),
                                capture_output=True, text=True, timeout=120, shell=True)
        if result.returncode != 0:
            logger.error("ImDisk install failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return False
        logger.info("ImDisk installed successfully")
        return True
    except Exception as e:
        logger.error("Failed to install ImDisk: %s", e)
        return False
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _normalize_drive(mount_point: str) -> str:
    """Normalize a Windows drive letter to 'X:' form (no trailing separator)."""
    drive = mount_point.rstrip("\\/")
    if not drive.endswith(":"):
        drive += ":"
    return drive.upper()


def _is_our_ramdisk(drive_root: str) -> bool:
    """Check if a drive root has our marker file (from a previous run or crash)."""
    return os.path.exists(os.path.join(drive_root, RAMDISK_MARKER))


def _drive_letters():
    """Yield drive letters Z: down to D: as 'X:' strings."""
    for letter in "ZYXWVUTSRQPONMLKJIHGFED":
        yield letter + ":"


def _find_existing_ramdisk() -> str:
    """Scan drive letters for a RAM disk left over from a previous crash. Returns 'X:' or empty."""
    for drive in _drive_letters():
        drive_root = drive + os.sep
        if os.path.exists(drive_root) and _is_our_ramdisk(drive_root):
            logger.info("Found existing RAM disk from previous run: %s", drive)
            return drive
    return ""


def _find_free_drive() -> str:
    """Find an unused drive letter, searching from Z: down. Returns 'X:' or empty."""
    for drive in _drive_letters():
        if not os.path.exists(drive + os.sep):
            return drive
    return ""


def _write_marker(mount: str):
    """Write a marker file to identify our RAM disk after a crash. Raises on failure."""
    marker = os.path.join(mount, RAMDISK_MARKER)
    with open(marker, "w") as f:
        f.write("pid=%d\n" % os.getpid())


def setup_ramdisk(worker_config) -> str:
    """Set up a RAM disk for engine temp files. Returns the mount path, or empty string.

    On Windows the mount path is 'X:\\' (drive root); on Linux it's a /dev/shm subdirectory.
    Raises RuntimeError on failure (ramdisk=true means the user wants it).
    Returns empty string only when ramdisk is disabled or not needed.
    """
    global _ramdisk_mount, _ramdisk_owned
    if not _needs_ramdisk(worker_config):
        return ""

    if sys.platform == "win32":
        size_mb = _estimate_ramdisk_mb(worker_config)
        # Check for leftover from a previous crash
        drive = _find_existing_ramdisk()
        if drive:
            drive_root = drive + os.sep
            _ramdisk_mount = drive_root
            _write_marker(drive_root)
            return drive_root
        # Use configured drive letter or auto-find one
        if worker_config.ramdisk_drive:
            drive = _normalize_drive(worker_config.ramdisk_drive)
            drive_root = drive + os.sep
            if os.path.exists(drive_root):
                logger.info("Using pre-existing drive %s (externally managed)", drive)
                _ramdisk_mount = drive_root
                return drive_root
        else:
            drive = _find_free_drive()
            if not drive:
                raise RuntimeError("No free drive letters available for RAM disk")
        # Ensure imdisk is available
        if not _has_imdisk():
            if not worker_config.auto_install_imdisk:
                raise RuntimeError("ImDisk not found and auto_install_imdisk is disabled")
            logger.info("ImDisk not found, attempting auto-install")
            if not _install_imdisk():
                raise RuntimeError("ImDisk auto-install failed")
            if not _has_imdisk():
                raise RuntimeError("ImDisk still not on PATH after install")
        logger.info("Creating RAM disk: %s (%d MB)", drive, size_mb)
        # Create unformatted virtual disk
        result = subprocess.run(["imdisk", "-a", "-s", "%dM" % size_mb, "-m", drive], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError("imdisk failed (rc=%d): %s" % (result.returncode, detail))
        # Format requires elevation — RunAs only works with .exe, so wrap in cmd.exe
        fmt_cmd = 'Start-Process cmd.exe -ArgumentList "/c format %s /fs:ntfs /q /y" -Verb RunAs -Wait' % drive
        result = subprocess.run(["powershell", "-Command", fmt_cmd], capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            subprocess.run(["imdisk", "-D", "-m", drive], capture_output=True, text=True, timeout=10)
            raise RuntimeError("format failed (rc=%d): %s" % (result.returncode, detail))
        drive_root = drive + os.sep
        _write_marker(drive_root)
        logger.info("RAM disk created: %s (%d MB)", drive_root, size_mb)
        _ramdisk_mount = drive_root
        _ramdisk_owned = True
        return drive_root
    else:
        # Linux/macOS: use /dev/shm (already tmpfs, no root needed)
        if not os.path.isdir("/dev/shm"):
            raise RuntimeError("/dev/shm not available — cannot create RAM-backed temp")
        mount = "/dev/shm/spsa_temp"
        os.makedirs(mount, exist_ok=True)
        logger.info("Using /dev/shm for engine temp: %s", mount)
        _ramdisk_mount = mount
        return mount


def teardown_ramdisk(worker_config):
    """Remove the RAM disk created by setup_ramdisk."""
    global _ramdisk_mount, _ramdisk_owned
    if not _ramdisk_mount:
        return
    if sys.platform == "win32":
        drive = _ramdisk_mount.rstrip("\\/")
        if _ramdisk_owned:
            logger.info("Removing RAM disk %s", drive)
            try:
                subprocess.run(["imdisk", "-D", "-m", drive], capture_output=True, text=True, timeout=30)
                logger.info("RAM disk removed: %s", drive)
            except Exception as e:
                logger.warning("Failed to remove RAM disk %s: %s", drive, e)
        else:
            logger.info("RAM disk %s was not created by this worker, skipping teardown", drive)
    else:
        # /dev/shm: just clean up our subdirectory, no unmount needed
        logger.info("Cleaning up %s", _ramdisk_mount)
        shutil.rmtree(_ramdisk_mount, ignore_errors=True)
    _ramdisk_mount = ""
    _ramdisk_owned = False


def _make_temp_env(worker_config) -> dict:
    """Build an env dict with temp dir overridden for subprocess, or None if no override."""
    if not _ramdisk_mount:
        return None
    env = os.environ.copy()
    if sys.platform == "win32":
        env["TEMP"] = _ramdisk_mount
        env["TMP"] = _ramdisk_mount
    else:
        env["TMPDIR"] = _ramdisk_mount
    return env


def http_get(url: str, timeout: int) -> dict:
    """GET request, return parsed JSON."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, data: dict, retry_timeout: int = 0) -> dict:
    """POST JSON, return parsed JSON response.

    On connection errors, retries with exponential backoff for up to
    retry_timeout seconds (0 = no retry).
    """
    body = json.dumps(data).encode()
    deadline = time.monotonic() + retry_timeout
    delay = 1
    while True:
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except (ConnectionError, TimeoutError, urllib.error.URLError) as e:
            if time.monotonic() + delay > deadline:
                raise
            logger.warning("Connection error (%s), retrying in %ds...", e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def parse_cutechess_output(output: str) -> tuple:
    """
    Parse cutechess-cli output for game results.

    cutechess-cli prints a running score after each game:
        Score of engine1 vs engine2: W - L - D  [pct]  N

    We need the LAST occurrence (final tally after all games).

    Returns:
        (wins, losses, draws) from engine1's perspective.
        engine1 = theta_plus, engine2 = theta_minus.
    """
    pattern = r"Score of .+ vs .+: (\d+) - (\d+) - (\d+)"
    matches = re.findall(pattern, output)
    if not matches:
        raise ValueError(
            f"Could not parse cutechess-cli output:\n{output[-500:]}"
        )
    wins, losses, draws = matches[-1]
    return int(wins), int(losses), int(draws)


def parse_gauntlet_output(output: str, seed_name: str) -> dict:
    """Parse cutechess-cli gauntlet output from per-game result lines.

    Tournament mode prints "Finished game N (A vs B): result" per game
    (no cumulative "Score of" lines).  Tally results per opponent from
    the seed engine's perspective.

    Returns:
        {opponent_name: (seed_wins, seed_losses, draws)}
    """
    pattern = r"Finished game \d+ \((.+?) vs (.+?)\): (1-0|0-1|1/2-1/2)"
    games = re.findall(pattern, output)
    if not games:
        raise ValueError(
            f"Could not parse cutechess-cli gauntlet output:\n{output[-500:]}"
        )
    wins = {}
    losses = {}
    draws = {}
    for white, black, result in games:
        if white == seed_name:
            opponent = black
        elif black == seed_name:
            opponent = white
        else:
            continue
        wins.setdefault(opponent, 0)
        losses.setdefault(opponent, 0)
        draws.setdefault(opponent, 0)
        if result == "1/2-1/2":
            draws[opponent] += 1
        elif (result == "1-0" and white == seed_name) or \
             (result == "0-1" and black == seed_name):
            wins[opponent] += 1
        else:
            losses[opponent] += 1
    return {opp: (wins[opp], losses[opp], draws[opp]) for opp in wins}


def build_cutechess_command(worker_config: WorkerConfig,
                            tuning_config: dict,
                            engine1_params: dict,
                            engine1_name: str,
                            engine2_params: dict,
                            engine2_name: str,
                            num_games: int,
                            pgn_file: str,
                            engine1_cmd: str = "",
                            engine2_cmd: str = "",
                            engine3_params: dict = None,
                            engine3_name: str = "",
                            engine3_cmd: str = "") -> list:
    """Build the command line for cutechess-cli or fastchess.

    Detects which tool is configured via worker_config.cutechess_cli and
    adjusts syntax accordingly.  fastchess requires `-output format=cutechess`
    so that our existing output parsers work unchanged.

    Args:
        engine1_cmd: override engine binary for engine1.
        engine2_cmd: override engine binary for engine2.
        engine3_params: if not None, adds a 3rd engine and uses gauntlet mode.
        engine3_cmd: override engine binary for engine3.
        All cmd overrides default to worker_config.engine when empty.
    """
    engine_cmd = worker_config.engine
    protocol = tuning_config["engine"].get("protocol", "uci")
    fixed_options = tuning_config["engine"].get("fixed_options", {})

    book_file = worker_config.opening_book
    book_format = worker_config.book_format
    book_depth = worker_config.book_depth

    depth = tuning_config.get("depth")
    tc = tuning_config.get("time_control", "1+0.1")

    # Apply worker-local overrides
    cc_overrides = worker_config.cutechess_overrides
    if "depth" in cc_overrides:
        depth = cc_overrides["depth"]
        logger.info("Worker override: depth=%s", depth)
    if "tc" in cc_overrides:
        tc = cc_overrides["tc"]
        logger.info("Worker override: tc=%s", tc)

    # Detect fast-chess vs cutechess-cli from the binary name
    tool_name = Path(worker_config.cutechess_cli).stem.lower()
    is_fastchess = "fastchess" in tool_name

    # Get parameter overrides from worker config (exclude _comment)
    param_overrides = {k: v for k, v in worker_config.parameter_overrides.items()
                       if not k.startswith('_')}

    # Build engine option strings
    def option_args(params: dict) -> list:
        args = []
        # Fixed options first
        for name, val in fixed_options.items():
            if isinstance(val, bool):
                val = "true" if val else "false"
            args.append(f"option.{name}={val}")
        # Tunable params
        for name, val in params.items():
            args.append(f"option.{name}={val}")
        # Parameter overrides (applied last, highest priority)
        for name, val in param_overrides.items():
            if isinstance(val, bool):
                val = "true" if val else "false"
            args.append(f"option.{name}={val}")
        return args

    cmd = [worker_config.cutechess_cli]

    # Engine 1
    e1_cmd = engine1_cmd or engine_cmd
    cmd += ["-engine", f"cmd={e1_cmd}", f"name={engine1_name}"]
    cmd += option_args(engine1_params)

    # Engine 2
    e2_cmd = engine2_cmd or engine_cmd
    cmd += ["-engine", f"cmd={e2_cmd}", f"name={engine2_name}"]
    cmd += option_args(engine2_params)

    # Engine 3 (gauntlet mode: engine1 is the seed, plays engines 2 and 3)
    if engine3_params is not None:
        e3_cmd = engine3_cmd or engine_cmd
        cmd += ["-engine", f"cmd={e3_cmd}", f"name={engine3_name}"]
        cmd += option_args(engine3_params)
        cmd += ["-tournament", "gauntlet"]

    # Common settings
    cmd += ["-each", f"proto={protocol}"]

    if depth is not None:
        cmd += [f"depth={depth}"]
        cmd += ["tc=inf"]
    else:
        cmd += [f"tc={tc}"]

    # Opening book
    if book_file:
        if not book_format:
            ext = Path(book_file).suffix.lower().lstrip(".")
            book_format = ext if ext in ("pgn", "epd") else "pgn"
        cmd += ["-openings", f"file={book_file}", f"format={book_format}"]
        if book_depth:
            cmd += [f"plies={book_depth}"]
        cmd += ["order=random"]
        if not is_fastchess:
            cmd += ["policy=round"]

    # Number of games: each round plays 2 games (color swap) per pairing.
    # 2-engine: rounds = games/2.  Gauntlet (2 pairings): rounds = games/4.
    games_per_round = 4 if engine3_params is not None else 2
    num_rounds = max(1, num_games // games_per_round)
    assert(num_games)
    assert(num_rounds)

    cmd += ["-rounds", str(num_rounds)]
    cmd += ["-games", "2"]  # 2 games per round (color swap)
    cmd += ["-repeat"]

    # Concurrency
    if worker_config.concurrency > 1:
        cmd += ["-concurrency", str(worker_config.concurrency)]
        if is_fastchess:
            cmd += ["-force-concurrency"]

    # PGN output
    if is_fastchess:
        cmd += ["-pgnout", f"file={pgn_file}"]
        cmd += ["-output", "format=cutechess"]
        cmd += ["-ratinginterval", "0"]
        cmd += ["-scoreinterval", "0"]
    else:
        cmd += ["-pgnout", pgn_file]

    # Debug: log all engine I/O
    if _cutechess_debug:
        if is_fastchess:
            cmd += ["-log", "engine=true"]
        else:
            cmd += ["-debug"]

    return cmd


def _handle_game_interrupt(proc):
    """Handle Ctrl+C while cutechess-cli games are running.

    Prompts the operator (whether or not the child is still alive):
        [w]ait — let games finish, report results, then stop
        [s]top — kill immediately
        Enter  — dismiss, keep working (default)
    A second Ctrl+C during the wait force-kills the child.
    """
    global _shutdown_requested

    if _shutdown_requested:
        logger.info("Force stop.")
        proc.kill()
        proc.wait()
        raise KeyboardInterrupt

    try:
        answer = input("\nGames in progress. [w]ait and stop | [s]top now | [Enter] to dismiss ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer == "w":
        _shutdown_requested = True
        logger.info("Waiting for current games to finish (Ctrl+C again to force stop)...")
    elif answer == "s":
        proc.kill()
        proc.wait()
        raise KeyboardInterrupt
    else:
        logger.info("Continuing...")


def _run_cutechess(cmd, work, worker_config, tuning_config):
    """Run a cutechess-cli process and return its stdout, or None if cancelled.

    Handles process lifecycle: logging, pipe draining, polling with
    timeout/validate/interrupt, return-code checks, error-line scanning.
    """
    # Summarize command for console when many tunable params make it noisy
    opt_count = sum(1 for a in cmd if a.startswith("option."))
    if opt_count > 10:
        brief = [a for a in cmd if not a.startswith("option.")]
        logger.info("Running: %s [%d engine options]", " ".join(brief), opt_count)
    else:
        logger.info("Running: %s", " ".join(cmd))

    # Isolate child process from Ctrl+C so we can offer a graceful stop.
    # Windows: CREATE_NEW_PROCESS_GROUP prevents CTRL_C_EVENT propagation.
    # Unix: start_new_session puts the child in its own session.
    global _current_process
    popen_kwargs = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    temp_env = _make_temp_env(worker_config)
    if temp_env:
        popen_kwargs["env"] = temp_env
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen_kwargs)
    _current_process = proc

    # Drain pipes in background threads to prevent deadlock while the
    # main thread polls proc.poll() (which directly sets returncode).
    # Output is streamed to cutechess_last.log for post-mortem inspection.
    stdout_buf = []
    stderr_buf = []
    log_path = Path(worker_config.log_file)
    cc_log_name = log_path.stem.replace("worker", "cutechess_last") + ".log"
    try:
        _cc_log = open(log_path.parent / cc_log_name, "w", buffering=1)
        _cc_log.write("=== chunk %s, iteration %d ===\n" % (work.chunk_id, work.iteration))
    except OSError:
        _cc_log = None

    def _drain(pipe, buf):
        lines = []
        for line in pipe:
            lines.append(line)
            if _cc_log:
                _cc_log.write(line)
        buf.append("".join(lines))

    out_t = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf), daemon=True)
    err_t = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf), daemon=True)
    out_t.start()
    err_t.start()

    # Poll for exit — main thread stays responsive to Ctrl+C.
    # The inner try re-enters after "ignore" so the worker keeps going.
    validate_interval = tuning_config.get("validate_interval", 0)
    base_url = worker_config.coordinator.rstrip("/")
    worker_name = worker_config.name or platform.node()
    deadline = time.monotonic() + GAME_TIMEOUT
    last_validate = time.monotonic()
    try:
        while proc.poll() is None:
            now = time.monotonic()
            if now > deadline:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd[0], GAME_TIMEOUT)
            # Periodic chunk-validity check
            if validate_interval and now - last_validate >= validate_interval:
                last_validate = now
                try:
                    resp = http_get("%s/validate?worker=%s&chunk=%s" % (base_url, worker_name, work.chunk_id), VALIDATE_TIMEOUT)
                    logger.debug("Validate chunk %s: %s", work.chunk_id, resp)
                    if not resp.get("valid", True):
                        logger.warning("Chunk %s cancelled by coordinator, aborting games", work.chunk_id)
                        proc.kill()
                        proc.wait()
                        return None
                except Exception as e:
                    logger.debug("Validate check failed: %s", e)
            try:
                time.sleep(POLL_INTERVAL)
            except KeyboardInterrupt:
                _handle_game_interrupt(proc)
    finally:
        _current_process = None

    # Collect pipe output (process is dead; readers will see EOF shortly)
    out_t.join(timeout=PIPE_DRAIN_TIMEOUT)
    err_t.join(timeout=PIPE_DRAIN_TIMEOUT)
    if _cc_log:
        try:
            _cc_log.close()
        except OSError:
            pass

    if proc.returncode != 0:
        rc = proc.returncode
        logger.error("cutechess-cli failed (rc=%s)", hex(rc))
        logger.error("stdout (last 1000 chars): %s", (stdout_buf[0][-1000:] if stdout_buf else "") or "(empty)")
        logger.error("stderr (last 1000 chars): %s", (stderr_buf[0][-1000:] if stderr_buf else "") or "(empty)")
        # Windows STATUS_ACCESS_VIOLATION (subprocess returns signed or unsigned)
        if rc & 0xFFFFFFFF == 0xc0000005:
            raise RetryableError(f"cutechess-cli access violation ({hex(rc)})")
        raise RetryableError(f"cutechess-cli exited with code {hex(rc)}")

    output = stdout_buf[0] if stdout_buf else ""
    stderr_output = stderr_buf[0] if stderr_buf else ""
    if stderr_output:
        logger.warning("cutechess-cli stderr: %s", stderr_output.strip()[-1000:])
        # "Cannot start engine" / "Cannot execute command" = config error, not transient
        if "cannot start engine" in stderr_output.lower() or "cannot execute command" in stderr_output.lower():
            raise RuntimeError("Engine failed to start (check paths/permissions):\n" + stderr_output.strip()[-500:])

    # Log all output lines that mention errors or crashes
    for line in output.splitlines():
        if any(kw in line.lower() for kw in ("abandoned", "error", "crash", "disconnect", "timeout", "illegal", "terminated", "forfeit")):
            logger.warning("cutechess: %s", line.strip())

    return output


def _execute_match(cmd, expected_games, work, worker_config, tuning_config):
    """
    Run a single cutechess-cli match and return (wins, draws, losses) or None.

    This is the standard-mode wrapper around _run_cutechess().
    """
    output = _run_cutechess(cmd, work, worker_config, tuning_config)
    if output is None:
        return None

    # Log score lines for diagnostics
    score_lines = re.findall(r"Score of .+", output)
    logger.info("cutechess-cli reported %d score line(s)", len(score_lines))
    for line in score_lines[:-1]:
        logger.debug("  %s", line)
    if score_lines:
        logger.info("Final: %s", score_lines[-1])

    wins, losses, draws = parse_cutechess_output(output)
    total = wins + losses + draws

    if total == 0:
        started = any("started game" in line.lower() for line in output.splitlines())
        if not started:
            raise RuntimeError(
                "cutechess-cli started no games — check engine configuration\n"
                + (output[-500:] or "(no output)")
            )
        raise RetryableError("No games were played")

    # Abort if too many games failed — results would be noise
    min_completion = 0.5
    if total < expected_games * min_completion:
        logger.error(
            "Only %d/%d games completed (W=%d L=%d D=%d) — aborting chunk",
            total, expected_games, wins, losses, draws,
        )
        raise RetryableError(
            f"Only {total}/{expected_games} games completed "
            f"({total/expected_games:.0%}), minimum is {min_completion:.0%}"
        )

    if total != expected_games:
        logger.warning(
            "Expected %d games but got %d (W=%d L=%d D=%d)",
            expected_games, total, wins, losses, draws,
        )

    logger.info("Results: W=%d D=%d L=%d (%d games)", wins, draws, losses, total)

    return wins, draws, losses


def run_games(worker_config: WorkerConfig, tuning_config: dict, work: WorkItem) -> tuple:
    """Run cutechess-cli match(es) and return results.

    Standard mode: returns (wins, draws, losses) for theta_plus vs theta_minus.
    Reference mode: returns (plus_W/D/L, minus_W/D/L) for theta+/- vs reference.
    Returns None if cancelled.
    """
    games_dir = Path(worker_config.games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)
    pgn_file = str(games_dir / f"games-{work.iteration}.pgn").replace("\\", "/")

    # KLUDGE: clean up orphaned PyInstaller _MEI* dirs on the RAM disk.
    # Crashed/timed-out engines leave extracted dirs behind, filling the disk.
    # Windows only: file locks protect dirs still in use by colocated workers.
    if _ramdisk_mount and sys.platform == "win32":
        for entry in os.scandir(_ramdisk_mount):
            if entry.is_dir() and entry.name.startswith("_MEI"):
                shutil.rmtree(entry.path, ignore_errors=True)

    ref_engine = worker_config.reference_engine

    if ref_engine:
        # Reference mode: gauntlet with reference as seed vs theta+, theta-
        half = work.num_games // 2
        cmd = build_cutechess_command(
            worker_config, tuning_config,
            engine1_params={}, engine1_name="reference",
            engine2_params=work.theta_plus, engine2_name="theta_plus",
            num_games=work.num_games, pgn_file=pgn_file,
            engine1_cmd=ref_engine,
            engine3_params=work.theta_minus, engine3_name="theta_minus",
        )

        output = _run_cutechess(cmd, work, worker_config, tuning_config)
        if output is None:
            return None

        # Log ranking table from gauntlet output
        in_rank = False
        for line in output.splitlines():
            if line.startswith("Rank "):
                in_rank = True
            if in_rank:
                if not line.strip():
                    break
                logger.info("  %s", line.rstrip())

        # Parse per-pairing results (from reference's perspective)
        try:
            pairing_results = parse_gauntlet_output(output, "reference")
        except ValueError:
            raise RetryableError("Could not parse gauntlet output (engine init failure?):\n" + output[-500:])
        for name in ("theta_plus", "theta_minus"):
            if name not in pairing_results:
                raise RetryableError(f"Missing gauntlet pairing for {name} in output:\n{output[-500:]}")

        # Swap perspective: reference W/L → theta W/L
        ref_pw, ref_pl, ref_pd = pairing_results["theta_plus"]
        ref_mw, ref_ml, ref_md = pairing_results["theta_minus"]
        plus_wins, plus_draws, plus_losses = ref_pl, ref_pd, ref_pw
        minus_wins, minus_draws, minus_losses = ref_ml, ref_md, ref_mw

        # Validate game counts per side
        min_completion = 0.5
        for side, w, d, l in [("theta+", plus_wins, plus_draws, plus_losses),
                               ("theta-", minus_wins, minus_draws, minus_losses)]:
            total = w + d + l
            if total == 0:
                raise RetryableError(f"No games completed for {side}")
            if total < half * min_completion:
                raise RetryableError(
                    f"Only {total}/{half} games completed for {side} "
                    f"({total/half:.0%}), minimum is {min_completion:.0%}"
                )
            if total != half:
                logger.warning("Expected %d games for %s but got %d", half, side, total)
            logger.info("Results [%s]: W=%d D=%d L=%d (%d games)", side, w, d, l, total)

        return plus_wins, plus_draws, plus_losses, minus_wins, minus_draws, minus_losses

    else:
        # Standard mode: theta_plus vs theta_minus
        cmd = build_cutechess_command(
            worker_config, tuning_config,
            engine1_params=work.theta_plus, engine1_name="theta_plus",
            engine2_params=work.theta_minus, engine2_name="theta_minus",
            num_games=work.num_games, pgn_file=pgn_file,
        )
        return _execute_match(cmd, work.num_games, work, worker_config, tuning_config)


def worker_loop(worker_config: WorkerConfig):
    """Main worker loop: poll for work, run games, report results."""
    base_url = worker_config.coordinator.rstrip("/")

    # Fetch tuning config from coordinator
    def fetch_config():
        nonlocal tuning_config, server_start, default_retry
        tuning_config = http_get(f"{base_url}/config", HTTP_TIMEOUT)
        server_start = tuning_config.get("server_start", 0)
        default_retry = tuning_config.get("retry_after", 5)

    tuning_config, server_start, default_retry = None, 0, 5
    logger.info("Connecting to coordinator at %s", base_url)
    fetch_config()
    logger.info("Received tuning config: %d parameters", len(tuning_config.get("parameters", {})))
    if worker_config.reference_engine:
        logger.info("Reference mode: theta+/- vs %s", worker_config.reference_engine)
    retry_timeout = worker_config.http_retry_timeout

    worker_name = worker_config.name or platform.node()

    # Collect cutechess overrides to send to coordinator (for timeout estimation)
    cc_overrides = {k: v for k, v in worker_config.cutechess_overrides.items()
                    if not k.startswith('_')}

    # Compute effective chunk size cap from max_rounds_per_chunk and concurrency
    chunk_size_cap = worker_config.max_chunk_size  # hard cap (0 = unlimited)
    if worker_config.max_rounds_per_chunk > 0:
        rounds_cap = worker_config.max_rounds_per_chunk * worker_config.concurrency * 2
        chunk_size_cap = min(chunk_size_cap, rounds_cap) if chunk_size_cap > 0 else rounds_cap
    logger.info("Chunk size cap: %d games (%d rounds x %d concurrency)",
                chunk_size_cap, worker_config.max_rounds_per_chunk, worker_config.concurrency)

    consecutive_errors = 0
    max_retries = worker_config.max_retries

    while True:
        try:
            # Request work
            work_request = {
                "chunk_size": chunk_size_cap,
                "worker": worker_name,
                "server_start": server_start,
            }
            if cc_overrides:
                work_request["cutechess_overrides"] = cc_overrides
            response = http_post(f"{base_url}/work", work_request, retry_timeout)

            # Detect coordinator restart and re-fetch config
            if response.get("status") == "config_changed":
                logger.warning("Coordinator restarted, re-fetching config")
                fetch_config()
                continue

            status = response.get("status")
            if status == "done":
                logger.info("Tuning complete, shutting down")
                break
            elif status == "retry":
                delay = response.get("retry_after", default_retry)
                logger.debug("No work available, retrying in %ds", delay)
                time.sleep(delay)
                continue

            # We got a work assignment
            work = WorkItem.from_dict(response)
            logger.info("Got work: iteration %d, %d games", work.iteration, work.num_games)

            # Run the games
            game_result = run_games(worker_config, tuning_config, work)
            if game_result is None:
                if _shutdown_requested:
                    break
                continue  # chunk cancelled, request new work

            # Report results (PGNs saved locally by cutechess-cli)
            result = {
                "iteration": work.iteration,
                "chunk_id": work.chunk_id,
                "worker": worker_name,
                "shutting_down": _shutdown_requested,
            }
            if len(game_result) == 6:
                # Reference mode: per-side scores, coordinator derives aggregates
                pw, pd, pl, mw, md, ml = game_result
                result["num_games"] = pw + pd + pl + mw + md + ml
                result["reference_mode"] = True
                result["plus_wins"] = pw
                result["plus_draws"] = pd
                result["plus_losses"] = pl
                result["minus_wins"] = mw
                result["minus_draws"] = md
                result["minus_losses"] = ml
            else:
                # Standard mode
                wins, draws, losses = game_result
                result["wins"] = wins
                result["draws"] = draws
                result["losses"] = losses
                result["num_games"] = wins + draws + losses
            resp = http_post(f"{base_url}/result", result, retry_timeout)
            logger.info("Result submitted: %s", resp.get("status"))
            consecutive_errors = 0

            if _shutdown_requested:
                logger.info("Shutdown requested, stopping after result reported.")
                break

        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down.")
            break
        except urllib.error.URLError as e:
            logger.warning("Connection error: %s, retrying in 5s", e)
            time.sleep(5)
        except (subprocess.TimeoutExpired, RetryableError) as e:
            consecutive_errors += 1
            msg = "cutechess-cli timed out" if isinstance(e, subprocess.TimeoutExpired) else str(e)
            logger.error("%s (%d/%s)", msg, consecutive_errors, max_retries or "inf")
            if max_retries and consecutive_errors >= max_retries:
                logger.error("Too many consecutive errors, terminating.")
                break
        except Exception as e:
            logger.exception("Terminating.")
            break

def setup_logging(log_file: str, debug: bool, rotate: bool):
    """Configure logging to file and stdout."""
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rotate:
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when="midnight", backupCount=30,
        )
    else:
        file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(description="SPSA Tuning Worker")
    parser.add_argument("-c", "--config", required=True, help="Path to worker config JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--cutechess-debug", action="store_true", help="Pass -debug to cutechess-cli (log engine I/O)")
    parser.add_argument("--clean", action="store_true", help="Wipe log file before starting")
    args = parser.parse_args()

    config = WorkerConfig.from_json(args.config)

    # --clean: remove log file and rotated copies (keep games)
    if args.clean:
        log_path = Path(config.log_file)
        for f in log_path.parent.glob(log_path.name + "*"):
            f.unlink()

    global _cutechess_debug
    _cutechess_debug = args.cutechess_debug
    setup_logging(config.log_file, debug=args.debug, rotate=config.log_rotation)

    logger.info("Starting worker")
    logger.info("Coordinator: %s", config.coordinator)
    logger.info("Engine: %s", config.engine)
    logger.info("Concurrency: %d", config.concurrency)
    tool_name = Path(config.cutechess_cli).stem.lower()
    tool_label = "fastchess" if "fastchess" in tool_name else "cutechess-cli"
    logger.info("%s: %s", tool_label, config.cutechess_cli)
    logger.info("Games dir: %s", config.games_dir)
    if config.opening_book:
        fmt = config.book_format or Path(config.opening_book).suffix.lower().lstrip(".") or "pgn"
        logger.info("Opening book: %s (%s)", config.opening_book, fmt)

    ramdisk_path = setup_ramdisk(config)
    if ramdisk_path:
        logger.info("Engine temp dir: %s", ramdisk_path)
    try:
        worker_loop(config)
    finally:
        teardown_ramdisk(config)


if __name__ == "__main__":
    main()
