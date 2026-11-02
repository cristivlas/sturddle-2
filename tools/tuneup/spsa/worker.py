#!/usr/bin/env python3
"""
SPSA tuning worker: gets games from the coordinator, plays them with cutechess-cli or fastchess.

Usage:
    python worker.py -c worker.json
"""

import argparse
import ctypes
import math
import json
import logging
import logging.handlers
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import WorkerConfig, WorkItem
from ramdisk import (
    estimate_ramdisk_mb, create_ramdisk, remove_ramdisk,
    find_existing_ramdisk, normalize_drive, write_marker,
    find_free_drive, is_script,
)

logger = logging.getLogger("worker")


class RetryableError(Exception):
    """Chunk failed; ask for new work."""


GAME_TIMEOUT = 3600        # max seconds for a cutechess-cli run
PIPE_DRAIN_TIMEOUT = 30    # seconds to wait for pipe readers after exit
POLL_INTERVAL = 0.5        # seconds between process-alive checks
HTTP_TIMEOUT = 30          # seconds for general HTTP requests
VALIDATE_TIMEOUT = 3       # seconds for chunk-validity HTTP checks
SHUTDOWN_NOTIFY_TIMEOUT = 5  # seconds for the best-effort shutdown notify

_current_process = None
_shutdown_requested = False   # operator chose "wait", or SIGTERM
_cutechess_debug = False
_managed_mode = False         # launcher owns the Ctrl+C prompt
_tool_label = "cutechess-cli"

_ramdisk_mount = ""
_ramdisk_owned = False  # created by this worker


def _get_numa_node() -> int:
    """Return the NUMA node of the current process, or -1 if unavailable."""
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        if libnuma.numa_available() < 0:
            return -1
        cpu = os.sched_getaffinity(0)
        first_cpu = min(cpu)
        return libnuma.numa_node_of_cpu(first_cpu)
    except (OSError, AttributeError):
        return -1


_numactl_prefix = None


def _wrap_numactl(cmd: list) -> list:
    """Pin cmd to this worker's NUMA node; no-op without NUMA or numactl."""
    global _numactl_prefix
    if _numactl_prefix is None:
        node = _get_numa_node()
        if node < 0 or not shutil.which("numactl"):
            _numactl_prefix = []
        else:
            _numactl_prefix = ["numactl", f"--cpunodebind={node}", f"--membind={node}"]
    return _numactl_prefix + cmd


def _needs_ramdisk(worker_config):
    if not worker_config.ramdisk:
        return False
    engines = [worker_config.engine]
    if worker_config.reference_engine:
        engines.append(worker_config.reference_engine)
    return any(not is_script(e) for e in engines)


def setup_ramdisk(worker_config) -> str:
    """RAM disk for engine temp files: drive root on Windows, /dev/shm dir on Linux, "" if not needed."""
    global _ramdisk_mount, _ramdisk_owned
    if not _needs_ramdisk(worker_config):
        return ""

    if sys.platform == "win32":
        size_mb = worker_config.ramdisk_size
        if size_mb <= 0:
            engine2 = worker_config.reference_engine or worker_config.engine
            size_mb = estimate_ramdisk_mb(worker_config.engine, engine2, worker_config.concurrency, worker_config.ramdisk_decompression)
        # left over from a crash
        drive = find_existing_ramdisk()
        if drive:
            drive_root = drive + os.sep
            _ramdisk_mount = drive_root
            write_marker(drive_root)
            return drive_root
        if worker_config.ramdisk_drive:
            drive = normalize_drive(worker_config.ramdisk_drive)
            drive_root = drive + os.sep
            if os.path.exists(drive_root):
                logger.info("Using pre-existing drive %s (externally managed)", drive)
                _ramdisk_mount = drive_root
                return drive_root
        else:
            drive = find_free_drive()
            if not drive:
                raise RuntimeError("No free drive letters available for RAM disk")
        drive_root = create_ramdisk(size_mb, drive, auto_install=worker_config.auto_install_imdisk)
        _ramdisk_mount = drive_root
        _ramdisk_owned = True
        return drive_root
    else:
        # per-worker dir so co-located workers don't race on teardown
        if not os.path.isdir("/dev/shm"):
            raise RuntimeError("/dev/shm not available")
        suffix = worker_config.name or platform.node()
        mount = f"/dev/shm/spsa_temp_{suffix}"
        os.makedirs(mount, exist_ok=True)
        logger.info("Using /dev/shm for engine temp: %s", mount)
        _ramdisk_mount = mount
        return mount


def teardown_ramdisk(worker_config):
    global _ramdisk_mount, _ramdisk_owned
    if not _ramdisk_mount:
        return
    if sys.platform == "win32":
        if _ramdisk_owned:
            remove_ramdisk(_ramdisk_mount.rstrip("\\/"))
        else:
            logger.info("RAM disk %s was not created by this worker, skipping teardown", _ramdisk_mount)
    else:
        logger.info("Cleaning up %s", _ramdisk_mount)
        shutil.rmtree(_ramdisk_mount, ignore_errors=True)
    _ramdisk_mount = ""
    _ramdisk_owned = False


def _make_temp_env(worker_config) -> dict:
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
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def http_post(url: str, data: dict, retry_timeout: int = 0, timeout: int = HTTP_TIMEOUT) -> dict:
    """On connection errors, retry with backoff for up to retry_timeout seconds."""
    body = json.dumps(data).encode()
    deadline = time.monotonic() + retry_timeout
    delay = 1
    while True:
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (ConnectionError, TimeoutError, urllib.error.URLError) as e:
            if time.monotonic() + delay > deadline:
                raise
            logger.warning("Connection error (%s), retrying in %ds...", e, delay)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def parse_cutechess_output(output: str) -> tuple:
    """(wins, losses, draws) for engine1 from the last "Score of" line."""
    pattern = r"Score of .+ vs .+: (\d+) - (\d+) - (\d+)"
    matches = re.findall(pattern, output)
    if not matches:
        raise ValueError(
            f"Could not parse cutechess-cli output:\n{output[-500:]}"
        )
    wins, losses, draws = matches[-1]
    return int(wins), int(losses), int(draws)


def parse_gauntlet_output(output: str, seed_name: str) -> dict:
    """{opponent: (seed_wins, seed_losses, draws)} from "Finished game" lines; gauntlets print no "Score of"."""
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
    """cutechess-cli or fastchess command; engine3_params adds a third engine in gauntlet mode."""
    engine_cmd = worker_config.engine
    protocol = tuning_config["engine"].get("protocol", "uci")
    fixed_options = tuning_config["engine"].get("fixed_options", {})

    book_file = worker_config.opening_book
    book_format = worker_config.book_format
    book_depth = worker_config.book_depth

    depth = tuning_config.get("depth")
    tc = tuning_config.get("time_control", "1+0.1")

    cc_overrides = worker_config.cutechess_overrides
    if "depth" in cc_overrides:
        depth = cc_overrides["depth"]
        logger.info("Worker override: depth=%s", depth)
    if "tc" in cc_overrides:
        tc = cc_overrides["tc"]
        logger.info("Worker override: tc=%s", tc)

    tool_name = Path(worker_config.cutechess_cli).stem.lower()
    is_fastchess = "fastchess" in tool_name

    param_overrides = {k: v for k, v in worker_config.parameter_overrides.items() if not k.startswith('_')}

    def option_args(params: dict) -> list:
        args = []
        for name, val in fixed_options.items():
            if isinstance(val, bool):
                val = "true" if val else "false"
            args.append(f"option.{name}={val}")
        for name, val in params.items():
            args.append(f"option.{name}={val}")
        # last, so they win
        for name, val in param_overrides.items():
            if isinstance(val, bool):
                val = "true" if val else "false"
            args.append(f"option.{name}={val}")
        return args

    cmd = [worker_config.cutechess_cli]

    e1_cmd = engine1_cmd or engine_cmd
    cmd += ["-engine", f"cmd={e1_cmd}", f"name={engine1_name}"]
    cmd += option_args(engine1_params)

    e2_cmd = engine2_cmd or engine_cmd
    cmd += ["-engine", f"cmd={e2_cmd}", f"name={engine2_name}"]
    cmd += option_args(engine2_params)

    # gauntlet: engine1 plays both
    if engine3_params is not None:
        e3_cmd = engine3_cmd or engine_cmd
        cmd += ["-engine", f"cmd={e3_cmd}", f"name={engine3_name}"]
        cmd += option_args(engine3_params)
        cmd += ["-tournament", "gauntlet"]

    cmd += ["-each", f"proto={protocol}"]

    if depth is not None:
        cmd += [f"depth={depth}"]
        cmd += ["tc=inf"]
    else:
        cmd += [f"tc={tc}"]

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

    # 2 games (color swap) per pairing per round
    games_per_round = 4 if engine3_params is not None else 2
    num_rounds = max(1, num_games // games_per_round)
    assert(num_games)
    assert(num_rounds)

    cmd += ["-rounds", str(num_rounds)]
    cmd += ["-games", "2"]
    cmd += ["-repeat"]

    if worker_config.concurrency > 1:
        cmd += ["-concurrency", str(worker_config.concurrency)]
        if is_fastchess:
            cmd += ["-force-concurrency"]

    if is_fastchess:
        cmd += ["-pgnout", f"file={pgn_file}"]
        cmd += ["-output", "format=cutechess"]
        cmd += ["-ratinginterval", "0"]
        cmd += ["-scoreinterval", "0"]
    else:
        cmd += ["-pgnout", pgn_file]

    if tuning_config.get("auto_resign", True):
        cmd += ["-resign", f"movecount={tuning_config.get('resign_movecount', 3)}", f"score={tuning_config.get('resign_score', 700)}"]
        cmd += ["-draw", f"movenumber={tuning_config.get('draw_movenumber', 40)}", f"movecount={tuning_config.get('draw_movecount', 8)}", f"score={tuning_config.get('draw_score', 10)}"]

    # TB adjudication; cutechess-cli's TB options differ
    syzygy_path = param_overrides.get("SyzygyPath")
    if is_fastchess and syzygy_path:
        cmd += ["-tb", str(syzygy_path)]

    if _cutechess_debug:
        if is_fastchess:
            cmd += ["-log", "engine=true"]
        else:
            cmd += ["-debug"]

    return cmd


def _handle_game_interrupt(proc):
    """Ctrl+C during games: wait, stop or dismiss; a second Ctrl+C while waiting kills. --managed always kills."""
    global _shutdown_requested

    if _managed_mode:
        logger.info("SIGINT under --managed: killing %s", _tool_label)
        proc.kill()
        proc.wait()
        raise KeyboardInterrupt

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
    """Run the match and return stdout, or None if the coordinator cancelled the chunk."""
    opt_count = sum(1 for a in cmd if a.startswith("option."))
    if opt_count > 10:
        brief = [a for a in cmd if not a.startswith("option.")]
        logger.info("Running: %s [%d engine options]", " ".join(brief), opt_count)
    else:
        logger.info("Running: %s", " ".join(cmd))

    cmd = _wrap_numactl(cmd)

    # keep Ctrl+C away from the child so we can offer a graceful stop
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

    # drain pipes in threads so the child never blocks; also tee to a log
    stdout_buf = []
    stderr_buf = []
    log_path = Path(worker_config.log_file)
    cc_log_name = f"{log_path.stem}-{_tool_label}.log"
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

    # poll so the main thread stays responsive to Ctrl+C
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

    out_t.join(timeout=PIPE_DRAIN_TIMEOUT)
    err_t.join(timeout=PIPE_DRAIN_TIMEOUT)
    if _cc_log:
        try:
            _cc_log.close()
        except OSError:
            pass

    if proc.returncode != 0:
        rc = proc.returncode
        logger.error("%s failed (rc=%s)", _tool_label, hex(rc))
        logger.error("stdout (last 1000 chars): %s", (stdout_buf[0][-1000:] if stdout_buf else "") or "(empty)")
        logger.error("stderr (last 1000 chars): %s", (stderr_buf[0][-1000:] if stderr_buf else "") or "(empty)")
        # STATUS_ACCESS_VIOLATION, signed or unsigned
        if rc & 0xFFFFFFFF == 0xc0000005:
            raise RetryableError(f"{_tool_label} access violation ({hex(rc)})")
        raise RetryableError(f"{_tool_label} exited with code {hex(rc)}")

    output = stdout_buf[0] if stdout_buf else ""
    stderr_output = stderr_buf[0] if stderr_buf else ""
    if stderr_output:
        logger.warning("%s stderr: %s", _tool_label, stderr_output.strip()[-1000:])
        # config error, not transient
        if "cannot start engine" in stderr_output.lower() or "cannot execute command" in stderr_output.lower():
            raise RuntimeError("Engine failed to start (check paths/permissions):\n" + stderr_output.strip()[-500:])

    for line in output.splitlines():
        if any(kw in line.lower() for kw in ("abandoned", "error", "crash", "disconnect", "timeout", "illegal", "terminated", "forfeit", "loses on time")):
            logger.warning("%s: %s", _tool_label, line.strip())

    return output


def _check_forfeits(output, expected_games, max_forfeit_pct):
    """Raise RetryableError on too many time forfeits ("loses on time" or "Draw by timeout")."""
    if not max_forfeit_pct or max_forfeit_pct <= 0:
        return
    forfeit_count = sum(1 for line in output.splitlines() if "loses on time" in line.lower() or "draw by timeout" in line.lower())
    if forfeit_count == 0:
        return
    threshold = math.ceil(expected_games * max_forfeit_pct)
    logger.warning("Forfeits: %d/%d (threshold %d at %.0f%%)", forfeit_count, expected_games, threshold, max_forfeit_pct * 100)
    if forfeit_count >= threshold:
        # TODO: the coordinator treats the next /work as a crash-reconnect and resets the speed EWMA
        raise RetryableError(f"{forfeit_count}/{expected_games} games forfeited ({forfeit_count/expected_games:.0%}), max is {max_forfeit_pct:.0%}")


def _execute_match(cmd, expected_games, work, worker_config, tuning_config):
    """Standard mode: (wins, draws, losses) for theta_plus, or None if cancelled."""
    output = _run_cutechess(cmd, work, worker_config, tuning_config)
    if output is None:
        return None

    score_lines = re.findall(r"Score of .+", output)
    logger.info("%s reported %d score line(s)", _tool_label, len(score_lines))
    for line in score_lines[:-1]:
        logger.debug("  %s", line)
    if score_lines:
        logger.info("Final: %s", score_lines[-1])

    wins, losses, draws = parse_cutechess_output(output)
    total = wins + losses + draws

    if total == 0:
        started = any("started game" in line.lower() for line in output.splitlines())
        if not started:
            raise RuntimeError(f"{_tool_label} started no games, check engine configuration\n" + (output[-500:] or "(no output)"))
        raise RetryableError("No games were played")

    min_completion = 1 - worker_config.max_forfeit_pct
    if total < expected_games * min_completion:
        logger.error(
            "Only %d/%d games completed (W=%d L=%d D=%d), aborting chunk",
            total, expected_games, wins, losses, draws,
        )
        raise RetryableError(f"Only {total}/{expected_games} games completed ({total/expected_games:.0%}), minimum is {min_completion:.0%}")

    if total != expected_games:
        logger.warning(
            "Expected %d games but got %d (W=%d L=%d D=%d)",
            expected_games, total, wins, losses, draws,
        )

    _check_forfeits(output, expected_games, worker_config.max_forfeit_pct)

    logger.info("Results: W=%d D=%d L=%d (%d games)", wins, draws, losses, total)

    return wins, draws, losses


def run_games(worker_config: WorkerConfig, tuning_config: dict, work: WorkItem) -> tuple:
    """(W, D, L) in standard mode, (plus W, D, L, minus W, D, L) in reference mode, None if cancelled."""
    games_dir = Path(worker_config.games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)
    pgn_file = str(games_dir / f"games-{work.iteration}.pgn").replace("\\", "/")

    # KLUDGE: crashed engines leave PyInstaller _MEI* dirs behind and fill the RAM disk.
    # Windows only, where file locks protect the dirs of co-located workers.
    if _ramdisk_mount and sys.platform == "win32":
        for entry in os.scandir(_ramdisk_mount):
            if entry.is_dir() and entry.name.startswith("_MEI"):
                shutil.rmtree(entry.path, ignore_errors=True)

    ref_engine = worker_config.reference_engine

    if ref_engine:
        # reference engine is the gauntlet seed
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

        _check_forfeits(output, work.num_games, worker_config.max_forfeit_pct)

        in_rank = False
        for line in output.splitlines():
            if line.startswith("Rank "):
                in_rank = True
            if in_rank:
                if not line.strip():
                    break
                logger.info("  %s", line.rstrip())

        try:
            pairing_results = parse_gauntlet_output(output, "reference")
        except ValueError:
            raise RetryableError("Could not parse gauntlet output (engine init failure?):\n" + output[-500:])
        for name in ("theta_plus", "theta_minus"):
            if name not in pairing_results:
                raise RetryableError(f"Missing gauntlet pairing for {name} in output:\n{output[-500:]}")

        # flip from the reference's point of view
        ref_pw, ref_pl, ref_pd = pairing_results["theta_plus"]
        ref_mw, ref_ml, ref_md = pairing_results["theta_minus"]
        plus_wins, plus_draws, plus_losses = ref_pl, ref_pd, ref_pw
        minus_wins, minus_draws, minus_losses = ref_ml, ref_md, ref_mw

        min_completion = 1 - worker_config.max_forfeit_pct
        for side, w, d, l in [("theta+", plus_wins, plus_draws, plus_losses), ("theta-", minus_wins, minus_draws, minus_losses)]:
            total = w + d + l
            if total == 0:
                raise RetryableError(f"No games completed for {side}")
            if total < half * min_completion:
                raise RetryableError(f"Only {total}/{half} games completed for {side} ({total/half:.0%}), minimum is {min_completion:.0%}")
            if total != half:
                logger.warning("Expected %d games for %s but got %d", half, side, total)
            logger.info("Results [%s]: W=%d D=%d L=%d (%d games)", side, w, d, l, total)

        return plus_wins, plus_draws, plus_losses, minus_wins, minus_draws, minus_losses

    else:
        cmd = build_cutechess_command(
            worker_config, tuning_config,
            engine1_params=work.theta_plus, engine1_name="theta_plus",
            engine2_params=work.theta_minus, engine2_name="theta_minus",
            num_games=work.num_games, pgn_file=pgn_file,
        )
        return _execute_match(cmd, work.num_games, work, worker_config, tuning_config)


def worker_loop(worker_config: WorkerConfig):
    base_url = worker_config.coordinator.rstrip("/")

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

    # sent so the coordinator can estimate timeouts
    cc_overrides = {k: v for k, v in worker_config.cutechess_overrides.items()
                    if not k.startswith('_')}

    chunk_size_cap = worker_config.max_chunk_size
    if worker_config.max_rounds_per_chunk > 0:
        rounds_cap = worker_config.max_rounds_per_chunk * worker_config.concurrency * 2
        chunk_size_cap = min(chunk_size_cap, rounds_cap) if chunk_size_cap > 0 else rounds_cap
    logger.info("Chunk size cap: %d games (%d rounds x %d concurrency)", chunk_size_cap, worker_config.max_rounds_per_chunk, worker_config.concurrency)

    consecutive_errors = 0
    max_retries = worker_config.max_retries

    while True:
        try:
            work_request = {
                "chunk_size": chunk_size_cap,
                "worker": worker_name,
                "server_start": server_start,
                "concurrency": worker_config.concurrency,
            }
            if cc_overrides:
                work_request["cutechess_overrides"] = cc_overrides
            response = http_post(f"{base_url}/work", work_request, retry_timeout)

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

            work = WorkItem.from_dict(response)
            logger.info("Got work: iteration %d, %d games", work.iteration, work.num_games)

            game_result = run_games(worker_config, tuning_config, work)
            if game_result is None:
                if _shutdown_requested:
                    # deregister now rather than time out as crashed
                    try:
                        http_post(f"{base_url}/result", {
                            "iteration": work.iteration,
                            "chunk_id": work.chunk_id,
                            "worker": worker_name,
                            "num_games": 0,
                            "reference_mode": bool(worker_config.reference_engine),
                            "shutting_down": True,
                        }, timeout=SHUTDOWN_NOTIFY_TIMEOUT)
                    except Exception as e:
                        logger.debug("Shutdown notify failed: %s", e)
                    break
                continue  # chunk cancelled

            result = {
                "iteration": work.iteration,
                "chunk_id": work.chunk_id,
                "worker": worker_name,
                "shutting_down": _shutdown_requested,
            }
            if len(game_result) == 6:
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
            # http_post already retried for http_retry_timeout seconds
            logger.error("Coordinator unreachable after %ds: %s. Exiting.", retry_timeout, e)
            break
        except (subprocess.TimeoutExpired, RetryableError) as e:
            consecutive_errors += 1
            msg = f"{_tool_label} timed out" if isinstance(e, subprocess.TimeoutExpired) else str(e)
            logger.error("%s (%d/%s)", msg, consecutive_errors, max_retries or "inf")
            if max_retries and consecutive_errors >= max_retries:
                logger.error("Too many consecutive errors, terminating.")
                break
        except Exception as e:
            logger.exception("Terminating.")
            break


def _install_managed_signals():
    """SIGTERM finishes the current games, reports, and exits; SIGINT kills cutechess right away."""
    global _managed_mode
    _managed_mode = True

    def _term_handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("SIGTERM received; will stop after current games complete")

    signal.signal(signal.SIGTERM, _term_handler)


def setup_logging(log_file: str, debug: bool, rotate: bool):
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

    level = logging.DEBUG if debug else logging.INFO
    for lg in (logger, logging.getLogger("ramdisk")):
        lg.setLevel(level)
        lg.addHandler(file_handler)
        lg.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(description="SPSA Tuning Worker")
    parser.add_argument("-c", "--config", required=True, help="Path to worker config JSON")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--cutechess-debug", action="store_true", help="Pass -debug to cutechess-cli (log engine I/O)")
    parser.add_argument("--clean", action="store_true", help="Wipe log file before starting")
    parser.add_argument("--managed", action="store_true",
                        help="Non-interactive mode: SIGTERM = graceful, SIGINT = stop now. "
                             "Used by launch_numa_workers.py.")
    args = parser.parse_args()

    config = WorkerConfig.from_json(args.config)

    if args.clean:
        log_path = Path(config.log_file)
        for f in log_path.parent.glob(log_path.name + "*"):
            f.unlink()

    global _cutechess_debug
    _cutechess_debug = args.cutechess_debug
    setup_logging(config.log_file, debug=args.debug, rotate=config.log_rotation)

    if args.managed:
        _install_managed_signals()
        logger.info("Running in --managed mode (launcher-driven shutdown)")

    logger.info("Starting worker")
    logger.info("Coordinator: %s", config.coordinator)
    logger.info("Engine: %s", config.engine)
    logger.info("Concurrency: %d", config.concurrency)
    global _tool_label
    tool_name = Path(config.cutechess_cli).stem.lower()
    _tool_label = "fastchess" if "fastchess" in tool_name else "cutechess-cli"
    logger.info("%s: %s", _tool_label, config.cutechess_cli)
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
