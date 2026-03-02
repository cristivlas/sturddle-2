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
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
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
    """Build the cutechess-cli command line.

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

    # Apply worker-local cutechess-cli overrides
    cc_overrides = worker_config.cutechess_overrides
    if "depth" in cc_overrides:
        depth = cc_overrides["depth"]
        logger.info("Worker override: depth=%s", depth)
    if "tc" in cc_overrides:
        tc = cc_overrides["tc"]
        logger.info("Worker override: tc=%s", tc)

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

    # PGN output
    cmd += ["-pgnout", pgn_file]

    # Debug: log all engine I/O
    if _cutechess_debug:
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

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **popen_kwargs)
    _current_process = proc

    # Drain pipes in background threads to prevent deadlock while the
    # main thread polls proc.poll() (which directly sets returncode).
    # Output is streamed to cutechess_last.log for post-mortem inspection.
    stdout_buf = []
    stderr_buf = []
    log_dir = Path(worker_config.log_file).parent
    try:
        _cc_log = open(log_dir / "cutechess_last.log", "w", buffering=1)
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

    ref_engine = worker_config.reference_engine

    if ref_engine:
        # Reference mode: gauntlet with reference as seed vs theta+, theta-
        half = work.num_games // 2
        pgn_file = str(games_dir / "games_ref.pgn").replace("\\", "/")
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
        pairing_results = parse_gauntlet_output(output, "reference")
        for name in ("theta_plus", "theta_minus"):
            if name not in pairing_results:
                raise ValueError(f"Missing gauntlet pairing for {name} in output:\n{output[-500:]}")

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
        pgn_file = str(games_dir / "games.pgn").replace("\\", "/")
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

            if _shutdown_requested:
                logger.info("Shutdown requested, stopping after result reported.")
                break

        except KeyboardInterrupt:
            logger.info("Interrupted, shutting down.")
            break
        except urllib.error.URLError as e:
            logger.warning("Connection error: %s, retrying in 5s", e)
            time.sleep(5)
        except subprocess.TimeoutExpired:
            logger.error("cutechess-cli timed out")
            time.sleep(2)
        except RetryableError as e:
            logger.error("%s, retrying", e)
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
    logger.info("cutechess-cli: %s", config.cutechess_cli)
    logger.info("Games dir: %s", config.games_dir)
    if config.opening_book:
        fmt = config.book_format or Path(config.opening_book).suffix.lower().lstrip(".") or "pgn"
        logger.info("Opening book: %s (%s)", config.opening_book, fmt)

    worker_loop(config)


if __name__ == "__main__":
    main()
