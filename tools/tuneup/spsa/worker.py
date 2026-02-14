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
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import WorkerConfig, WorkItem

logger = logging.getLogger("worker")


def http_get(url: str) -> dict:
    """GET request, return parsed JSON."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
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


def build_cutechess_command(worker_config: WorkerConfig,
                            tuning_config: dict,
                            work: WorkItem,
                            pgn_file: str) -> list:
    """Build the cutechess-cli command line."""
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

    # Engine 1: theta_plus
    cmd += ["-engine", f"cmd={engine_cmd}", "name=theta_plus"]
    cmd += option_args(work.theta_plus)

    # Engine 2: theta_minus
    cmd += ["-engine", f"cmd={engine_cmd}", "name=theta_minus"]
    cmd += option_args(work.theta_minus)

    # Common settings
    cmd += ["-each", f"proto={protocol}"]

    if depth is not None:
        cmd += [f"depth={depth}"]
        cmd += ["tc=inf"]
    else:
        cmd += [f"tc={tc}"]

    # Opening book
    if book_file:
        cmd += ["-openings", f"file={book_file}", f"format={book_format}"]
        if book_depth:
            cmd += [f"plies={book_depth}"]
        cmd += ["order=random"]
        cmd += ["policy=round"]

    # Number of games (rounds = games/2 for alternating colors)
    num_rounds = max(1, work.num_games // 2)
    assert(work.num_games)
    assert(num_rounds)

    cmd += ["-rounds", str(num_rounds)]
    cmd += ["-games", "2"]  # 2 games per round (color swap)
    cmd += ["-repeat"]

    # Concurrency
    if worker_config.concurrency > 1:
        cmd += ["-concurrency", str(worker_config.concurrency)]

    # PGN output
    cmd += ["-pgnout", pgn_file]

    return cmd


def run_games(worker_config: WorkerConfig, tuning_config: dict, work: WorkItem) -> tuple:
    """
    Run cutechess-cli and return (wins, draws, losses).

    PGNs are saved directly by cutechess-cli into worker's games_dir.

    wins   = games won by engine1 (theta_plus)
    draws  = drawn games
    losses = games lost by engine1 (= won by theta_minus)
    """
    games_dir = Path(worker_config.games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    pgn_file = str(games_dir / "games.pgn").replace("\\", "/")

    cmd = build_cutechess_command(
        worker_config, tuning_config, work, pgn_file
    )

    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600,
    )

    if result.returncode != 0:
        rc = result.returncode
        logger.error("cutechess-cli failed (rc=%s)", hex(rc))
        logger.error("stdout (last 1000 chars): %s",
                      result.stdout[-1000:] if result.stdout else "(empty)")
        logger.error("stderr (last 1000 chars): %s",
                      result.stderr[-1000:] if result.stderr else "(empty)")
        # Windows STATUS_ACCESS_VIOLATION (subprocess returns signed or unsigned)
        if rc & 0xFFFFFFFF == 0xc0000005:
            raise OSError(f"cutechess-cli access violation ({hex(rc)})")
        raise RuntimeError(f"cutechess-cli exited with code {hex(rc)}")

    output = result.stdout
    if result.stderr:
        logger.warning("cutechess-cli stderr: %s", result.stderr.strip()[-1000:])

    # Log all output lines that mention errors or crashes
    for line in output.splitlines():
        if any(kw in line.lower() for kw in ("abandoned", "error", "crash", "disconnect", "timeout", "illegal", "terminated", "forfeit")):
            logger.warning("cutechess: %s", line.strip())

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
        raise RuntimeError("No games were played")

    # Abort if too many games failed — results would be noise
    min_completion = 0.5
    if total < work.num_games * min_completion:
        logger.error(
            "Only %d/%d games completed (W=%d L=%d D=%d) — aborting chunk",
            total, work.num_games, wins, losses, draws,
        )
        raise RuntimeError(
            f"Only {total}/{work.num_games} games completed "
            f"({total/work.num_games:.0%}), minimum is {min_completion:.0%}"
        )

    if total != work.num_games:
        logger.warning(
            "Expected %d games but got %d (W=%d L=%d D=%d)",
            work.num_games, total, wins, losses, draws,
        )

    logger.info(
        "Results: W=%d D=%d L=%d (%d games)",
        wins, draws, losses, total,
    )

    return wins, draws, losses


def worker_loop(worker_config: WorkerConfig):
    """Main worker loop: poll for work, run games, report results."""
    base_url = worker_config.coordinator.rstrip("/")

    # Fetch tuning config from coordinator
    logger.info("Connecting to coordinator at %s", base_url)
    tuning_config = http_get(f"{base_url}/config")
    logger.info("Received tuning config: %d parameters",
                len(tuning_config.get("parameters", {})))

    # Default retry interval when coordinator asks workers to retry
    default_retry = tuning_config.get("retry_after", 5)
    retry_timeout = worker_config.http_retry_timeout

    hostname = worker_config.name or platform.node()

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
                "worker": hostname,
            }
            if cc_overrides:
                work_request["cutechess_overrides"] = cc_overrides
            response = http_post(f"{base_url}/work", work_request, retry_timeout)

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
            logger.info(
                "Got work: iteration %d, %d games",
                work.iteration, work.num_games,
            )

            # Run the games
            wins, draws, losses = run_games(
                worker_config, tuning_config, work
            )

            # Report results (PGNs saved locally by cutechess-cli)
            result = {
                "iteration": work.iteration,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "num_games": wins + draws + losses,
                "chunk_id": work.chunk_id,
                "worker": hostname,
            }
            resp = http_post(f"{base_url}/result", result, retry_timeout)
            logger.info("Result submitted: %s", resp.get("status"))

        except urllib.error.URLError as e:
            logger.warning("Connection error: %s, retrying in 5s", e)
            time.sleep(5)
        except subprocess.TimeoutExpired:
            logger.error("cutechess-cli timed out")
            time.sleep(2)
        except OSError as e:
            logger.error("%s, retrying", e)
        except Exception as e:
            logger.exception("Terminating.")
            break

def setup_logging(log_file: str, debug: bool = False, rotate: bool = True):
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
    parser.add_argument("--clean", action="store_true", help="Wipe log file before starting")
    args = parser.parse_args()

    config = WorkerConfig.from_json(args.config)

    # --clean: remove log file and rotated copies (keep games)
    if args.clean:
        log_path = Path(config.log_file)
        for f in log_path.parent.glob(log_path.name + "*"):
            f.unlink()

    setup_logging(config.log_file, debug=args.debug, rotate=config.log_rotation)

    logger.info("Starting worker")
    logger.info("Coordinator: %s", config.coordinator)
    logger.info("Engine: %s", config.engine)
    logger.info("Concurrency: %d", config.concurrency)
    logger.info("cutechess-cli: %s", config.cutechess_cli)
    logger.info("Games dir: %s", config.games_dir)
    if config.opening_book:
        logger.info("Opening book: %s (%s)", config.opening_book, config.book_format)

    worker_loop(config)


if __name__ == "__main__":
    main()
