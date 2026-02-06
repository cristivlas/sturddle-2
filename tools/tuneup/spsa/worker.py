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


def http_post(url: str, data: dict) -> dict:
    """POST JSON, return parsed JSON response."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def parse_cutechess_output(output: str) -> tuple:
    """
    Parse cutechess-cli output for game results.

    Looks for the score summary line:
        Score of engine1 vs engine2: W - L - D  [pct]  N

    Returns:
        (wins, losses, draws) from engine1's perspective.
        engine1 = theta_plus, engine2 = theta_minus.
    """
    pattern = r"Score of .+ vs .+: (\d+) - (\d+) - (\d+)"
    match = re.search(pattern, output)
    if not match:
        raise ValueError(
            f"Could not parse cutechess-cli output:\n{output[-500:]}"
        )
    wins = int(match.group(1))
    losses = int(match.group(2))
    draws = int(match.group(3))
    return wins, losses, draws


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
    cmd += ["-engine", f"cmd={engine_cmd}"]
    cmd += option_args(work.theta_plus)

    # Engine 2: theta_minus
    cmd += ["-engine", f"cmd={engine_cmd}"]
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

    # Number of games (rounds = games/2 for alternating colors)
    num_rounds = max(1, work.num_games // 2)
    cmd += ["-rounds", str(num_rounds)]
    cmd += ["-games", "2"]  # 2 games per round (color swap)
    cmd += ["-repeat"]

    # Concurrency
    if worker_config.concurrency > 1:
        cmd += ["-concurrency", str(worker_config.concurrency)]

    # PGN output
    cmd += ["-pgnout", pgn_file]

    return cmd


def run_games(worker_config: WorkerConfig, tuning_config: dict,
              work: WorkItem) -> tuple:
    """
    Run cutechess-cli and return (score_plus, score_minus).

    PGNs are saved directly by cutechess-cli into worker's games_dir.

    score_plus = win rate for engine1 (theta_plus)
    score_minus = win rate for engine2 (theta_minus)
    """
    games_dir = Path(worker_config.games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    hostname = platform.node()
    pgn_file = str(games_dir / f"iter_{work.iteration:04d}_{hostname}.pgn")

    cmd = build_cutechess_command(
        worker_config, tuning_config, work, pgn_file
    )

    logger.info("Running: %s", " ".join(cmd))

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600,
    )

    if result.returncode != 0:
        logger.error(f"cutechess-cli failed (rc={hex(result.returncode)})")
        logger.error("stderr: %s", result.stderr[-500:] if result.stderr else "(empty)")
        raise RuntimeError(f"cutechess-cli exited with code {hex(result.returncode)}")

    output = result.stdout
    logger.debug("cutechess output:\n%s", output)

    wins, losses, draws = parse_cutechess_output(output)
    total = wins + losses + draws

    if total == 0:
        raise RuntimeError("No games were played")

    score_plus = (wins + draws * 0.5) / total
    score_minus = (losses + draws * 0.5) / total

    logger.info(
        "Results: +%d -%d =%d (score_plus=%.3f, score_minus=%.3f)",
        wins, losses, draws, score_plus, score_minus,
    )

    return score_plus, score_minus


def worker_loop(worker_config: WorkerConfig):
    """Main worker loop: poll for work, run games, report results."""
    base_url = worker_config.coordinator.rstrip("/")

    # Fetch tuning config from coordinator
    logger.info("Connecting to coordinator at %s", base_url)
    tuning_config = http_get(f"{base_url}/config")
    logger.info("Received tuning config: %d parameters",
                len(tuning_config.get("parameters", {})))

    hostname = platform.node()

    while True:
        try:
            # Request work
            response = http_post(
                f"{base_url}/work",
                {"chunk_size": 0, "worker": hostname},
            )

            status = response.get("status")
            if status == "done":
                logger.info("Tuning complete, shutting down")
                break
            elif status == "retry":
                delay = response.get("retry_after", 2)
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
            score_plus, score_minus = run_games(
                worker_config, tuning_config, work
            )

            # Report results (PGNs saved locally by cutechess-cli)
            result = {
                "iteration": work.iteration,
                "score_plus": score_plus,
                "score_minus": score_minus,
                "num_games": work.num_games,
            }
            resp = http_post(f"{base_url}/result", result)
            logger.info("Result submitted: %s", resp.get("status"))

        except urllib.error.URLError as e:
            logger.warning("Connection error: %s, retrying in 5s", e)
            time.sleep(5)
        except subprocess.TimeoutExpired:
            logger.error("cutechess-cli timed out")
            time.sleep(2)
        except Exception as e:
            logger.error("Error: %s", e, exc_info=True)
            time.sleep(5)


def setup_logging(log_file: str):
    """Configure logging to file and stdout."""
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(description="SPSA Tuning Worker")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to worker config JSON")
    args = parser.parse_args()

    config = WorkerConfig.from_json(args.config)
    setup_logging(config.log_file)

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
