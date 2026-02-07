#!/usr/bin/env python3
"""
SPSA Tuning Coordinator.

HTTP server that manages SPSA state and distributes work to workers.
Zero external dependencies — uses only Python stdlib.

Usage:
    python coordinator.py -c tuning.json [-p 8080] [--clean]
"""

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from config import TuningConfig, WorkItem, WorkResult
from spsa import SPSAOptimizer, SPSAState

logger = logging.getLogger("coordinator")


class CoordinatorState:
    """Thread-safe coordinator state managing one SPSA iteration at a time."""

    # Seconds before an assigned chunk is considered timed-out
    CHUNK_TIMEOUT = 600

    def __init__(self, tuning_config: TuningConfig, resume: bool = False):
        self.config = tuning_config
        self.lock = threading.Lock()

        output = Path(tuning_config.output_dir)
        self.logs_dir = output / "logs"
        self.state_file = output / "spsa_state.json"

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load or initialize SPSA
        state = None
        if resume and self.state_file.exists():
            with open(self.state_file) as f:
                state = SPSAState.from_dict(json.load(f))
            logger.info("Resumed from iteration %d", state.iteration)

        self.optimizer = SPSAOptimizer(
            params=tuning_config.parameters,
            spsa_config=tuning_config.spsa,
            max_iterations=tuning_config.max_iterations(),
            state=state,
        )

        # Current iteration work tracking
        self.current_delta = None
        self.current_work = None  # WorkItem template for this iteration
        self.games_assigned = 0
        self.games_completed = 0
        self.total_score_plus = 0.0
        self.total_score_minus = 0.0
        self.total_games_scored = 0
        self.pending_chunks = []  # (num_games, assign_time) for timeout tracking
        self._prepare_iteration()

    def _prepare_iteration(self):
        """Set up work for the current iteration."""
        if self.optimizer.is_done():
            return

        self.current_delta = self.optimizer.generate_perturbation()
        theta_plus, theta_minus = self.optimizer.compute_candidates(
            self.current_delta
        )
        self.current_work = WorkItem(
            iteration=self.optimizer.iteration,
            theta_plus=theta_plus,
            theta_minus=theta_minus,
            num_games=0,  # filled per chunk
        )
        self.games_assigned = 0
        self.games_completed = 0
        self.total_score_plus = 0.0
        self.total_score_minus = 0.0
        self.total_games_scored = 0
        self.pending_chunks = []

        logger.info(
            "Iteration %d: c_k=%.4f, a_k=%.6f",
            self.optimizer.iteration,
            self.optimizer.c_k(),
            self.optimizer.a_k(),
        )

    def _reclaim_timed_out_chunks(self):
        """Return timed-out assigned games to the pool."""
        now = time.time()
        still_pending = []
        reclaimed = 0
        for num_games, assign_time in self.pending_chunks:
            if now - assign_time > self.CHUNK_TIMEOUT:
                self.games_assigned -= num_games
                reclaimed += num_games
            else:
                still_pending.append((num_games, assign_time))
        self.pending_chunks = still_pending
        if reclaimed:
            logger.warning("Reclaimed %d timed-out games", reclaimed)

    def get_work(self, chunk_size: int = 0) -> dict:
        """
        Assign a chunk of games to a worker.

        Args:
            chunk_size: requested games (0 = let coordinator decide)

        Returns:
            WorkItem dict, or {"status": "done"/"retry"}.
        """
        with self.lock:
            if self.optimizer.is_done():
                return {"status": "done"}

            self._reclaim_timed_out_chunks()

            gpi = self.config.games_per_iteration
            remaining = gpi - self.games_assigned

            if remaining <= 0:
                # All games assigned for this iteration, wait for results
                return {"status": "retry", "retry_after": self.config.retry_after}

            if chunk_size <= 0:
                # Default: give a reasonable chunk
                chunk_size = max(2, min(remaining, gpi // 4))

            # Must be even (each game pair is +c vs -c)
            num_games = min(remaining, chunk_size)
            num_games = max(2, num_games - (num_games % 2))

            self.games_assigned += num_games
            self.pending_chunks.append((num_games, time.time()))

            work = WorkItem(
                iteration=self.current_work.iteration,
                theta_plus=self.current_work.theta_plus,
                theta_minus=self.current_work.theta_minus,
                num_games=num_games,
            )

            logger.info(
                "Assigned %d games (iter %d, %d/%d assigned)",
                num_games, work.iteration,
                self.games_assigned, gpi,
            )
            return work.to_dict()

    def submit_result(self, result: WorkResult) -> dict:
        """
        Accept results from a worker and advance iteration if complete.

        Returns:
            {"status": "ok"} or {"status": "ignored", ...} if stale.
        """
        with self.lock:
            if result.iteration != self.optimizer.iteration:
                logger.warning(
                    "Ignoring stale result for iteration %d (current: %d)",
                    result.iteration, self.optimizer.iteration,
                )
                return {"status": "ignored", "reason": "stale iteration"}

            self.games_completed += result.num_games
            self.total_score_plus += result.score_plus * result.num_games
            self.total_score_minus += result.score_minus * result.num_games
            self.total_games_scored += result.num_games

            # Remove matching pending chunk
            for i, (n, _) in enumerate(self.pending_chunks):
                if n == result.num_games:
                    self.pending_chunks.pop(i)
                    break

            logger.info(
                "Result: iter %d, %d games, +%.3f / -%.3f (%d/%d done)",
                result.iteration, result.num_games,
                result.score_plus, result.score_minus,
                self.games_completed, self.config.games_per_iteration,
            )

            # Check if iteration is complete
            if self.games_completed >= self.config.games_per_iteration:
                self._complete_iteration()

            return {"status": "ok"}

    def _complete_iteration(self):
        """Finalize current iteration: update theta, save state, log."""
        avg_score_plus = self.total_score_plus / self.total_games_scored
        avg_score_minus = self.total_score_minus / self.total_games_scored

        k = self.optimizer.iteration
        old_theta = dict(self.optimizer.theta)

        new_theta = self.optimizer.update(
            self.current_delta, avg_score_plus, avg_score_minus
        )

        # ELO estimates
        elo_plus = self.optimizer.elo_estimate(avg_score_plus)
        elo_minus = self.optimizer.elo_estimate(avg_score_minus)

        logger.info("=" * 60)
        logger.info("Iteration %d complete (%d games)", k, self.total_games_scored)
        logger.info(
            "Scores: +%.4f / -%.4f (diff: %+.4f, ELO diff: %+.1f)",
            avg_score_plus, avg_score_minus,
            avg_score_plus - avg_score_minus,
            elo_plus - elo_minus,
        )
        logger.info("Updated parameters:")
        for name in new_theta:
            param = self.optimizer.params[name]
            engine_val = param.to_engine_value(new_theta[name])
            step = new_theta[name] - old_theta[name]
            r = param.upper - param.lower
            logger.info(
                "  %s: %.4f -> %.4f (engine: %s, step: %+.4f, %.1f%% of range)",
                name, old_theta[name], new_theta[name], engine_val,
                step, 100.0 * abs(step) / r if r > 0 else 0,
            )
        logger.info("=" * 60)

        # Checkpoint
        self._save_state()

        # Prepare next iteration
        if not self.optimizer.is_done():
            self._prepare_iteration()
        else:
            logger.info("SPSA tuning complete after %d iterations", k + 1)
            logger.info("Final parameters:")
            for name, val in self.optimizer.get_engine_values().items():
                logger.info("  %s = %s", name, val)

    def _save_state(self):
        """Persist SPSA state to disk."""
        with open(self.state_file, "w") as f:
            json.dump(self.optimizer.state.to_dict(), f, indent=2)

    def get_status(self) -> dict:
        """Current tuning status for display."""
        with self.lock:
            return {
                "iteration": self.optimizer.iteration,
                "max_iterations": self.optimizer.max_iterations,
                "is_done": self.optimizer.is_done(),
                "theta": self.optimizer.get_engine_values(),
                "games_completed": self.games_completed,
                "games_per_iteration": self.config.games_per_iteration,
                "c_k": self.optimizer.c_k() if not self.optimizer.is_done() else 0,
                "a_k": self.optimizer.a_k() if not self.optimizer.is_done() else 0,
            }

    def get_tuning_config_dict(self) -> dict:
        """Tuning config as dict for workers to fetch."""
        return json.loads(self.config.to_json())

    def get_coordinator_dashboard(self) -> dict:
        """Rich coordinator data for graphical dashboard."""
        with self.lock:
            pct_complete = 0
            if not self.optimizer.is_done():
                pct_complete = (
                    self.optimizer.iteration / self.optimizer.max_iterations * 100
                )

            pct_games = 0
            gpi = self.config.games_per_iteration
            if gpi > 0:
                pct_games = (
                    self.games_completed / gpi * 100
                ) if not self.optimizer.is_done() else 100

            history = self.optimizer.state.history

            return {
                "iteration": self.optimizer.iteration,
                "max_iterations": self.optimizer.max_iterations,
                "is_done": self.optimizer.is_done(),
                "progress_pct": min(100, pct_complete),
                "current_iteration_progress_pct": min(100, pct_games),
                "games_completed_in_iteration": self.games_completed,
                "games_per_iteration": gpi,
                "games_assigned": self.games_assigned,
                "games_pending": self.games_assigned - self.games_completed,
                "theta": self.optimizer.get_engine_values(),
                "c_k": self.optimizer.c_k() if not self.optimizer.is_done() else 0,
                "a_k": self.optimizer.a_k() if not self.optimizer.is_done() else 0,
                "history": history[-50:] if history else [],
            }



class CoordinatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the coordinator."""

    coordinator: CoordinatorState  # set on class before server starts

    def log_message(self, format, *args):
        # Route http.server logs through our logger
        logger.debug(format, *args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}


    def _send_html(self, html: str, status: int = 200):
        """Send HTML response."""
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


    def _render_coordinator_dashboard(self) -> str:
        """Render dashboard from template with data."""
        template_path = Path(__file__).parent / "dashboard.tmpl"

        if not template_path.exists():
            return "<h1>Error: dashboard.tmpl template not found</h1>"

        chart_js_path = Path(__file__).parent / "chart.umd.min.js"
        with open(chart_js_path, "r") as f:
            chart_js = f.read()

        with open(template_path) as f:
            template = f.read()

        data = self.coordinator.get_coordinator_dashboard()

        progress = data["progress_pct"]
        iter_progress = data["current_iteration_progress_pct"]
        iteration = data["iteration"]
        max_iters = data["max_iterations"]
        is_done = data["is_done"]
        games_done = data["games_completed_in_iteration"]
        games_total = data["games_per_iteration"]
        games_assigned = data["games_assigned"]
        games_pending = data["games_pending"]

        # Build parameter table rows
        theta_rows = ""
        for name, value in data["theta"].items():
            theta_rows += f"""        <tr>
            <td style="font-family: monospace; padding: 8px;">{name}</td>
            <td style="font-family: monospace; text-align: right; padding: 8px;">{value}</td>
        </tr>
"""

        # Build history section
        history_section = ""
        if data.get("history"):
            history_rows = ""
            for h in list(reversed(data.get("history", [])))[-10:]:
                iter_num = h.get("iteration", "?")
                score_diff = h.get("score_diff", 0)
                elo_diff = h.get("elo_diff", 0)
                history_rows += f"""        <tr>
            <td style="padding: 8px; text-align: center;">{iter_num}</td>
            <td style="padding: 8px; text-align: center;">{score_diff:+.4f}</td>
            <td style="padding: 8px; text-align: center;">{elo_diff:+.1f}</td>
        </tr>
"""
            history_section = f"""
            <div class="section">
                <h3>Recent Iterations</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Iteration</th>
                            <th>Score Diff</th>
                            <th>ELO Diff</th>
                        </tr>
                    </thead>
                    <tbody>
                        {history_rows}
                    </tbody>
                </table>
            </div>
"""

        # JSON blob for client-side charts
        history_json = json.dumps(data.get("history", []))

        status_color = "#4CAF50" if not is_done else "#2196F3"
        status_text = "COMPLETE" if is_done else "IN PROGRESS"

        return template.format(
            chart_js=chart_js,
            status_color=status_color,
            status_text=status_text,
            iteration=iteration,
            max_iters=max_iters,
            progress=progress,
            games_done=games_done,
            games_total=games_total,
            iter_progress=iter_progress,
            games_pending=games_pending,
            a_k=data["a_k"],
            c_k=data["c_k"],
            games_assigned=games_assigned,
            theta_rows=theta_rows,
            history_section=history_section,
            history_json=history_json,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def do_GET(self):
        if self.path in ("", "/", "/dashboard"):
            html = self._render_coordinator_dashboard()
            self._send_html(html)
        elif self.path == "/config":
            self._send_json(self.coordinator.get_tuning_config_dict())
        elif self.path == "/status":
            self._send_json(self.coordinator.get_status())
        else:
            self.send_error(404)


    def do_POST(self):
        if self.path == "/work":
            data = self._read_json()
            chunk_size = data.get("chunk_size", 0)
            result = self.coordinator.get_work(chunk_size)
            self._send_json(result)
        elif self.path == "/result":
            data = self._read_json()
            work_result = WorkResult.from_dict(data)
            result = self.coordinator.submit_result(work_result)
            self._send_json(result)
        else:
            self.send_error(404)


def setup_logging(log_dir: Path):
    """Configure logging to file and stdout."""
    log_file = log_dir / "coordinator.log"

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
    parser = argparse.ArgumentParser(description="SPSA Tuning Coordinator")
    parser.add_argument("-c", "--config", required=True,
                        help="Path to tuning config JSON")
    parser.add_argument("-p", "--port", type=int, default=8080,
                        help="Server port (default: 8080)")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe state and logs, start fresh")
    args = parser.parse_args()

    config = TuningConfig.from_json(args.config)

    output = Path(config.output_dir)
    logs_dir = output / "logs"
    state_file = output / "spsa_state.json"

    # --clean: remove state and logs
    if args.clean:
        if state_file.exists():
            state_file.unlink()
        for log in logs_dir.glob("*.log"):
            log.unlink()

    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(logs_dir)

    if args.clean:
        logger.info("Clean start: state and logs wiped")

    logger.info("Starting SPSA coordinator on port %d", args.port)
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", config.output_dir)
    logger.info("Parameters: %s", list(config.parameters.keys()))
    logger.info("Budget: %d games, %d per iteration, %d iterations",
                config.spsa.budget, config.games_per_iteration,
                config.max_iterations())
    if config.depth is not None:
        logger.info("Search: depth %d", config.depth)
    else:
        logger.info("Search: TC %s", config.time_control)

    coordinator = CoordinatorState(config, resume=not args.clean)
    CoordinatorHandler.coordinator = coordinator

    server = HTTPServer(("0.0.0.0", args.port), CoordinatorHandler)
    logger.info("Coordinator ready, waiting for workers...")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        coordinator._save_state()
        server.shutdown()


if __name__ == "__main__":
    main()
