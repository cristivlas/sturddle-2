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
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

from config import TuningConfig, WorkItem, WorkResult
from spsa import SPSAOptimizer, SPSAState

VERSION = "1.0.0"

logger = logging.getLogger("coordinator")


@dataclass
class WorkerInfo:
    """Tracked state for a connected worker."""
    name: str
    last_seen: float
    chunks_completed: int = 0
    games_completed: int = 0
    games_completed_iter: int = 0 # games completed in current iteration
    _speed_ewma: float = 0.0      # exponentially weighted moving average (games/sec)
    _ewma_alpha: float = 0.3      # smoothing factor: higher = more weight on recent
    cutechess_overrides: dict = field(default_factory=dict)  # worker-local tc/depth

    @property
    def games_per_second(self) -> float:
        return self._speed_ewma

    def update_speed(self, games: int, elapsed: float):
        """Update EWMA speed estimate from a completed chunk."""
        if elapsed <= 0 or games <= 0:
            return
        sample = games / elapsed
        old = self._speed_ewma
        if self._speed_ewma <= 0:
            self._speed_ewma = sample  # first observation
        else:
            # Blend: alpha * new_sample + (1-alpha) * old_estimate
            self._speed_ewma = (
                self._ewma_alpha * sample
                + (1 - self._ewma_alpha) * self._speed_ewma
            )
        logger.debug(
            "Speed update %s: %d games in %.1fs (%.2f g/s), ewma %.2f -> %.2f",
            self.name, games, elapsed, sample, old, self._speed_ewma,
        )


@dataclass
class ChunkInfo:
    """A single assigned chunk, tracked by unique ID."""
    chunk_id: str
    worker_name: str
    num_games: int
    assign_time: float
    timeout: float


class CoordinatorState:
    """Thread-safe coordinator state managing one SPSA iteration at a time."""

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

        # Stamp session start on fresh state
        if self.optimizer.state.created_at == 0.0:
            self.optimizer.state.created_at = time.time()

        # Current iteration work tracking (restored from state or reset)
        self.current_delta = None
        self.current_work = None  # WorkItem template for this iteration
        self.games_assigned = 0
        self.games_completed = 0
        self.total_score_plus = 0.0
        self.total_score_minus = 0.0
        self.total_games_scored = 0
        self.pending_chunks = {}  # chunk_id -> ChunkInfo

        # Worker registry
        self.workers = {}  # name -> WorkerInfo

        # Time estimates and timeouts
        self._base_sec_per_game = self._estimate_game_duration()
        self.chunk_timeout_multiplier = 5.0
        self.min_chunk_timeout = 60.0
        # Max timeout: at least 30 min, or enough for the largest possible chunk
        max_chunk_games = tuning_config.games_per_iteration // 2
        self.max_chunk_timeout = max(
            1800.0,
            max_chunk_games * self._base_sec_per_game * self.chunk_timeout_multiplier,
        )
        self.worker_timeout = max(120.0, self._base_sec_per_game * 4.0)
        self.server_start_time = time.time()
        self._prepare_iteration()

        # Log if resuming with partial progress
        if self.games_completed > 0:
            logger.info(
                "Resuming iteration %d with %d/%d games already scored",
                self.optimizer.iteration, self.games_completed,
                self.config.games_per_iteration,
            )

    def _estimate_game_duration(self, overrides: dict = None) -> float:
        """Estimate wall-clock seconds per game from time control or search depth.

        Worker-local cutechess_overrides take priority over tuning config.
        """
        cfg = self.config
        depth = overrides.get("depth") if overrides else cfg.depth
        tc = overrides.get("tc") if overrides else cfg.time_control

        if depth is not None:
            return 0.15 * (1.35 ** depth)
        if tc:
            if isinstance(tc, str) and "+" in tc:
                base, inc = tc.split("+", 1)
                per_side = float(base) + 40.0 * float(inc)
            else:
                per_side = float(tc)
            return 2.0 * per_side  # both sides
        return 10.0

    def _worker_sec_per_game(self, worker_name: str) -> float:
        """Per-game time: observed EWMA speed or config-based fallback."""
        w = self.workers.get(worker_name)
        if w and w.games_per_second > 0:
            return 1.0 / w.games_per_second
        if w and w.cutechess_overrides:
            return self._estimate_game_duration(w.cutechess_overrides)
        return self._base_sec_per_game

    def _touch_worker(self, name: str):
        """Register or update a worker's last-seen timestamp."""
        if not name:
            return
        now = time.time()
        if name not in self.workers:
            self.workers[name] = WorkerInfo(name=name, last_seen=now)
            logger.info("Worker registered: %s", name)
        else:
            self.workers[name].last_seen = now

    def _is_worker_alive(self, name: str) -> bool:
        """Worker is alive if recently seen or has a pending chunk."""
        w = self.workers.get(name)
        if not w:
            return False
        now = time.time()
        if (now - w.last_seen) < self.worker_timeout:
            return True
        # Still alive if running a chunk that hasn't timed out
        return any(
            c.worker_name == name and (now - c.assign_time) < c.timeout
            for c in self.pending_chunks.values()
        )

    def _active_workers(self) -> list:
        """Return list of workers considered alive."""
        return [w for w in self.workers.values()
                if self._is_worker_alive(w.name)]

    def _chunk_timeout_for(self, worker_name: str, num_games: int) -> float:
        """Timeout for a chunk based on expected duration."""
        expected = num_games * self._worker_sec_per_game(worker_name)
        timeout = expected * self.chunk_timeout_multiplier
        return max(self.min_chunk_timeout, min(self.max_chunk_timeout, timeout))

    def _compute_chunk_size(self, worker_name: str, remaining: int) -> int:
        """
        Split remaining games proportional to worker speed.

        Bootstrap (no speed data): split evenly across active workers.
        Capped at 50% of remaining to leave work for other workers.
        Even-rounding and final clamping handled by get_work().
        """
        active = self._active_workers()
        num_workers = max(1, len(active))
        total_speed = sum(w.games_per_second for w in active)

        if total_speed > 0:
            w = self.workers.get(worker_name)
            my_speed = w.games_per_second if (w and w.games_per_second > 0) else (total_speed / num_workers)
            chunk = int(remaining * my_speed / total_speed)
        else:
            chunk = remaining // num_workers

        # Cap at 50% to leave work for other workers
        chunk = min(chunk, max(remaining // 2, 2))

        # If remainder is too small to split among other workers, take it all
        if remaining - chunk < num_workers:
            chunk = remaining

        return chunk

    def _prepare_iteration(self):
        """Set up work for the current iteration.

        If the persisted state has a delta and partial scores for the
        current iteration, restore them instead of starting fresh.
        This lets the coordinator resume a partially-completed iteration
        after a restart, re-playing only the remaining games.
        """
        if self.optimizer.is_done():
            return

        st = self.optimizer.state

        # Restore persisted delta + partial scores, or generate fresh
        if st.current_delta:
            self.current_delta = st.current_delta
            self.games_completed = st.games_completed
            self.total_score_plus = st.total_score_plus
            self.total_score_minus = st.total_score_minus
            self.total_games_scored = st.total_games_scored
        else:
            self.current_delta = self.optimizer.generate_perturbation()
            self.games_completed = 0
            self.total_score_plus = 0.0
            self.total_score_minus = 0.0
            self.total_games_scored = 0

        theta_plus, theta_minus = self.optimizer.compute_candidates(
            self.current_delta
        )
        self.current_work = WorkItem(
            iteration=self.optimizer.iteration,
            theta_plus=theta_plus,
            theta_minus=theta_minus,
            num_games=0,  # filled per chunk
        )
        # Only assign remaining games (games_completed already scored)
        self.games_assigned = self.games_completed
        self.pending_chunks = {}

        # Reset per-iteration worker counters
        for w in self.workers.values():
            w.games_completed_iter = 0

        logger.info(
            "Iteration %d: c_k=%.4f, a_k=%.6f",
            self.optimizer.iteration,
            self.optimizer.c_k(),
            self.optimizer.a_k(),
        )

    def _try_steal_chunk(self, worker_name: str) -> bool:
        """Try to reclaim a chunk from a slower worker for reassignment.

        Steals only when both workers have observed speed data, the target
        is at least 2x slower, and the chunk is still early (< 25% of
        expected time elapsed).  Returns True if games were freed up.
        """
        if not self.pending_chunks:
            return False

        fast = self.workers.get(worker_name)
        if not fast or fast.games_per_second <= 0:
            return False

        fast_spg = 1.0 / fast.games_per_second
        now = time.time()

        best_cid = None
        best_games = 0

        for cid, chunk in self.pending_chunks.items():
            slow = self.workers.get(chunk.worker_name)
            if not slow or slow.games_per_second <= 0:
                continue

            slow_spg = 1.0 / slow.games_per_second
            if slow_spg < fast_spg * 2:
                continue

            elapsed = now - chunk.assign_time
            expected = chunk.num_games * slow_spg
            if expected <= 0 or elapsed / expected >= 0.25:
                continue

            if chunk.num_games > best_games:
                best_cid = cid
                best_games = chunk.num_games

        if best_cid is None:
            return False

        chunk = self.pending_chunks.pop(best_cid)
        self.games_assigned -= chunk.num_games
        logger.info(
            "Work steal: reclaimed %d games from %s [%s] for %s (%.0fs into chunk)",
            chunk.num_games, chunk.worker_name, best_cid,
            worker_name, now - chunk.assign_time,
        )
        return True

    def _reclaim_timed_out_chunks(self):
        """Reclaim chunks that have exceeded their per-worker timeout."""
        now = time.time()
        timed_out = [cid for cid, chunk in self.pending_chunks.items()
                     if now - chunk.assign_time > chunk.timeout]
        for cid in timed_out:
            chunk = self.pending_chunks.pop(cid)
            self.games_assigned -= chunk.num_games
            logger.warning(
                "Reclaimed chunk %s from %s: %d games (%.0fs elapsed, timeout was %ds)",
                cid, chunk.worker_name, chunk.num_games,
                now - chunk.assign_time, int(chunk.timeout),
            )

    def get_work(self, chunk_size: int = 0, worker_name: str = "",
                 cutechess_overrides: dict = None) -> dict:
        """
        Assign a chunk of games to a worker.

        Flow: touch worker (heartbeat) -> reclaim timed-out chunks ->
        compute adaptive chunk size -> assign and track.

        Args:
            chunk_size: requested games (0 = let coordinator decide)
            worker_name: worker hostname for tracking

        Returns:
            WorkItem dict, or {"status": "done"/"retry"}.
        """
        with self.lock:
            self._touch_worker(worker_name)
            if cutechess_overrides and worker_name in self.workers:
                self.workers[worker_name].cutechess_overrides = cutechess_overrides

            if self.optimizer.is_done():
                return {"status": "done"}

            # Reclaim games from workers that disappeared mid-chunk
            self._reclaim_timed_out_chunks()

            gpi = self.config.games_per_iteration
            remaining = gpi - self.games_assigned

            if remaining <= 0:
                # Try work stealing: reclaim a chunk from a slower worker
                if self.config.work_stealing and self._try_steal_chunk(worker_name):
                    remaining = gpi - self.games_assigned
                else:
                    return {"status": "retry", "retry_after": self.config.retry_after}

            # Adaptive chunk sizing; worker's max_chunk_size is a ceiling
            adaptive = self._compute_chunk_size(worker_name, remaining)
            if chunk_size > 0:
                adaptive = min(adaptive, chunk_size)

            # Must be even (each game pair is +c vs -c)
            num_games = min(remaining, adaptive)
            num_games = max(2, num_games - (num_games % 2))

            # Generate unique chunk ID and compute timeout
            chunk_id = uuid.uuid4().hex[:12]
            timeout = self._chunk_timeout_for(worker_name, num_games)

            self.games_assigned += num_games
            self.pending_chunks[chunk_id] = ChunkInfo(
                chunk_id=chunk_id,
                worker_name=worker_name,
                num_games=num_games,
                assign_time=time.time(),
                timeout=timeout,
            )

            work = WorkItem(
                iteration=self.current_work.iteration,
                theta_plus=self.current_work.theta_plus,
                theta_minus=self.current_work.theta_minus,
                num_games=num_games,
                chunk_id=chunk_id,
            )

            logger.info(
                "Assigned %d games to %s [%s] (iter %d, %d/%d, timeout=%ds)",
                num_games, worker_name or "?", chunk_id,
                work.iteration, self.games_assigned, gpi, int(timeout),
            )
            return work.to_dict()

    def submit_result(self, result: WorkResult) -> dict:
        """
        Accept results from a worker and advance iteration if complete.

        Returns:
            {"status": "ok"} or {"status": "ignored", ...} if stale.
        """
        with self.lock:
            self._touch_worker(result.worker)

            if result.iteration != self.optimizer.iteration:
                logger.warning(
                    "Ignoring stale result for iteration %d (current: %d)",
                    result.iteration, self.optimizer.iteration,
                )
                return {"status": "ignored", "reason": "stale iteration"}

            # Reject results for chunks not in pending (reclaimed or unknown)
            if not result.chunk_id or result.chunk_id not in self.pending_chunks:
                logger.warning(
                    "Ignoring result for unknown/reclaimed chunk %s from %s",
                    result.chunk_id or "?", result.worker or "?",
                )
                return {"status": "ignored", "reason": "unknown chunk"}

            chunk = self.pending_chunks.pop(result.chunk_id)
            elapsed = time.time() - chunk.assign_time

            self.games_completed += result.num_games
            self.total_score_plus += result.score_plus * result.num_games
            self.total_score_minus += result.score_minus * result.num_games
            self.total_games_scored += result.num_games

            # Update EWMA speed estimate; this drives adaptive chunk sizing
            # so future get_work() calls distribute proportionally
            if result.worker and result.worker in self.workers:
                w = self.workers[result.worker]
                w.chunks_completed += 1
                w.games_completed += result.num_games
                w.games_completed_iter += result.num_games
                w.update_speed(result.num_games, elapsed)

            logger.info(
                "Result: iter %d, %d games from %s [%s], +%.3f / -%.3f (%d/%d done, %.1fs)",
                result.iteration, result.num_games,
                result.worker or "?", result.chunk_id or "?",
                result.score_plus, result.score_minus,
                self.games_completed, self.config.games_per_iteration,
                elapsed,
            )

            # Check if iteration is complete
            if self.games_completed >= self.config.games_per_iteration:
                self._complete_iteration()
            else:
                # Checkpoint partial progress so a restart doesn't lose it
                self._sync_and_save()

            return {"status": "ok"}

    def _sync_and_save(self):
        """Sync iteration progress to SPSAState and checkpoint."""
        st = self.optimizer.state
        st.current_delta = self.current_delta
        st.games_completed = self.games_completed
        st.total_score_plus = self.total_score_plus
        st.total_score_minus = self.total_score_minus
        st.total_games_scored = self.total_games_scored
        self._save_state()

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

        # Log worker stats
        active = self._active_workers()
        if active:
            logger.info("Worker stats:")
            for w in active:
                logger.info(
                    "  %s: %d games, %.2f games/sec",
                    w.name, w.games_completed, w.games_per_second,
                )
        logger.info("=" * 60)

        # Clear iteration progress and checkpoint
        st = self.optimizer.state
        st.current_delta = {}
        st.games_completed = 0
        st.total_score_plus = 0.0
        st.total_score_minus = 0.0
        st.total_games_scored = 0
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
        """Persist SPSA state atomically: write temp file, then rename.

        On POSIX os.replace is atomic. On Windows it's not strictly atomic
        but it is an overwrite-or-fail operation, avoiding partial writes.
        """
        state_dir = self.state_file.parent
        try:
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp", prefix="spsa_state_", dir=state_dir
            )
            with os.fdopen(fd, "w") as f:
                json.dump(self.optimizer.state.to_dict(), f, indent=2)
            os.replace(tmp_path, self.state_file)
        except Exception:
            logger.exception("Failed to save state to %s", self.state_file)
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def get_status(self) -> dict:
        """Current tuning status for display."""
        now = time.time()
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
                "workers": {
                    name: {
                        "alive": self._is_worker_alive(name),
                        "last_seen_ago": round(now - w.last_seen, 1),
                        "games_completed": w.games_completed,
                        "games_completed_iter": w.games_completed_iter,
                    }
                    for name, w in self.workers.items()
                },
                "pending_chunks": len(self.pending_chunks),
            }

    def get_tuning_config_dict(self) -> dict:
        """Tuning config as dict for workers to fetch."""
        return json.loads(self.config.to_json())

    def get_charts_data(self) -> dict:
        """Full history for the charts page (no slicing)."""
        with self.lock:
            return {"history": list(self.optimizer.state.history)}

    def get_coordinator_dashboard(self) -> dict:
        """Rich coordinator data for graphical dashboard."""
        now = time.time()
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

            # Compute per-worker assigned (in-flight) games from pending chunks
            assigned_per_worker = {}
            for chunk in self.pending_chunks.values():
                assigned_per_worker[chunk.worker_name] = (
                    assigned_per_worker.get(chunk.worker_name, 0) + chunk.num_games
                )

            worker_data = []
            for name, w in self.workers.items():
                worker_data.append({
                    "name": name,
                    "alive": self._is_worker_alive(name),
                    "last_seen_ago": round(now - w.last_seen, 1),
                    "games_assigned": assigned_per_worker.get(name, 0),
                    "games_completed_iter": w.games_completed_iter,
                    "games_completed": w.games_completed,
                })

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
                "total_games": self.optimizer.iteration * gpi + self.games_completed,
                "theta": self.optimizer.get_engine_values(),
                "c_k": self.optimizer.c_k() if not self.optimizer.is_done() else 0,
                "a_k": self.optimizer.a_k() if not self.optimizer.is_done() else 0,
                "history": list(
                    history[-self.config.dashboard_history:]
                    if self.config.dashboard_history else history
                ) if history else [],
                "workers": worker_data,
                "session_start": self.optimizer.state.created_at,
                "server_start": self.server_start_time,
            }



class CoordinatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the coordinator."""

    coordinator: CoordinatorState  # set on class before server starts
    dashboard_template: str = ""   # cached at startup
    chart_js: str = ""             # cached at startup

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
        """Render dashboard from cached template with live data."""
        if not self.dashboard_template:
            return "<h1>Error: dashboard template not loaded</h1>"

        data = self.coordinator.get_coordinator_dashboard()

        progress = data["progress_pct"]
        iter_progress = data["current_iteration_progress_pct"]
        iteration = data["iteration"]
        max_iters = data["max_iterations"]
        is_done = data["is_done"]
        games_done = data["games_completed_in_iteration"]
        games_total = data["games_per_iteration"]
        games_pending = data["games_pending"]

        # Build parameter table rows
        theta_rows = ""
        for name, value in data["theta"].items():
            theta_rows += f"""        <tr>
            <td style="font-family: monospace;">{name}</td>
            <td style="font-family: monospace;">{value}</td>
        </tr>
"""

        # Build history section
        history_section = ""
        if data.get("history"):
            history_rows = ""
            for h in reversed(data.get("history", [])):
                iter_num = h.get("iteration", "?")
                score_diff = h.get("score_diff", 0)
                elo_diff = h.get("elo_diff", 0)
                a_k = h.get("a_k", 0)
                c_k = h.get("c_k", 0)
                history_rows += f"""        <tr>
            <td>{iter_num}</td>
            <td>{score_diff:+.4f}</td>
            <td>{elo_diff:+.1f}</td>
            <td>{a_k:.6f}</td>
            <td>{c_k:.4f}</td>
        </tr>
"""
            history_section = f"""
            <div class="section" style="margin-top: 0;">
                <h3>History</h3>
                <div style="padding-right: 17px;">
                <table style="table-layout:fixed; width:100%;">
                    <colgroup><col style="width:20%"><col style="width:20%"><col style="width:20%"><col style="width:20%"><col style="width:20%"></colgroup>
                    <thead><tr><th>Iter</th><th>Score Diff</th><th>ELO Diff</th><th>a_k</th><th>c_k</th></tr></thead>
                </table>
                </div>
                <div style="max-height: 260px; overflow-y: auto;">
                <table style="table-layout:fixed; width:100%;">
                    <colgroup><col style="width:20%"><col style="width:20%"><col style="width:20%"><col style="width:20%"><col style="width:20%"></colgroup>
                    <tbody>
                        {history_rows}
                    </tbody>
                </table>
                </div>
            </div>
"""

        # Build workers section
        workers_section = ""
        if data.get("workers"):
            workers_rows = ""
            for w in sorted(data["workers"], key=lambda w: w["games_completed"], reverse=True):
                if w["alive"]:
                    status_style = "color: #4CAF50"
                    status_text = "online"
                else:
                    status_style = "color: #f44336; font-weight: bold"
                    status_text = "offline"
                workers_rows += f"""        <tr>
            <td style="font-family: monospace;">{w["name"]}</td>
            <td style="{status_style}">{status_text}</td>
            <td>{w["last_seen_ago"]:.0f}s ago</td>
            <td>{w["games_assigned"]}</td>
            <td>{w["games_completed_iter"]}</td>
            <td>{w["games_completed"]}</td>
        </tr>
"""
            workers_section = f"""
            <div class="section" style="margin-top: 0;">
                <h3>Workers</h3>
                <table>
                    <colgroup><col style="width:20%"><col style="width:12%"><col style="width:17%"><col style="width:17%"><col style="width:17%"><col style="width:17%"></colgroup>
                    <thead><tr><th>Name</th><th>Status</th><th>Last Seen</th><th>Assigned</th><th>Iter Done</th><th>Session Done</th></tr></thead>
                </table>
                <div style="max-height: 260px; overflow-y: auto;">
                <table>
                    <colgroup><col style="width:20%"><col style="width:12%"><col style="width:17%"><col style="width:17%"><col style="width:17%"><col style="width:17%"></colgroup>
                    <tbody>
                        {workers_rows}
                    </tbody>
                </table>
                </div>
            </div>
"""

        # JSON blob for client-side charts
        history_json = json.dumps(data.get("history", []))

        status_color = "#4CAF50" if not is_done else "#2196F3"
        status_text = "COMPLETE" if is_done else "IN PROGRESS"

        refresh_sec = self.coordinator.config.dashboard_refresh
        session_start = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(data["session_start"])
        )
        server_start = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(data["server_start"])
        )
        return self.dashboard_template.format(
            version=VERSION,
            chart_js=self.chart_js,
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
            total_games=data["total_games"],
            theta_rows=theta_rows,
            history_section=history_section,
            workers_section=workers_section,
            history_json=history_json,
            refresh_sec=refresh_sec,
            refresh_ms=refresh_sec * 1000,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_start=session_start,
            server_start=server_start,
        )

    def _render_charts_page(self) -> str:
        """Render charts page, re-reading template from disk each request."""
        template_path = Path(__file__).parent / "charts.tmpl"
        try:
            with open(template_path) as f:
                template = f.read()
        except FileNotFoundError:
            return "<h1>Error: charts.tmpl not found</h1>"

        data = self.coordinator.get_charts_data()
        history_json = json.dumps(data.get("history", []))

        return template.format(
            version=VERSION,
            chart_js=self.chart_js,
            history_json=history_json,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def do_GET(self):
        if self.path in ("", "/", "/dashboard"):
            html = self._render_coordinator_dashboard()
            self._send_html(html)
        elif self.path == "/charts":
            html = self._render_charts_page()
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
            worker_name = data.get("worker", "")
            cc_overrides = data.get("cutechess_overrides")
            result = self.coordinator.get_work(
                chunk_size, worker_name, cc_overrides
            )
            self._send_json(result)
        elif self.path == "/result":
            data = self._read_json()
            work_result = WorkResult.from_dict(data)
            result = self.coordinator.submit_result(work_result)
            self._send_json(result)
        else:
            self.send_error(404)


def setup_logging(log_dir: Path, debug: bool = False):
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

    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(description="SPSA Tuning Coordinator")
    parser.add_argument("-c", "--config", required=True, help="Path to tuning config JSON")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--clean", action="store_true", help="Wipe state and logs, start fresh")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
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
    setup_logging(logs_dir, debug=args.debug)

    if args.clean:
        logger.info("Clean start: state and logs wiped")

    # Cache dashboard template and chart.js at startup
    spsa_dir = Path(__file__).parent
    template_path = spsa_dir / "dashboard.tmpl"
    chart_js_path = spsa_dir / "chart.umd.min.js"

    if template_path.exists():
        with open(template_path) as f:
            CoordinatorHandler.dashboard_template = f.read()
    else:
        logger.warning("dashboard.tmpl not found, dashboard disabled")

    if chart_js_path.exists():
        with open(chart_js_path) as f:
            CoordinatorHandler.chart_js = f.read()

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

    # Threaded so dashboard renders don't block worker requests
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", args.port), CoordinatorHandler)
    logger.info("Coordinator ready, waiting for workers...")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        coordinator._save_state()
        server.shutdown()


if __name__ == "__main__":
    main()
