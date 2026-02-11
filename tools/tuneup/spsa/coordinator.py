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
import mimetypes
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

from config import TuningConfig, WorkItem, WorkResult
from spsa import SPSAOptimizer, SPSAState

VERSION = "1.0.1"

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
    expected_duration: float = 0.0


class CoordinatorState:
    """Thread-safe coordinator state managing one SPSA iteration at a time."""

    def __init__(self, tuning_config: TuningConfig, resume: bool = False):
        self.config = tuning_config
        self.lock = threading.Lock()
        self.dashboard_changed = threading.Condition()
        self.dashboard_version = 0
        self.draining = False
        self._restart = False
        self.drain_complete = threading.Event()

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
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0
        self.pending_chunks = {}  # chunk_id -> ChunkInfo
        self.stolen_chunks = {}   # stolen_cid -> replacement_cid

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

        # If remainder is too small to split, take it all
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

        # Restore persisted delta + partial game counts, or generate fresh
        if st.current_delta:
            self.current_delta = st.current_delta
            self.games_completed = st.games_completed
            self.total_wins = st.total_wins
            self.total_draws = st.total_draws
            self.total_losses = st.total_losses
        else:
            self.current_delta = self.optimizer.generate_perturbation()
            self.games_completed = 0
            self.total_wins = 0
            self.total_draws = 0
            self.total_losses = 0

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
        self.stolen_chunks = {}

        # Reset per-iteration worker counters
        for w in self.workers.values():
            w.games_completed_iter = 0

        logger.info(
            "Iteration %d: c_k=%.4f, a_k=%.6f",
            self.optimizer.iteration,
            self.optimizer.c_k(),
            self.optimizer.a_k(),
        )

    def _try_steal_chunk(self, worker_name: str) -> tuple[str, int] | None:
        """Try to reclaim a chunk from a slower worker for reassignment.

        Compares the fast worker's redo-from-scratch time against the
        slow worker's expected_duration recorded at assignment time.
        Steal if: the holder is overdue and we're faster, or we can
        finish before the holder's original deadline.

        Returns (stolen_chunk_id, num_games), or None.
        """
        if not self.pending_chunks:
            return None

        fast_spg = self._worker_sec_per_game(worker_name)
        now = time.time()

        best_cid = None
        best_saving = 0.0

        for cid, chunk in self.pending_chunks.items():
            if chunk.worker_name == worker_name:
                continue

            elapsed = now - chunk.assign_time
            expected = chunk.expected_duration
            fast_total = chunk.num_games * fast_spg

            overdue = elapsed > expected
            if overdue and fast_total < expected:
                saving = expected - fast_total
                logger.debug(
                    "Work steal: %s eyeing %s [%s] %d games (overdue) — elapsed=%.1fs expected=%.1fs fast=%.1fs saving=%.1fs",
                    worker_name, chunk.worker_name, cid, chunk.num_games, elapsed, expected, fast_total, saving,
                )
            elif fast_total + elapsed < expected:
                saving = expected - elapsed - fast_total
                logger.debug(
                    "Work steal: %s eyeing %s [%s] %d games — elapsed=%.1fs expected=%.1fs fast=%.1fs saving=%.1fs",
                    worker_name, chunk.worker_name, cid, chunk.num_games, elapsed, expected, fast_total, saving,
                )
            else:
                logger.debug(
                    "Work steal: %s skip %s [%s] %d games — elapsed=%.1fs expected=%.1fs fast=%.1fs",
                    worker_name, chunk.worker_name, cid, chunk.num_games, elapsed, expected, fast_total,
                )
                continue

            if saving > best_saving:
                best_cid = cid
                best_saving = saving

        if best_cid is None:
            return None

        chunk = self.pending_chunks.pop(best_cid)
        self.games_assigned -= chunk.num_games
        logger.info(
            "Work steal: %d games from %s [%s] for %s (%.0fs elapsed, est. saving %.1fs)",
            chunk.num_games, chunk.worker_name, best_cid,
            worker_name, now - chunk.assign_time, best_saving,
        )
        return best_cid, chunk.num_games

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

    def _notify_dashboard(self):
        """Wake up any SSE listeners so they push fresh data immediately."""
        with self.dashboard_changed:
            self.dashboard_version += 1
            self.dashboard_changed.notify_all()

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

            cap = self.config.max_pending_per_worker
            if cap > 0 and worker_name:
                pending = sum(1 for c in self.pending_chunks.values() if c.worker_name == worker_name)
                if pending >= cap:
                    logger.debug("Throttling %s: %d pending chunks (cap %d)", worker_name, pending, cap)
                    return {"status": "retry", "retry_after": self.config.retry_after}

            gpi = self.config.games_per_iteration
            remaining = gpi - self.games_assigned
            stolen_cid = None
            stolen_games = 0

            if remaining <= 0:
                if self.draining:
                    if not self._restart:
                        return {"status": "done"}
                    return {"status": "retry", "retry_after": self.config.retry_after}
                # Try work stealing: reclaim a chunk from a slower worker
                steal = None
                if self.config.work_stealing:
                    steal = self._try_steal_chunk(worker_name)
                if steal:
                    stolen_cid, stolen_games = steal
                    remaining = gpi - self.games_assigned
                else:
                    return {"status": "retry", "retry_after": self.config.retry_after}

            if stolen_cid:
                num_games = stolen_games
            else:
                adaptive = self._compute_chunk_size(worker_name, remaining)
                if chunk_size > 0:
                    adaptive = min(adaptive, chunk_size)
                num_games = min(remaining, adaptive)

            # Must be even (each game pair is +c vs -c)
            num_games = max(2, num_games - (num_games % 2))

            # Generate unique chunk ID and compute timeout
            chunk_id = uuid.uuid4().hex[:12]
            timeout = self._chunk_timeout_for(worker_name, num_games)
            expected_duration = num_games * self._worker_sec_per_game(worker_name)

            self.games_assigned += num_games
            self.pending_chunks[chunk_id] = ChunkInfo(
                chunk_id=chunk_id,
                worker_name=worker_name,
                num_games=num_games,
                assign_time=time.time(),
                timeout=timeout,
                expected_duration=expected_duration,
            )

            # Record steal race so either finisher can resolve it
            if stolen_cid:
                self.stolen_chunks[stolen_cid] = chunk_id

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
            self._notify_dashboard()
            return work.to_dict()

    def submit_result(self, result: WorkResult) -> dict:
        """
        Accept results from a worker and advance iteration if complete.

        Returns:
            {"status": "ok"} or {"status": "ignored", ...} if stale.
        """
        with self.lock:
            self._touch_worker(result.worker)

            if not result.chunk_id or not result.worker:
                logger.error("Malformed result: %s", result)
                return {"status": "ignored", "reason": "malformed result"}

            if result.iteration != self.optimizer.iteration:
                logger.warning(
                    "Ignoring stale result (%d games) for iteration %d from %s (current: %d)",
                    result.num_games, result.iteration, result.worker, self.optimizer.iteration,
                )
                return {"status": "ignored", "reason": "stale iteration"}

            # Resolve chunk: pending (normal), stolen (replaced), or unknown
            chunk = None
            if result.chunk_id in self.pending_chunks:
                chunk = self.pending_chunks.pop(result.chunk_id)
                stolen_cid = next((k for k, v in self.stolen_chunks.items() if v == result.chunk_id), None)
                if stolen_cid:
                    self.stolen_chunks.pop(stolen_cid)
                    logger.debug("Replacement [%s] from %s won against [%s]", result.chunk_id, result.worker, stolen_cid)
            elif result.chunk_id in self.stolen_chunks:
                replacement_cid = self.stolen_chunks.pop(result.chunk_id)
                if replacement_cid in self.pending_chunks:
                    replacement = self.pending_chunks.pop(replacement_cid)
                    self.games_assigned -= replacement.num_games
                    logger.info("Got [%s] from %s, cancelled [%s]", result.chunk_id, result.worker, replacement_cid)
                else:
                    return {"status": "ignored", "reason": f"replaced by {replacement_cid}"}
            else:
                logger.warning("Ignoring result for unknown/reclaimed chunk %s from %s", result.chunk_id, result.worker)
                return {"status": "ignored", "reason": "unknown chunk"}

            self.games_completed += result.num_games
            self.total_wins += result.wins
            self.total_draws += result.draws
            self.total_losses += result.losses

            if result.worker in self.workers:
                w = self.workers[result.worker]
                w.chunks_completed += 1
                w.games_completed += result.num_games
                w.games_completed_iter += result.num_games
                if chunk:
                    w.update_speed(result.num_games, time.time() - chunk.assign_time)

            logger.info(
                "Result: iter %d, %d games from %s [%s], W=%d D=%d L=%d (%d/%d done)",
                result.iteration, result.num_games,
                result.worker, result.chunk_id,
                result.wins, result.draws, result.losses,
                self.games_completed, self.config.games_per_iteration,
            )

            # Check if iteration is complete
            if self.games_completed >= self.config.games_per_iteration:
                self._complete_iteration()
            else:
                # Checkpoint partial progress so a restart doesn't lose it
                self._sync_and_save()

            self._notify_dashboard()
            return {"status": "ok"}

    def _sync_and_save(self):
        """Sync iteration progress to SPSAState and checkpoint."""
        st = self.optimizer.state
        st.current_delta = self.current_delta
        st.games_completed = self.games_completed
        st.total_wins = self.total_wins
        st.total_draws = self.total_draws
        st.total_losses = self.total_losses
        self._save_state()

    def _complete_iteration(self):
        """Finalize current iteration: update theta, save state, log."""
        total = self.total_wins + self.total_draws + self.total_losses
        avg_score_plus = (self.total_wins + 0.5 * self.total_draws) / total
        avg_score_minus = (self.total_losses + 0.5 * self.total_draws) / total

        k = self.optimizer.iteration
        old_theta = dict(self.optimizer.theta)

        new_theta = self.optimizer.update(
            self.current_delta, avg_score_plus, avg_score_minus
        )

        # ELO estimates
        elo_plus = self.optimizer.elo_estimate(avg_score_plus)
        elo_minus = self.optimizer.elo_estimate(avg_score_minus)

        logger.info("=" * 60)
        logger.info("Iteration %d complete (%d games, W=%d D=%d L=%d)",
                     k, total, self.total_wins, self.total_draws, self.total_losses)
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
        st.total_wins = 0
        st.total_draws = 0
        st.total_losses = 0
        self._save_state()

        # Prepare next iteration
        if self.draining:
            logger.info("Drain complete after iteration %d", k)
            self.drain_complete.set()
        elif not self.optimizer.is_done():
            self._prepare_iteration()
        else:
            logger.info("SPSA tuning complete after %d iterations", k + 1)
            logger.info("Final parameters:")
            for name, val in self.optimizer.get_engine_values().items():
                logger.info("  %s = %s", name, val)

    def _save_state(self):
        """Persist SPSA state with backup of previous version.

        Renames current state to .bak before writing new state.
        If the write fails, .bak is available for manual recovery.
        Raises on failure — continuing would risk overwriting the good .bak.
        """
        backup_file = self.state_file.with_suffix(".bak")
        if self.state_file.exists():
            os.replace(self.state_file, backup_file)
        with open(self.state_file, "w") as f:
            json.dump(self.optimizer.state.to_dict(), f, indent=2)

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
                "history_total": len(history),
                "dashboard_history": self.config.dashboard_history,
                "workers": worker_data,
                "session_start": self.optimizer.state.created_at,
                "server_start": self.server_start_time,
            }



class CoordinatorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the coordinator."""

    coordinator: CoordinatorState  # set on class before server starts
    chart_js: str = ""             # cached at startup
    static_dir: str = ""           # set from config

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

    def _serve_static(self):
        """Serve static files from configured static_dir."""
        if not self.static_dir:
            self.send_error(404)
            return
        if not self.path.startswith("/static/"):
            self.send_error(404)
            return
        rel = self.path[len("/static/"):]
        # Resolve and prevent path traversal
        base = Path(self.static_dir).resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            self.send_error(404)
            return
        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        with open(target, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def _render_coordinator_dashboard(self) -> str:
        """Render dashboard, re-reading template from disk each request."""
        template_path = Path(__file__).parent / "dashboard.tmpl"
        try:
            with open(template_path) as f:
                dashboard_template = f.read()
        except FileNotFoundError:
            return "<h1>Error: dashboard.tmpl not found</h1>"

        data = self.coordinator.get_coordinator_dashboard()

        is_done = data["is_done"]
        status_color = "#4CAF50" if not is_done else "#2196F3"
        status_text = "COMPLETE" if is_done else "IN PROGRESS"

        session_start = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(data["session_start"])
        )
        server_start = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(data["server_start"])
        )
        return dashboard_template.format(
            version=VERSION,
            chart_js=self.chart_js,
            status_color=status_color,
            status_text=status_text,
            iteration=data["iteration"],
            max_iters=data["max_iterations"],
            progress=data["progress_pct"],
            games_done=data["games_completed_in_iteration"],
            games_total=data["games_per_iteration"],
            iter_progress=data["current_iteration_progress_pct"],
            games_pending=data["games_pending"],
            a_k=data["a_k"],
            c_k=data["c_k"],
            total_games=data["total_games"],
            theta_json=json.dumps(data.get("theta", {})),
            workers_json=json.dumps(data.get("workers", [])),
            history_json=json.dumps(data.get("history", [])),
            history_total=data["history_total"],
            dashboard_history=data["dashboard_history"] or "all",
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

    def _render_logs_page(self) -> str:
        """Render logs page with coordinator logs."""
        template_path = Path(__file__).parent / "logs.tmpl"
        try:
            with open(template_path) as f:
                template = f.read()
        except FileNotFoundError:
            return "<h1>Error: logs.tmpl not found</h1>"

        # Read log file
        log_file = self.coordinator.logs_dir / "coordinator.log"
        log_lines_html = ""
        total_lines = 0

        try:
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip('\n\r')
                        # Escape HTML special characters
                        line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

                        # Add color coding based on log level
                        css_class = "log-line"
                        if "[ERROR]" in line:
                            css_class += " log-line-error"
                        elif "[WARNING]" in line:
                            css_class += " log-line-warning"
                        elif "[INFO]" in line:
                            css_class += " log-line-info"
                        elif "[DEBUG]" in line:
                            css_class += " log-line-debug"

                        log_lines_html += f'<div class="{css_class}">{line}</div>\n'
                        total_lines += 1
            else:
                log_lines_html = '<div class="log-line">Log file not found</div>'
                total_lines = 1
        except Exception as e:
            log_lines_html = f'<div class="log-line log-line-error">Error reading log file: {str(e)}</div>'
            total_lines = 1

        return template.format(
            version=VERSION,
            log_lines=log_lines_html,
            total_lines=total_lines,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _handle_sse(self):
        """Stream dashboard updates as Server-Sent Events.

        Sends live data plus only the latest history entry (to keep
        payloads small).  The client appends new history entries and
        falls back to a full page reload if it detects a gap.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        coord = self.coordinator
        changed = coord.dashboard_changed
        max_interval = coord.config.dashboard_refresh
        last_version = -1  # force first send
        try:
            while True:
                with changed:
                    if coord.dashboard_version == last_version:
                        changed.wait(timeout=max_interval)
                    last_version = coord.dashboard_version

                data = coord.get_coordinator_dashboard()
                is_done = data["is_done"]
                history = data.get("history", [])
                payload = json.dumps({
                    "status_color": "#2196F3" if is_done else "#4CAF50",
                    "status_text": "COMPLETE" if is_done else "IN PROGRESS",
                    "iteration": data["iteration"],
                    "max_iters": data["max_iterations"],
                    "progress": data["progress_pct"],
                    "games_done": data["games_completed_in_iteration"],
                    "games_total": data["games_per_iteration"],
                    "iter_progress": data["current_iteration_progress_pct"],
                    "games_pending": data["games_pending"],
                    "a_k": data["a_k"],
                    "c_k": data["c_k"],
                    "total_games": data["total_games"],
                    "theta": data.get("theta", {}),
                    "workers": data.get("workers", []),
                    "last_history": history[-1] if history else None,
                    "history_len": data["history_total"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "server_start": time.strftime(
                        "%Y-%m-%d %H:%M:%S",
                        time.localtime(data["server_start"]),
                    ),
                })
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected

    def do_GET(self):
        if self.path in ("", "/", "/dashboard"):
            html = self._render_coordinator_dashboard()
            self._send_html(html)
        elif self.path == "/sse":
            self._handle_sse()
        elif self.path == "/charts":
            html = self._render_charts_page()
            self._send_html(html)
        elif self.path == "/logs":
            html = self._render_logs_page()
            self._send_html(html)
        elif self.path.startswith("/static/"):
            self._serve_static()
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

    # Cache chart.js at startup (large, doesn't change)
    spsa_dir = Path(__file__).parent
    chart_js_path = spsa_dir / "chart.umd.min.js"

    if chart_js_path.exists():
        with open(chart_js_path) as f:
            CoordinatorHandler.chart_js = f.read()

    if getattr(config, "static_dir", ""):
        CoordinatorHandler.static_dir = config.static_dir

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

    while True:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            if coordinator.draining:
                logger.info("Force stop.")
                break
            iter_k = coordinator.optimizer.iteration
            if not coordinator.optimizer.is_done():
                can_restart = sys.platform != "win32"
                try:
                    if can_restart:
                        prompt = f"\nWait for iteration {iter_k}? [r]estart / [s]top / [N]o "
                    else:
                        prompt = f"\nWait for iteration {iter_k} to complete? [s]top / [N]o "
                    answer = input(prompt).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = ""
                if answer in ("r", "s") if can_restart else answer == "s":
                    coordinator.draining = True
                    coordinator._restart = (answer == "r")
                    coordinator._notify_dashboard()
                    logger.info("Draining — waiting for iteration %d to complete (Ctrl+C again to force stop)...", iter_k)
                    # Break out of serve_forever() once the drain completes;
                    # needs a thread because serve_forever() blocks.
                    def drain_watcher():
                        coordinator.drain_complete.wait()
                        server.shutdown()
                    threading.Thread(target=drain_watcher, daemon=True).start()
                    continue
            break
        else:
            break  # serve_forever returned normally (drain complete)

    coordinator._save_state()
    server.server_close()
    if coordinator.draining and coordinator._restart:
        logger.info("Restarting coordinator...")
        restart_cmd = [sys.argv[0], "-c", args.config, "-p", str(args.port)]
        os.execv(sys.executable, [sys.executable] + restart_cmd)
    else:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
