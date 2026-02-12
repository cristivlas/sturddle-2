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
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

from config import TuningConfig, WorkItem, WorkResult
from spsa import SPSAOptimizer, SPSAState

VERSION = "1.0.2"

logger = logging.getLogger("coordinator")


class WorkerStatus(Enum):
    ONLINE = 0
    OVERDUE = 1
    TIMED_OUT = 2


@dataclass
class WorkerInfo:
    """Tracked state for a connected worker."""
    name: str
    last_seen: float
    chunks_completed: int = 0
    games_completed: int = 0
    games_completed_iter: int = 0 # games completed in current iteration
    _spg_ewma: float = 0.0        # exponentially weighted moving average (sec/game)
    _ewma_alpha: float = 0.3      # smoothing factor: higher = more weight on recent
    cutechess_overrides: dict = field(default_factory=dict)  # worker-local tc/depth

    @property
    def sec_per_game(self) -> float:
        return self._spg_ewma

    @property
    def games_per_second(self) -> float:
        return 1.0 / self._spg_ewma if self._spg_ewma > 0 else 0.0

    def update_speed(self, games: int, elapsed: float):
        """Update EWMA speed estimate from a completed chunk."""
        if elapsed <= 0 or games <= 0:
            return
        sample = elapsed / games
        old = self._spg_ewma
        if self._spg_ewma <= 0:
            self._spg_ewma = sample  # first observation
        else:
            self._spg_ewma = (
                self._ewma_alpha * sample
                + (1 - self._ewma_alpha) * self._spg_ewma
            )
        logger.debug(
            "Speed update %s: %d games in %.1fs (%.2f s/g), ewma %.2f -> %.2f",
            self.name, games, elapsed, sample, old, self._spg_ewma,
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
        self.games_completed = 0
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0
        self.pending_chunks = {}  # chunk_id -> ChunkInfo
        self.stolen_chunks = {}   # stolen_cid -> replacement_cid

        # Worker registry: keyed by self-reported name (hostname or worker config
        # "name" field).  Fine for trusted homelab / LAN setups; a public-facing
        # deployment would need IP-based validation or auth tokens.
        self.workers = {}  # name -> WorkerInfo

        # Time estimates and timeouts
        self._base_sec_per_game = self._estimate_game_duration()
        # Max timeout: at least 30 min, or enough for the largest possible chunk
        max_chunk_games = tuning_config.games_per_iteration // 2
        self.max_chunk_timeout = max(
            1800.0,
            max_chunk_games * self._base_sec_per_game * self.config.chunk_timeout_factor,
        )
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

    def _games_in_flight(self) -> int:
        """Games currently assigned to workers (pending results)."""
        return sum(c.num_games for c in self.pending_chunks.values())

    def _games_assigned(self) -> int:
        """Completed + in-flight games (computed from ground truth)."""
        return self.games_completed + self._games_in_flight()

    def _worker_sec_per_game(self, worker_name: str) -> float:
        """Per-game time: observed EWMA or config-based fallback."""
        w = self.workers.get(worker_name)
        if w:
            if w.sec_per_game > 0:
                return w.sec_per_game
            if w.cutechess_overrides:
                return self._estimate_game_duration(w.cutechess_overrides)
        return self._base_sec_per_game

    def _touch_worker(self, name: str):
        """Register or update a worker's last-seen timestamp."""
        assert name, "worker name required (enforced at HTTP boundary)"
        now = time.time()
        if name not in self.workers:
            self.workers[name] = WorkerInfo(name=name, last_seen=now)
            logger.info("Worker registered: %s", name)
        else:
            self.workers[name].last_seen = now

    def _is_overdue(self, now: float, assign_time: float, expected: float) -> bool:
        return (now - assign_time) > expected * self.config.overdue_factor

    def _worker_status(self, name: str) -> WorkerStatus:
        """Determine worker health: online, overdue, or timed out."""
        w = self.workers.get(name)
        if not w:
            return WorkerStatus.TIMED_OUT
        now = time.time()
        chunks = [c for c in self.pending_chunks.values() if c.worker_name == name]
        if not chunks:
            if (now - w.last_seen) < self.config.worker_idle_timeout:
                return WorkerStatus.ONLINE
            return WorkerStatus.TIMED_OUT
        if not any(now - c.assign_time < c.timeout for c in chunks):
            return WorkerStatus.TIMED_OUT
        if all(self._is_overdue(now, c.assign_time, c.expected_duration) for c in chunks):
            return WorkerStatus.OVERDUE
        return WorkerStatus.ONLINE

    def _is_worker_alive(self, name: str) -> bool:
        return self._worker_status(name) != WorkerStatus.TIMED_OUT

    def _active_workers(self) -> list:
        """Return list of workers considered alive."""
        return [w for w in self.workers.values()
                if self._is_worker_alive(w.name)]

    def _chunk_eta_per_worker(self, now: float) -> dict[str, float]:
        """Sum of remaining expected seconds per worker across pending chunks."""
        eta = {}
        for chunk in self.pending_chunks.values():
            remaining = max(0, chunk.expected_duration - (now - chunk.assign_time))
            eta[chunk.worker_name] = eta.get(chunk.worker_name, 0) + remaining
        return eta

    def _chunk_timeout_for(self, worker_name: str, num_games: int) -> float:
        """Timeout for a chunk based on expected duration."""
        expected = num_games * self._worker_sec_per_game(worker_name)
        timeout = expected * self.config.chunk_timeout_factor
        return max(self.config.min_chunk_timeout, min(self.max_chunk_timeout, timeout))

    def _compute_chunk_size(self, worker_name: str, remaining: int) -> int:
        """
        Split remaining games proportional to worker speed.

        Bootstrap (no speed data): split evenly across active workers.
        Unproven workers (no EWMA) get a small bootstrap chunk so they
        establish a speed estimate before receiving full-sized work.
        Even-rounding and final clamping handled by get_work().
        """
        active = self._active_workers()
        num_workers = max(1, len(active))
        total_speed = sum(w.games_per_second for w in active)

        w = self.workers.get(worker_name)
        bootstrap = w and w.sec_per_game <= 0

        if total_speed > 0:
            my_speed = w.games_per_second if (w and w.games_per_second > 0) else (total_speed / num_workers)
            chunk = int(remaining * my_speed / total_speed)
        else:
            chunk = remaining // num_workers

        # Cap unproven workers until they establish an EWMA
        if bootstrap:
            cap = max(10, int(self.config.min_chunk_timeout / self._base_sec_per_game))
            chunk = min(chunk, cap)

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

        spg = self._worker_sec_per_game(worker_name)
        now = time.time()

        best_cid = None
        # Saving must exceed per-chunk startup overhead. EWMA tracks
        # per-game time so the fixed cost (process spawn, engine init)
        # is amortized away in large chunks and not visible in the estimate.
        best_saving = self.config.min_chunk_expected_duration

        for cid, chunk in self.pending_chunks.items():
            if chunk.worker_name == worker_name:
                continue

            elapsed = now - chunk.assign_time
            expected = chunk.expected_duration
            new_expected = chunk.num_games * spg

            overdue = self._is_overdue(now, chunk.assign_time, expected)
            if overdue:
                logger.warning(
                    "Work steal: %s taking %s [%s] %d games (overdue) — elapsed=%.1fs expected=%.1fs",
                    worker_name, chunk.worker_name, cid, chunk.num_games, elapsed, expected,
                )
                best_cid = cid
                break

            # Don't speed-steal from freshly assigned chunks (prevents chain stealing)
            if elapsed < self.config.min_chunk_expected_duration:
                continue

            if new_expected + elapsed < expected:
                saving = expected - elapsed - new_expected
                logger.debug(
                    "Work steal: %s eyeing %s [%s] %d games — elapsed=%.1fs expected=%.1fs new=%.1fs saving=%.1fs",
                    worker_name, chunk.worker_name, cid, chunk.num_games, elapsed, expected, new_expected, saving,
                )
            else:
                logger.debug(
                    "Work steal: %s skip %s [%s] %d games — elapsed=%.1fs expected=%.1fs new=%.1fs",
                    worker_name, chunk.worker_name, cid, chunk.num_games, elapsed, expected, new_expected,
                )
                continue

            if saving > best_saving:
                best_cid = cid
                best_saving = saving

        if best_cid is None:
            return None

        chunk = self.pending_chunks.pop(best_cid)
        logger.info(
            "Work steal: %d games from %s [%s] for %s (%.0fs elapsed, est. saving %.1fs, %.2f s/g)",
            chunk.num_games, chunk.worker_name, best_cid,
            worker_name, now - chunk.assign_time, best_saving, spg,
        )
        return best_cid, chunk.num_games

    def _release_chunk(self, cid: str, reason: str):
        """Remove a chunk from pending tracking and clean up stolen_chunks."""
        chunk = self.pending_chunks.pop(cid)
        stolen_cid = next((k for k, v in self.stolen_chunks.items() if v == cid), None)
        if stolen_cid:
            self.stolen_chunks.pop(stolen_cid)
        if cid in self.stolen_chunks:
            self.stolen_chunks.pop(cid)
        logger.warning("Released [%s] (%d games) from %s: %s", cid, chunk.num_games, chunk.worker_name, reason)

    def _reclaim_timed_out_chunks(self):
        """Reclaim chunks that have exceeded their per-worker timeout."""
        now = time.time()
        timed_out = [(cid, now - c.assign_time, c.timeout)
                     for cid, c in self.pending_chunks.items()
                     if now - c.assign_time > c.timeout]
        for cid, elapsed, timeout in timed_out:
            self._release_chunk(cid, "timed out (%.0fs elapsed, timeout %ds)" % (elapsed, int(timeout)))

    def _notify_dashboard(self):
        """Wake up any SSE listeners so they push fresh data immediately."""
        with self.dashboard_changed:
            self.dashboard_version += 1
            self.dashboard_changed.notify_all()

    def get_work(self, chunk_size: int, worker_name: str, cutechess_overrides: dict) -> dict:
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
        assert worker_name, "worker_name required; enforced at HTTP boundary"
        with self.lock:
            self._touch_worker(worker_name)
            if cutechess_overrides and worker_name in self.workers:
                self.workers[worker_name].cutechess_overrides = cutechess_overrides

            if self.optimizer.is_done():
                return {"status": "done"}

            # Reclaim games from workers that disappeared mid-chunk
            self._reclaim_timed_out_chunks()

            # Worker reconnected: release its old chunks and reset EWMA
            stale = [cid for cid, c in self.pending_chunks.items() if c.worker_name == worker_name]
            if stale:
                for cid in stale:
                    self._release_chunk(cid, "worker %s reconnected" % worker_name)
                w = self.workers.get(worker_name)
                if w:
                    w._spg_ewma = 0.0

            gpi = self.config.games_per_iteration
            remaining = gpi - self._games_assigned()
            logger.debug("get_work: %s, remaining=%d, completed=%d, in_flight=%d",
                         worker_name, remaining, self.games_completed, self._games_in_flight())
            stolen_cid = None
            stolen_games = 0

            if remaining <= 0:
                steal = None
                if self.config.work_stealing:
                    steal = self._try_steal_chunk(worker_name)
                if steal:
                    stolen_cid, stolen_games = steal
                    remaining = gpi - self._games_assigned()
                elif self.draining and not self._restart:
                    return {"status": "done"}
                else:
                    return {"status": "retry", "retry_after": self.config.retry_after}

            if stolen_cid:
                num_games = stolen_games
            else:
                adaptive = self._compute_chunk_size(worker_name, remaining)
                if chunk_size > 0:
                    adaptive = min(adaptive, chunk_size)
                num_games = min(remaining, adaptive)

            if num_games == 0:
                return {"status": "retry", "retry_after": self.config.retry_after}

            # Must be even (each game pair is +c vs -c); round up
            num_games += num_games % 2

            # Generate unique chunk ID and compute timeout
            chunk_id = uuid.uuid4().hex[:12]
            timeout = self._chunk_timeout_for(worker_name, num_games)
            expected_duration = max(
                self.config.min_chunk_expected_duration,
                num_games * self._worker_sec_per_game(worker_name),
            )

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
                work.iteration, self._games_assigned(), gpi, int(timeout),
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
                    logger.info("Replacement [%s] from %s won against [%s]", result.chunk_id, result.worker, stolen_cid)
            elif result.chunk_id in self.stolen_chunks:
                replacement_cid = self.stolen_chunks.pop(result.chunk_id)
                if replacement_cid in self.pending_chunks:
                    replacement = self.pending_chunks.pop(replacement_cid)
                    logger.info("Got [%s] from %s, cancelled [%s] (%d games)", result.chunk_id, result.worker, replacement_cid, replacement.num_games)
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
        display = self._get_display_values()
        for name in new_theta:
            param = self.optimizer.params[name]
            step = new_theta[name] - old_theta[name]
            r = param.upper - param.lower
            logger.info(
                "  %s: %.4f -> %.4f (engine: %s, step: %+.4f, %.1f%% of range)",
                name, old_theta[name], new_theta[name], display[name],
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
            for name, val in self._get_display_values().items():
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
            eta_per_worker = self._chunk_eta_per_worker(now)
            return {
                "iteration": self.optimizer.iteration,
                "max_iterations": self.optimizer.max_iterations,
                "is_done": self.optimizer.is_done(),
                "theta": self._get_display_values(),
                "games_completed": self.games_completed,
                "games_per_iteration": self.config.games_per_iteration,
                "c_k": self.optimizer.c_k() if not self.optimizer.is_done() else 0,
                "a_k": self.optimizer.a_k() if not self.optimizer.is_done() else 0,
                "workers": {
                    name: {
                        "alive": self._is_worker_alive(name),
                        "status": self._worker_status(name).value,
                        "last_seen_ago": round(now - w.last_seen, 1),
                        "games_completed": w.games_completed,
                        "games_completed_iter": w.games_completed_iter,
                        "sec_per_game": round(self._worker_sec_per_game(name), 1),
                        "chunk_eta": round(eta_per_worker.get(name, 0), 0),
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

    def _has_normalized_params(self) -> bool:
        """Check if any parameters are normalized (have original range info)."""
        return any(getattr(p, 'is_normalized', False) for p in self.config.parameters.values())

    def _get_display_values(self) -> dict:
        """Engine-space display values: denormalized integers for normalized params."""
        result = {}
        for name, val in self.optimizer.theta.items():
            param = self.config.parameters.get(name)
            if param and getattr(param, 'is_normalized', False):
                result[name] = param.denormalize(val)
            elif param:
                result[name] = param.to_engine_value(val)
            else:
                result[name] = val
        return result

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

            eta_per_worker = self._chunk_eta_per_worker(now)
            worker_data = []
            for name, w in self.workers.items():
                worker_data.append({
                    "name": name,
                    "alive": self._is_worker_alive(name),
                    "status": self._worker_status(name).value,
                    "last_seen_ago": round(now - w.last_seen, 1),
                    "games_assigned": assigned_per_worker.get(name, 0),
                    "games_completed_iter": w.games_completed_iter,
                    "games_completed": w.games_completed,
                    "sec_per_game": round(self._worker_sec_per_game(name), 1),
                    "chunk_eta": round(eta_per_worker.get(name, 0), 0),
                })

            has_normalized = self._has_normalized_params()
            result = {
                "iteration": self.optimizer.iteration,
                "max_iterations": self.optimizer.max_iterations,
                "is_done": self.optimizer.is_done(),
                "progress_pct": min(100, pct_complete),
                "current_iteration_progress_pct": min(100, pct_games),
                "games_completed_in_iteration": self.games_completed,
                "games_per_iteration": gpi,
                "games_assigned": self._games_assigned(),
                "games_pending": self._games_in_flight(),
                "total_games": self.optimizer.iteration * gpi + self.games_completed,
                "theta": self._get_display_values(),
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
                "has_normalized": has_normalized,
            }
            if has_normalized:
                result["theta_internal"] = dict(self.optimizer.theta)
            return result



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
            theta_internal_json=json.dumps(data.get("theta_internal", {})),
            has_normalized=json.dumps(data.get("has_normalized", False)),
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
                sse_data = {
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
                }
                if "theta_internal" in data:
                    sse_data["theta_internal"] = data["theta_internal"]
                payload = json.dumps(sse_data)
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
            if not worker_name:
                self._send_json({"status": "error", "reason": "worker name required"})
                return
            cc_overrides = data.get("cutechess_overrides")
            result = self.coordinator.get_work(chunk_size, worker_name, cc_overrides)
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
