#!/usr/bin/env python3
"""
SPSA tuning coordinator: HTTP server that owns the SPSA state and hands out games.

Usage:
    python coordinator.py -c tuning.json [-p 8080] [--clean] [--debug]
"""

import argparse
import json
import logging
import logging.handlers
import math
import mimetypes
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

from config import TuningConfig, WorkItem, WorkResult
from spsa import SPSAOptimizer, SPSAState

VERSION = "1.2.2"

ITER_DURATION_EWMA_ALPHA = 0.3  # session ETA smoothing

logger = logging.getLogger("coordinator")


class WorkerStatus(Enum):
    ONLINE = 0
    OVERDUE = 1
    TIMED_OUT = 2


@dataclass
class WorkerInfo:
    name: str
    last_seen: float
    chunks_completed: int = 0
    games_completed: int = 0
    games_completed_iter: int = 0
    _spg_ewma: float = 0.0        # sec/game
    cutechess_overrides: dict = field(default_factory=dict)  # worker-local tc/depth
    concurrency: int = 0          # 0 = unknown

    @property
    def sec_per_game(self) -> float:
        return self._spg_ewma

    @property
    def games_per_second(self) -> float:
        return 1.0 / self._spg_ewma if self._spg_ewma > 0 else 0.0

    def update_speed(self, games: int, elapsed: float, tau_seconds: float):
        """Time-weighted EWMA: alpha = elapsed / (elapsed + tau_seconds), so long chunks weigh more."""
        if elapsed <= 0 or games <= 0 or tau_seconds <= 0:
            return
        sample = elapsed / games
        old = self._spg_ewma
        if self._spg_ewma <= 0:
            self._spg_ewma = sample
        else:
            alpha = elapsed / (elapsed + tau_seconds)
            self._spg_ewma = alpha * sample + (1 - alpha) * self._spg_ewma
        logger.debug("Speed update %s: %d games in %.1fs (%.2f s/g), ewma %.2f -> %.2f", self.name, games, elapsed, sample, old, self._spg_ewma)


@dataclass
class ChunkInfo:
    chunk_id: str
    worker_name: str
    num_games: int
    assign_time: float
    timeout: float
    expected_duration: float = 0.0


class CoordinatorState:
    """Thread-safe SPSA state, one iteration at a time."""

    def __init__(self, tuning_config: TuningConfig, resume: bool):
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

        if self.optimizer.state.created_at == 0.0:
            self.optimizer.state.created_at = time.time()

        self.current_delta = None
        self.current_work = None  # WorkItem template
        self.games_completed = 0
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0
        # reference mode
        self.total_plus_wins = 0
        self.total_plus_draws = 0
        self.total_plus_losses = 0
        self.total_minus_wins = 0
        self.total_minus_draws = 0
        self.total_minus_losses = 0
        self.iteration_reference_mode = None  # set by the first result
        self.iteration_start_time = time.time()
        self.pending_chunks = {}  # chunk_id -> ChunkInfo

        # for the per-iteration timing log
        self._iter_overflow_retries = 0
        self._iter_boundary_retries = 0
        self._iter_stale_submissions = 0

        # Not persisted: no session ETA until an iteration completes after a (re)start.
        self._iter_duration_ewma: float = 0.0

        # Keyed by self-reported name, no auth: trusted LAN only.
        self.workers = {}  # name -> WorkerInfo
        self._worker_stats = {}  # disconnected: name -> (games_completed, chunks_completed, games_completed_iter)
        self._worker_connect_times = {}
        self._mode_mismatch_workers: set = set()  # told "done" on their next /work

        self._base_sec_per_game = self._estimate_game_duration()
        self.server_start_time = time.time()
        self._prepare_iteration()
        self.total_games_at_start = (
            self._historical_games_total() + self.games_completed
        )

        if self.games_completed > 0:
            logger.info(
                "Resuming iteration %d with %d/%d games already scored",
                self.optimizer.iteration, self.games_completed,
                self.config.games_per_iteration,
            )

    def _historical_games_total(self) -> int:
        """Games over completed iterations; old entries without "games" count as games_per_iteration."""
        gpi = self.config.games_per_iteration
        return sum(h["games"] if "games" in h else gpi for h in self.optimizer.state.history)

    def _estimate_game_duration(self, overrides: dict = None) -> float:
        """Seconds per game guessed from depth or time control; worker overrides win."""
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
        return sum(c.num_games for c in self.pending_chunks.values())

    def _games_assigned(self) -> int:
        return self.games_completed + self._games_in_flight()

    def _worker_sec_per_game(self, worker_name: str) -> float:
        w = self.workers.get(worker_name)
        if w:
            if w.sec_per_game > 0:
                return w.sec_per_game
            if w.cutechess_overrides:
                return self._estimate_game_duration(w.cutechess_overrides)
        return self._base_sec_per_game

    def _touch_worker(self, name: str):
        assert name
        now = time.time()
        if name not in self.workers:
            if self.optimizer.is_done():
                return
            w = WorkerInfo(name=name, last_seen=now)
            saved = self._worker_stats.pop(name, None)
            if saved:
                w.games_completed, w.chunks_completed, w.games_completed_iter = saved
                logger.info("Worker reconnected: %s (%d games)", name, w.games_completed)
            else:
                logger.info("Worker registered: %s", name)
            self.workers[name] = w
        else:
            if self._worker_status(name) == WorkerStatus.TIMED_OUT:
                logger.info("Worker %s back after timeout, resetting speed estimate", name)
                self.workers[name]._spg_ewma = 0.0
            self.workers[name].last_seen = now

    def _is_overdue(self, now: float, assign_time: float, expected: float) -> bool:
        return (now - assign_time) > expected * self.config.overdue_factor

    def _worker_status(self, name: str) -> WorkerStatus:
        w = self.workers.get(name)
        if not w:
            return WorkerStatus.TIMED_OUT
        now = time.time()
        # validate heartbeats keep last_seen fresh during games
        if self.config.validate_interval:
            if (now - w.last_seen) >= self.config.worker_idle_timeout:
                return WorkerStatus.TIMED_OUT
            chunks = [c for c in self.pending_chunks.values() if c.worker_name == name]
            if chunks and all(self._is_overdue(now, c.assign_time, c.expected_duration) for c in chunks):
                return WorkerStatus.OVERDUE
            return WorkerStatus.ONLINE
        # no heartbeats: use chunk timeouts
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
        return [w for w in self.workers.values()
                if self._is_worker_alive(w.name)]

    def _chunk_eta_per_worker(self, now: float) -> dict[str, float]:
        """Remaining expected seconds per worker."""
        eta = {}
        for chunk in self.pending_chunks.values():
            remaining = max(0, chunk.expected_duration - (now - chunk.assign_time))
            eta[chunk.worker_name] = eta.get(chunk.worker_name, 0) + remaining
        return eta

    def _worker_throughput(self) -> float:
        """EWMA games/sec over active workers; ignores idle time and retries."""
        return sum(w.games_per_second for w in self._active_workers())

    def _iter_eta_seconds(self) -> float:
        """Seconds left in this iteration (0 = unknown); a bit optimistic, but smooth."""
        tp = self._worker_throughput()
        if tp <= 0:
            return 0.0
        games_left = max(0, self.config.games_per_iteration - self.games_completed)
        return games_left / tp

    def _compute_etas(self, now: float) -> dict:
        """ETAs for the dashboard; clients count down from arrival time."""
        if self.optimizer.is_done():
            return {"iter_eta_s": 0, "session_eta_s": 0, "avg_iter_sec": 0}

        iter_eta_s = self._iter_eta_seconds()
        avg_iter = self._iter_duration_ewma
        iter_elapsed = max(0.0, now - self.iteration_start_time)
        full_iters_left = max(0, self.optimizer.max_iterations - self.optimizer.iteration - 1)

        if avg_iter > 0:
            current_iter_remaining = max(0.0, avg_iter - iter_elapsed)
            session_eta_s = full_iters_left * avg_iter + current_iter_remaining
        else:
            # throughput extrapolation is too optimistic, wait for the EWMA
            session_eta_s = 0.0

        return {
            "iter_eta_s": round(iter_eta_s, 1),
            "session_eta_s": round(session_eta_s, 1),
            "avg_iter_sec": round(avg_iter, 1),
        }

    def _chunk_timeout_for(self, worker_name: str, num_games: int) -> float:
        expected = num_games * self._worker_sec_per_game(worker_name)
        timeout = expected * self.config.chunk_timeout_factor
        return max(self.config.min_chunk_timeout, timeout)

    def _compute_chunk_size(self, worker_name: str, remaining: int) -> int:
        """Share of remaining games by worker speed; even split without data, capped while warming up."""
        active = self._active_workers()
        num_workers = max(1, len(active))
        total_speed = sum(w.games_per_second for w in active)

        w = self.workers.get(worker_name)
        warming_up = w and (w.sec_per_game <= 0 or w.chunks_completed < self.config.ewma_warmup_chunks)

        if total_speed > 0:
            my_speed = w.games_per_second if (w and w.games_per_second > 0) else (total_speed / num_workers)
            chunk = int(remaining * my_speed / total_speed)
        else:
            chunk = remaining // num_workers

        if warming_up:
            if w and w.concurrency > 0:
                cap = w.concurrency
            else:
                cap = max(12, int(self.config.min_chunk_timeout / self._base_sec_per_game))
            chunk = min(chunk, cap)

        if remaining - chunk < num_workers:
            chunk = remaining

        return chunk

    def _prepare_iteration(self):
        """Set up the current iteration, resuming partial progress from saved state."""
        if self.optimizer.is_done():
            return

        st = self.optimizer.state

        if st.current_delta:
            self.current_delta = st.current_delta
            self.games_completed = st.games_completed
            self.total_wins = st.total_wins
            self.total_draws = st.total_draws
            self.total_losses = st.total_losses
            self.total_plus_wins = st.total_plus_wins
            self.total_plus_draws = st.total_plus_draws
            self.total_plus_losses = st.total_plus_losses
            self.total_minus_wins = st.total_minus_wins
            self.total_minus_draws = st.total_minus_draws
            self.total_minus_losses = st.total_minus_losses
        else:
            self.current_delta = self.optimizer.generate_perturbation()
            self.games_completed = 0
            self.total_wins = 0
            self.total_draws = 0
            self.total_losses = 0
            self.total_plus_wins = 0
            self.total_plus_draws = 0
            self.total_plus_losses = 0
            self.total_minus_wins = 0
            self.total_minus_draws = 0
            self.total_minus_losses = 0

        theta_plus, theta_minus = self.optimizer.compute_candidates(self.current_delta)
        self.current_work = WorkItem(
            iteration=self.optimizer.iteration,
            theta_plus=theta_plus,
            theta_minus=theta_minus,
            num_games=0,  # filled per chunk
        )
        self.pending_chunks = {}

        self._iter_overflow_retries = 0
        self._iter_boundary_retries = 0
        self._iter_stale_submissions = 0

        # disconnected workers too
        for w in self.workers.values():
            w.games_completed_iter = 0
        for n in self._worker_stats:
            g, c, _ = self._worker_stats[n]
            self._worker_stats[n] = (g, c, 0)

        logger.info(
            "Iteration %d: c_k=%.4f, a_k=%.6f",
            self.optimizer.iteration,
            self.optimizer.c_k(),
            self.optimizer.a_k(),
        )

    def _release_chunk(self, cid: str, reason: str):
        chunk = self.pending_chunks.pop(cid)
        logger.warning("Released [%s] (%d games) from %s: %s", cid, chunk.num_games, chunk.worker_name, reason)

    def _graceful_disconnect(self, worker_name: str):
        """Release a departing worker's chunks and keep its stats; no-op if already gone."""
        if worker_name not in self.workers:
            return
        w = self.workers[worker_name]
        for cid in [c for c, ci in self.pending_chunks.items() if ci.worker_name == worker_name]:
            self._release_chunk(cid, "worker %s shutting down" % worker_name)
        self._worker_stats[worker_name] = (w.games_completed, w.chunks_completed, w.games_completed_iter)
        del self.workers[worker_name]
        logger.info("Worker %s disconnected gracefully", worker_name)

    def _reclaim_timed_out_chunks(self):
        """Release chunks of dead workers; overdue chunks of live ones get raced by overflow chunks instead."""
        now = time.time()
        alive = {w.name for w in self._active_workers()}
        stale = []
        for cid, c in self.pending_chunks.items():
            if c.worker_name not in alive:
                w = self.workers.get(c.worker_name)
                ago = "%.0fs ago" % (now - w.last_seen) if w else "unknown"
                stale.append((cid, "worker %s disconnected (last seen %s)" % (c.worker_name, ago)))
        for cid, reason in stale:
            self._release_chunk(cid, reason)

    def _notify_dashboard(self):
        with self.dashboard_changed:
            self.dashboard_version += 1
            self.dashboard_changed.notify_all()

    def get_work(self, chunk_size: int, worker_name: str, cutechess_overrides: dict, worker_server_start: float, worker_concurrency: int = 0) -> "dict | bool":
        """WorkItem dict, a status dict, or False to deny a crash-looping worker."""
        assert worker_name
        with self.lock:
            self._touch_worker(worker_name)
            if worker_name in self._mode_mismatch_workers:
                self._mode_mismatch_workers.discard(worker_name)
                self.workers.pop(worker_name, None)
                logger.warning("Rejecting banned worker %s (mode mismatch); fix config and reconnect", worker_name)
                return {"status": "done"}
            if worker_server_start and worker_server_start != self.server_start_time:
                return {"status": "config_changed"}
            if cutechess_overrides and worker_name in self.workers:
                self.workers[worker_name].cutechess_overrides = cutechess_overrides
            if worker_concurrency > 0 and worker_name in self.workers:
                self.workers[worker_name].concurrency = worker_concurrency

            if self.optimizer.is_done():
                return {"status": "done"}

            self._reclaim_timed_out_chunks()

            # still holding chunks: it crashed and reconnected
            stale = [cid for cid, c in self.pending_chunks.items() if c.worker_name == worker_name]
            if stale:
                # Rate-limit crash-reconnects
                max_r = self.config.max_retries
                if max_r > 0:
                    now = time.time()
                    window = max_r * self.config.retry_after
                    last = self._worker_connect_times.get(worker_name, 0)
                    if now - last < window:
                        logger.warning("Worker %s denied: crash-reconnect within %.0fs", worker_name, window)
                        return False
                    self._worker_connect_times[worker_name] = now
                for cid in stale:
                    self._release_chunk(cid, "worker %s reconnected" % worker_name)
                w = self.workers.get(worker_name)
                if w:
                    w._spg_ewma = 0.0

            gpi = self.config.games_per_iteration
            remaining = gpi - self._games_assigned()
            overflow = False
            if remaining <= 0:
                # Stragglers hold the rest: race them, late results get dropped as stale.
                if not self.pending_chunks:
                    if self.draining and not self._restart:
                        return {"status": "done"}
                    self._iter_boundary_retries += 1
                    return {"status": "retry", "retry_after": self.config.retry_after, "server_start": self.server_start_time}
                overflow = True
                remaining = max(c.num_games for c in self.pending_chunks.values())

            logger.debug("get_work: %s, remaining=%d, completed=%d, in_flight=%d, overflow=%s",
                         worker_name, remaining, self.games_completed, self._games_in_flight(), overflow)

            adaptive = self._compute_chunk_size(worker_name, remaining)
            if chunk_size > 0:
                adaptive = min(adaptive, chunk_size)
            num_games = min(remaining, adaptive)

            mode_unit = 4 if self.iteration_reference_mode else 2
            wc = self.workers[worker_name].concurrency if worker_name in self.workers else 0
            unit = wc * mode_unit // math.gcd(wc, mode_unit) if wc > 0 else mode_unit

            if wc > 0 and self.config.validate_interval > 0:
                # Round up (fewer, bigger chunks) if it still fits in the iteration ETA.
                chunk_up = ((num_games + unit - 1) // unit) * unit
                spg = self.workers[worker_name].sec_per_game
                use_round_up = True
                if spg > 0:
                    iter_eta = self._iter_eta_seconds()
                    if iter_eta > 0:
                        use_round_up = chunk_up * spg <= iter_eta
                if use_round_up:
                    rounded = chunk_up
                else:
                    rounded = ((num_games + unit // 2) // unit) * unit
                if rounded > remaining:
                    rounded = (num_games // unit) * unit
                num_games = max(unit, rounded)
            else:
                if num_games == 0:
                    return {"status": "retry", "retry_after": self.config.retry_after, "server_start": self.server_start_time}
                num_games = ((num_games + mode_unit - 1) // mode_unit) * mode_unit

            # after rounding, so overflow_factor is a real ceiling
            if overflow:
                needed = gpi - self.games_completed
                if self._games_in_flight() + num_games > needed * self.config.overflow_factor:
                    # no bigger than the largest pending chunk: not a new worst straggler
                    largest_pending = max(c.num_games for c in self.pending_chunks.values())
                    if num_games > largest_pending:
                        self._iter_overflow_retries += 1
                        return {"status": "retry", "retry_after": self.config.retry_after, "server_start": self.server_start_time}

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

            work = WorkItem(
                iteration=self.current_work.iteration,
                theta_plus=self.current_work.theta_plus,
                theta_minus=self.current_work.theta_minus,
                num_games=num_games,
                chunk_id=chunk_id,
            )

            logger.info(
                "Assigned %d games to %s [%s] (iter %d, %d/%d%s, conc=%s, timeout=%ds)",
                num_games, worker_name or "?", chunk_id,
                work.iteration, self._games_assigned(), gpi,
                " overflow" if overflow else "",
                wc if wc > 0 else "?",
                int(timeout),
            )
            self._notify_dashboard()
            d = work.to_dict()
            d["server_start"] = self.server_start_time
            return d

    def submit_result(self, result: WorkResult) -> dict:
        """Record a chunk's results; completes the iteration once enough games are in."""
        with self.lock:
            self._touch_worker(result.worker)

            if not result.chunk_id or not result.worker:
                logger.error("Malformed result: %s", result)
                if result.shutting_down:
                    self._graceful_disconnect(result.worker)
                return {"status": "ignored", "reason": "malformed result"}

            if result.iteration != self.optimizer.iteration:
                self._iter_stale_submissions += 1
                logger.warning(
                    "Ignoring stale result (%d games) for iteration %d from %s (current: %d)",
                    result.num_games, result.iteration, result.worker, self.optimizer.iteration,
                )
                if result.shutting_down:
                    self._graceful_disconnect(result.worker)
                return {"status": "ignored", "reason": "stale iteration"}

            if result.chunk_id not in self.pending_chunks:
                logger.warning("Ignoring result for unknown/reclaimed chunk %s from %s", result.chunk_id, result.worker)
                if result.shutting_down:
                    self._graceful_disconnect(result.worker)
                return {"status": "ignored", "reason": "unknown chunk"}
            chunk = self.pending_chunks.pop(result.chunk_id)

            if self.iteration_reference_mode is None:
                self.iteration_reference_mode = result.reference_mode
            elif result.reference_mode != self.iteration_reference_mode:
                mode = "reference" if result.reference_mode else "standard"
                expected = "reference" if self.iteration_reference_mode else "standard"
                logger.error("Mode mismatch from %s (got %s, expected %s), evicting", result.worker, mode, expected)
                self.workers.pop(result.worker, None)
                self._mode_mismatch_workers.add(result.worker)
                return {"status": "ignored", "reason": f"mode mismatch: expected {expected}"}

            # odd count: crashed run, unknown which side got shorted
            if result.num_games % 2 != 0:
                logger.warning("Rejecting odd result (%d games) from %s [%s]", result.num_games, result.worker, result.chunk_id)
                if result.shutting_down:
                    self._graceful_disconnect(result.worker)
                return {"status": "ignored", "reason": "odd game count"}

            self.games_completed += result.num_games

            if result.reference_mode:
                self.total_plus_wins += result.plus_wins
                self.total_plus_draws += result.plus_draws
                self.total_plus_losses += result.plus_losses
                self.total_minus_wins += result.minus_wins
                self.total_minus_draws += result.minus_draws
                self.total_minus_losses += result.minus_losses
                self.total_wins += result.plus_wins + result.minus_wins
                self.total_draws += result.plus_draws + result.minus_draws
                self.total_losses += result.plus_losses + result.minus_losses
            else:
                self.total_wins += result.wins
                self.total_draws += result.draws
                self.total_losses += result.losses

            if result.worker in self.workers:
                w = self.workers[result.worker]
                w.chunks_completed += 1
                w.games_completed += result.num_games
                w.games_completed_iter += result.num_games
                if chunk:
                    # Skip short chunks: startup overhead shrinks the estimate, then the chunks.
                    spg = w.sec_per_game or self._base_sec_per_game
                    if w.sec_per_game <= 0 or result.num_games * spg >= self.config.min_chunk_expected_duration:
                        w.update_speed(result.num_games, time.time() - chunk.assign_time, self.config.min_chunk_expected_duration)
                if result.shutting_down:
                    self._graceful_disconnect(result.worker)

            if result.reference_mode:
                logger.info(
                    "Result: iter %d, %d games from %s [%s], +W=%d/D=%d/L=%d -W=%d/D=%d/L=%d (%d/%d done)",
                    result.iteration, result.num_games,
                    result.worker, result.chunk_id,
                    result.plus_wins, result.plus_draws, result.plus_losses,
                    result.minus_wins, result.minus_draws, result.minus_losses,
                    self.games_completed, self.config.games_per_iteration,
                )
            else:
                logger.info(
                    "Result: iter %d, %d games from %s [%s], W=%d D=%d L=%d (%d/%d done)",
                    result.iteration, result.num_games,
                    result.worker, result.chunk_id,
                    result.wins, result.draws, result.losses,
                    self.games_completed, self.config.games_per_iteration,
                )

            if self.games_completed >= self.config.games_per_iteration:
                self._complete_iteration()
            else:
                self._sync_and_save()

            self._notify_dashboard()
            return {"status": "ok"}

    def _sync_and_save(self):
        st = self.optimizer.state
        st.current_delta = self.current_delta
        st.games_completed = self.games_completed
        st.total_wins = self.total_wins
        st.total_draws = self.total_draws
        st.total_losses = self.total_losses
        st.total_plus_wins = self.total_plus_wins
        st.total_plus_draws = self.total_plus_draws
        st.total_plus_losses = self.total_plus_losses
        st.total_minus_wins = self.total_minus_wins
        st.total_minus_draws = self.total_minus_draws
        st.total_minus_losses = self.total_minus_losses
        self._save_state()

    def _complete_iteration(self):
        iter_wall_clock = time.time() - self.iteration_start_time
        chunks_killed = len(self.pending_chunks)

        if self._iter_duration_ewma <= 0:
            self._iter_duration_ewma = iter_wall_clock
        else:
            self._iter_duration_ewma = (
                ITER_DURATION_EWMA_ALPHA * iter_wall_clock
                + (1 - ITER_DURATION_EWMA_ALPHA) * self._iter_duration_ewma
            )

        total = self.total_wins + self.total_draws + self.total_losses
        dw = self.config.spsa.draw_weight

        total_plus = self.total_plus_wins + self.total_plus_draws + self.total_plus_losses
        total_minus = self.total_minus_wins + self.total_minus_draws + self.total_minus_losses

        if total_plus > 0 and total_minus > 0:
            avg_score_plus = (self.total_plus_wins + dw * self.total_plus_draws) / total_plus
            avg_score_minus = (self.total_minus_wins + dw * self.total_minus_draws) / total_minus
        else:
            avg_score_plus = (self.total_wins + dw * self.total_draws) / total
            avg_score_minus = (self.total_losses + dw * self.total_draws) / total

        k = self.optimizer.iteration
        old_theta = dict(self.optimizer.theta)
        old_display = self._get_display_values()

        # reference mode: skip the update if both sides lost to the reference
        if total_plus > 0 and total_minus > 0 and max(avg_score_plus, avg_score_minus) < 0.5:
            logger.warning(
                "Iteration %d: both sides below reference (%.4f / %.4f), skipping update",
                k, avg_score_plus, avg_score_minus,
            )
            new_theta = old_theta
            self.optimizer.advance(avg_score_plus, avg_score_minus, games=self.games_completed)
        else:
            new_theta = self.optimizer.update(self.current_delta, avg_score_plus, avg_score_minus, games=self.games_completed)

        elo_plus = self.optimizer.elo_estimate(avg_score_plus)
        elo_minus = self.optimizer.elo_estimate(avg_score_minus)

        logger.info("=" * 60)
        logger.info("Iteration %d complete (%d games, W=%d D=%d L=%d)",
                     k, total, self.total_wins, self.total_draws, self.total_losses)
        logger.info(
            "Timing: games_phase=%.1fs, killed_chunks=%d, overflow_retries=%d, boundary_retries=%d, stale_submissions=%d",
            iter_wall_clock, chunks_killed,
            self._iter_overflow_retries, self._iter_boundary_retries, self._iter_stale_submissions,
        )
        if total_plus > 0 and total_minus > 0:
            logger.info(
                "Reference mode: theta+ %dW/%dD/%dL (%d games), theta- %dW/%dD/%dL (%d games)",
                self.total_plus_wins, self.total_plus_draws, self.total_plus_losses, total_plus,
                self.total_minus_wins, self.total_minus_draws, self.total_minus_losses, total_minus,
            )
        logger.info(
            "Scores: +%.4f / -%.4f (diff: %+.4f, ELO diff: %+.1f)",
            avg_score_plus, avg_score_minus,
            avg_score_plus - avg_score_minus,
            elo_plus - elo_minus,
        )
        skipped = new_theta is old_theta
        if skipped:
            logger.info("Parameters unchanged (skipped).")
        else:
            logger.info("Updated parameters:")
            display = self._get_display_values()
            for name in new_theta:
                param = self.optimizer.params[name]
                step = new_theta[name] - old_theta[name]
                r = param.upper - param.lower
                logger.info(
                    "  %s: %s -> %s (step: %+.4f, %.1f%% of range)",
                    name, old_display[name], display[name],
                    step, 100.0 * abs(step) / r if r > 0 else 0,
                )

        active = self._active_workers()
        if active or self._worker_stats:
            logger.info("Worker stats:")
            for w in sorted(active, key=lambda w: w.games_per_second, reverse=True):
                logger.info("  %s: %d games, %.2f games/sec", w.name, w.games_completed, w.games_per_second)
            for name, (games, chunks, _) in sorted(self._worker_stats.items()):
                logger.info("  %s: %d games (disconnected)", name, games)
        logger.info("=" * 60)

        self.games_completed = 0
        self.iteration_reference_mode = None
        self.total_plus_wins = 0
        self.total_plus_draws = 0
        self.total_plus_losses = 0
        self.total_minus_wins = 0
        self.total_minus_draws = 0
        self.total_minus_losses = 0
        st = self.optimizer.state
        st.current_delta = {}
        st.games_completed = 0
        st.total_wins = 0
        st.total_draws = 0
        st.total_losses = 0
        st.total_plus_wins = 0
        st.total_plus_draws = 0
        st.total_plus_losses = 0
        st.total_minus_wins = 0
        st.total_minus_draws = 0
        st.total_minus_losses = 0
        self.iteration_start_time = time.time()
        self._save_state()

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
            self.workers.clear()

    def _save_state(self):
        """Keep the previous state as .bak; raise on failure so the good .bak survives."""
        backup_file = self.state_file.with_suffix(".bak")
        if self.state_file.exists():
            os.replace(self.state_file, backup_file)
        with open(self.state_file, "w") as f:
            json.dump(self.optimizer.state.to_dict(), f, indent=2)

    def get_status(self) -> dict:
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

    def check_chunk(self, chunk_id: str, worker_name: str) -> bool:
        with self.lock:
            self._touch_worker(worker_name)
            return chunk_id in self.pending_chunks

    def get_tuning_config_dict(self) -> dict:
        d = json.loads(self.config.to_json())
        d["server_start"] = self.server_start_time
        return d

    def get_charts_data(self) -> dict:
        with self.lock:
            return {"history": list(self.optimizer.state.history)}

    def _has_normalized_params(self) -> bool:
        return any(getattr(p, 'is_normalized', False) for p in self.config.parameters.values())

    def _get_display_values(self) -> dict:
        result = {}
        for name, val in self.optimizer.theta.items():
            param = self.config.parameters.get(name)
            if param:
                result[name] = param.denormalize(val)
            else:
                result[name] = val
        return result

    def get_coordinator_dashboard(self) -> dict:
        now = time.time()
        with self.lock:
            if self.optimizer.is_done():
                pct_complete = 100
            else:
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
                    "concurrency": w.concurrency,
                })

            has_normalized = self._has_normalized_params()
            throughput = sum(
                w.games_per_second for name, w in self.workers.items()
                if self._is_worker_alive(name)
            )
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
                "total_games": self._historical_games_total() + self.games_completed,
                "total_games_at_start": self.total_games_at_start,
                "throughput": round(throughput, 2),
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
                "iteration_start": self.iteration_start_time,
                "has_normalized": has_normalized,
                **self._compute_etas(now),
            }
            if has_normalized:
                result["theta_internal"] = dict(self.optimizer.theta)
            return result


class CoordinatorHandler(BaseHTTPRequestHandler):
    coordinator: CoordinatorState  # set before the server starts
    chart_js: str = ""
    static_dir: str = ""

    def log_message(self, format, *args):
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
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self):
        if not self.static_dir:
            self.send_error(404)
            return
        if not self.path.startswith("/static/"):
            self.send_error(404)
            return
        rel = self.path[len("/static/"):]
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
        """Templates are re-read on every request."""
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
            total_games_at_start=data["total_games_at_start"],
            throughput=data["throughput"],
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
            iteration_start=data["iteration_start"],
            iter_eta_s=data["iter_eta_s"],
            session_eta_s=data["session_eta_s"],
            avg_iter_sec=data["avg_iter_sec"],
        )

    def _render_charts_page(self) -> str:
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
        template_path = Path(__file__).parent / "logs.tmpl"
        try:
            with open(template_path) as f:
                template = f.read()
        except FileNotFoundError:
            return "<h1>Error: logs.tmpl not found</h1>"

        log_file = self.coordinator.logs_dir / "coordinator.log"
        log_entries = []

        try:
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip('\n\r')
                        level = ""
                        if "[ERROR]" in line:
                            level = "error"
                        elif "[WARNING]" in line:
                            level = "warning"
                        elif "[INFO]" in line:
                            level = "info"
                        elif "[DEBUG]" in line:
                            level = "debug"
                        log_entries.append([line, level])
            else:
                log_entries.append(["Log file not found", ""])
        except Exception as e:
            log_entries.append([f"Error reading log file: {str(e)}", "error"])

        log_data_json = json.dumps(log_entries).replace('{', '{{').replace('}', '}}')

        return template.format(
            version=VERSION,
            log_data_json=log_data_json,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _handle_sse(self):
        """SSE updates carry only the latest history entry; the client reloads on a gap."""
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
                    "total_games_at_start": data["total_games_at_start"],
                    "iter_eta_s": data["iter_eta_s"],
                    "session_eta_s": data["session_eta_s"],
                    "avg_iter_sec": data["avg_iter_sec"],
                    "theta": data.get("theta", {}),
                    "workers": data.get("workers", []),
                    "last_history": history[-1] if history else None,
                    "history_len": data["history_total"],
                    "iteration_start": data["iteration_start"],
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
        elif self.path.startswith("/validate?"):
            qs = parse_qs(urlparse(self.path).query)
            worker = qs.get("worker", [""])[0]
            chunk = qs.get("chunk", [""])[0]
            if not worker or not chunk:
                self._send_json({"error": "worker and chunk required"}, status=400)
            else:
                self._send_json({"valid": self.coordinator.check_chunk(chunk, worker)})
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
            worker_server_start = data.get("server_start", 0)
            worker_concurrency = data.get("concurrency", 0)
            result = self.coordinator.get_work(chunk_size, worker_name, cc_overrides, worker_server_start, worker_concurrency)
            if result is False:
                self.send_error(403)
                return
            self._send_json(result)
        elif self.path == "/result":
            data = self._read_json()
            work_result = WorkResult.from_dict(data)
            result = self.coordinator.submit_result(work_result)
            self._send_json(result)
        else:
            self.send_error(404)


def setup_logging(log_dir: Path, debug: bool, rotate: bool):
    log_file = log_dir / "coordinator.log"

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

    if args.clean:
        if state_file.exists():
            state_file.unlink()
        for log in logs_dir.glob("*.log*"):
            log.unlink()

    logs_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(logs_dir, debug=args.debug, rotate=config.log_rotation)

    if args.clean:
        logger.info("Clean start: state and logs wiped")

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

        def handle_error(self, _request, client_address):
            exc = sys.exc_info()[1]
            if isinstance(exc, ConnectionError):
                logger.warning("Connection error from %s: %s", client_address, exc)
            else:
                logger.error("Request error from %s:", client_address, exc_info=True)

    server = ThreadedHTTPServer(("0.0.0.0", args.port), CoordinatorHandler)
    if coordinator.optimizer.is_done():
        logger.info("Tuning already complete (%d iterations).", coordinator.optimizer.iteration)
    else:
        logger.info("Coordinator ready, waiting for workers...")

    restart_debug = False
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
                    restart_opts = "[r]estart | restart in [d]ebug mode | " if can_restart else ""
                    prompt = f"\nWait for iteration {iter_k}? {restart_opts}[w]ait and stop | [s]top now | [Enter] to dismiss "
                    answer = input(prompt).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = ""
                if answer == "s":
                    break
                if answer in ("r", "d", "w") if can_restart else answer == "w":
                    coordinator.draining = True
                    coordinator._restart = answer in ("r", "d")
                    restart_debug = (answer == "d")
                    coordinator._notify_dashboard()
                    hint = "restart now" if coordinator._restart else "force stop"
                    logger.info("Draining: waiting for iteration %d to complete (Ctrl+C again to %s)...", iter_k, hint)
                    def drain_watcher():
                        coordinator.drain_complete.wait()
                        server.shutdown()
                    threading.Thread(target=drain_watcher, daemon=True).start()
                    continue
                continue  # dismissed
            break
        else:
            break  # drain complete

    coordinator._save_state()
    server.server_close()
    if coordinator.draining and coordinator._restart:
        logger.info("Restarting coordinator%s...", " in debug mode" if restart_debug else "")
        restart_cmd = [sys.argv[0], "-c", args.config, "-p", str(args.port)]
        if restart_debug:
            restart_cmd.append("--debug")
        os.execv(sys.executable, [sys.executable] + restart_cmd)
    else:
        logger.info("Shutting down.")


if __name__ == "__main__":
    main()
