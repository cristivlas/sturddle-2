#!/usr/bin/env python3
"""
Launch one worker per NUMA node, each pinned with numactl.

Given a base worker JSON path (e.g. worker.json), locates the
per-node configs created by split_numa_config.py (worker-0.json,
worker-1.json, ...) and launches each as:

    numactl --cpunodebind=K --membind=K <python> worker.py -c worker-I.json --managed

where I is the sequential config index and K is the matching
CPU-bearing NUMA node id discovered on this machine.

Ctrl+C handling:
  Workers are started in their own session, so Ctrl+C is not
  forwarded to them by the terminal. The launcher catches SIGINT
  and prompts:
      [w]ait and stop | [s]top now | [Enter] to dismiss
  'w' sends SIGTERM to every worker (graceful: finish current games).
  's' sends SIGKILL (immediate).
  A second Ctrl+C during the wait phase force-kills everyone.

Usage:
    python launch_numa_workers.py <worker.json>
"""

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from split_numa_config import cpu_bearing_nodes, numa_available


def _discover_configs(base: Path) -> list:
    """Return existing <base-stem>-{i}<suffix> paths, sequential from 0."""
    configs = []
    i = 0
    while True:
        p = base.with_name(f"{base.stem}-{i}{base.suffix}")
        if not p.is_file():
            break
        configs.append(p)
        i += 1
    return configs


def _alive(procs) -> list:
    return [p for p in procs if p.poll() is None]


def _signal_all(procs, sig) -> None:
    for p in _alive(procs):
        try:
            p.send_signal(sig)
        except OSError:
            pass


def _wait_all(procs) -> int:
    rc = 0
    for p in procs:
        p.wait()
        if p.returncode != 0:
            rc = p.returncode
    return rc


def _prompt_action() -> str:
    """Return 'wait', 'stop', or 'dismiss'."""
    try:
        answer = input(
            "\nGames in progress. [w]ait and stop | [s]top now | "
            "[Enter] to dismiss "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "dismiss"
    if answer == "w":
        return "wait"
    if answer == "s":
        return "stop"
    return "dismiss"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Launch per-NUMA-node workers with numactl pinning.",
    )
    ap.add_argument(
        "config",
        help="Base worker JSON path (e.g. worker.json); "
             "per-node files are worker-0.json, worker-1.json, ...",
    )
    args = ap.parse_args()

    if not numa_available():
        print("No NUMA architecture detected -- use worker.py directly.", file=sys.stderr)
        return 1

    if not shutil.which("numactl"):
        print("Error: numactl not found in PATH.", file=sys.stderr)
        return 1

    base = Path(args.config)
    configs = _discover_configs(base)
    if not configs:
        print(
            f"Error: no per-node configs found for {base} -- "
            "run split_numa_config.py first.",
            file=sys.stderr,
        )
        return 1

    nodes = cpu_bearing_nodes()
    if len(configs) != len(nodes):
        print(
            f"Error: found {len(configs)} config(s) but {len(nodes)} "
            f"CPU-bearing NUMA node(s). Regenerate configs for this machine.",
            file=sys.stderr,
        )
        return 1

    worker_py = Path(__file__).resolve().parent / "worker.py"
    if not worker_py.is_file():
        print(f"Error: worker.py not found at {worker_py}", file=sys.stderr)
        return 1

    procs = []
    for node_id, cfg in zip(nodes, configs):
        cmd = [
            "numactl",
            f"--cpunodebind={node_id}",
            f"--membind={node_id}",
            sys.executable,
            str(worker_py),
            "-c",
            str(cfg),
            "--managed",
        ]
        print("Launching:", " ".join(cmd))
        # start_new_session isolates the worker from the terminal's
        # SIGINT so the launcher can own the interrupt prompt.
        procs.append(subprocess.Popen(cmd, start_new_session=True))

    graceful_shutdown = False
    while _alive(procs):
        try:
            _wait_all(procs)
            break
        except KeyboardInterrupt:
            if graceful_shutdown:
                # Second Ctrl+C during the wait phase -- escalate.
                print("\nForce killing workers.", file=sys.stderr)
                _signal_all(procs, signal.SIGKILL)
                _wait_all(procs)
                return 130

            action = _prompt_action()
            if action == "wait":
                print(
                    "Waiting for workers to finish current games "
                    "(Ctrl+C again to force kill)...",
                    file=sys.stderr,
                )
                graceful_shutdown = True
                _signal_all(procs, signal.SIGTERM)
            elif action == "stop":
                print("Stopping workers now.", file=sys.stderr)
                _signal_all(procs, signal.SIGTERM)
                # Give them a moment to clean up, then SIGKILL stragglers.
                deadline = time.monotonic() + 5.0
                while _alive(procs) and time.monotonic() < deadline:
                    time.sleep(0.2)
                if _alive(procs):
                    _signal_all(procs, signal.SIGKILL)
                _wait_all(procs)
                return 130
            # else: dismiss -- resume waiting

    return _wait_all(procs)


if __name__ == "__main__":
    sys.exit(main())
