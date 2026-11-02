#!/usr/bin/env python3
"""
Launch one --managed worker per NUMA node, pinned with numactl, from the configs
written by split_numa_config.py. On Ctrl+C, [w]ait lets workers finish their games,
[s]top kills them; a second Ctrl+C force-kills.

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
    """<stem>-0<suffix>, <stem>-1<suffix>, ... while they exist."""
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
    """'wait', 'stop' or 'dismiss'; Ctrl+C here propagates."""
    try:
        answer = input(
            "\nGames in progress. [w]ait and stop | [s]top now | "
            "[Enter] to dismiss "
        ).strip().lower()
    except EOFError:
        return "dismiss"
    if answer == "w":
        return "wait"
    if answer == "s":
        return "stop"
    return "dismiss"


def _hard_kill(procs) -> None:
    _signal_all(procs, signal.SIGKILL)
    for p in procs:
        try:
            p.wait()
        except KeyboardInterrupt:
            continue


def _wait_with_grace(procs, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while _alive(procs) and time.monotonic() < deadline:
        time.sleep(0.2)


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
        print("No NUMA architecture detected, use worker.py directly.", file=sys.stderr)
        return 1

    if not shutil.which("numactl"):
        print("Error: numactl not found in PATH.", file=sys.stderr)
        return 1

    base = Path(args.config)
    configs = _discover_configs(base)
    if not configs:
        print(
            f"Error: no per-node configs found for {base}; "
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
        # own session: terminal Ctrl+C reaches only the launcher
        procs.append(subprocess.Popen(cmd, start_new_session=True))

    graceful_shutdown = False
    while _alive(procs):
        try:
            _wait_all(procs)
            break
        except KeyboardInterrupt:
            if graceful_shutdown:
                print("\nForce killing workers.", file=sys.stderr)
                _hard_kill(procs)
                return 130

            # Ctrl+C at the prompt or during the grace period force-kills
            try:
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
                    # under --managed, SIGINT kills cutechess and exits
                    _signal_all(procs, signal.SIGINT)
                    _wait_with_grace(procs, 5.0)
                    if _alive(procs):
                        _hard_kill(procs)
                    return 130
            except KeyboardInterrupt:
                print("\nForce killing workers.", file=sys.stderr)
                _hard_kill(procs)
                return 130

    return _wait_all(procs)


if __name__ == "__main__":
    sys.exit(main())
