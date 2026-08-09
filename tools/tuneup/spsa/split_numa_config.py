#!/usr/bin/env python3
"""
Split a worker JSON config into per-NUMA-node copies.

Given a worker config, creates <config>-0.json .. <config>-(N-1).json
next to it, where N is the number of CPU-bearing NUMA nodes on this
machine. Each copy has:
  - name suffixed with the NUMA node id (defaults to hostname if unset)
  - concurrency divided evenly across nodes

Prints a message and exits without writing anything if:
  - the machine is not NUMA (libnuma unavailable or numa_available < 0)
  - fewer than 2 CPU-bearing NUMA nodes exist
  - concurrency does not divide evenly across the detected nodes
  - any target output file already exists

Usage:
    python split_numa_config.py <worker.json>
"""

import argparse
import ctypes
import json
import platform
import sys
from pathlib import Path


def numa_available() -> bool:
    """Return True iff libnuma reports a working NUMA system."""
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        return libnuma.numa_available() >= 0
    except OSError:
        return False


def cpu_bearing_nodes() -> list:
    """Return sorted list of NUMA node ids that have at least one CPU.

    Reads /sys/devices/system/node -- skips memory-only nodes whose
    cpulist is empty.
    """
    node_root = Path("/sys/devices/system/node")
    if not node_root.is_dir():
        return []
    nodes = []
    for entry in node_root.iterdir():
        if not entry.is_dir() or not entry.name.startswith("node"):
            continue
        try:
            node_id = int(entry.name[4:])
        except ValueError:
            continue
        cpulist_file = entry / "cpulist"
        try:
            cpulist = cpulist_file.read_text().strip()
        except OSError:
            continue
        if cpulist:
            nodes.append(node_id)
    return sorted(nodes)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split a worker JSON config into per-NUMA-node copies.",
    )
    ap.add_argument("config", help="Path to worker JSON config")
    args = ap.parse_args()

    src_path = Path(args.config)
    if not src_path.is_file():
        print(f"Error: {src_path} does not exist or is not a file", file=sys.stderr)
        return 1

    if not numa_available():
        print("No NUMA architecture detected -- nothing to do.")
        return 0

    nodes = cpu_bearing_nodes()
    if len(nodes) < 2:
        print(f"Only {len(nodes)} CPU-bearing NUMA node(s) detected -- nothing to split.")
        return 0

    n = len(nodes)

    with open(src_path) as f:
        config = json.load(f)

    concurrency = config.get("concurrency", 1)
    if not isinstance(concurrency, int) or concurrency <= 0:
        print(f"Error: concurrency must be a positive integer (got {concurrency!r})", file=sys.stderr)
        return 1

    if concurrency % n != 0:
        lower = (concurrency // n) * n
        upper = lower + n
        suggestions = [v for v in (lower, upper) if v > 0]
        print(
            f"concurrency={concurrency} does not divide evenly across {n} NUMA nodes. "
            f"Try {' or '.join(map(str, suggestions))}. Nothing written."
        )
        return 1

    per_node_concurrency = concurrency // n
    base_name = config.get("name") or platform.node()
    base_log = config.get("log_file") or "worker.log"
    base_games_dir = config.get("games_dir") or "./games"

    targets = [
        src_path.with_name(f"{src_path.stem}-{i}{src_path.suffix}")
        for i in range(n)
    ]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        print("Error: would overwrite existing file(s):", file=sys.stderr)
        for p in existing:
            print(f"  {p}", file=sys.stderr)
        return 1

    log_path = Path(base_log)
    games_path = Path(base_games_dir)

    for node_id, out_path in zip(nodes, targets):
        cfg = dict(config)
        cfg["name"] = f"{base_name}-{node_id}"
        cfg["concurrency"] = per_node_concurrency
        cfg["log_file"] = str(
            log_path.with_name(f"{log_path.stem}-{node_id}{log_path.suffix}")
        )
        cfg["games_dir"] = str(games_path.with_name(f"{games_path.name}-{node_id}"))
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
        print(
            f"Wrote {out_path}  "
            f"(node={node_id}, concurrency={per_node_concurrency}, "
            f"log={cfg['log_file']}, games_dir={cfg['games_dir']})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
