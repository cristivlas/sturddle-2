#!/usr/bin/env python3
"""Row counts per h5 file from a list file (one path per line), with per-source rollup."""

import re
import sys

import h5py

paths = [line.strip() for line in open(sys.argv[1]) if line.strip()]
sizes = {p: len(h5py.File(p, "r")["data"]) for p in paths}
total = sum(sizes.values())

groups = {}
for p, n in sizes.items():
    name = p.rsplit("/", 1)[-1]
    key = re.split(r"[-_.\d]", name)[0] or name
    groups[key] = groups.get(key, 0) + n

for p, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
    print(f"{n:>16,} {100.0 * n / total:>6.2f}%  {p}")
print()
for key, n in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"{n:>16,} {100.0 * n / total:>6.2f}%  {key}")
print(f"{total:>16,} 100.00%  total")
