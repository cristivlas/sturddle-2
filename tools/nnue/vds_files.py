#!/usr/bin/env python3
import argparse
import math
import os
import re

import h5py


# rows: what the parent VDS maps from this file
def expand(path, recurse, out, depth=0, rows=None, chain=()):
    path = os.path.abspath(path)
    if path in chain:
        return
    with h5py.File(path, "r") as hf:
        data = hf["data"]
        if not data.is_virtual or (depth > 0 and not recurse):
            out.append((path, len(data) if rows is None else rows))
            return
        base = os.path.dirname(path)
        row_size = math.prod(data.shape[1:])
        for vs in data.virtual_sources():
            p = vs.file_name
            if p == ".":
                continue
            mapped = vs.vspace.get_select_npoints() // row_size
            expand(p if os.path.isabs(p) else os.path.join(base, p), recurse, out, depth + 1, mapped, chain + (path,))


def print_sizes(files):
    sizes = {}
    for path, n in files:
        sizes[path] = sizes.get(path, 0) + n
    total = sum(sizes.values())
    if not total:
        print(f"{0:>16,} 100.00%  total")
        return
    for path, n in sorted(sizes.items(), key=lambda f: -f[1]):
        print(f"{n:>16,} {100.0 * n / total:>6.2f}%  {path}")
    print()
    groups = {}
    for path, n in sizes.items():
        name = os.path.basename(path)
        key = re.split(r"[-_.\d]", name)[0] or name
        groups[key] = groups.get(key, 0) + n
    for key, n in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{n:>16,} {100.0 * n / total:>6.2f}%  {key}")
    print(f"{total:>16,} 100.00%  total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", nargs="+", help="h5/VDS container(s) and/or text file(s) listing h5 paths")
    parser.add_argument("-r", "--recurse", action="store_true", help="expand nested VDS members down to real files")
    parser.add_argument("--sizes", action="store_true", help="row counts per file and per name prefix")
    args = parser.parse_args()

    files = []
    for arg in args.input:
        if arg.endswith(".h5"):
            expand(arg, args.recurse, files)
        else:
            # listed files are leaves unless --recurse
            for line in open(arg):
                if line.strip():
                    expand(line.strip(), args.recurse, files, depth=1)

    if args.sizes:
        print_sizes(files)
    else:
        for path in dict.fromkeys(path for path, _ in files):
            print(path)
