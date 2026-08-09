#!/usr/bin/env python3
"""
List the source files behind an h5/VDS container, one path per line —
suitable as a file list for label_check.py et al. Direct members by default;
-r/--recurse expands nested VDS members down to the real files.

    ./vds_files.py -r dmix.h5 > dmix_files.txt
"""

import argparse
import os

import h5py


def expand(path, seen, recurse, depth=0):
    path = os.path.abspath(path)
    if path in seen:
        return
    seen.add(path)
    with h5py.File(path, "r") as hf:
        data = hf["data"]
        if not data.is_virtual or (depth > 0 and not recurse):
            print(path)
            return
        base = os.path.dirname(path)
        for vs in data.virtual_sources():
            p = vs.file_name
            if p == ".":
                continue
            expand(p if os.path.isabs(p) else os.path.join(base, p), seen, recurse, depth + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", nargs="+", help="h5/VDS container path(s)")
    parser.add_argument("-r", "--recurse", action="store_true", help="expand nested VDS members down to real files")
    args = parser.parse_args()

    seen = set()
    for path in args.input:
        expand(path, seen, args.recurse)
