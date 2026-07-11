#!/usr/bin/env python3
"""
Versioned backup / restore of a dataset's sidecar profiles, for experiments.

Resolves the same sidecars the trainers use: the container's own
<file>.h5.profile.json plus each member's (one level, not recursive), and
stores them in a portable tar.gz next to the container:

    <vds>.profiles-<tag>.tar.gz

    ./profiles.py backup mix.h5              new archive, auto-numbered tag
    ./profiles.py backup mix.h5 --tag base   named tag
    ./profiles.py list mix.h5                show archives and their contents
    ./profiles.py restore mix.h5 --tag base  deploy an archive's profiles

Restore auto-backups the current profiles first (unless an identical archive
already exists), so nothing is ever lost.
"""

import argparse
import glob
import io
import os
import re
import tarfile

import h5py


def sidecars(vds_path):
    """All existing/potential sidecar paths: container first, then members."""
    paths = [vds_path]
    with h5py.File(vds_path, "r") as hf:
        data = hf["data"]
        if data.is_virtual:
            # HDF5 resolves relative member paths against the container's directory
            base = os.path.dirname(os.path.abspath(vds_path))
            for vs in data.virtual_sources():
                p = vs.file_name
                if p != ".":
                    paths.append(p if os.path.isabs(p) else os.path.join(base, p))
    seen = set()
    unique = [p for p in paths if not (p in seen or seen.add(p))]
    return [os.path.abspath(p) + ".profile.json" for p in unique]


def archive_name(vds, tag):
    return f"{vds}.profiles-{tag}.tar.gz"


def archives(vds):
    return sorted(glob.glob(glob.escape(vds) + ".profiles-*.tar.gz"))


def next_tag(vds):
    rgx = re.compile(re.escape(os.path.basename(vds)) + r"\.profiles-(\d+)\.tar\.gz$")
    nums = [int(m.group(1)) for a in archives(vds) if (m := rgx.search(os.path.basename(a)))]
    return str(max(nums, default=0) + 1)


def snapshot(vds):
    """{sidecar path: content bytes} for all existing sidecars."""
    snap = {}
    for sc in sidecars(vds):
        if os.path.exists(sc):
            with open(sc, "rb") as f:
                snap[sc] = f.read()
        else:
            print(f"{sc}: no profile, skipped")
    return snap


def read_archive(path):
    """{stored path: content bytes}"""
    out = {}
    with tarfile.open(path, "r:gz") as tar:
        for ti in tar.getmembers():
            if ti.isfile():
                out["/" + ti.name if not re.match(r"^[A-Za-z]:", ti.name) else ti.name] = tar.extractfile(ti).read()
    return out


def write_archive(path, snap):
    with tarfile.open(path, "w:gz") as tar:
        for sc, content in snap.items():
            name = sc.replace(os.sep, "/")
            name = name[1:] if name.startswith("/") else name  # single slash only: keep UNC //server
            ti = tarfile.TarInfo(name=name)
            ti.size = len(content)
            tar.addfile(ti, io.BytesIO(content))


def canon(path):
    return os.path.normpath(path.replace("/", os.sep))


def backup(args):
    snap = snapshot(args.vds)
    if not snap:
        print("nothing to back up")
        return
    if not args.tag:  # auto-numbered backups dedup; an explicit tag may alias existing content
        snap_c = {canon(p): c for p, c in snap.items()}
        for a in archives(args.vds):
            if {canon(p): c for p, c in read_archive(a).items()} == snap_c:
                print(f"identical to existing {a}, not duplicating")
                return
    tag = args.tag or next_tag(args.vds)
    dest = archive_name(args.vds, tag)
    if os.path.exists(dest):
        print(f"{dest}: exists — pick another --tag")
        return
    write_archive(dest, snap)
    print(f"wrote {dest} ({len(snap)} profile(s))")


def list_archives(args):
    found = archives(args.vds)
    if not found:
        print("no archives")
    for a in found:
        print(a)
        for p in read_archive(a):
            print(f"    {p}")


def restore(args):
    found = archives(args.vds)
    if not found:
        print("no archives")
        return
    src = archive_name(args.vds, args.tag) if args.tag else max(found, key=os.path.getmtime)
    if not os.path.exists(src):
        print(f"{src}: not found; available:")
        for a in found:
            print(f"    {a}")
        return

    # preserve the current profiles first, unless already archived identically
    backup(argparse.Namespace(vds=args.vds, tag=None))

    # only write paths that belong to this dataset's sidecars (guards against archives
    # made on a different machine/layout writing to stale absolute paths)
    valid = {canon(p) for p in sidecars(args.vds)}
    restored = skipped = 0
    for stored, content in read_archive(src).items():
        path = canon(stored)
        if path not in valid:
            print(f"{path}: not a sidecar of {args.vds}, skipped")
            skipped += 1
            continue
        with open(path, "wb") as f:
            f.write(content)
        print(f"restored {path}")
        restored += 1
    print(f"{src}: restored {restored}, skipped {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("action", choices=["backup", "restore", "list"])
    parser.add_argument("vds", help="dataset (h5/VDS container) path")
    parser.add_argument("--tag", help="archive tag (backup: default auto-number; restore: default latest)")
    args = parser.parse_args()

    {"backup": backup, "restore": restore, "list": list_archives}[args.action](args)
