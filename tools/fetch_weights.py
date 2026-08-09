#!/usr/bin/env python3
"""
Ensure weights.bin is present at the repo root and matches weights.bin.sha256.

If weights.bin is missing or stale, decompress weights.bin.gz and verify.
Idempotent; safe to call from setup.py and make-native.py before build.

Exits non-zero on any verification failure.
"""
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN = REPO_ROOT / 'weights.bin'
GZ = REPO_ROOT / 'weights.bin.gz'
SHA = REPO_ROOT / 'weights.bin.sha256'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def expected_digest():
    if not SHA.exists():
        sys.exit(f'ERROR: missing {SHA.relative_to(REPO_ROOT)}')
    return SHA.read_text().strip().split()[0]


def ensure():
    want = expected_digest()

    if BIN.exists() and sha256(BIN) == want:
        return BIN

    if not GZ.exists():
        sys.exit(f'ERROR: {BIN.name} missing or stale and {GZ.name} not found')

    print(f'Decompressing {GZ.name} -> {BIN.name}')
    tmp = BIN.with_suffix('.bin.tmp')
    try:
        with gzip.open(GZ, 'rb') as src, open(tmp, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        got = sha256(tmp)
        if got != want:
            sys.exit(f'ERROR: decompressed sha256 mismatch\n  expected {want}\n  got      {got}')
        tmp.replace(BIN)
    finally:
        if tmp.exists():
            tmp.unlink()
    return BIN


if __name__ == '__main__':
    path = ensure()
    print(f'OK: {path}')
