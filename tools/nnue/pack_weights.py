#!/usr/bin/env python3
"""
Pack a fresh weights.bin into weights.bin.gz and write weights.bin.sha256.

Run after re-exporting weights from the trainer. Inverse of fetch_weights.py.
"""
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BIN = REPO_ROOT / 'weights.bin'
GZ = REPO_ROOT / 'weights.bin.gz'
SHA = REPO_ROOT / 'weights.bin.sha256'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pack():
    if not BIN.exists():
        sys.exit(f'ERROR: {BIN} not found')

    digest = sha256(BIN)
    SHA.write_text(digest + '\n')
    print(f'{SHA.name}: {digest}')

    with open(BIN, 'rb') as src, gzip.open(GZ, 'wb', compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    print(f'{GZ.name}: {GZ.stat().st_size:,} bytes ({100 * GZ.stat().st_size / BIN.stat().st_size:.1f}% of raw)')


if __name__ == '__main__':
    pack()
