#!/usr/bin/env python3
'''
Build the native single-file bundle: all engine flavors, book.bin and
weights.bin, compressed and appended to the launcher.

Needs the zstandard package.

Part of Sturddle Chess 2
Copyright (c) 2023 - 2026 Cristian Vlasceanu.
'''
import argparse
import importlib.util
import os
import platform
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import zstandard
except ImportError:
    sys.exit('ERROR: zstandard not found (pip install zstandard)')

ROOT = Path(__file__).resolve().parents[2]
MAKE_NATIVE = ROOT / 'tools' / 'make-native.py'
OUT_DIR = ROOT / 'dist' / 'bundle'

ASSETS = ['book.bin', 'weights.bin']
MAGIC = b'STURDBND'
CHUNK_SIZE = 1 << 20

spec = importlib.util.spec_from_file_location('make_native', MAKE_NATIVE)
make_native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_native)


# same as tools/build.py
def flavors():
    if platform.machine() in ['x86_64', 'AMD64']:
        return ['AVX512', 'AVX512_BF16', 'AVX2', 'AVX2_VNNI', 'AVX', 'generic']
    if platform.machine() == 'aarch64':
        return ['ARMv8_2', 'generic']
    sys.exit(f'ERROR: {platform.machine()} not supported')


def exe_name(name):
    return f'{name}.exe' if sys.platform.startswith('win') else name


# same naming as tools/build.py
def bundle_name(base, version):
    if sys.platform.startswith('win'):
        return f'{base}-{version}.exe'
    return f'{base}-{version}-{platform.system()}-{platform.machine()}'


def banner(text):
    print('*********************************************************')
    print(text)
    print('*********************************************************')


def make(out_dir, *args):
    if subprocess.call([sys.executable, str(MAKE_NATIVE), *args, '--out-dir', str(out_dir)], cwd=ROOT):
        sys.exit(f'ERROR: make-native.py {" ".join(args)} failed')


def write_record(writer, path, is_exe):
    name = path.name.encode()
    writer.write(struct.pack('<H', len(name)) + name)
    writer.write(struct.pack('<BQ', int(is_exe), path.stat().st_size))
    with open(path, 'rb') as f:
        while chunk := f.read(CHUNK_SIZE):
            writer.write(chunk)


# launcher, zstd payload, zero padding, trailer (magic, payload offset and size)
def write_bundle(out_path, launcher, files, level):
    compressor = zstandard.ZstdCompressor(level=level, write_checksum=True, threads=-1)

    with open(out_path, 'wb') as out:
        out.write(launcher.read_bytes())
        offset = out.tell()

        with compressor.stream_writer(out, closefd=False) as writer:
            for path, is_exe in files:
                print(f'Adding {path.name}')
                write_record(writer, path, is_exe)
            writer.write(struct.pack('<H', 0))

        size = out.tell() - offset

        # signtool pads to 8 bytes before the signature
        out.write(b'\0' * (-(out.tell() + 24) % 8))
        out.write(MAGIC + struct.pack('<QQ', offset, size))

    if not sys.platform.startswith('win'):
        os.chmod(out_path, 0o755)


def main():
    parser = argparse.ArgumentParser(description='Build the native single-file bundle')
    parser.add_argument('-n', '--name', default='sturddle', help='Executable base name (default: "sturddle")')
    parser.add_argument('--level', type=int, default=19, help='zstd compression level (default: 19)')
    args = parser.parse_args()

    archs = flavors()
    version = make_native.parse_version()

    with tempfile.TemporaryDirectory(prefix='sturddle-bundle-build-') as tmp:
        build_dir = Path(tmp)
        for arch in archs:
            banner(f'Building {arch}')
            make(build_dir, arch)
        banner('Building launcher')
        make(build_dir, '--launcher')
        banner('Bundling')

        files = [(build_dir / exe_name(f'sturddle-{version}{make_native.ARCH_SUFFIX[arch]}'), True) for arch in archs]
        files += [(build_dir / asset, False) for asset in ASSETS]
        launcher = build_dir / exe_name(f'sturddle-{version}-launcher')

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / bundle_name(args.name, version)
        write_bundle(out_path, launcher, files, args.level)

    dgst = make_native.write_sha256(out_path)
    print(f'\nBuilt {out_path}')
    print(f'Digest {dgst}')


if __name__ == '__main__':
    main()
