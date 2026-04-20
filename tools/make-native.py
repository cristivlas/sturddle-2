#!/usr/bin/env python3
"""
Build native Sturddle executable (no Python runtime embedded).

Usage:
    python tools/make-native.py [ARCH]

ARCH is one of: native (default), AVX, AVX2, AVX2_VNNI, AVX512, AVX512_BF16.

Output: dist/native/sturddle-<version><arch-suffix>.exe (Windows)
        dist/native/sturddle-<version><arch-suffix>      (Linux)
Copies weights.bin and book.bin next to the binary.
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_weights

ARCH_FLAGS = {
    'native':      ['-march=native'],
    'AVX':         ['-march=corei7-avx', '-mtune=corei7-avx'],
    'AVX2':        ['-march=core-avx2', '-mtune=znver3'],
    'AVX2_VNNI':   ['-march=alderlake', '-mtune=raptorlake'],
    'AVX512':      ['-march=skylake-avx512', '-mtune=skylake-avx512'],
    'AVX512_BF16': ['-march=cooperlake', '-mtune=znver4'],
}

ARCH_SUFFIX = {
    'native': '', 'AVX': '-avx', 'AVX2': '-avx2',
    'AVX2_VNNI': '-avx2-vnni', 'AVX512': '-avx512', 'AVX512_BF16': '-avx512-bf16',
}

SOURCES = ['chess.cpp', 'context.cpp', 'search.cpp', 'uci_native.cpp', 'tbprobe.cpp', 'main_native.cpp']
INCLUDES = ['.', 'libpopcnt', 'magic-bits/include', 'version2', 'Fathom/src']
DEFINES = ['NATIVE_BUILD=1', 'NATIVE_UCI=1', 'NATIVE_BOOK=1', 'WITH_NNUE', 'CALLBACK_PERIOD=8192', 'NO_ASSERT']

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / 'dist' / 'native'


def parse_version():
    text = (REPO_ROOT / 'version.h').read_text()
    major = re.search(r'^#define\s+STURDDLE_VERSION_MAJOR\s+(\d+)', text, re.M).group(1)
    minor = re.search(r'^#define\s+STURDDLE_VERSION_MINOR\s+(\d+)', text, re.M).group(1)
    patch = re.search(r'^#define\s+STURDDLE_VERSION_PATCH\s+"([^"]+)"', text, re.M).group(1)
    return f'{major}.{minor}.{patch}'


def init_msvc_env():
    if os.environ.get('INCLUDE'):
        return
    vcvars = Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat')
    if not vcvars.exists():
        sys.exit(f'ERROR: vcvars64.bat not found at {vcvars}')
    result = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            os.environ[k] = v


def build_windows(arch, version, build_stamp, embed):
    cl_exe = Path(r'C:\Program Files\LLVM\bin\clang-cl.exe')
    if not cl_exe.exists():
        sys.exit(f'ERROR: clang-cl not found at {cl_exe}')
    init_msvc_env()

    exe = OUT_DIR / f'sturddle-{version}{ARCH_SUFFIX[arch]}.exe'

    cxxflags = ['/std:c++20', '/EHsc', '/fp:fast', '/O2', '/MT'] + ARCH_FLAGS[arch] + [
        '-O3', '-Ofast', '-Werror', '-Wmissing-field-initializers',
        '-Wno-deprecated-declarations', '-Wno-unused-command-line-argument',
        '-Wno-unused-label', '-Wno-unused-variable', '-Wno-nan-infinity-disabled',
        '-Wno-c23-extensions',  # #embed in C++ mode
        '/D_FORTIFY_SOURCE=0', '/GS-',
    ]
    defines = DEFINES if embed else DEFINES + ['SHARED_WEIGHTS']
    define_args = [f'-D{d}' for d in defines] + [f'-DBUILD_STAMP={build_stamp}']
    include_args = [f'-I{d}' for d in INCLUDES]

    with tempfile.TemporaryDirectory(prefix='sturddle-native-') as tmp:
        tmp_dir = Path(tmp)

        def compile_one(src):
            obj = tmp_dir / (Path(src).stem + '.obj')
            cmd = [str(cl_exe), '/c'] + cxxflags + define_args + include_args + [src, f'/Fo{obj}']
            print(' '.join(cmd))
            rc = subprocess.call(cmd, cwd=REPO_ROOT)
            if rc != 0:
                raise RuntimeError(f'compile failed: {src} (rc={rc})')
            return obj

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            objs = list(pool.map(compile_one, SOURCES))

        link_cmd = [str(cl_exe)] + [str(o) for o in objs] + [
            f'/Fe{exe}',
            '/link', '/SUBSYSTEM:CONSOLE', '/LTCG:OFF', '/STACK:33554432',
        ]
        print(' '.join(link_cmd))
        rc = subprocess.call(link_cmd, cwd=REPO_ROOT)
        if rc != 0:
            sys.exit(rc)

    return exe


def build_linux(arch, version, build_stamp, embed):
    cxx = os.environ.get('CXX')
    if not cxx:
        sys.exit('ERROR: set CXX (e.g. CXX=clang++-20) before invoking this script.')

    cxx_ver = subprocess.check_output([cxx, '--version'], text=True)
    m = re.search(r'(\d+)\.\d+\.\d+', cxx_ver)
    if not m:
        sys.exit(f'ERROR: cannot parse {cxx} version from:\n{cxx_ver}')
    cc_ver = int(m.group(1))
    is_clang = 'clang' in Path(cxx).name
    triplet = subprocess.check_output([cxx, '-dumpmachine'], text=True).strip()

    if is_clang and cc_ver < 16:
        sys.exit(f'ERROR: {cxx} v{cc_ver}. clang >= 16 required.')
    if (not is_clang) and cc_ver < 13:
        sys.exit(f'ERROR: {cxx} v{cc_ver}. gcc >= 13 required.')

    exe = OUT_DIR / f'sturddle-{version}{ARCH_SUFFIX[arch]}'

    cxxflags = ['-std=c++20', '-O3', '-ffast-math'] + ARCH_FLAGS[arch] + [
        '-Wextra', '-Werror',
        '-Wno-unused-label', '-Wno-unknown-pragmas',
        '-Wno-unused-parameter', '-Wno-unused-variable',
        '-Wno-empty-body', '-Wno-int-in-bool-context',
        '-fno-stack-protector', '-D_FORTIFY_SOURCE=0',
    ]
    if is_clang:
        cxxflags += [
            '-Wno-macro-redefined', '-Wno-deprecated-declarations',
            '-Wno-nan-infinity-disabled',
            '-Wno-c23-extensions',  # #embed in C++ mode
            '-stdlib=libc++', '-fexperimental-library',
        ]

    defines = DEFINES if embed else DEFINES + ['SHARED_WEIGHTS']
    define_args = [f'-D{d}' for d in defines] + [f'-DBUILD_STAMP={build_stamp}']
    include_args = [f'-I{d}' for d in INCLUDES]

    with tempfile.TemporaryDirectory(prefix='sturddle-native-') as tmp:
        tmp_dir = Path(tmp)

        def compile_one(src):
            obj = tmp_dir / (Path(src).stem + '.o')
            cmd = [cxx, '-c'] + cxxflags + define_args + include_args + [src, '-o', str(obj)]
            print(' '.join(cmd))
            rc = subprocess.call(cmd, cwd=REPO_ROOT)
            if rc != 0:
                raise RuntimeError(f'compile failed: {src} (rc={rc})')
            return obj

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            objs = list(pool.map(compile_one, SOURCES))

        link_cmd = [cxx] + [str(o) for o in objs] + ['-o', str(exe), '-lpthread']
        if is_clang:
            link_cmd += [
                '-fuse-ld=lld',
                f'-L/usr/lib/llvm-{cc_ver}/lib/',
                f'-L/usr/lib/llvm-{cc_ver}/lib/{triplet}',
                '-L/usr/local/opt/llvm/lib/c++',
                '-stdlib=libc++', '-lc++', '-lc++experimental',
            ]
        print(' '.join(link_cmd))
        rc = subprocess.call(link_cmd, cwd=REPO_ROOT)
        if rc != 0:
            sys.exit(rc)

    return exe


def write_sha256(exe_path):
    h = hashlib.sha256()
    with open(exe_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    dgst_path = Path(f'{exe_path}-sha256.txt')
    dgst_path.write_bytes(f'{h.hexdigest()} *{exe_path.name}\n'.encode())
    return dgst_path


def main():
    parser = argparse.ArgumentParser(description='Build native Sturddle binary')
    parser.add_argument('arch', nargs='?', default='native', choices=list(ARCH_FLAGS.keys()), help='Target SIMD architecture')
    parser.add_argument('--embed', action='store_true',
                        help='Embed NNUE weights in the binary via C23/C++26 #embed (requires GCC 15+, Clang 19+, or MSVC 17.15+)')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    version = parse_version()
    build_stamp = datetime.now().strftime('%m%d%y')

    fetch_weights.ensure()

    embed = args.embed
    if sys.platform.startswith('win'):
        exe = build_windows(args.arch, version, build_stamp, embed)
    elif sys.platform.startswith('linux'):
        exe = build_linux(args.arch, version, build_stamp, embed)
    else:
        sys.exit(f'Unsupported platform: {sys.platform}')

    assets = ('book.bin',) if embed else ('weights.bin', 'book.bin')
    for asset in assets:
        src = REPO_ROOT / asset
        if src.exists():
            shutil.copy(src, OUT_DIR / asset)

    dgst = write_sha256(exe)

    print(f'\nBuilt {exe}')
    print(f'Digest {dgst}')


if __name__ == '__main__':
    main()
