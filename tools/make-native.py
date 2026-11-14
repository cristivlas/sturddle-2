#!/usr/bin/env python3
"""
Build the native Sturddle executable (no Python runtime).

Usage:
    python tools/make-native.py [ARCH] [--embed] [--out-dir DIR]
    python tools/make-native.py --launcher [--out-dir DIR]

ARCH: native (default), generic, AVX, AVX2, AVX2_VNNI, AVX512, AVX512_BF16, ARMv8_2.

weights.bin is loaded at runtime from next to the executable, unless --embed
builds it in. book.bin and weights.bin are copied to the output directory,
dist/native by default.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# magic bits are a bit faster than the attack tables without PEXT
ARCH_FLAGS = {
    'native':      ['-march=native'],
    'generic':     ['-DUSE_MAGIC_BITS'],
    'AVX':         ['-DUSE_MAGIC_BITS', '-march=corei7-avx', '-mtune=corei7-avx'],
    'AVX2':        ['-march=core-avx2', '-mtune=znver3'],
    'AVX2_VNNI':   ['-march=alderlake', '-mtune=raptorlake'],
    'AVX512':      ['-march=skylake-avx512', '-mtune=skylake-avx512'],
    'AVX512_BF16': ['-march=cooperlake', '-mtune=znver4'],
    'ARMv8_2':     ['-DUSE_MAGIC_BITS', '-march=armv8.2-a+fp16+dotprod'],
}

ARCH_SUFFIX = {
    'native': '', 'generic': '-generic', 'AVX': '-avx', 'AVX2': '-avx2',
    'AVX2_VNNI': '-avx2-vnni', 'AVX512': '-avx512', 'AVX512_BF16': '-avx512-bf16',
    'ARMv8_2': '-armv8-2',
}

LAUNCHER_SOURCES = ['tools/native-bundle/launcher.cpp', 'tools/native-bundle/zstd_decoder.cpp']
SOURCES = ['chess.cpp', 'context.cpp', 'search.cpp', 'uci_native.cpp', 'utility.cpp', 'tbprobe.cpp', 'main_native.cpp', 'nnue/model.cpp']
INCLUDES = ['.', 'libpopcnt', 'magic-bits/include', 'nnue/version2', 'Fathom/src']
# Windows only, same icon and manifest as tools/build.py
RESOURCES = ['sturddle.rc']
MANIFEST = 'manifest.xml'
DEFINES = ['NATIVE_BUILD=1', 'NATIVE_UCI=1', 'NATIVE_BOOK=1', 'CALLBACK_PERIOD=8192', 'NO_ASSERT']

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / 'tools'))
import fetch_weights
OUT_DIR = REPO_ROOT / 'dist' / 'native'


def parse_version():
    text = (REPO_ROOT / 'version.h').read_text()
    major = re.search(r'^#define\s+STURDDLE_VERSION_MAJOR\s+(\d+)', text, re.M).group(1)
    minor = re.search(r'^#define\s+STURDDLE_VERSION_MINOR\s+(\d+)', text, re.M).group(1)
    patch = re.search(r'^#define\s+STURDDLE_VERSION_PATCH\s+"([^"]+)"', text, re.M).group(1)
    return f'{major}.{minor}.{patch}'


MSVC_WARNINGS = [
    '/wd4068',  # unknown pragma
]

MSVC_ARCH_FLAGS = {
    '-march=corei7-avx': '/arch:AVX',
    '-march=core-avx2': '/arch:AVX2',
    '-march=skylake-avx512': '/arch:AVX512',
}


def find_visual_studio():
    program_files = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    vswhere = Path(program_files) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.exists():
        return None
    query = ['-latest', '-products', '*', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64']
    result = subprocess.run([vswhere, *query, '-property', 'installationPath'], capture_output=True, text=True)
    install_dir = result.stdout.strip()
    return Path(install_dir) if install_dir else None


def init_msvc_env(vs_dir):
    if os.environ.get('INCLUDE'):
        return
    vcvars = vs_dir / 'VC/Auxiliary/Build/vcvars64.bat' if vs_dir else None
    if not vcvars or not vcvars.exists():
        sys.exit('ERROR: Visual Studio C++ tools not found')
    result = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            os.environ[k] = v


def find_tool(names, dirs):
    search_path = os.pathsep.join(str(d) for d in dirs)
    for name in names:
        if path := shutil.which(name, path=search_path):
            return Path(path)
    sys.exit(f'ERROR: {" or ".join(names)} not found')


def msvc_arch_flags(arch_flags):
    if not arch_flags:
        return []  # the launcher
    # MSVC never defines __BMI2__, so there is no PEXT
    flags = ['-DUSE_MAGIC_BITS']
    for flag in arch_flags:
        if flag.startswith('-march=') and flag != '-march=native':
            if flag not in MSVC_ARCH_FLAGS:
                print(f'WARNING: {flag} needs clang-cl, skipping')
                sys.exit(0)
            flags.append(MSVC_ARCH_FLAGS[flag])
    return flags


def build_windows(out_dir, name, arch_flags, sources, define_args):
    vs_dir = find_visual_studio()
    init_msvc_env(vs_dir)

    llvm_dirs = [Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'LLVM/bin']
    if vs_dir:
        llvm_dirs.append(vs_dir / 'VC/Tools/Llvm/x64/bin')
    names = [os.environ['CL_EXE']] if os.environ.get('CL_EXE') else ['clang-cl', 'cl']
    cl_exe = find_tool(names, [os.environ['PATH'], *llvm_dirs])
    is_clang = cl_exe.stem.lower() == 'clang-cl'
    rc_exe = find_tool(['llvm-rc' if is_clang else 'rc'], [cl_exe.parent, os.environ['PATH']])

    exe = out_dir / f'{name}.exe'

    cxxflags = ['/nologo', '/std:c++latest', '/EHsc', '/fp:fast', '/O2', '/MT']
    if is_clang:
        cxxflags += arch_flags + [
            '-O3', '-Ofast', '-Werror', '-Wmissing-field-initializers',
            '-Wno-deprecated-declarations', '-Wno-unused-command-line-argument',
            '-Wno-unused-label', '-Wno-unused-variable', '-Wno-nan-infinity-disabled',
        ]
    else:
        cxxflags += msvc_arch_flags(arch_flags) + MSVC_WARNINGS
        # MSVC has no native target
        define_args = ['-DBUILD_FLAVOR=generic' if d == '-DBUILD_FLAVOR=native' else d for d in define_args]
    cxxflags += ['/D_FORTIFY_SOURCE=0', '/GS-'] + shlex.split(os.environ.get('CXXFLAGS', ''))
    include_args = [f'-I{d}' for d in INCLUDES]

    with tempfile.TemporaryDirectory(prefix='sturddle-native-') as tmp:
        tmp_dir = Path(tmp)

        def compile_one(src):
            if src.endswith('.rc'):
                obj = tmp_dir / (Path(src).stem + '.res')
                cmd = [str(rc_exe), '/FO', str(obj), src]
            else:
                obj = tmp_dir / (Path(src).stem + '.obj')
                cmd = [str(cl_exe), '/c'] + cxxflags + define_args + include_args + [src, f'/Fo{obj}']
            print(' '.join(cmd))
            rc = subprocess.call(cmd, cwd=REPO_ROOT)
            if rc != 0:
                raise RuntimeError(f'compile failed: {src} (rc={rc})')
            return obj

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            objs = list(pool.map(compile_one, sources + RESOURCES))

        link_cmd = [str(cl_exe)] + [str(o) for o in objs] + [
            f'/Fe{exe}',
            '/link', '/SUBSYSTEM:CONSOLE', '/LTCG:OFF', '/STACK:33554432',
            '/MANIFEST:EMBED', f'/MANIFESTINPUT:{REPO_ROOT / MANIFEST}',
        ]
        print(' '.join(link_cmd))
        rc = subprocess.call(link_cmd, cwd=REPO_ROOT)
        if rc != 0:
            sys.exit(rc)

    return exe


def build_linux(out_dir, name, arch_flags, sources, define_args):
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

    if is_clang and cc_ver < 18:
        sys.exit(f'ERROR: {cxx} v{cc_ver}. clang >= 18 required.')
    if (not is_clang) and cc_ver < 14:
        sys.exit(f'ERROR: {cxx} v{cc_ver}. gcc >= 14 required.')

    exe = out_dir / name

    cxxflags = ['-std=c++23', '-O3', '-ffast-math'] + arch_flags + [
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
            '-stdlib=libc++', '-fexperimental-library',
        ]
    cxxflags += shlex.split(os.environ.get('CXXFLAGS', ''))

    include_args = [f'-I{d}' for d in INCLUDES]
    if triplet.startswith(('aarch64', 'arm')):
        include_args.append('-Innue/simde')

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
            objs = list(pool.map(compile_one, sources))

        link_cmd = [cxx] + [str(o) for o in objs] + ['-o', str(exe), '-lpthread']
        if is_clang:
            link_cmd += [
                '-fuse-ld=lld',
                f'-L/usr/lib/llvm-{cc_ver}/lib/',
                f'-L/usr/lib/llvm-{cc_ver}/lib/{triplet}',
                '-L/usr/local/opt/llvm/lib/c++',
                # most distros do not ship libc++
                '-stdlib=libc++', '-nostdlib++',
                '-Wl,-Bstatic', '-lc++experimental', '-lc++', '-lc++abi', '-Wl,-Bdynamic',
            ]
        print(' '.join(link_cmd))
        rc = subprocess.call(link_cmd, cwd=REPO_ROOT)
        if rc != 0:
            sys.exit(rc)

    return exe


def write_sha256(exe_path):
    dgst_path = Path(f'{exe_path}-sha256.txt')
    dgst_path.write_bytes(f'{fetch_weights.sha256(exe_path)} *{exe_path.name}\n'.encode())
    return dgst_path


def main():
    parser = argparse.ArgumentParser(description='Build native Sturddle binary')
    parser.add_argument('arch', nargs='?', default='native', choices=list(ARCH_FLAGS.keys()), help='Target SIMD architecture')
    parser.add_argument('--embed', action='store_true', help='Build weights.bin into the executable.')
    parser.add_argument('--launcher', action='store_true', help='Build the launcher instead of the engine.')
    parser.add_argument('--out-dir', type=Path, default=OUT_DIR, help='Output directory (default: dist/native).')
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    version = parse_version()
    build_stamp = datetime.now().strftime('%m%d%y')

    if sys.platform.startswith('win'):
        build = build_windows
    elif sys.platform.startswith('linux'):
        build = build_linux
    else:
        sys.exit(f'Unsupported platform: {sys.platform}')

    if args.launcher:
        # no -march, must run on any CPU
        exe = build(out_dir, f'sturddle-{version}-launcher', [], LAUNCHER_SOURCES, ['-DNATIVE_BUILD=1', '-DNO_ASSERT'])
    else:
        embed = args.embed

        fetch_weights.ensure()

        defines = DEFINES if embed else DEFINES + ['SHARED_WEIGHTS']
        define_args = [f'-D{d}' for d in defines] + [f'-DBUILD_STAMP={build_stamp}', f'-DBUILD_FLAVOR={args.arch}']

        exe = build(out_dir, f'sturddle-{version}{ARCH_SUFFIX[args.arch]}', ARCH_FLAGS[args.arch], SOURCES, define_args)

        assets = ('book.bin',) if embed else ('weights.bin', 'book.bin')
        for asset in assets:
            src = REPO_ROOT / asset
            if not src.exists():
                sys.exit(f'ERROR: required asset {asset} not found at {src}')
            shutil.copy(src, out_dir / asset)

    dgst = write_sha256(exe)

    print(f'\nBuilt {exe}')
    print(f'Digest {dgst}')


if __name__ == '__main__':
    main()
