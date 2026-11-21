#!/usr/bin/env python3
import argparse
import io
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


# GNU-style flags, used by clang++, g++ and clang-cl
# MSVC uses MSVC_ARCH_FLAGS instead.
# Don't reorder: PGO profiles are matched on the exact flag list.
#
# magic bits are a bit faster than the attack tables without PEXT
# https://github.com/goutham/magic-bits
#
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

# gcc ignores unknown -Wno- flags
SHARED_WARNINGS = [
    '-Wno-unused-label',
    '-Wno-unused-variable',
    '-Wno-nan-infinity-disabled',
]

LAUNCHER_SOURCES = [
    'tools/native-bundle/launcher.cpp',
    'tools/native-bundle/zstd_decoder.cpp',
]
SOURCES = [
    'chess.cpp',
    'context.cpp',
    'search.cpp',
    'uci_native.cpp',
    'utility.cpp',
    'tbprobe.cpp',
    'main_native.cpp',
    'nnue/model.cpp',
]
INCLUDES = [
    '.',
    'libpopcnt',
    'magic-bits/include',
    'nnue/version2',
    'Fathom/src',
]
# Windows only, same icon and manifest as tools/build.py
RESOURCES = ['sturddle.rc']
MANIFEST = 'manifest.xml'
DEFINES = [
    'NATIVE_BUILD=1',
    'NATIVE_UCI=1',
    'NATIVE_BOOK=1',
    'CALLBACK_PERIOD=8192',
    'NO_ASSERT',
]

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / 'tools'))
import fetch_weights
OUT_DIR = REPO_ROOT / 'dist' / 'native'
PGO_DIR = REPO_ROOT / 'pgo'
PGN_DIR = REPO_ROOT / 'test' / 'runs'

# For profiling: opening, middlegame, endgame
PHASE_PLIES = [
    (0, 30),
    (30, 80),
    (80, 1000),
]
MAX_MISMATCH = 0.1
# --pgo outcome, read by tools/native-bundle/build.py
PGO_STATUS = 'PGO: '


def parse_version():
    text = (REPO_ROOT / 'version.h').read_text()

    def define(name, value):
        return re.search(rf'^#define\s+STURDDLE_VERSION_{name}\s+{value}', text, re.M).group(1)

    major = define('MAJOR', r'(\d+)')
    minor = define('MINOR', r'(\d+)')
    patch = define('PATCH', r'"([^"]+)"')
    return f'{major}.{minor}.{patch}'


def arch_suffix(arch):
    return '' if arch == 'native' else '-' + arch.lower().replace('_', '-')


MSVC_WARNINGS = [
    '/wd4068',  # unknown pragma
]

# MSVC never defines __BMI2__, so there is no PEXT; missing archs need clang-cl
MSVC_ARCH_FLAGS = {
    'native':  ['-DUSE_MAGIC_BITS'],
    'generic': ['-DUSE_MAGIC_BITS'],
    'AVX':     ['-DUSE_MAGIC_BITS', '/arch:AVX'],
    'AVX2':    ['-DUSE_MAGIC_BITS', '/arch:AVX2'],
    'AVX512':  ['-DUSE_MAGIC_BITS', '/arch:AVX512'],
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


class BuildError(RuntimeError):
    def __init__(self, message, output):
        super().__init__(message)
        self.output = output


def run_step(cmd, step):
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace'
    )
    # one print, so parallel steps don't interleave
    print(f"{' '.join(map(str, cmd))}\n{result.stdout}", end='')
    if result.returncode != 0:
        raise BuildError(f'{step} failed (rc={result.returncode})', result.stdout)
    return result.stdout


def find_tool(names, dirs):
    search_path = os.pathsep.join(str(d) for d in dirs)
    for name in names:
        if path := shutil.which(name, path=search_path):
            return Path(path)
    sys.exit(f'ERROR: {" or ".join(names)} not found')


class Build:
    def __init__(self, exe, sources, cxxflags, define_args):
        self.exe = exe
        self.sources = sources
        self.cxxflags = cxxflags + shlex.split(os.environ.get('CXXFLAGS', ''))
        self.define_args = define_args
        self.include_args = [f'-I{d}' for d in INCLUDES]

    def make(self, exe, compile_flags=(), link_flags=()):
        with tempfile.TemporaryDirectory(prefix='sturddle-native-') as tmp:
            obj_dir = Path(tmp)

            def compile_one(src):
                obj = obj_dir / (Path(src).stem + self.obj_suffix(src))
                return obj, run_step(self.compile_cmd(src, obj, compile_flags), f'compile {src}')

            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
                objs, outputs = zip(*pool.map(compile_one, self.sources))

            return [*outputs, run_step(self.link_cmd(exe, objs, compile_flags, link_flags), f'link {exe.name}')]

    def compile_args(self, src, compile_flags):
        return [*self.cxxflags, *compile_flags, *self.define_args, *self.include_args, src]

    def run(self, pgo=None):
        toolchain = self.pgo_toolchain() if pgo else None
        if pgo and not toolchain:
            print(f'WARNING: {self.no_pgo}, building without PGO')

        if toolchain:
            # BUILD_STAMP changes daily
            flags = self.cxxflags + [d for d in self.define_args if not d.startswith('-DBUILD_STAMP=')]
            status = pgo.build(self.exe, self.make, toolchain, flags)
        else:
            self.make(self.exe)
            status = f'none ({self.no_pgo})' if pgo else None

        if pgo:
            print(f'{PGO_STATUS}{status}')
        return self.exe


class WindowsBuild(Build):
    no_pgo = '--pgo needs clang-cl' # TODO: add MSVC support

    def __init__(self, out_dir, name, arch, sources, define_args):
        vs_dir = find_visual_studio()
        init_msvc_env(vs_dir)

        llvm_dirs = [Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'LLVM/bin']
        if vs_dir:
            llvm_dirs.append(vs_dir / 'VC/Tools/Llvm/x64/bin')
        names = [os.environ['CL_EXE']] if os.environ.get('CL_EXE') else ['clang-cl', 'cl']
        self.cl_exe = find_tool(names, [os.environ['PATH'], *llvm_dirs])
        self.is_clang = self.cl_exe.stem.lower() == 'clang-cl'
        self.rc_exe = find_tool(['llvm-rc' if self.is_clang else 'rc'], [self.cl_exe.parent, os.environ['PATH']])

        cxxflags = [
            '/nologo',
            '/std:c++latest',
            '/EHsc',
            '/fp:fast',
            '/O2',
            '/MT',
        ]
        if self.is_clang:
            cxxflags += ARCH_FLAGS.get(arch, []) + [
                '-O3',
                '-Ofast',
                '-Werror',
                '-Wmissing-field-initializers',
                '-Wno-deprecated-declarations',
                '-Wno-unused-command-line-argument',
                *SHARED_WARNINGS,
            ]
        else:
            if arch and arch not in MSVC_ARCH_FLAGS:
                print(f'WARNING: {arch} needs clang-cl, skipping')
                sys.exit(0)
            cxxflags += MSVC_ARCH_FLAGS.get(arch, []) + MSVC_WARNINGS
            # MSVC has no native target
            define_args = ['-DBUILD_FLAVOR=generic' if d == '-DBUILD_FLAVOR=native' else d for d in define_args]
        cxxflags += ['/D_FORTIFY_SOURCE=0', '/GS-']

        super().__init__(out_dir / f'{name}.exe', sources + RESOURCES, cxxflags, define_args)

    def obj_suffix(self, src):
        return '.res' if src.endswith('.rc') else '.obj'

    def compile_cmd(self, src, obj, compile_flags):
        if src.endswith('.rc'):
            return [str(self.rc_exe), '/FO', str(obj), src]
        return [str(self.cl_exe), '/c', *self.compile_args(src, compile_flags), f'/Fo{obj}']

    def link_cmd(self, exe, objs, compile_flags, link_flags):
        ltcg = [] if any(flag.upper().startswith('/LTCG') for flag in link_flags) else ['/LTCG:OFF']
        return [str(self.cl_exe), *map(str, objs), *compile_flags] + [
            f'/Fe{exe}',
            '/link',
            '/SUBSYSTEM:CONSOLE',
            *ltcg,
            '/STACK:33554432',
            '/MANIFEST:EMBED',
            f'/MANIFESTINPUT:{REPO_ROOT / MANIFEST}',
            *link_flags,
        ]

    def pgo_toolchain(self):
        return ClangPgo(['llvm-profdata'], [self.cl_exe.parent, os.environ['PATH']]) if self.is_clang else None


class LinuxBuild(Build):
    def __init__(self, out_dir, name, arch, sources, define_args):
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

        cxxflags = [
            '-std=c++23',
            '-O3',
            '-ffast-math',
        ]
        cxxflags += ARCH_FLAGS.get(arch, []) + [
            '-Wextra',
            '-Werror',
            *SHARED_WARNINGS,
            '-Wno-unknown-pragmas',
            '-Wno-unused-parameter',
            '-Wno-empty-body',
            '-Wno-int-in-bool-context',
            '-fno-stack-protector',
            '-D_FORTIFY_SOURCE=0',
        ]
        self.link_flags = ['-lpthread']
        if is_clang:
            cxxflags += [
                '-Wno-macro-redefined',
                '-Wno-deprecated-declarations',
                '-stdlib=libc++',
                '-fexperimental-library',
            ]
            self.link_flags += [
                '-fuse-ld=lld',
                f'-L/usr/lib/llvm-{cc_ver}/lib/',
                f'-L/usr/lib/llvm-{cc_ver}/lib/{triplet}',
                '-L/usr/local/opt/llvm/lib/c++',
                # most distros do not ship libc++
                '-stdlib=libc++',
                '-nostdlib++',
                '-Wl,-Bstatic',
                '-lc++experimental',
                '-lc++',
                '-lc++abi',
                '-Wl,-Bdynamic',
            ]
        else:
            # same as clang: VNNI-VL intrinsics cast their args, float -> __bf16 narrows
            cxxflags += ['-flax-vector-conversions', '-Wno-narrowing']
        self.cxx = cxx
        self.cc_ver = cc_ver
        self.is_clang = is_clang

        super().__init__(out_dir / name, sources, cxxflags, define_args)
        if triplet.startswith(('aarch64', 'arm')):
            self.include_args.append('-Innue/simde')

    def obj_suffix(self, src):
        return '.o'

    def compile_cmd(self, src, obj, compile_flags):
        return [self.cxx, '-c', *self.compile_args(src, compile_flags), '-o', str(obj)]

    def link_cmd(self, exe, objs, compile_flags, link_flags):
        return [self.cxx, *map(str, objs), *compile_flags, '-o', str(exe), *self.link_flags, *link_flags]

    def pgo_toolchain(self):
        if not self.is_clang:
            return GccPgo([f'gcov-dump-{self.cc_ver}', 'gcov-dump'], [os.environ['PATH']])
        return ClangPgo([f'llvm-profdata-{self.cc_ver}', 'llvm-profdata'], [f'/usr/lib/llvm-{self.cc_ver}/bin', os.environ['PATH']])


def sample_positions(pgn_dir, count):
    """
    Positions to search while profiling for PGO.
    Random non-book positions from the games in pgn_dir, rotating through game phases.
    """
    import chess.pgn

    games = []
    for path in sorted(pgn_dir.rglob('*.pgn')):
        data = path.read_bytes()
        starts = [m.start() for m in re.finditer(rb'^\[Event ', data, re.M)]
        games += [(path, start, end) for start, end in zip(starts, starts[1:] + [len(data)])]
    if not games:
        sys.exit(f'ERROR: no games in {pgn_dir}')

    rng = random.Random(0)
    positions = []
    attempts = count * 100
    while len(positions) < count and attempts:
        attempts -= 1
        path, start, end = rng.choice(games)
        with open(path, 'rb') as f:
            f.seek(start)
            game = chess.pgn.read_game(io.StringIO(f.read(end - start).decode(errors='replace')))

        low, high = PHASE_PLIES[len(positions) % len(PHASE_PLIES)]
        board = game.board()
        searched = []
        for ply, node in enumerate(game.mainline()):
            if low <= ply < high and node.comment != 'book':
                searched.append(board.fen())
            board.push(node.move)
        if searched:
            positions.append(rng.choice(searched))
    if len(positions) < count:
        sys.exit(f'ERROR: not enough positions in {pgn_dir}')
    return positions


def run_workload(engine, positions, depth):
    proc = subprocess.Popen([engine], cwd=engine.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def exited():
        sys.exit(f'ERROR: {engine.name} exited (rc={proc.wait()})')

    def send(*commands):
        try:
            proc.stdin.write(''.join(f'{command}\n' for command in commands))
            proc.stdin.flush()
        except OSError:
            exited()

    def wait_for(token):
        for line in proc.stdout:
            if line.startswith(token):
                return
        exited()

    send('uci', 'setoption name Threads value 1', 'setoption name OwnBook value false', 'isready')
    wait_for('readyok')
    for i, fen in enumerate(positions, 1):
        print(f'Profiling position {i}/{len(positions)}', end='\r', flush=True)
        send('ucinewgame', f'position fen {fen}', f'go depth {depth}')
        wait_for('bestmove')
    print()
    send('quit')
    if rc := proc.wait():
        sys.exit(f'ERROR: {engine.name} failed (rc={rc})')


class PgoToolchain(ABC):
    @property
    @abstractmethod
    def suffix(self):
        """
        goes on the profile name in --pgo-dir
        """

    @abstractmethod
    def instrument(self, raw_dir):
        """
        compile and link flags for a build that profiles into raw_dir
        """

    @abstractmethod
    def use(self, profile):
        """
        compile and link flags for a build that uses profile
        """

    @abstractmethod
    def merge(self, raw_dir, profile):
        """
        raw_dir data -> profile
        """

    @abstractmethod
    def unusable(self, output):
        """
        failed build output says profile is unreadable
        """

    @abstractmethod
    def mismatch(self, outputs, profile):
        """
        mismatched and total function counts
        """


class ClangPgo(PgoToolchain):
    suffix = '.profdata'
    mismatch_warning = re.compile(r'of (\d+) functions?, (\d+) ha(?:s|ve) mismatched data')

    def __init__(self, profdata_names, profdata_dirs):
        self.profdata_names = profdata_names
        self.profdata_dirs = profdata_dirs

    def instrument(self, raw_dir):
        return [f'-fprofile-instr-generate={raw_dir / "sturddle.profraw"}'], []

    def use(self, profile):
        return [f'-fprofile-instr-use={profile}', '-Wno-error=profile-instr-out-of-date'], []

    def merge(self, raw_dir, profile):
        profdata = find_tool(self.profdata_names, self.profdata_dirs)
        subprocess.check_call([profdata, 'merge', '-o', profile, *raw_dir.glob('*.profraw')])

    def unusable(self, output):
        return 'Error in reading profile' in output

    # only files with mismatches report, so total is not program-wide
    def mismatch(self, outputs, profile):
        counts = self.mismatch_warning.findall(''.join(outputs))
        return sum(int(m) for _, m in counts), sum(int(t) for t, _ in counts)


class GccPgo(PgoToolchain):
    # the profile is a directory with one .gcda file per source
    suffix = '-gcc'
    # gcc names .gcda files after the object path, which is a new temp dir on every build;
    # these flags name them after the source instead. Any aux files go to the (ignored) pgo/
    naming = [f'-fprofile-prefix-path={REPO_ROOT}', '-dumpdir', 'pgo/']
    stale_warnings = ('coverage-mismatch', 'missing-profile')
    # first quoted name on the line: the function, or the .gcda file if missing altogether
    stale_warning = re.compile(r"[\u2018'](.+?)[\u2019'].*\[-W(?:coverage-mismatch|missing-profile)\]")

    def __init__(self, dump_names, dump_dirs):
        self.dump_names = dump_names
        self.dump_dirs = dump_dirs

    def instrument(self, raw_dir):
        return [f'-fprofile-generate={raw_dir}', *self.naming], []

    def use(self, profile):
        return [
            f'-fprofile-use={profile}',
            *(f'-Wno-error={w}' for w in self.stale_warnings),
            *self.naming,
        ], []

    def merge(self, raw_dir, profile):
        if profile.exists():
            shutil.rmtree(profile)
        shutil.copytree(raw_dir, profile)

    def unusable(self, output):
        # 'is version': written by another gcc release
        return any(e in output for e in ('is not a gcov data file', 'is version', 'is corrupted'))

    # gcc only names the stale functions, so count them all in the profile itself;
    # per output, as the total counts a header function once per source
    def mismatch(self, outputs, profile):
        mismatched = sum(len(set(self.stale_warning.findall(output))) for output in outputs)
        gcda = sorted(profile.glob('*.gcda'))
        if not mismatched or not gcda:
            return mismatched, 0
        dump = subprocess.check_output([find_tool(self.dump_names, self.dump_dirs), *gcda], text=True)
        return mismatched, dump.count(':FUNCTION ident=')


class Pgo:
    def __init__(self, args, build_launcher):
        self.arch = args.arch
        self.positions = args.pgo_positions
        self.depth = args.pgo_depth
        self.games = args.pgo_games.resolve()
        self.name = f'{sys.platform}-{args.arch}' + ('-embed' if args.embed else '')
        self.pgo_dir = args.pgo_dir.resolve()
        self.launcher_exe = args.launcher_exe
        self.build_launcher = build_launcher

    def build(self, exe, make, toolchain, flags):
        profile = self.pgo_dir / (self.name + toolchain.suffix)
        meta_path = Path(f'{profile}.json')
        meta = {'flags': flags}
        built = False

        try:
            matches = profile.exists() and json.loads(meta_path.read_text()) == meta
        except (OSError, ValueError):
            matches = False

        if matches:
            try:
                outputs = make(exe, *toolchain.use(profile))
            except BuildError as e:
                if not toolchain.unusable(e.output):
                    raise
                reason = 'unusable profile'
            else:
                built = True
                mismatched, total = toolchain.mismatch(outputs, profile)
                if mismatched <= MAX_MISMATCH * total:
                    return 'reused'
                reason = f'{mismatched} of {total} functions mismatched'
        else:
            reason = 'no matching profile'

        if not self.can_profile():
            print(f'WARNING: {reason}, cannot profile {self.arch} on this CPU')
            if built:
                return f'stale ({reason}, cannot profile on this CPU)'
            make(exe)
            return f'none ({reason}, cannot profile on this CPU)'

        print(f'{reason}, profiling {self.arch}')
        self.collect(exe.name, make, toolchain, profile)
        meta_path.write_text(json.dumps(meta, indent=2))
        make(exe, *toolchain.use(profile))
        return f'profiled ({reason})'

    def can_profile(self):
        if self.arch == 'native':
            return True
        if self.launcher_exe:
            return self.supported_by(self.launcher_exe)
        with tempfile.TemporaryDirectory(prefix='sturddle-launcher-') as tmp:
            return self.supported_by(self.build_launcher(Path(tmp)))

    def supported_by(self, launcher):
        return self.arch in subprocess.check_output([launcher, '--flavors'], text=True).split()

    def collect(self, name, make, toolchain, profile):
        with tempfile.TemporaryDirectory(prefix='sturddle-pgo-') as tmp:
            engine = Path(tmp) / name
            raw_dir = Path(tmp) / 'raw'
            raw_dir.mkdir()
            make(engine, *toolchain.instrument(raw_dir))
            shutil.copy(REPO_ROOT / 'weights.bin', tmp)
            run_workload(engine, sample_positions(self.games, self.positions), self.depth)
            profile.parent.mkdir(parents=True, exist_ok=True)
            toolchain.merge(raw_dir, profile)


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
    parser.add_argument('--no-digest', action='store_true', help='Skip writing the -sha256.txt digest.')
    parser.add_argument('--pgo', action='store_true', help='Profile-guided optimization (clang-cl, clang++ and g++ only).')
    parser.add_argument('--pgo-dir', type=Path, default=PGO_DIR, help='Profile directory (default: pgo).')
    parser.add_argument('--pgo-positions', type=int, default=100, help='Positions to profile (default: 100).')
    parser.add_argument('--pgo-games', type=Path, default=PGN_DIR, help='Directory of PGN games to sample --pgo positions from (default: test/runs).')
    parser.add_argument('--pgo-depth', type=int, default=20, help='Search depth per position (default: 20).')
    parser.add_argument('--launcher-exe', type=Path, help='Launcher used by --pgo to check CPU support (default: build one).')
    args = parser.parse_args()
    if args.launcher_exe and not args.launcher_exe.is_file():
        sys.exit(f'ERROR: launcher not found: {args.launcher_exe}')

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    version = parse_version()
    build_stamp = datetime.now().strftime('%m%d%y')

    if sys.platform.startswith('win'):
        build = WindowsBuild
    elif sys.platform.startswith('linux'):
        build = LinuxBuild
    else:
        sys.exit(f'Unsupported platform: {sys.platform}')

    # no -march, must run on any CPU
    def build_launcher(launcher_dir):
        return build(launcher_dir, f'sturddle-{version}-launcher', None, LAUNCHER_SOURCES, ['-DNATIVE_BUILD=1', '-DNO_ASSERT']).run()

    if args.launcher:
        exe = build_launcher(out_dir)
    else:
        embed = args.embed

        fetch_weights.ensure()

        defines = DEFINES if embed else DEFINES + ['SHARED_WEIGHTS']
        define_args = [f'-D{d}' for d in defines] + [f'-DBUILD_STAMP={build_stamp}', f'-DBUILD_FLAVOR={args.arch}']

        pgo = Pgo(args, build_launcher) if args.pgo else None
        exe = build(out_dir, f'sturddle-{version}{arch_suffix(args.arch)}', args.arch, SOURCES, define_args).run(pgo)

        assets = ('book.bin',) if embed else ('weights.bin', 'book.bin')
        for asset in assets:
            src = REPO_ROOT / asset
            if not src.exists():
                sys.exit(f'ERROR: required asset {asset} not found at {src}')
            shutil.copy(src, out_dir / asset)

    print(f'\nBuilt {exe}')
    if not args.no_digest:
        print(f'Digest {write_sha256(exe)}')


if __name__ == '__main__':
    try:
        main()
    except BuildError as e:
        sys.exit(f'ERROR: {e}')
