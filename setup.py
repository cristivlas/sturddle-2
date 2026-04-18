import re
import subprocess
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from os import cpu_count, environ, pathsep

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import armcpu

MIN_CLANG_VER = 16
MIN_GCC_VER = 13


def _parallel_build(build_ext_self):
    compiler = build_ext_self.compiler
    pool = ThreadPoolExecutor(max_workers=cpu_count() or 4)
    futures = []

    if cl_exe:
        # MSVCCompiler.compile() calls spawn() per file and never calls _compile();
        # patch compile() + spawn() instead.
        _compiling = False
        original_spawn = compiler.spawn
        original_compile = compiler.compile

        def _spawn(cmd):
            if _compiling:
                futures.append(pool.submit(original_spawn, cmd))
            else:
                original_spawn(cmd)

        def _compile_msvc(*args, **kwargs):
            nonlocal _compiling
            _compiling = True
            try:
                result = original_compile(*args, **kwargs)
                for f in futures:
                    f.result()
                futures.clear()
            finally:
                _compiling = False
            return result

        compiler.spawn = _spawn
        compiler.compile = _compile_msvc
    else:
        # Unix: _compile() is called per file, link() is called once after.
        original_compile = compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            futures.append(pool.submit(original_compile, obj, src, ext, cc_args, extra_postargs, pp_opts))

        original_link = compiler.link

        def _link(*args, **kwargs):
            for f in futures:
                f.result()
            futures.clear()
            original_link(*args, **kwargs)

        compiler._compile = _compile
        compiler.link = _link

    try:
        build_ext.build_extensions(build_ext_self)
    finally:
        pool.shutdown(wait=False)


'''
Monkey-patch MSVCCompiler to use clang-cl.exe on Windows.
'''
cl_exe = environ.get('CL_EXE', '')
if cl_exe:
    try:
        from setuptools._distutils._msvccompiler import MSVCCompiler, _find_exe
    except:
        # setuptools >= 80
        from setuptools._distutils._msvccompiler import MSVCCompiler
        from setuptools._distutils.compilers.C.msvc import _find_exe

    _initialize = MSVCCompiler.initialize

    def initialize(self, platform=None):
        _initialize(self, platform)
        paths = self._paths.split(pathsep)
        self.cc = _find_exe(cl_exe, paths)
        print(self.cc)

    class BuildExt(build_ext):
        def build_extensions(self):
            self.compiler.__class__.initialize = initialize
            _parallel_build(self)
else:
    class BuildExt(build_ext):
        def build_extensions(self):
            _parallel_build(self)


def get_compiler_major_version(compiler=None):

    if compiler is None:
        compiler = environ.get('CC', 'gcc')

    version_string = subprocess.check_output([compiler, '--version']).decode('utf-8')

    # This pattern looks for the first digit(s), followed by a dot, followed by any digit(s)
    # and then a dash or space. That should match the major.minor part of the version.
    version_pattern = re.compile(r'(\d+)\.\d+\.\d+')
    version = version_pattern.search(version_string)
    if version:
        # The major version number is before the first dot
        return int(version.group(1).split('.')[0])
    else:
        raise ValueError('Could not parse ' + compiler + ' version from string: ' + version_string)


# build_stamp = datetime.now().strftime('%m%d%y.%H%M')
build_stamp = datetime.now().strftime('%m%d%y')

sourcefiles = [
    '__init__.pyx',
    'chess.cpp',
    'context.cpp',
    'search.cpp',
    'uci_native.cpp',
    'tbprobe.cpp',
]


cxx = environ.get('CXX')
if cxx and 'CC' not in environ:
    if cxx.startswith('clang++'):
        cc = cxx.replace('clang++', 'clang')
        environ['CC'] = cc
    elif cxx.startswith('g++'):
        cc = cxx.replace('g++', 'gcc')
        environ['CC'] = cc

"""
Compiler args.
"""
inc_dirs = [
    '-I./libpopcnt',
    '-I./magic-bits/include',
    '-I./version2',
    '-I.',
    '-I./Fathom/src',
]

link = []

if environ.get('BUILD_ASSERT', None):
    args = []
else:
    args = ['-DNO_ASSERT']  # Release build

platform = sysconfig.get_platform()

NATIVE_UCI = environ.get('NATIVE_UCI', '1').lower() in ['1', 'true', 'yes']

# Debug build
if environ.get('BUILD_DEBUG', None):
    if platform.startswith('win'):
        args = [ '/Od', '/Zi' ]
        link = [ '/DEBUG' ]
    else:
        args = [ '-O0', '-D_DEBUG' ]

args.append('-DBUILD_STAMP=' + build_stamp)
args += environ.get("CXXFLAGS", '').split()


arm_arch = armcpu.get_arch()
if not arm_arch is None:
    # Emulate SSE on ARM using: https://github.com/simd-everywhere/simde
    args += [ '-I./simde', '-Wno-bitwise-instead-of-logical' ]
    if arm_arch == 'armv7':
        args += [ '-mfpu=neon-vfpv4', '-mfloat-abi=hard' ]

if platform.startswith('win'):
    # Windows build
    args += [
        '/fp:fast',
        '/std:c++20',
        '/DWITH_NNUE',
        '/DCALLBACK_PERIOD=8192',
        '/DCYTHON_WITHOUT_ASSERTIONS',
    ]

    if environ.get('BUILD_DEBUG', None):
        # Enable runtime checks in debug build
        # args += [ '/RTCc', '-D_ALLOW_RTCc_IN_STL' ]
        args += [ '/guard:cf', '/RTCs', '/RTCu' ]
        link += [ '/GUARD:CF' ]
    else:
        args += [ '/D_FORTIFY_SOURCE=0', '/GS-' ]
        link += [ '/GUARD:NO' ]

    args.append(f'/DNATIVE_UCI={int(NATIVE_UCI)}')

    # clang specific
    if cl_exe.lower().endswith('clang-cl.exe'):
        args += [
            '-Wno-deprecated-declarations',
            '-Wno-unused-command-line-argument',
            '-Wno-unused-label',
            '-Wno-unused-variable',
            '-Wno-nan-infinity-disabled',
        ]
        if not environ.get('BUILD_DEBUG', None):
            args += [ '-O3', '-Ofast' ]

    else:
        # assume Microsoft compiler
        args += [
            '-D_CRT_SECURE_NO_WARNINGS',
            '/wd4068',
            '/wd4305', # warning C4305: '=': truncation from 'int' to 'bool'
            '/wd4101', # warning C4101: '__pyx_t_1': unreferenced local variable
            '/wd4551', # warning C4551: function call missing argument list
            '/wd4244', # warning C4244: '=': conversion from 'Py_ssize_t' to 'long', possible loss of data
        ]

    link += ['/LTCG:OFF']  # MSFT linker args
else:
    # Linux, Mac
    # STDCPP=20 if NATIVE_UCI else 17
    STDCPP=20

    # Linux and Mac
    if '-O0' not in args:
        args.append('-O3')
    args += [
        f'-std=c++{STDCPP}',
        '-Wextra',
        '-Wno-unused-label',
        '-Wno-unknown-pragmas',
        '-Wno-unused-parameter',
        '-Wno-unused-variable',
        '-DCYTHON_WITHOUT_ASSERTIONS',
        '-DCALLBACK_PERIOD=8192',
        '-fno-stack-protector',
        '-DWITH_NNUE',
        '-Wno-empty-body',
        '-Wno-int-in-bool-context',
    ]

    # Silence off Py_DEPRECATED warnings for clang;
    # clang is the default compiler on macosx.
    cc = 'clang' if platform.startswith('macos') else environ.get('CC')
    if cc and cc.startswith('clang'):
        args += [
            '-Wno-macro-redefined',
            '-D_FORTIFY_SOURCE=0',  # Avoid the overhead.
            '-Wno-deprecated-declarations',
            '-fvisibility=hidden',
            '-DPyMODINIT_FUNC=__attribute__((visibility("default"))) extern "C" PyObject*',
        ]
        args.append(f'-DNATIVE_UCI={int(NATIVE_UCI)}')

        if STDCPP >= 20:
            cc_ver = get_compiler_major_version(cc)
            if cc_ver < MIN_CLANG_VER:
                raise RuntimeError(f'{cc} ver={cc_ver}. clang {MIN_CLANG_VER} or higher required.')

            if '-arch arm64' in environ.get('ARCHFLAGS', ''):
                print('ARM64 Target, skipping extra compiler and linker flags.')
            else:
                triplet = subprocess.check_output([cc, '-dumpmachine'], text=True).strip()
                args += ['-stdlib=libc++', '-fexperimental-library']
                link += [
                    '-fuse-ld=lld',
                    f'-L/usr/lib/llvm-{cc_ver}/lib/',
                    f'-L/usr/lib/llvm-{cc_ver}/lib/{triplet}',
                    '-L/usr/local/opt/llvm/lib/c++',
                    '-lc++',
                    '-lc++experimental',
                ]
        else:
            args.append('-DNATIVE_UCI=false')

    else:
        # Not Clang
        if NATIVE_UCI:
            if get_compiler_major_version() < MIN_GCC_VER:
                raise RuntimeError(f'NATIVE_UCI uses C++20 and requires GCC >= {MIN_GCC_VER} or Clang >= {MIN_CLANG_VER}')

            args.append('-DNATIVE_UCI=true')
        else:
            args.append('-DNATIVE_UCI=false')

        args.append('-DUSE_MAGIC_BITS')

"""
end of compiler args.
"""

extensions = [
    Extension(
        name=environ.get('TARGET', 'chess_engine'),
        sources=sourcefiles,
        extra_compile_args=args + inc_dirs,
        extra_link_args=link
    )
]
if not NATIVE_UCI:
    extensions.append(Extension(
        name='uci',
        sources=['uci.pyx'],
        extra_compile_args=args + inc_dirs,
        extra_link_args=link
    ))

ext_modules = cythonize(extensions)

setup(ext_modules=ext_modules, cmdclass={'build_ext': BuildExt})
