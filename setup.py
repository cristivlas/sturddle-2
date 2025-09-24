import re
import subprocess
import sysconfig
from datetime import datetime
from os import environ, pathsep

from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import armcpu

MIN_CLANG_VER = 16
MIN_GCC_VER = 13

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
            build_ext.build_extensions(self)
else:
    class BuildExt(build_ext):
        pass


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
if cxx and cxx.startswith('clang++') and 'CC' not in environ:
    cc = cxx.replace('clang++', 'clang')
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

NATIVE_UCI = environ.get('NATIVE_UCI', '').lower() in ['1', 'true', 'yes']
SHARED_WEIGHTS = environ.get('SHARED_WEIGHTS', '').lower() in ['1', 'true', 'yes']

# Debug build
if environ.get('BUILD_DEBUG', None):
    if platform.startswith('win'):
        args = [ '/Od', '/Zi' ]
        link = [ '/DEBUG' ]
    else:
        args = [ '-O0', '-D_DEBUG' ]


if SHARED_WEIGHTS:
    args.append('-DSHARED_WEIGHTS')

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

    if NATIVE_UCI:
        args.append('/DNATIVE_UCI=true')

    # clang specific
    if cl_exe.lower().endswith('clang-cl.exe'):
        args += [
            '-Wno-unused-command-line-argument',
            '-Wno-unused-variable',
            '-Wno-nan-infinity-disabled',
        ]
        if not environ.get('BUILD_DEBUG', None):
            args += [ '-Ofast']

    link += ['/LTCG:OFF']  # MSFT linker args
else:
    # Linux, Mac
    STDCPP=20 if NATIVE_UCI else 17

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
        if NATIVE_UCI:
            cc_ver = get_compiler_major_version(cc)
            if cc_ver < MIN_CLANG_VER:
                raise RuntimeError(f'{cc} ver={cc_ver}. NATIVE_UCI requires clang {MIN_CLANG_VER} or higher')

            args.append('-DNATIVE_UCI=true')

            if '-arch arm64' in environ.get('ARCHFLAGS', ''):
                print('ARM64 Target, skipping extra compiler and linker flags.')
            else:
                args += ['-stdlib=libc++', '-fexperimental-library']
                link += [
                    '-fuse-ld=lld',
                    f'-L/usr/lib/llvm-{cc_ver}/lib/',
                    f'-L/usr/lib/llvm-{cc_ver}/lib/x86_64-pc-linux-gnu',
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
                raise RuntimeError(f'NATIVE_UCI uses C++20 and requires GCC {MIN_GCC_VER} or later')

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

if SHARED_WEIGHTS:
    weights = Extension(
        name='weights',
        sources=['weights.cpp'],
        extra_compile_args=args + inc_dirs,
        extra_link_args=link
    )
    ext_modules.append(weights)

setup(ext_modules=ext_modules, cmdclass={'build_ext': BuildExt})
