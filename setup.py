from setuptools import Extension, setup
from Cython.Build import cythonize
from os import environ
import sysconfig

sourcefiles = [
    '__init__.pyx',
    'captures.cpp',
    'chess.cpp',
    'context.cpp',
    'search.cpp',
]

platform = sysconfig.get_platform()
"""
Compiler args.
"""
inc_dirs = ['-I./libpopcnt', '-I./thread-pool']

# Assert-enabled build
args = []
link = []

# Debug build
# if platform.startswith('win'):
#     args = [ '/Od', '/Zi' ]
#     link = ['/DEBUG']
# else:
#     args = [ '-O0', '-D_DEBUG' ]

# Release build
args = ['-DNO_ASSERT', '-DLAZY_STATE_COPY' ]

if platform.startswith('win'):
    args.append('/std:c++17')
else:
    if '-O0' not in args:
        args.append('-O3')
    args.append('-std=c++17')
    args.append('-Wall')
    args.append('-Wextra')
    args.append('-Wno-unused-label')
    args.append('-Wno-unknown-pragmas')
    args.append('-Wno-unused-parameter')
    args.append('-Wno-unused-variable')
    args.append('-DCYTHON_WITHOUT_ASSERTIONS')

    cc = 'clang' if platform.startswith('macos') else environ.get('CC', None)
    if cc and cc.startswith('clang'):
        args.append('-Wno-deprecated-declarations')
    else:
        args.append('-fno-extern-tls-init')


extensions = [Extension('chess_engine', sources=sourcefiles, extra_compile_args=args + inc_dirs, extra_link_args=link)]

setup(ext_modules=cythonize(extensions))