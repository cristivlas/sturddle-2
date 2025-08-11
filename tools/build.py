'''
Build all-in-one executable using pyinstaller.

Part of Sturddle Chess 2.0
Copyright (c) 2023 - 2025 Cristian Vlasceanu.
'''
import argparse
import glob
import os
import platform
import shutil
import sys

BOOK = 'book.bin'


def find_editbin():
    """Find editbin using distutils MSVC detection"""
    try:
        from setuptools._distutils._msvccompiler import MSVCCompiler
    except:
        # setuptools >= 80
        from setuptools._distutils._msvccompiler import MSVCCompiler

    # Create MSVC compiler instance to get environment
    compiler = MSVCCompiler()
    compiler.initialize()

    # Get the compiler executable path
    if hasattr(compiler, 'cc') and compiler.cc:
        cl_exe = compiler.cc
        # print(f"Found compiler: {cl_exe}")

        # editbin should be in the same directory as cl.exe
        compiler_dir = os.path.dirname(cl_exe)
        editbin_path = os.path.join(compiler_dir, 'editbin.exe')

        if os.path.exists(editbin_path):
            return editbin_path

    raise RuntimeError('Could not locate editbin')

def delete_file_or_dir(path):
    paths = glob.glob(path)
    for p in paths:
        if os.path.exists(p):
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)

def delete(list):
    for path in list:
        delete_file_or_dir(path)

def is_windows():
    return os.name == 'nt' or sys.platform in ['win32', 'cygwin']

def run_cmd(command):
    print(command)
    return os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build all-in-one executable')
    parser.add_argument('-v', '--venv')
    parser.add_argument('--native-uci', dest='native_uci', action='store_true', default=True)
    parser.add_argument('--no-native-uci', dest='native_uci', action='store_false')

    args = parser.parse_args()

    if args.native_uci:
        os.environ['NATIVE_UCI'] = '1'

    mods = '*.pyd' if is_windows() else '*.so'
    editbin = find_editbin() if is_windows() else None
    cl_exe = os.environ.get('CL_EXE', '')

    delete(['*.spec', 'build', mods]) # cleanup

    exe = f'"{sys.executable}"' # the Python interpreter

    ARCHS = [''] # default

    if args.native_uci:
        if platform.machine() in ['x86_64', 'AMD64']:
            # TODO: Re-evaluate VNNI support in next release
            # ARCHS = ['AVX512', 'AVX2', 'AVX2_VNNI', '']
            ARCHS = ['AVX512', 'AVX2', '']
        elif platform.machine() == 'aarch64':
            ARCHS = ['ARMv8_2', '']

    for i, arch in enumerate(ARCHS):
        delete(['uci.cpp', '__init__.cpp']) # force re-cythonize
        print('*********************************************************')
        print(f'Building {arch if arch else "generic"} module')
        print('*********************************************************')

        arch_flags = ''
        if is_windows() and not cl_exe.lower().startswith('clang-cl'):
            if arch:
                if arch.endswith('_VNNI'):
                    continue
                arch_flags = f'/arch:{arch}'
        # otherwise assume Clang or GCC on POSIX
        elif arch == 'AVX2':
            arch_flags = '-march=core-avx2 -mtune=core-avx2'
        elif arch == 'AVX2_VNNI':
            arch_flags = '-march=core-avx2 -mtune=core-avx2 -mavxvnni'
        elif arch == 'AVX512':
            arch_flags = '-march=skylake-avx512 -mtune=skylake-avx512'
        elif arch == 'ARMv8_2':
            arch_flags = '-march=armv8.2-a+fp16'

        # os.environ['CXXFLAGS'] = f'{arch_flags} -DUSE_MMAP_HASH_TABLE -DSHARED_WEIGHTS'
        os.environ['CXXFLAGS'] = f'{arch_flags} -DSHARED_WEIGHTS'

        # Build the shared weights on the 1st flavor only
        if i == 0:
            os.environ['SHARED_WEIGHTS'] = 'true'
        else:
            os.environ.pop('SHARED_WEIGHTS', None)

        arch = arch.lower()
        os.environ['TARGET'] = f'chess_engine_{arch}' if arch else 'chess_engine'

        if run_cmd(f'{exe} setup.py clean --all') or run_cmd(f'{exe} setup.py build_ext --inplace'):
            print('Build failed.')
            sys.exit(-1)

    if args.venv:
        scripts_dir = 'Scripts' if is_windows() else 'bin'
        installer = os.path.join(args.venv, scripts_dir, 'pyinstaller')
    else:
        installer = 'pyinstaller'

    # PyInstaller arguments:
    script = 'main.py' if args.native_uci else 'sturddle.py'

    libs = [ f'--add-binary={mods}{os.path.pathsep}.' ]

    if args.native_uci:
        for libcxx in [
            '/usr/lib/llvm-15/lib/libc++.1.so',
            '/usr/lib/llvm-15/lib/libc++abi.1.so',
            '/usr/local/opt/llvm/lib/c++/libc++.1.dylib',
            '/usr/local/opt/llvm/lib/c++/libc++abi.1.dylib'
        ]:
            if os.path.exists(libcxx):
                libs.append(f'--add-binary={libcxx}{os.path.pathsep}.')

    data = f'--add-data={BOOK}{os.path.pathsep}.'

    # run PyInstaller
    if run_cmd(f'{installer} {script} -p . --onefile {" ".join(libs)} {data} --icon chess.ico'):
        print('pyinstaller failed')
        sys.exit(-2)

    # import the engine we just built, to determine its version
    sys.path.append('.')
    import chess_engine

    MAIN = os.path.join('dist', 'main' if args.native_uci else 'sturddle')
    NAME = os.path.join('dist', f'sturddle-{".".join(chess_engine.__build__[:2])}')
    if is_windows():
        MAIN += '.exe'
        NAME += '.exe'
        # Configure 32M stack
        if run_cmd(f'"{editbin}" /STACK:33554432 {MAIN}'):
            print('failed to set stack size')
            sys.exit(-2)
    else:
        NAME += f'-{platform.system()}-{platform.machine()}'

    while True:
        try:
            print(f'rename {MAIN} as {NAME}')
            os.replace(MAIN, NAME)
            break
        except Exception as e:
            print(e)
            os.unlink(NAME)
