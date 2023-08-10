'''
Build all-in-one executable using pyinstaller.

Part of Sturddle Chess 2.0
Copyright (c) 2023 Cristian Vlasceanu.
'''
import argparse
import glob
import os
import platform
import secrets
import shutil
import string
import sys

BOOK = 'book.bin'

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

    # cleanup
    delete(['*.spec', 'build', mods])

    exe = f'"{sys.executable}"' # the Python interpreter

    ARCHS = [''] # default

    if platform.machine() in ['x86_64', 'AMD64']:
        ARCHS = ['AVX512', 'AVX2', '']
    elif platform.machine() == 'aarch64':
        ARCHS = ['ARMv8_2', '']

    if len(ARCHS) == 1 and not args.native_uci:
        print('Python UCI implementation not supported on this platform')
        os._exit(-1)

    for arch in ARCHS:
        delete(['uci.cpp', '__init__.cpp']) # force re-cythonize
        print('*********************************************************')
        print(f'Building {arch if arch else "generic"} module')
        print('*********************************************************')

        arch_flags = ''
        if is_windows():
            if arch:
                arch_flags = f'/arch:{arch}'
        # otherwise assume Clang or GCC on POSIX
        elif arch == 'AVX2':
            arch_flags = '-march=core-avx2 -mtune=core-avx2' # '-DUSE_AVX2'
        elif arch == 'AVX512':
            arch_flags = '-march=skylake-avx512 -mtune=skylake-avx512' # '-DUSE_AVX512'
        elif arch == 'ARMv8_2':
            arch_flags = '-march=armv8.2-a+fp16'

        os.environ['CXXFLAGS'] = arch_flags
        arch = arch.lower()
        os.environ['TARGET'] = f'chess_engine_{arch}' if arch else 'chess_engine'

        if run_cmd(f'{exe} setup.py clean --all') or run_cmd(f'{exe} setup.py build_ext --inplace'):
            print('Build failed.')
            sys.exit(-1)

    # Generate key
    KEY = ''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(16))

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
    if run_cmd(f'{installer} {script} -p . --onefile {" ".join(libs)} {data} --key="{KEY}"'):
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
