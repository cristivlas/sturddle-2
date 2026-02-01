#! /usr/bin/env python3
'''
Alternative engine bootloader that uses the native UCI implementation.
'''
# import everything for the benefit of pyinstaller
import argparse
import importlib
import logging
import os

import chess
import chess.pgn
import chess.polyglot
import psutil

import armcpu

'''
Import the chess engine module flavor that best matches the CPU capabilities.
'''
if armcpu.get_arch() is not None:
    # ARM
    def _is_fp16_supported():
        cpuinfo = {}
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = map(str.strip, line.split(':', 1))
                        cpuinfo[key] = value
        except FileNotFoundError:
            pass
        return 'Features' in cpuinfo and 'asimdhp' in cpuinfo['Features']

    flavors = {
        'chess_engine_armv8_2': _is_fp16_supported,
        'chess_engine': lambda *_: True
    }
elif armcpu.is_apple_silicon():
    # importing cpufeature on Apple Silicon under Rosetta
    # emulation crashes with 'Floating point exception: 8'
    flavors = {
        'chess_engine': lambda *_: True
    }
else:
    # Select AVX, AVX2 or AVX512 on x86/AMD64
    import cpufeature

    def _is_avx_supported():
        return cpufeature.extension.CPUFeature['AVX']

    def _is_avx2_supported():
        # Check for AVX2 with FMA
        return (cpufeature.extension.CPUFeature['AVX2'] and
                (cpufeature.extension.CPUFeature['FMA3'] or cpufeature.extension.CPUFeature['FMA4']))

    def _is_avx512_supported():
        return (cpufeature.extension.CPUFeature['AVX512f'] and
                cpufeature.extension.CPUFeature['AVX512bw'])

    # Check for AVX512 Cooper Lake extensions (BF16, VNNI).
    def _is_avx512_cooperlake():
        try:
            import cpuid
            _, ebx, ecx, _ = cpuid.cpuid_count(7, 0)
            eax = cpuid.cpuid_count(7, 1)[0]

            required_ebx = (
                (1 << 16) |  # AVX512F
                (1 << 17) |  # AVX512DQ
                (1 << 28) |  # AVX512CD
                (1 << 30) |  # AVX512BW
                (1 << 31)    # AVX512VL
            )
            has_base = (ebx & required_ebx) == required_ebx
            has_vnni = bool((ecx >> 11) & 1)      # AVX512_VNNI
            has_bf16 = bool((eax >> 5) & 1)       # AVX512_BF16

            return has_base and has_vnni and has_bf16
        except:
            return False

    def _is_avx2_vnni_supported():
        if not _is_avx2_supported():
            return False
        try:
            import cpuid
            eax = cpuid.cpuid_count(7, 1)[0]
            return bool((eax >> 4) & 1)  # AVX-VNNI in CPUID(7,1) EAX bit 4
        except:
            return False

    flavors = {
        'chess_engine_avx512_bf16': _is_avx512_cooperlake,
        'chess_engine_avx512': _is_avx512_supported,
        'chess_engine_avx2_vnni': _is_avx2_vnni_supported,
        'chess_engine_avx2': _is_avx2_supported,
        'chess_engine_avx': _is_avx_supported,
        'chess_engine': lambda *_: True,
    }

def load_engine():
    for eng in flavors:
        if not flavors[eng]():
            continue
        try:
            engine = importlib.import_module(eng)
            globals().update({k:v for k, v in engine.__dict__.items() if not k.startswith('_')})
            logging.info(f'Loaded {engine.__name__}')
            return engine
        except Exception as e:
            logging.warning(e)

def _configure_logging(args):
    log = logging.getLogger()
    for h in log.handlers[:]:
        log.removeHandler(h)
    format = '%(asctime)s %(levelname)-8s %(process)d %(message)s'
    filename = f'{args.logfile}.{os.getpid()}' if args.separate_logs else args.logfile
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, filename=filename, format=format)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sturddle Chess Engine')
    parser.add_argument('-D', '--dev-mode', action='store_true', help='enable developer-mode features')
    parser.add_argument('-l', '--logfile', default='sturddle.log')
    parser.add_argument('-s', '--separate-logs', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose logging')
    args = parser.parse_args()

    _configure_logging(args)

    engine = load_engine()
    assert engine, 'Failed to load engine.'
    try:
        engine.uci('Sturddle', debug=args.verbose, dev_mode=args.dev_mode)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
        logging.exception('UCI')
        os._exit(-1)

    os._exit(0)
