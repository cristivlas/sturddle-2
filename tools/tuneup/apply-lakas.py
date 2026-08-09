#! /usr/bin/env python3
import argparse
import re
import ast
import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def root_path():
    return os.path.abspath(os.path.join(os.path.split(sys.argv[0])[0], '../..'))

sys.path.append(root_path())
from chess_engine import *

params = get_param_info()
#print(params)


def scale_param(name, val):
    p = params.get(name)
    if p:
        (default_val, lo, hi, grp, normal) = p
        if normal:
            val = int((val + 1) * (hi - lo) / 2 + lo)

    return val

def parse_best_params(logfile, recommended=False):
    pat = ("recommended param", r"recommended param: ({.*})") if recommended else ("best param", r"best param: ({.*})")
    logging.info(f"Reading log file: {logfile}")
    with open(logfile, 'rb') as f:
        f.seek(-4096, os.SEEK_END)
        content = f.read().decode()

    chunks = content.strip().split('\n\n')
    logging.info(f"Found {len(chunks)} chunk(s) in the log file")

    best_params = None
    for i, chunk in enumerate(reversed(chunks)):
        logging.info(f"Processing chunk {-1 - i}")
        for line in chunk.split('\n'):
            if pat[0] in line:
                match = re.search(pat[1], line)
                if match:
                    best_params = ast.literal_eval(match.group(1))
                    logging.info(f"{pat[0]}: {best_params}")
                    return best_params

    logging.warning(f"No {pat[0]} found in any chunk")
    return None


def update_header(header_file, best_params):
    logging.info(f"Reading header file: {header_file}")
    with open(header_file, 'r') as f:
        lines = f.readlines()

    mod_count = 0
    updated_lines = []
    for line in lines:
        original_line = line
        for param, value in best_params.items():
            value = scale_param(param, value)

            # This pattern matches lines like: DECLARE_VALUE(  PARAM_NAME, VALUE, MIN, MAX)
            pattern = re.compile(rf'(DECLARE_VALUE\s*\(\s*{param}\s*,\s*)(-?\d+)(\s*,\s*-?\d+\s*,\s*-?\d+\s*\))')
            match = pattern.search(line)

            if not match:
                pattern = re.compile(rf'(DECLARE_PARAM\s*\(\s*{param}\s*,\s*)(-?\d+)(\s*,\s*-?\d+\s*,\s*-?\d+\s*\))')
                match = pattern.search(line)

            if not match:
                pattern = re.compile(rf'(DECLARE_NORMAL\s*\(\s*{param}\s*,\s*)(-?\d+)(\s*,\s*-?\d+\s*,\s*-?\d+\s*\))')
                match = pattern.search(line)

            if match:
                before_value = match.group(1)
                old_value = match.group(2)
                after_value = match.group(3)

                # Calculate the spaces required to keep the alignment
                old_value_len = len(old_value)
                new_value = str(value)
                new_value_len = len(new_value)
                if new_value_len < old_value_len:
                    new_value = ' ' * (old_value_len - new_value_len) + new_value
                elif new_value_len > old_value_len:
                    before_value = before_value[:- (new_value_len - old_value_len)]

                # Construct the updated line
                replacement = f'{before_value}{new_value}{after_value}'
                line = pattern.sub(replacement, line)
                if line != original_line:
                    logging.info(f"Updated line: '{original_line.strip()}' to '{line.strip()}'")
                    mod_count += 1
        updated_lines.append(line)

    if mod_count > 0:
        logging.info(f"Writing updated header file: {header_file}")
        with open(header_file, 'w') as f:
            f.writelines(updated_lines)
    else:
        logging.info(f"Unmodified: {header_file}")


def get_endgame_adjustments(best_params):
    m_sym = {
        'ENDGAME_PAWN_ADJUST': 1,
        'ENDGAME_KNIGHT_ADJUST': 2,
        'ENDGAME_BISHOP_ADJUST': 3,
        'ENDGAME_ROOK_ADJUST': 4,
        'ENDGAME_QUEEN_ADJUST': 5,
        'ENDGAME_KING_ADJUST': 6
    }
    m_map = { k:0 for k in range(0, 7) }

    for k in m_sym:
        if k in best_params:
            val = scale_param(k, best_params[k])
        elif k in params:
            val = params[k][0]
        else:
            val = 0
        m_map[m_sym[k]] = val

    if all([v == 0 for v in m_map.values()]):
        return None

    weights = ', '.join(map(str, m_map.values()))
    return (f'#define ENDGAME_ADJUST {{ {weights} }}')


def get_weights(best_params):
    m_sym = {
        'PAWN': 1,
        'KNIGHT': 2,
        'BISHOP': 3,
        'ROOK': 4,
        'QUEEN': 5,
        'KING': 6
    }
    m_map = { k:0 for k in range(0, 7) }

    for k in m_sym:
        if k in best_params:
            val = scale_param(k, best_params[k])
        elif k in params:
            val = params[k][0]
        else:
            val = 20000 if k == 'KING' else 0

        if val <= 1:
            return None

        m_map[m_sym[k]] = val

    weights = ', '.join(map(str, m_map.values()))
    return f'#define PIECE_VALUES {{ {weights} }}'


def patch_header(header_file, best_params):

    weights = get_weights(best_params)

    if weights is None:
        logging.warning('Piece weights not tuned')
        return

    logging.info(f"Reading header file: {header_file}")
    with open(header_file, 'r', encoding='utf-8') as f:
        text = f.read()

    new_text = re.sub(r"#define PIECE_VALUES .*", weights, text)

    adjust = get_endgame_adjustments(best_params)
    if adjust:
        new_text = re.sub(r"#define ENDGAME_ADJUST .*", adjust, new_text)

    if new_text == text:
        logging.info(f'Unmodified: {header_file}')
    else:
        with open(header_file, 'w', encoding='utf-8') as f:
            f.write(new_text)
        logging.info(f'Patched: {header_file}')


def print_piece_square_tables(best_params):
    piece_name = ['PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING']

    print("\nint SQUARE_TABLE[][64] = {")
    print("    {}/* NONE */,")
    for piece in range(1, 7):
        print(f"    {{ /* {piece_name[piece-1]} */")
        print(f"     ", end='')
        for i in range(64):
            key = f"PS_{piece}_{i}"
            val = best_params.get(key)
            if val:
                val = scale_param(key, val)
            else:
                # Use default value
                val = params.get(key, (0,))[0]
            end_char = ', ' if (i % 8 != 7) else ',\n'
            if i % 8 == 0 and i != 0:
                print("     ", end='')  # align rows
            print(f"{val:>4}", end=end_char)
        suffix = ',' if piece != 6 else ''
        print(f"    }}{suffix}")
    print("};")

    print("\nint ENDGAME_KING_SQUARE_TABLE[64] = {")
    print("    ", end='')
    for i in range(64):
        key = f"PS_KEG_{i}"
        val = scale_param(key, best_params.get(key, 0))
        end_char = ', ' if (i % 8 != 7) else (',\n' if i != 63 else '\n')
        if i % 8 == 0 and i != 0:
            print("    ", end='')  # align rows
        print(f"{val:>4}", end=end_char)
    print("};")


def main():
    parser = argparse.ArgumentParser(description='Update C++ header file with best parameters from log file.')
    parser.add_argument('logfile', help='Path to the log file')
    parser.add_argument('--config', default='config.h', help='Path to the C++ header file')
    parser.add_argument('-p', '--patch', help="Optional file to patch (normally chess.h)")
    parser.add_argument('-r', '--recommended', action='store_true', help='Use recommended param instead of best')

    args = parser.parse_args()

    best_params = parse_best_params(args.logfile, args.recommended)
    if best_params:
        update_header(args.config, best_params)

        if args.patch:
            patch_header(args.patch, best_params)

        if any(k.startswith('PS_') for k in best_params):
            print_piece_square_tables(best_params)
    else:
        logging.warning("No best params found to update the header file.")

if __name__ == '__main__':
    main()
