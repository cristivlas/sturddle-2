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


# Grading adjust param names: ADJUST_<pawn bucket>_<piece> -> (row, column) in GRADING_ADJUST
GRADING_ADJUST_RE = re.compile(r'ADJUST_(\d+)_(PAWN|KNIGHT|BISHOP|ROOK|QUEEN)$')
GRADING_COLUMN = { 'PAWN': 1, 'KNIGHT': 2, 'BISHOP': 3, 'ROOK': 4, 'QUEEN': 5 }


def get_grading_adjustments(best_params):
    """Collect tuned ADJUST_<bucket>_<piece> values as {(bucket, column): value}."""
    updates = {}
    for k, v in best_params.items():
        m = GRADING_ADJUST_RE.match(k)
        if m:
            updates[(int(m.group(1)), GRADING_COLUMN[m.group(2)])] = scale_param(k, v)
    return updates


def read_eval_piece_grading(header_dir):
    """Read EVAL_PIECE_GRADING from common.h next to the patched header; None if unknown."""
    try:
        with open(os.path.join(header_dir or '.', 'common.h'), 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(r'#define\s+EVAL_PIECE_GRADING\s+(\w+)', line.strip())
                if m:
                    return m.group(1) == 'true'
    except OSError:
        pass
    return None


def parse_piece_values(text, grading):
    """PIECE_VALUES of the active EVAL_PIECE_GRADING branch as a list of 7 ints."""
    in_if = in_else = False
    fallback = None
    for line in text.splitlines():
        s = line.lstrip()
        if re.match(r'#if\s+EVAL_PIECE_GRADING\b', s):
            in_if, in_else = True, False
        elif in_if and s.startswith('#else'):
            in_else = True
        elif in_if and s.startswith('#endif'):
            in_if = in_else = False
        m = re.match(r'#define\s+PIECE_VALUES\s*\{([^}]+)\}', s)
        if m:
            vals = [int(x) for x in m.group(1).split(',')]
            if in_if and not in_else and grading is not False:
                return vals
            if (in_else and grading is False) or not in_if:
                fallback = vals
    return fallback


def patch_piece_values(text, weights, grading):
    """Replace the PIECE_VALUES define of the active EVAL_PIECE_GRADING branch only."""
    lines = text.splitlines(keepends=True)
    in_if = in_else = False
    for i, line in enumerate(lines):
        s = line.lstrip()
        if re.match(r'#if\s+EVAL_PIECE_GRADING\b', s):
            in_if, in_else = True, False
        elif in_if and s.startswith('#else'):
            in_else = True
        elif in_if and s.startswith('#endif'):
            in_if = in_else = False
        if re.match(r'#define\s+PIECE_VALUES\b', s):
            wrong_branch = grading is not None and in_if and (in_else == grading)
            if not wrong_branch:
                lines[i] = re.sub(r'\{[^}]*\}', '{ ' + ', '.join(map(str, weights)) + ' }', line)
    return ''.join(lines)


def patch_grading_adjust(text, updates, weights):
    """Patch the multi-line GRADING_ADJUST macro (one bucket row per line): update
    elements, normalize column widths, refresh the effective-piece-value comment."""
    lines = text.splitlines(keepends=True)
    row = -1
    applied = set()
    for i, line in enumerate(lines):
        if re.match(r'\s*#define\s+GRADING_ADJUST\b', line):
            row = 0
            continue
        if row < 0:
            continue
        m = re.search(r'\{([^}]*)\}', line)
        if not m:
            break
        values = [int(tok) for tok in m.group(1).split(',')]
        for (bucket, col), val in updates.items():
            if bucket == row and col < len(values):
                applied.add((bucket, col))
                if values[col] != int(val):
                    logging.info(f'GRADING_ADJUST[{bucket}][{col}]: {values[col]} -> {val}')
                    values[col] = int(val)
        indent = line[:len(line) - len(line.lstrip())]
        vals_txt = ', '.join([str(values[0])] + [f"{v:>4}" for v in values[1:6]] + [str(values[6])])
        label = '0-4' if row == 0 else f"{4 * row + 1}-{4 * row + 4}"
        comment = f" /* {label:>5} pawns: " + ', '.join(f"{values[c] + weights[c]:>4}" for c in range(1, 6)) + " */" if weights else ""
        lines[i] = f"{indent}{{ {vals_txt} }},{comment} \\\n"
        row += 1

    for key in sorted(set(updates) - applied):
        logging.warning(f'GRADING_ADJUST{list(key)}: no matching macro row, update dropped')
    return ''.join(lines)


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

    return [m_map[k] for k in range(0, 7)]


def patch_header(header_file, best_params):
    logging.info(f"Reading header file: {header_file}")
    with open(header_file, 'r', encoding='utf-8') as f:
        text = f.read()

    grading = read_eval_piece_grading(os.path.dirname(header_file))
    weights = get_weights(best_params)

    new_text = text
    if weights is None:
        logging.warning('Piece weights not tuned')
    else:
        new_text = patch_piece_values(new_text, weights, grading)

    adjust = get_grading_adjustments(best_params)
    if adjust:
        if weights is None:
            weights = parse_piece_values(new_text, grading)
        new_text = patch_grading_adjust(new_text, adjust, weights)

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
