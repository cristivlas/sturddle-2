#! /usr/bin/env python3
import argparse
import re
import ast
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PST_SCALE = 1000


def parse_best_params(logfile):
    logging.info(f"Reading log file: {logfile}")
    with open(logfile, 'r') as f:
        content = f.read()

    # Split the log content into chunks separated by empty lines
    chunks = content.strip().split('\n\n')
    logging.info(f"Found {len(chunks)} chunks in the log file")

    # Process the last chunk
    last_chunk = chunks[-1]
    logging.info(f"Processing the last chunk")

    best_params = None
    for line in last_chunk.split('\n'):
        if 'best param' in line:
            match = re.search(r"best param: ({.*})", line)
            if match:
                best_params = ast.literal_eval(match.group(1))
                logging.info(f"Best params: {best_params}")
                break

    if not best_params:
        logging.warning("No best params found in the last chunk")

    return best_params


def update_header(header_file, best_params):
    logging.info(f"Reading header file: {header_file}")
    with open(header_file, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        original_line = line
        for param, value in best_params.items():
            # This pattern matches lines like: DECLARE_VALUE(  PARAM_NAME, VALUE, MIN, MAX)
            pattern = re.compile(rf'(DECLARE_VALUE\s*\(\s*{param}\s*,\s*)(-?\d+)(\s*,\s*-?\d+\s*,\s*-?\d+\s*\))')
            match = pattern.search(line)

            if not match:
                pattern = re.compile(rf'(DECLARE_PARAM\s*\(\s*{param}\s*,\s*)(-?\d+)(\s*,\s*-?\d+\s*,\s*-?\d+\s*\))')
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
        updated_lines.append(line)

    logging.info(f"Writing updated header file: {header_file}")
    with open(header_file, 'w') as f:
        f.writelines(updated_lines)


def print_weights(best_params):
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
            val = best_params[k]
        else:
            val = 20000
        m_map[m_sym[k]] = val

    weights = ', '.join(map(str, m_map.values()))
    print(f'#define DEFAULT_WEIGHTS {{ {weights} }}')


#def print_fp_margins(best_params):
#    print("    int fp_margins[] = {")
#    print("        0,")
#    for i in range(1, 17):
#        k = f"FP_{i}"
#        val = best_params.get(k, 0)
#        print(f"        {val}, /* {i} */")
#    print("    };")


def print_piece_square_tables(best_params):
    piece_name = ['PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING']

    print("\nint SQUARE_TABLE[][64] = {")
    print("    {}/* NONE */,")
    for piece in range(1, 7):
        print(f"    {{ /* {piece_name[piece-1]} */")
        print(f"     ", end='')
        for i in range(64):
            key = f"PS_{piece}_{i}"
            val = int(best_params.get(key, 0) * PST_SCALE)
            end_char = ', ' if (i % 8 != 7) else (',\n' if i != 63 else '\n')
            if i % 8 == 0 and i != 0:
                print("     ", end='')  # align rows
            print(f"{val:>4}", end=end_char)
        suffix = ',' if piece != 6 else ''
        print(f"    }}{suffix}")
    print("};")

    print("\nint ENDGAME_KING_SQUARE_TABLE[][64] = {")
    print("    ", end='')
    for i in range(64):
        key = f"PS_KEG_{i}"
        val = best_params.get(key, 0)
        end_char = ', ' if (i % 8 != 7) else (',\n' if i != 63 else '\n')
        if i % 8 == 0 and i != 0:
            print("    ", end='')  # align rows
        print(f"{val:>4}", end=end_char)
    print("};")

def main():
    parser = argparse.ArgumentParser(description='Update C++ header file with best parameters from log file.')
    parser.add_argument('logfile', help='Path to the log file')
    parser.add_argument('--config', default='config.h', help='Path to the C++ header file')

    args = parser.parse_args()

    best_params = parse_best_params(args.logfile)
    if best_params:
        update_header(args.config, best_params)
        logging.info("Header file updated successfully.")
        print_weights(best_params)
        # print_fp_margins(best_params)

        if any(k.startswith('PS_') for k in best_params):
            print_piece_square_tables(best_params)
    else:
        logging.warning("No best params found to update the header file.")

if __name__ == '__main__':
    main()
