#!/usr/bin/env python3
import argparse
import mmap
import os
import sys

from tqdm import tqdm


def split_pgn_file(input_file, output_prefix, n):
    file_size = os.path.getsize(input_file)

    with open(input_file, 'r', encoding='latin1') as pgn_file:
        mmapped_file = mmap.mmap(pgn_file.fileno(), 0, access=mmap.ACCESS_READ)
        lines = mmapped_file.read().decode('latin1').splitlines()
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Reading file')
        for line in lines:
            delta = len(line.encode('latin1')) + 1
            pbar.update(min(delta, file_size - pbar.n))
        pbar.close()

    event_indices = [i for i, line in tqdm(enumerate(lines), total=len(lines), unit='game', desc='Counting games') if line.startswith('[Event ')]
    num_games = len(event_indices)
    games_per_file = num_games // n

    for i in tqdm(range(n), total=n, unit='file', desc='Splitting files'):
        start_index = i * games_per_file
        if i == n - 1:
            end_index = num_games
        else:
            end_index = start_index + games_per_file
        start_line = event_indices[start_index]
        end_line = event_indices[end_index] if end_index < num_games else len(lines)

        output_file = f'{output_prefix}_{i+1:02d}.pgn'
        with open(output_file, 'w') as out_file:
            for line in lines[start_line:end_line]:
                out_file.write(line)
    print(f'Successfully split PGN file into {n} parts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split PGN files')
    parser.add_argument('input_file', type=str, help='Input PGN file')
    parser.add_argument('output_prefix', type=str, help='Output file prefix')
    parser.add_argument('n', type=int, help='Number of files to split into')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f'Input file {args.input_file} does not exist.')
        sys.exit(1)

    if args.n <= 0:
        print('Number of parts (n) should be a positive integer.')
        sys.exit(1)

    split_pgn_file(args.input_file, args.output_prefix, args.n)
