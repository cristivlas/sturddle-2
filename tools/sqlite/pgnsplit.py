#!/usr/bin/env python3
import argparse
import mmap
import os
import sys

from tqdm import tqdm


def split_pgn_file(input_file, output_prefix, n):
    with open(input_file, 'rb') as pgn_file:
        mmapped_file = mmap.mmap(pgn_file.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = os.path.getsize(input_file)
        event_indices = []

        event_bytes = b'[Event '
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Counting games')
        pos = 0
        while pos < len(mmapped_file):
            new_pos = mmapped_file.find(event_bytes, pos)
            if new_pos == -1:
                break
            event_indices.append(new_pos)
            pbar.update(new_pos - pos)
            pos = new_pos + len(event_bytes)
        pbar.update(file_size - pos)
        pbar.close()

        num_games = len(event_indices)
        games_per_file = num_games // n

        pbar = tqdm(total=num_games, unit='game', desc='Splitting file')
        for i in range(n):
            start_index = i * games_per_file
            if i == n - 1:
                end_index = num_games
            else:
                end_index = start_index + games_per_file
            start_line = event_indices[start_index]
            end_line = event_indices[end_index] if end_index < num_games else len(mmapped_file)

            output_file = f'{output_prefix}_{i+1:02d}.pgn'
            with open(output_file, 'wb') as out_file:
                out_file.write(mmapped_file[start_line:end_line])
            pbar.update(end_index - start_index)
        pbar.close()
    print('Successfully split PGN file into {} parts.'.format(n))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split PGN files')
    parser.add_argument('input_file', type=str, help='Input PGN file')
    parser.add_argument('output_prefix', type=str, help='Output file prefix')
    parser.add_argument('n', type=int, help='Number of files to split into')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print('Input file {} does not exist.'.format(args.input_file))
        sys.exit(1)

    if args.n <= 0:
        print('Number of parts (n) should be a positive integer.')
        sys.exit(1)

    split_pgn_file(args.input_file, args.output_prefix, args.n)
