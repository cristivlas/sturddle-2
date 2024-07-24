#!/usr/bin/env python3
"""
Convert sqlite3 databases into h5 files, in parallel.
"""
import argparse
import os
import sqlite3

import libtmux


def get_total_rows(db_file):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM position')
    total_rows = cursor.fetchone()[0]
    conn.close()
    return total_rows

def create_tmux_session(session_name, db_file, start_row, row_count, total_rows, clip):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    server = libtmux.Server()
    session = server.new_session(session_name=session_name)
    window = session.attached_window
    pane = window.attached_pane

    for begin in range(start_row, total_rows, row_count):
        command = f'{script_directory}/toh5.py {db_file} --begin {begin} --row-count {row_count} -c {clip}'
        #pane.send_keys(f'echo {command}')
        pane.send_keys(command)
        try:
            if begin + row_count < total_rows:
                pane = window.split_window(attach=False)
        except:
            break
    session.attach_session()

# https://bugs.python.org/issue34803
large_int = lambda x: int(float(x))

def main():
    parser = argparse.ArgumentParser(description='Make datasets from database.')
    parser.add_argument('db_file', help='Database file to slice.')
    parser.add_argument('--row-count', required=True, type=large_int, help='Row count for each slice.')
    parser.add_argument('--session-name', default='default_session', help='The name of the tmux session.')
    parser.add_argument('--start-row', type=large_int, default=0)
    parser.add_argument('--total-rows', type=large_int)
    parser.add_argument('--clip', '-c', required=True, type=large_int)

    args = parser.parse_args()

    total_rows = get_total_rows(args.db_file) if args.total_rows is None else args.total_rows
    create_tmux_session(args.session_name, args.db_file, args.start_row, args.row_count, total_rows, args.clip)

if __name__ == '__main__':
    main()
