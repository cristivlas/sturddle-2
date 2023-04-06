#!/usr/bin/env python3
import argparse
import os

import chess
import chess.pgn as pgn
from dbutils.sqlite import SQLConn
from tqdm import tqdm


def pgn_to_epd(game, mate_score):
    '''
    Extracts positions and scores from a PGN game object
    Returns a list of (epd, score) tuples
    '''
    board = game.board()
    epd_list = []
    for node in game.mainline():
        board.push(node.move)
        epd = board.epd()
        comment = node.comment.strip()
        if comment.startswith('[%eval'):
            # Parse score in centipawns
            score_str = comment.split()[1][:-1]
            if score_str.startswith('#'):
                # Convert mate score to centipawns
                mate_in = int(score_str[1:])
                score = mate_score - mate_in
            else:
                score = int(float(score_str) * 100)
            if not board.turn:
                score = -score

            #print(epd, node.move, score_str, score)
            epd_list.append((epd, score))

    return epd_list



def main(args):
    with SQLConn(args.output) as sqlconn:
        sqlconn.exec('''CREATE TABLE IF NOT EXISTS position(
                        epd text PRIMARY KEY,
                        depth integer,
                        score integer
                        )''')

        # Estimate total number of games based on file size
        file_size = os.path.getsize(args.pgn_file)
        avg_game_size = 500  # bytes (adjust this value as needed)
        num_games = file_size // avg_game_size

        # Open PGN file
        with open(args.pgn_file, 'r') as pgn_data:
            game_iter = iter(lambda: pgn.read_game(pgn_data), None)
            for game in tqdm(game_iter, total=num_games):
                epd_list = pgn_to_epd(game, args.mate_score)
                for epd, cp_score in epd_list:
                    sqlconn.exec('''INSERT OR IGNORE INTO position(epd, depth, score)
                                    VALUES(?, ?, ?)''', (epd, -1, int(cp_score)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse PGN file and output SQLite database of evaluated positions')
    parser.add_argument('pgn_file', help='PGN input file')
    parser.add_argument('-o', '--output', default='output.db', help='SQLite output file (default: output.db)')
    parser.add_argument('--mate-score', type=int, default=29999, help='Mate score in centipawns (default: 29999)')
    args = parser.parse_args()

    main(args)
