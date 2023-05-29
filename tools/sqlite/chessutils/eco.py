import csv
import os
import chess
import chess.pgn

class EcoAPI:
    '''
    Wrapper for https://github.com/lichess-org/chess-openings
    '''
    def __init__(self, data_dir=None):
        '''
        Read TSV files and index by FEN and name.
        '''
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'eco')
        self.by_fen = {}

        for fname in self.tsv_files():
            self.read_tsv_file(fname)


    def tsv_files(self):
        for dir, _subdirs, files in os.walk(self.data_dir):
            for f in sorted(files):
                if f.endswith('.tsv'):
                    yield os.path.join(dir, f)


    def read_tsv_file(self, fname):
        with open(fname) as f:
            reader = csv.DictReader(f, dialect='excel-tab')
            for row in reader:
                self.by_fen[row['epd']] = row


    def lookup(self, board, transpose=True):
        '''
        Lookup by board position (FEN)
        '''
        row = self.by_fen.get(board.epd(), None)
        if row is None and board._stack:
            prev = chess.Board()
            board._stack[-1].restore(prev)
            row = self.by_fen.get(prev.epd(), None)

        if row and not transpose:
            pgn = chess.pgn.read_game(io.StringIO(row['pgn']))
            for i, move in enumerate(pgn.mainline_moves()):
                if i >= len(board.move_stack) or move != board.move_stack[i]:
                    return None
        return row


    def openings(self):
        for _, v in self.by_fen.items():
            yield v
