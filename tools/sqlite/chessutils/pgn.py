import chess, chess.pgn
import os
import re
from .eco import EcoAPI
from functools import partial

ecoAPI = EcoAPI()

RESULT = [WIN, LOSS, DRAW] = [1, 0, 1/2]

# normalize result encodings
_result = {
    '1/2': [ DRAW, DRAW  ],
'1/2-1/2': [ DRAW, DRAW  ],
'1/2 1/2': [ DRAW, DRAW  ],
    '1-0': [ LOSS, WIN  ],
    '0-1': [ WIN,  LOSS ],
    '1-O': [ LOSS, WIN  ],
    'O-1': [ WIN,  LOSS ],
    '+/-': [ LOSS, WIN  ],
    '-/+': [ WIN,  LOSS ],
    '(+)-(-)': [ LOSS, WIN  ],
    '(-)-(+)': [ WIN,  LOSS ],
}


class Visitor(chess.pgn.GameBuilder):
    __unique_games = set()

    def __init__(self, on_meta, on_move, on_eval, *args, **kwds):
        self.on_meta = on_meta
        self.on_move = on_move
        self.on_eval = on_eval
        self.meta = None
        self.valid = True
        self.unique = kwds.pop('unique', True)
        super().__init__(*args, **kwds)

    def begin_game(self):
        self.valid = True
        self.duplicate = False
        self.filtered = False
        self.board = None
        return super().begin_game()

    def end_headers(self):
        super().end_headers()

        self.reshdr = self.game.headers['Result']
        if not self.reshdr in _result:
            self.filtered = True
            return chess.pgn.SKIP

        self.meta = game_metadata(self.game)

        # de-dupe by header info (Event, players, etc.)
        if self.unique:
            k = str(self.meta)
            if k in Visitor.__unique_games:
                self.duplicate = True
                return chess.pgn.SKIP
            elif not self.on_meta(self.game, self.meta):
                self.filtered = True
                return chess.pgn.SKIP
            Visitor.__unique_games.add(k)

    def visit_board(self, board):
        assert not self.duplicate
        assert not self.filtered
        self.board = board
        self.valid = not self.game.errors and bool(board.move_stack) # and board.is_valid()
        if self.valid:
            # Make sure the winner was recorded correctly
            reshdr = self.reshdr
            if board.is_checkmate():
                if _result[reshdr][board.turn] != LOSS:
                    self.valid = False
                if _result[reshdr][not board.turn] != WIN:
                    self.valid = False
            elif board.is_stalemate():
                self.valid = _result[reshdr][board.turn]==DRAW and _result[reshdr][not board.turn]==DRAW

        return super().visit_board(board)

    def visit_move(self, board, move):
        nag = self.variation_stack[-1]
        if nag and nag.move:
            if eval := nag.eval():
                self.on_eval(self.meta, board, nag.move, eval)
        self.on_move(self.meta, board, move)
        return super().visit_move(board, move)

    def result(self):
        if self.game:
            self.game.duplicate = self.duplicate
            self.game.filtered = self.filtered
            self.game.meta = self.meta
            self.game.valid = self.valid and self.board and self.board.is_valid()
        return super().result()


class GameFileReader:
    def __init__(self, stream, on_meta=None, on_move=None, on_eval=None, unique=True):
        self.stream = stream
        self.on_meta = on_meta or (lambda *_: True)
        self.on_move = on_move or (lambda *_: None)
        self.on_eval = on_eval or (lambda *_: None)
        self.unique = unique
        self.errors = None

    def __iter__(self):
        return self.games()

    def games(self):
        while True:
            game = chess.pgn.read_game(self.stream, Visitor=partial(Visitor, self.on_meta, self.on_move, self.on_eval, unique=self.unique))
            if not game:
                break
            if game.errors:
                self.errors = game.errors
                continue
            if not game.valid:
                continue
            if game.duplicate or game.filtered:
                assert not list(game.mainline_moves())
                continue
            yield game


def read_games(f, on_meta=None, on_move=None, on_eval=None, unique=True):
    return GameFileReader(f, on_meta, on_move, on_eval, unique)


def __cleanup(info):
    event = info['event']['name']
    match = re.match('(.*)Round: (.*)', event)
    if match:
        info['event']['name'] = match.group(1).strip()
        info['event']['round'] = match.group(2).strip()


def __normalize_name(name):
    if not name:
        return ('', '')

    def capitalize(n):
        return ' '.join([t.strip().capitalize() for t in n]).strip()

    tok = name.split(',')
    FIRST, LAST = -1, 0
    if len(tok)==1:
        tok = name.split()
        FIRST, LAST = LAST, FIRST
    first = tok[FIRST]
    last = tok[LAST]
    middle = tok[1:-1] if len(tok)>2 else []
    return capitalize(last.split()), capitalize(first.split() + middle)


def game_metadata(game):
    headers = game.headers
    info = { 'white': {}, 'black': {}, 'event': {} }
    result = _result[headers['Result']]
    info['white']['name'] = __normalize_name(headers.get('White', None))
    info['black']['name'] = __normalize_name(headers.get('Black', None))
    info['white']['result'] = result[chess.WHITE]
    info['black']['result'] = result[chess.BLACK]
    info['event']['name'] = headers.get('Event', None)
    info['event']['round'] = headers.get('Round', None)
    __cleanup(info)
    return info


def game_moves(game):
    return ' '.join([move.uci() for move in game.mainline_moves()]).strip()


def game_opening(game):
    opening = None
    board = game.board()
    # go through the moves and apply them to the board
    for move in game.mainline_moves():
        board.push(move)
        # lookup the opening that matches the current board configuration
        entry = ecoAPI.lookup(board)
        if entry:
            opening = entry
    # return the last match, if any
    return opening
