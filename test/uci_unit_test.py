#! /usr/bin/env python3
import argparse
import asyncio
import logging
import sys
import time

import chess
import chess.engine


class EngineTestScope:
    def __init__(self, args):
        self.args = args
        self.start_time = None
        self.end_time = None
        # On Windows, CreateProcess won't launch a .py script directly; prepend
        # the active Python interpreter so the script runs as expected.
        if args.engine.endswith('.py'):
            engine_command_line = [sys.executable, args.engine]
        else:
            engine_command_line = [args.engine]
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_command_line)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.end_time = time.perf_counter()
        self.engine.quit()

    @property
    def elapsed(self):
        return self.end_time - self.start_time


'''
Test the UCI "position" command.
'''
def test_position(test):
    for _ in range(args.iterations):
        for pos in [
            'r1bqkbnr/p1pp1ppp/1pn5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4',
            '8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - -',
            '8/8/4R3/2r3pk/6Pp/7P/1PPB1P2/1K1R4 b - g3',
            'r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq -',
            'r1bqkb1r/ppp2ppp/2n2n2/4p3/4p3/N2PB3/PPPQ1PPP/R3KBNR w KQkq -',
            'r1bqk2r/ppp2ppp/2nb1n2/4p3/4P3/N3BP2/PPPQ2PP/R3KBNR b KQkq -',
            'r3kb1r/pppbqppp/5n2/3Pp3/1nP5/N2PBN2/PP1Q1PPP/R3KB1R b KQkq -',
            'r3kb1r/pppbqppp/5n2/3Pp3/1nP5/N2PBN2/PP1Q1PPP/R3KB1R b KQkq -',
            'r3kb1r/pppbqppp/5n2/3Pp3/1nP5/N2PBN2/PP1Q1PPP/R3KB1R b KQk -',
            'r3kb1r/pppbqppp/2N2n2/3Pp3/1nP5/3PBN2/PP1Q1PPP/R3KB1R b KQkq -',
            'r3kb1r/pppbqppp/5N2/3Pp3/1nP5/3PBN2/PP1Q1PPP/R3KB1R b KQkq -',
            'r3k2r/pppbqppp/5n2/3Pp3/1nP5/N2PBN2/PP1Q1PPP/R3K2R b KQkq -',
            'r1bqk2r/ppp2p1p/5B2/4b3/4p3/N2P4/P1PQ1PPP/R3KBNR w KQkq -',
            'r1bqk2r/ppp2p1p/5B2/4b3/4p3/N2P4/P1PQ1PPP/R3KBNR b KQkq -',
        ]:
            board = chess.Board(pos)
            test.engine.protocol._position(board)
            assert board == test.engine.protocol.board


def extract_depth(info_line):
    # Split the line by space to get a list of words
    words = info_line.split()
    try:
        # Find the index of the word "depth"
        depth_index = words.index('depth')
    except:
        return 0
    # The depth value is the word at the index after "depth", so we return the word at depth_index + 1
    return int(words[depth_index + 1])


'''
Send a "go" command to the engine, optionally setting up the position.
'''
class GoCommand(chess.engine.BaseCommand[chess.engine.UciProtocol, str]):
    depth = 0

    def __init__(self, engine, **kwargs):
        super().__init__(engine)
        self.pos = kwargs.pop('fen', None)
        self.time = kwargs.pop('movetime', 0)
        self.go_depth = kwargs.pop('depth', 0)
        self.moves = kwargs.pop('moves', [])
        self.searchmoves = kwargs.pop('searchmoves', [])
        # Raw UCI lines to send before the "position"/"go" sequence; used by
        # leak tests to inject "ucinewgame", "go perft", etc.
        self.pre_commands = kwargs.pop('pre_commands', [])
        # Override the assembled "go ..." line entirely. Used by terminator
        # tests to send specific token orderings.
        self.raw_go = kwargs.pop('raw_go', None)

    def start(self, engine):
        for line in self.pre_commands:
            engine.send_line(line)
        if self.pos:
            if self.moves:
                engine.send_line(f'position fen {self.pos} moves {" ".join(self.moves)}')
            else:
                engine.send_line(f'position fen {self.pos}')
        # Do not use the opening book
        if 'stockfish' not in args.engine:
            engine.send_line('setoption name OwnBook value false')
        engine.send_line('setoption name Ponder value false')

        if self.raw_go:
            engine.send_line(self.raw_go)
            return
        if self.go_depth:
            cmd = f'go depth {self.go_depth}'
        else:
            cmd = f'go movetime {self.time}'
        if self.searchmoves:
            cmd += f' searchmoves {" ".join(self.searchmoves)}'
        engine.send_line(cmd)

    def line_received(self, engine, line):
        if line.startswith('info '):
            if args.verbose:
                print(line)
            GoCommand.depth = extract_depth(line)
        elif line.startswith('bestmove'):
            self.result.set_result(line)
            self.set_finished()
        # ignore everything else (perft output, debug messages, etc.)

'''
Test the UCI "go" command.
'''
def test_go(test):
    for pos, moves in [
        ('8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - -', []),
        ('8/8/4R3/2r3pk/6Pp/7P/1PPB1P2/1K1R4 b - g3', []),
        ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -',[]),
        ('r1bqkbnr/1ppn1ppp/4p3/p3N3/3Pp3/2P5/PP1N1PPP/R1BQKB1R b KQkq -', ['d7e5', 'd4e5', 'e4e3', 'f2f3', 'e3d2', 'c1d2', 'c8d7']),
    ]:
        def _go(engine):
            # Use depth limit, not movetime, so the Python build doesn't blow
            # its main-thread stack on endgame positions that iterate deeply.
            return GoCommand(engine, fen=pos, moves=moves, depth=12)

        response = test.engine.communicate(_go)
        print (f'depth {GoCommand.depth} {response}')
        assert response.startswith('bestmove '), response


'''
Test the UCI "go searchmoves" directive.
Uses "go depth N" rather than "go movetime" to keep the search bounded
and avoid the Python-build stack overflow on Windows at deeper iterations.
'''
def test_searchmoves(test):
    startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'

    # Restrict root to two unusual moves; bestmove must be one of them.
    restricted = ['a2a3', 'h2h3']
    def _go_restricted(engine):
        return GoCommand(engine, fen=startpos, depth=8, searchmoves=restricted)
    response = test.engine.communicate(_go_restricted)
    assert response.startswith('bestmove '), response
    move = response.split()[1]
    assert move in restricted, f'expected one of {restricted}, got {move}'
    print(f'searchmoves restricted ........ bestmove {move} (ok)')

    # Follow-up unrestricted go: filter must reset; engine should pick a
    # mainstream first move, not one of the restricted ones.
    def _go_unrestricted(engine):
        return GoCommand(engine, fen=startpos, depth=8)
    response = test.engine.communicate(_go_unrestricted)
    assert response.startswith('bestmove '), response
    move = response.split()[1]
    assert move not in restricted, \
        f'filter leaked from previous go: bestmove {move} still in {restricted}'
    print(f'searchmoves cleared ........... bestmove {move} (ok)')

    # Single-move filter: bestmove must be exactly that move.
    only = ['g1f3']
    def _go_only(engine):
        return GoCommand(engine, fen=startpos, depth=8, searchmoves=only)
    response = test.engine.communicate(_go_only)
    assert response.startswith('bestmove '), response
    move = response.split()[1]
    assert move == only[0], f'expected {only[0]}, got {move}'
    print(f'searchmoves single move ....... bestmove {move} (ok)')


'''
Verify shape-based termination of the searchmoves list correctly handles
"searchmoves" appearing anywhere in the "go" command, with various UCI
keywords as terminators.
'''
def test_searchmoves_terminators(test):
    startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'
    restricted = ['a2a3', 'h2h3']

    def run_raw(scenario, raw_go):
        def _go(engine):
            return GoCommand(engine, fen=startpos, raw_go=raw_go)
        response = test.engine.communicate(_go)
        assert response.startswith('bestmove '), response
        move = response.split()[1]
        assert move in restricted, \
            f'{scenario}: expected one of {restricted}, got {move} (raw: {raw_go!r})'
        print(f'terminator {scenario:.<22s} bestmove {move} (ok)')

    # 1) searchmoves last (no terminator) — runs to end of line.
    run_raw('end-of-line',
            f'go depth 8 searchmoves {" ".join(restricted)}')

    # 2) searchmoves followed by "depth" keyword.
    run_raw('depth after',
            f'go searchmoves {" ".join(restricted)} depth 8')

    # 3) searchmoves followed by "wtime"/"btime". Include "depth" to cap the
    #    search — wtime/btime alone would let it run to a depth that blows
    #    the Python build's main-thread stack.
    run_raw('wtime after',
            f'go searchmoves {" ".join(restricted)} wtime 60000 btime 60000 depth 8')

    # 4) searchmoves between two other params.
    run_raw('mid-command',
            f'go wtime 60000 searchmoves {" ".join(restricted)} btime 60000 depth 8')

    # 5) searchmoves with zero moves (immediate non-move terminator): no
    #    restriction installed, depth still honored.
    def _go_empty(engine):
        return GoCommand(engine, fen=startpos,
                         raw_go='go searchmoves depth 6')
    response = test.engine.communicate(_go_empty)
    assert response.startswith('bestmove '), response
    move = response.split()[1]
    # With zero moves in filter, any legal move is OK; verify the move is
    # legal and the search actually ran (didn't crash or return null move).
    board = chess.Board(fen=startpos)
    assert chess.Move.from_uci(move) in board.legal_moves, \
        f'empty searchmoves yielded illegal move {move}'
    print(f'terminator {"empty list":.<22s} bestmove {move} (ok)')


'''
Verify the "searchmoves" filter cannot leak across commands. A leaking filter
would silently corrupt subsequent searches, so this is critical to check.

Each scenario: install a filter with two unusual moves, run an intervening
UCI command, then issue a plain "go" and assert the bestmove is NOT one of
the restricted moves (the engine should now consider the full move list).
'''
def test_searchmoves_no_leak(test):
    startpos = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'
    restricted = ['a2a3', 'h2h3']

    def install_filter():
        def _go(engine):
            return GoCommand(engine, fen=startpos, depth=6, searchmoves=restricted)
        response = test.engine.communicate(_go)
        assert response.startswith('bestmove '), response
        move = response.split()[1]
        assert move in restricted, f'install failed: {move} not in {restricted}'

    def assert_no_leak(scenario, pre_commands):
        def _go(engine):
            return GoCommand(engine, fen=startpos, depth=8, pre_commands=pre_commands)
        response = test.engine.communicate(_go)
        assert response.startswith('bestmove '), response
        move = response.split()[1]
        assert move not in restricted, \
            f'{scenario}: filter leaked, bestmove {move} still in {restricted}'
        print(f'no leak after {scenario:.<22s} bestmove {move} (ok)')

    # 1) Plain follow-up go (baseline; already covered by test_searchmoves,
    #    repeated here for completeness).
    install_filter()
    assert_no_leak('plain go', [])

    # 2) "go perft" between filter-go and next go. Perft has its own early
    #    return path; make sure that path doesn't leave the filter installed.
    install_filter()
    assert_no_leak('go perft', ['go perft 3'])

    # 3) "ucinewgame" between. A fresh game must start with no restriction.
    install_filter()
    assert_no_leak('ucinewgame', ['ucinewgame'])

    # 4) "stop" between. A no-op stop (no search running) shouldn't leak.
    install_filter()
    assert_no_leak('stop', ['stop'])

    # 5) Filter replaced by a different filter, then unrestricted go. The
    #    second go's filter must fully replace the first, then clear cleanly.
    install_filter()
    def _go_replace(engine):
        return GoCommand(engine, fen=startpos, depth=6, searchmoves=['g1f3'])
    response = test.engine.communicate(_go_replace)
    assert response.startswith('bestmove g1f3'), \
        f'replacement filter failed: {response}'
    assert_no_leak('filter replace', [])


'''
Half-move (fifty-move) clock tests.

The clock is read via the "debug" command, which prints a "halfmove clock: N"
line. If the engine does not report it, the probe returns None and the affected
tests are skipped.
'''
class HalfmoveProbe(chess.engine.BaseCommand[chess.engine.UciProtocol, int]):
    def __init__(self, engine, *, setup_lines):
        super().__init__(engine)
        self.setup_lines = setup_lines
        self.halfmove = None

    def start(self, engine):
        if 'stockfish' not in args.engine:
            engine.send_line('setoption name OwnBook value false')
        for line in self.setup_lines:
            engine.send_line(line)
        engine.send_line('debug')
        engine.send_line('isready')  # readyok is the last line; it ends the probe

    def line_received(self, engine, line):
        s = line.strip()
        low = s.lower()
        if low.startswith('halfmove clock:'):
            try:
                self.halfmove = int(s.split(':', 1)[1])
            except ValueError:
                pass
        elif low == 'readyok':
            self.result.set_result(self.halfmove)
            self.set_finished()


def probe_halfmove(test, setup_lines):
    def _cmd(engine):
        return HalfmoveProbe(engine, setup_lines=setup_lines)
    return test.engine.communicate(_cmd)


def test_halfmove_startpos_zero(test):
    hm = probe_halfmove(test, ['position startpos'])
    if hm is None:
        print('halfmove startpos ............. skipped (no debug clock on this build)')
        return
    assert hm == 0, f'expected halfmove 0 at startpos, got {hm}'
    print('halfmove startpos ............. clock 0 (ok)')


def test_fen_halfmove_parsed(test):
    fen = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 49 13'
    hm = probe_halfmove(test, [f'position fen {fen}'])
    if hm is None:
        print('fen halfmove parsed ........... skipped')
        return
    assert hm == 49, f'FEN half-move field ignored: expected 49, got {hm}'
    print('fen halfmove parsed ........... clock 49 (ok)')


def test_fen_halfmove_plus_moves(test):
    fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 10 6'
    hm = probe_halfmove(test, [f'position fen {fen} moves g1f3 g8f6'])
    if hm is None:
        print('fen halfmove + moves .......... skipped')
        return
    assert hm == 12, f'expected 10 + 2 = 12, got {hm}'
    print('fen halfmove + moves .......... clock 12 (ok)')


def test_halfmove_resets_on_new_position(test):
    hm1 = probe_halfmove(test, ['position startpos moves g1f3 g8f6 f3g1 f6g8'])
    if hm1 is None:
        print('halfmove reset (position) ..... skipped')
        return
    assert hm1 == 4, f'expected clock 4 after knight shuffle, got {hm1}'
    hm2 = probe_halfmove(test, ['position startpos'])
    assert hm2 == 0, f'clock leaked across position commands: expected 0, got {hm2}'
    print('halfmove reset (position) ..... clock 0 (ok)')


def test_halfmove_resets_on_ucinewgame(test):
    hm1 = probe_halfmove(test, ['position startpos moves g1f3 g8f6 f3g1 f6g8'])
    if hm1 is None:
        print('halfmove reset (ucinewgame) ... skipped')
        return
    assert hm1 == 4, f'expected clock 4 after knight shuffle, got {hm1}'
    hm2 = probe_halfmove(test, ['ucinewgame', 'position startpos'])
    assert hm2 == 0, f'clock leaked across ucinewgame: expected 0, got {hm2}'
    print('halfmove reset (ucinewgame) ... clock 0 (ok)')


def test_halfmove_fen_overrides_leftover(test):
    hm1 = probe_halfmove(test, ['position startpos moves g1f3 g8f6 f3g1 f6g8'])
    if hm1 is None:
        print('halfmove fen overrides ........ skipped')
        return
    assert hm1 == 4, f'expected clock 4 after knight shuffle, got {hm1}'

    # FEN with a half-move field: its value must win over the leftover clock.
    fen5 = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 49 13'
    hm2 = probe_halfmove(test, [f'position fen {fen5}'])
    assert hm2 == 49, f'FEN clock did not override leftover: expected 49, got {hm2}'

    # Re-dirty, then a FEN without a half-move field must reset to 0.
    probe_halfmove(test, ['position startpos moves g1f3 g8f6 f3g1 f6g8'])
    fen4 = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -'
    hm3 = probe_halfmove(test, [f'position fen {fen4}'])
    assert hm3 == 0, f'FEN path leaked leftover: expected 0, got {hm3}'
    print('halfmove fen overrides ........ clock 49 then 0 (ok)')


'''
Run the engine on the Kiwipete position — a tactically dense test board
widely used in engine development. Verifies the engine completes a search
without crashing and returns a legal move; exercises move ordering and
the singular-extension path.
'''
def test_kiwipete(test):
    fen = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
    def _go(engine):
        return GoCommand(engine, fen=fen, depth=12)
    response = test.engine.communicate(_go)
    assert response.startswith('bestmove '), response
    move = response.split()[1]
    board = chess.Board(fen=fen)
    assert chess.Move.from_uci(move) in board.legal_moves, \
        f'engine returned illegal move {move}'
    print(f'kiwipete ...................... bestmove {move} (depth 12, ok)')


def test_tricky(test):
    tests = [
        '3K4/3P2k1/8/8/8/8/2r5/5R2 w - -',  # Lucena
    ]
    for fen in tests:
        board = chess.Board(fen=fen)
        # Use depth limit rather than time so the Python build doesn't blow
        # its 1 MB main-thread stack on this endgame study (engine needs 32 MB
        # for deep ID; only the native binary links with /STACK:33554432).
        info = test.engine.analyse(board, chess.engine.Limit(depth=15))
        print(f'{info["score"]}, depth={info["depth"]}')

def run_tests(args):
    for test in [
        test_position,
        test_halfmove_startpos_zero,
        test_fen_halfmove_parsed,
        test_fen_halfmove_plus_moves,
        test_halfmove_resets_on_new_position,
        test_halfmove_resets_on_ucinewgame,
        test_halfmove_fen_overrides_leftover,
        test_tricky,
        test_go,
        test_searchmoves,
        test_searchmoves_terminators,
        test_searchmoves_no_leak,
        test_kiwipete,
    ]:
        with EngineTestScope(args) as test_scope:
            test(test_scope)
        print(f'{test.__name__:.<25s} Elapsed time: {test_scope.elapsed:.4f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', default='./main.py')
    parser.add_argument('-i', '--iterations', type=int, default=1, help='number of iterations to run')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose logging')

    args = parser.parse_args()
    run_tests(args)
