import csv
import os
import re


class EcoAPI:
    """
        API for looking up moves in The Encyclopaedia of Chess Openings
        (see https://en.wikipedia.org/wiki/Encyclopaedia_of_Chess_Openings)

        Powered by: https://github.com/niklasf/eco
    """
    def __init__(self, data_dir=None):
        """ read tab-separated-value files and store them in db dictionary by FEN """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'eco')
        self.db = {}

        for f in self._tsv_files():
            self._read_tsv_file(f)

    @staticmethod
    def __normalize_opening_name(entry):
        """
        Use a regular expression to separate the main name of
        the opening from its variation, if it's in a format like:

        Ruy Lopez: Morphy Variation
        """
        name = entry['name']
        match = re.match('(.*): (.*)', name)
        if match:
            entry['name'] = match.group(1)
            entry['variation'] = match.group(2)
        return entry

    def _tsv_files(self):
        for dir, _subdirs, files in os.walk(self.data_dir):
            for f in sorted(files):
                if f.endswith('.tsv'):
                    yield os.path.join(dir, f)

    def _read_tsv_file(self, fname):
        with open(fname) as f:
            reader = csv.DictReader(f, dialect='excel-tab')
            for row in reader:
                self.db[row['fen']] = EcoAPI.__normalize_opening_name(row)

    def lookup(self, board):
        row = self.db.get(board.epd(), None)
        if row:
            for i, move in enumerate(row['moves'].split(' ')):
                if i > len(board.move_stack) or move != board.move_stack[i].uci():
                    return None
            return row
