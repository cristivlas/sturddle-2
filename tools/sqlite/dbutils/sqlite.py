import sqlite3

class SQLConn:
    """ Sqlite3 Connection Wrapper """
    def __init__(self, db_file):
        self._conn = sqlite3.connect(db_file)
        self._cursor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._conn:
            if self._cursor:
                self._conn.commit()
            self._conn.close()

    def commit(self):
        assert self._cursor
        self._conn.commit()
        rowid = self._cursor.lastrowid
        self._cursor = None
        return rowid

    def exec(self, *args, echo=False):
        if echo:
            print(*args)
        self._cursor = self._conn.cursor()
        return self._cursor.execute(*args)

    def row_count(self, table):
        return self.exec(f'SELECT COUNT(_ROWID_) FROM {table}').fetchone()[0]

    def row_max_count(self, table):
        return self.exec(f'SELECT MAX(_ROWID_) FROM {table}').fetchone()[0]
