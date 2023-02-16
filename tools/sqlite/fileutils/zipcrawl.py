"""
Extend the FileCrawler with the ability to look inside ZIP files.
ZIPs are expanded into temporary directories which are automatically deleted.
"""
import shutil
import tempfile
from functools import partial
from zipfile import ZipFile

from .filecrawl import FileCrawler, walk_directory


class ZipCrawler(FileCrawler):
    """ A FileCrawler that also digs into zip files """
    def __init__(self, paths, default_callback=lambda _: None):
        super().__init__(paths, default_callback)
        self.set_action('.zip', partial(self._extract_from_zip, default_callback))
        self._tmpdirs = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for tmp in self._tmpdirs:
            # print(f'cleaning up: {tmp}')
            shutil.rmtree(tmp)

    def _extract_from_zip(self, default_callback, fname):
        dir = tempfile.mkdtemp()
        self._tmpdirs.append(dir)
        with ZipFile(fname, 'r') as archive:
            print(f'Expanding {fname}\033[K')
            archive.extractall(dir)
            if walk_directory(dir, self._callbacks, default_callback):
                return True

