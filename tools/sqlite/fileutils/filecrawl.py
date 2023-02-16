import os


""" Lookup callback by file extension in callbacks dictionary and invoke it """
def file_ext_callback(root, fname, callbacks, default_callback):
    ext = os.path.splitext(fname.lower())[1]
    return callbacks.get(ext, default_callback)(os.path.join(root, fname))


def walk_directory(dir, callbacks, default_callback):
    """
    Walk a directory tree.
    
    For each non-directory entry encountered: 
    lookup callback by file extension and invoke it.
    
    Stop when callback returns True.
    """
    for root, dirs, files in os.walk(dir):
        for f in sorted(files):
            if file_ext_callback(root, f, callbacks, default_callback):
                return True


class FileCrawler:
    def __init__(self, paths, default_callback=lambda _: None):
        self._paths = paths
        self._default_callback = default_callback
        self._callbacks = {}

    def crawl(self):
        for path in self._paths:
            if os.path.isdir(path):
                if walk_directory(path, self._callbacks, self._default_callback):
                    return True
            elif file_ext_callback('.', path, self._callbacks, self._default_callback):
                return True

    def set_action(self, file_extension, action):
        self._callbacks[file_extension.lower()]=action


"""
Example:
    print the name of all Python source code files in the current directory tree
"""
if __name__ == '__main__':
    dir_crawler = FileCrawler('.')
    dir_crawler.set_action('.py', lambda f: print(f))
    dir_crawler.crawl()
