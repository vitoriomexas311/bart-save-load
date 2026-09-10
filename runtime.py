"""Keep compiled PyTensor extensions from different test stacks apart."""

import importlib.metadata as md
import os
from pathlib import Path
import sys


def configure_compiler_cache():
    """Choose a version-specific cache BEFORE importing PyMC or PyTensor.

    Virtual environments do not automatically isolate PyTensor's shared user
    cache. Running the PyMC 5 and 6 examples concurrently exposed a lazylinker
    version mismatch. Give each PyTensor/Python version its own directory.

    Explicit user cache settings take precedence. Other PYTENSOR_FLAGS options
    are preserved. PyTensor itself creates the directory when it needs it.
    """
    flags = os.environ.get('PYTENSOR_FLAGS', '')
    keys = {part.partition('=')[0].strip() for part in flags.split(',')}
    if {'compiledir', 'base_compiledir'} & keys:
        return
    version = md.version('pytensor')
    interpreter = f'{sys.version_info.major}.{sys.version_info.minor}'
    cache = Path(__file__).resolve().parent / '.pytensor-cache' / f'{version}-py{interpreter}'
    setting = f'base_compiledir={cache.as_posix()}'
    os.environ['PYTENSOR_FLAGS'] = f'{flags},{setting}' if flags else setting
