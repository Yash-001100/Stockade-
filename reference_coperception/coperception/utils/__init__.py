from . import *
from .AverageMeter import AverageMeter

try:
    from .mapping import mapping
except ImportError:
    # The bundled `mapping` extension is a Python-3.7-only compiled module
    # (cp37-win_amd64.pyd / cpython-37m-x86_64-linux-gnu.so) and won't load
    # on newer interpreters. It's only used by tools/det/create_data_det.py
    # (raw LiDAR -> preprocessed .npy), not by training/eval on data that's
    # already preprocessed, so it's safe to skip here.
    mapping = None