"""Minimal local stand-in for mmcv.utils.

coperception's mean_ap.py only uses mmcv for print_log(), always called
with logger=None in this project's eval path -- mmcv's own documented
behavior for logger=None is to just print to stdout, which this replicates
without pulling in the full mmcv package (version-pinned to exact
torch/CUDA builds that don't exist yet for this machine's RTX 5060 Ti /
torch 2.11+cu128 combo).
"""


def print_log(msg, logger=None, level=None):
    if logger is None or logger == "silent":
        print(msg)
    else:
        print(msg)
