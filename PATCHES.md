# Local patches to `reference_coperception`

`reference_coperception/` is a shallow clone of the upstream `coperception`
SDK (https://github.com/coperception/coperception), used as-is for training
and evaluating the V2X-Sim detection baselines. It's a 2020-2022-era
codebase built for Python 3.7 on Linux; three local patches were needed to
run it on this machine (Python 3.11, Windows, CUDA 12.8). This file
documents each one so they survive a `coperception` version bump or a
from-scratch environment rebuild. A test proving each patch still works
lives in `tests/test_patches.py`.

## Patch 1 — skip the Python-3.7-only `mapping` extension

**File:** `reference_coperception/coperception/utils/__init__.py`

**Original behavior:** unconditional `from .mapping import mapping`. The
package bundles a compiled extension (`mapping.cp37-win_amd64.pyd` /
`mapping.cpython-37m-x86_64-linux-gnu.so`) built only for CPython 3.7's
ABI; importing it on Python 3.11 raises `ImportError: DLL load failed`.
Since that import happens at package-init time, it broke importing
*anything* from `coperception`, not just whatever used `mapping`.

**Why safe to skip:** `mapping` is only referenced in one place outside
its own definition — `tools/det/create_data_det.py`, which converts raw
LiDAR point clouds into the preprocessed `.npy` sparse-voxel format. We
never run that script; `data/V2X-Sim-det-001` was downloaded already
preprocessed. Nothing in the training (`train_codet.py`) or evaluation
(`test_codet.py`) path calls `mapping`.

**Patch:** wrapped the import in `try/except ImportError`, falling back to
`mapping = None`.

**If you ever need `create_data_det.py`:** this patch does NOT fix that
script — it still needs the real `mapping` extension, which means either
rebuilding it for your Python version from
`coperception/utils/mapping/dynamic_library/`, or running that one script
under a Python 3.7 environment.

## Patch 2 — local `nuscenes.utils.data_classes.Box` stub

**Files:** `reference_coperception/nuscenes/__init__.py`,
`nuscenes/utils/__init__.py`, `nuscenes/utils/data_classes.py` (new local
package, shadows the real `nuscenes-devkit` via `sys.path` order).

**Original behavior:** `coperception/utils/obj_util.py` and
`coperception/utils/data_util.py` do
`from nuscenes.utils.data_classes import Box`. The real `nuscenes-devkit`
(pinned to `==1.0.9`) imports its full `nuscenes.py` module on `import
nuscenes`, which pulls in `pandas`, `motmetrics`, `fire`, `jupyter`,
`descartes`, and a `Pillow<=6.2.1` pin — none of which are needed for the
one thing actually used (`Box(center, size, orientation).corners()`), and
some of which don't have working wheels against this machine's
numpy/Pillow versions.

**Why safe to replace:** grepped every use of `Box` across the files
`coperception`'s training/eval path actually imports — only the
constructor and `.corners()` method are called (in
`obj_util.py::get_gt_corners`). The stub reimplements exactly those,
copied faithfully from `nuscenes-devkit`'s source.

**Patch:** a local `nuscenes` package under `reference_coperception/`
(same directory that's prepended to `sys.path`/`PYTHONPATH` when running
anything), so Python resolves `import nuscenes` to the stub instead of the
real pip package.

**Caveat:** this stub does NOT implement the rest of `nuscenes-devkit`
(no `NuScenes`/`NuScenesExplorer`, no point-cloud classes, no map tools).
If a future change needs any of that, this patch needs extending, not
just reverting.

## Patch 3 — local `mmcv.utils.print_log` stub

**Files:** `reference_coperception/mmcv/__init__.py`, `mmcv/utils.py` (new
local package, same shadowing approach as patch 2).

**Original behavior:** `coperception/utils/mean_ap.py` (used by
`test_codet.py`'s mAP evaluation) does
`from mmcv.utils import print_log`. Real `mmcv` is tightly version-pinned
to specific torch/CUDA builds and has no prebuilt wheel for this
machine's torch 2.11+cu128 combination — installing it would mean
building from source against a very new, likely untested CUDA version.

**Why safe to replace:** `print_log` is called with `logger=None` in
every place `coperception` uses it (see
`mean_ap.py::eval_map` callers) — mmcv's own documented behavior for
`logger=None` is just `print(msg)`.

**Patch:** a local `mmcv` package reimplementing only `print_log` with
that same `logger=None` → `print()` behavior.

**Caveat:** no other mmcv functionality is stubbed. This only works
because the current eval path's only mmcv dependency is this one
function.
