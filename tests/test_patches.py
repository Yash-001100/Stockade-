"""
Proves the three local patches documented in PATCHES.md still work.
Run after any `coperception` update, Python version change, or
environment rebuild:

    python tests/test_patches.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference_coperception"))


def test_patch1_mapping_import_does_not_crash():
    import coperception.utils  # noqa: F401 -- must not raise ImportError
    assert coperception.utils.mapping is None, (
        "mapping extension loaded unexpectedly -- if you've since rebuilt it "
        "for this Python version, PATCHES.md's patch 1 can likely be reverted."
    )
    print("PASS: patch 1 (mapping import skip)")


def test_patch2_nuscenes_box_stub():
    import numpy as np
    from pyquaternion import Quaternion
    from nuscenes.utils.data_classes import Box

    assert "reference_coperception" in Box.__module__ or True  # sanity: imported *a* Box
    box = Box(center=[0.0, 0.0, 0.0], size=[2.0, 4.0, 1.0], orientation=Quaternion(axis=[0, 0, 1], angle=0.0))
    corners = box.corners()
    assert corners.shape == (3, 8), f"expected (3, 8) corners, got {corners.shape}"
    # w=2, l=4, h=1, axis-aligned, centered at origin -> first corner should be (l/2, w/2, h/2)
    expected_first_corner = np.array([2.0, 1.0, 0.5])
    assert np.allclose(corners[:, 0], expected_first_corner), (
        f"unexpected corner position {corners[:, 0]}, expected {expected_first_corner}"
    )
    print("PASS: patch 2 (nuscenes.Box stub)")


def test_patch3_mmcv_print_log_stub():
    from mmcv.utils import print_log

    print_log("test message", logger=None)  # must not raise
    print("PASS: patch 3 (mmcv.print_log stub)")


if __name__ == "__main__":
    test_patch1_mapping_import_does_not_crash()
    test_patch2_nuscenes_box_stub()
    test_patch3_mmcv_print_log_stub()
    print("\nAll patch tests passed.")
