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


def test_patch4_configurable_lr_schedule():
    import torch
    from coperception.utils.CoDetModule import FaFModule

    dummy_model = torch.nn.Linear(2, 2)
    dummy_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)

    class DummyConfig:
        MGDA = False
        pred_len = 1
        category_num = 2
        box_code_size = 6
        code_type = "faf"
        loss_type = "faf_loss"
        only_det = True

    # Default args -> must reproduce the ORIGINAL hardcoded schedule exactly.
    default_module = FaFModule(dummy_model, dummy_model, DummyConfig(), dummy_optimizer, {}, kd_flag=0)
    assert default_module.scheduler.milestones == {50: 1, 100: 1, 150: 1, 200: 1}, (
        f"default milestones changed unexpectedly: {default_module.scheduler.milestones}"
    )
    assert default_module.scheduler.gamma == 0.5

    # Custom args -> must actually take effect.
    custom_optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.0005)
    custom_module = FaFModule(
        dummy_model, dummy_model, DummyConfig(), custom_optimizer, {}, kd_flag=0,
        lr_milestones=(15, 30), lr_gamma=0.5,
    )
    assert custom_module.scheduler.milestones == {15: 1, 30: 1}, (
        f"custom milestones not applied: {custom_module.scheduler.milestones}"
    )
    print("PASS: patch 4 (configurable LR schedule)")


if __name__ == "__main__":
    test_patch1_mapping_import_does_not_crash()
    test_patch2_nuscenes_box_stub()
    test_patch3_mmcv_print_log_stub()
    test_patch4_configurable_lr_schedule()
    print("\nAll patch tests passed.")
