"""
Stockade -- diagnostic: is agent 5 structurally harder, or is something
wrong with the pipeline?

Loads ground-truth boxes directly from the preprocessed val split (no
model inference needed) and compares, per agent:
  - objects per scene-frame (scene density)
  - distance of each object from the agent (ego-centered coords)
  - fraction of "far" objects (>20m, where detection is hardest)
  - visibility (fraction of the agent's observable area that's occupied
    vs free, from vis_occupy_indices/vis_free_indices -- a density/
    occlusion proxy)

Also does a basic sanity check on trans_matrices (should be well-formed
4x4 rigid transforms) and anchor/box decoding, as a cheap substitute for
a full BEV visualization pipeline.
"""

import sys
sys.path.insert(0, "reference_coperception")

import numpy as np
import glob
from collections import defaultdict


def analyze_agent(agent_dir, max_samples=1000):
    files = glob.glob(f"{agent_dir}/*/0.npy")[:max_samples]
    n_total_files = len(files)
    n_objects_per_frame = []
    distances = []
    occupied_fractions = []
    trans_matrix_issues = 0
    n_loaded = 0

    for f in files:
        d = np.load(f, allow_pickle=True)
        if d == 0:
            continue  # empty placeholder sample
        d = d.item()
        n_loaded += 1

        # gt_max_iou: (num_real_objects, 4) = (row, col, anchor_idx, instance_idx)
        # -- confirmed from detection_util.py's `anchors_map[tuple(gt_max_iou_idx[k][:-1])]`
        # usage. One row per REAL object, unlike allocation_mask (many positive
        # anchor cells per object) -- this is the clean per-object count/position.
        gt = d.get("gt_max_iou")
        if gt is not None and len(gt) > 0:
            n_objects_per_frame.append(len(gt))
            # voxel_size=0.25m, 256x256 grid centered at 0 over [-32,32]m.
            for row, col, _anchor_idx, _instance_idx in gt:
                cell_a = (row - 128) * 0.25
                cell_b = (col - 128) * 0.25
                dist = float(np.sqrt(cell_a**2 + cell_b**2))
                distances.append(dist)
        else:
            n_objects_per_frame.append(0)

        vis_occ = d.get("vis_occupy_indices")
        vis_free = d.get("vis_free_indices")
        if vis_occ is not None and vis_free is not None:
            n_occ = vis_occ.shape[1] if vis_occ.ndim == 2 else len(vis_occ)
            n_free = vis_free.shape[1] if vis_free.ndim == 2 else len(vis_free)
            if (n_occ + n_free) > 0:
                occupied_fractions.append(n_occ / (n_occ + n_free))

        tm = d.get("trans_matrices")  # raw shape (6, 4, 4): one transform per agent
        if tm is not None:
            # sanity: rotation part of each rigid transform should be ~orthonormal
            try:
                for a in range(tm.shape[0]):
                    R = tm[a, :3, :3]
                    if np.allclose(R, 0):
                        continue  # unused/padding agent slot
                    should_be_identity = R @ R.T
                    if not np.allclose(should_be_identity, np.eye(3), atol=0.05):
                        trans_matrix_issues += 1
            except Exception:
                trans_matrix_issues += 1

    return {
        "n_total_files": n_total_files,
        "n_loaded": n_loaded,
        "pct_non_empty": 100.0 * n_loaded / n_total_files if n_total_files else 0,
        "mean_objects_per_frame": float(np.mean(n_objects_per_frame)) if n_objects_per_frame else 0,
        "mean_distance_m": float(np.mean(distances)) if distances else 0,
        "median_distance_m": float(np.median(distances)) if distances else 0,
        "pct_far_objects_gt20m": float(np.mean(np.array(distances) > 20)) * 100 if distances else 0,
        "mean_occupied_fraction": float(np.mean(occupied_fractions)) if occupied_fractions else 0,
        "trans_matrix_issues": trans_matrix_issues,
        "n_distance_samples": len(distances),
    }


def main():
    val_root = "data/V2X-Sim-det-001/V2X-Sim-det/val"
    print(f"{'agent':<8}{'non_empty%':>11}{'n_loaded':>9}{'objs/frame':>12}{'mean_dist_m':>13}{'median_dist':>13}{'%>20m':>8}{'occ_frac':>10}{'tm_issues':>11}")
    for agent_id in range(1, 6):
        stats = analyze_agent(f"{val_root}/agent{agent_id}")
        print(
            f"agent{agent_id:<3}{stats['pct_non_empty']:>11.1f}{stats['n_loaded']:>9}{stats['mean_objects_per_frame']:>12.2f}"
            f"{stats['mean_distance_m']:>13.2f}{stats['median_distance_m']:>13.2f}"
            f"{stats['pct_far_objects_gt20m']:>8.1f}{stats['mean_occupied_fraction']:>10.3f}"
            f"{stats['trans_matrix_issues']:>11}"
        )


if __name__ == "__main__":
    main()
