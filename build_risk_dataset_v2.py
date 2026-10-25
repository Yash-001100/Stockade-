"""
Stockade -- Phase 2 (learned classifier), v2 feature set: adds ego-motion-
compensated, physically-grounded closing-speed/TTC features on top of the
v1 geometric features from build_risk_dataset.py.

v1 used a unitless "frames to double size" proxy for closing speed. v2
instead estimates real depth per frame (pinhole approximation -- see
motion_features.py for the stated assumptions: focal_px ~= frame_width,
typical per-class vehicle height), corrects it for the ego camera's own
motion using background optical flow, and fits closing speed/TTC on that
corrected depth in real units (m/s, seconds).

New columns (added alongside the v1 ones, not replacing them, so the two
feature sets can be compared head-to-head):
  - max_closing_speed_mps: strongest ego-compensated closing speed seen
  - min_real_ttc_s: smallest ego-compensated time-to-collision seen
  - mean_ego_correction: mean |scale - 1| of the background optical-flow
    fit across the clip (diagnostic: how much ego-motion correction
    actually mattered for this clip)

Usage:
    python build_risk_dataset_v2.py --tracks-dir outputs_dad_full --videos-dir data/DAD --output risk_dataset_v2.csv [--limit N]
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from motion_features import (
    MIN_MATCHED_POINTS,
    background_scale_factors,
    cumulative_scale,
    estimate_depth_m,
    estimate_focal_px,
)
from score_risk import MIN_CONFIDENCE, MIN_HISTORY, MIN_PROXIMITY_FRACTION, WINDOW, get_frame_size

# A real highway closing speed rarely exceeds this even under hard braking;
# anything above is treated as a numerical artifact (noisy box height at
# range, amplified by the depth inversion), not a real reading.
MAX_PLAUSIBLE_CLOSING_SPEED_MPS = 40.0

EMA_ALPHA = 0.4
FPS = 25.0  # confirmed fixed across sampled DAD clips


def load_tracks_with_class(csv_path: Path):
    """track_id -> sorted list of (frame, (x1,y1,x2,y2), conf, class_id),
    plus boxes_by_frame: frame -> list of boxes for ALL confident detections
    (used to mask out vehicles from the background optical-flow estimate)."""
    by_track = defaultdict(list)
    boxes_by_frame = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            conf = float(row["confidence"])
            if conf < MIN_CONFIDENCE:
                continue
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            frame = int(row["frame"])
            by_track[int(row["track_id"])].append((frame, box, conf, int(row["class_id"])))
            boxes_by_frame[frame].append(box)
    for tid in by_track:
        by_track[tid].sort(key=lambda r: r[0])
    return by_track, boxes_by_frame


def windowed_slope_real(values, window):
    """Same trick as score_risk.windowed_slope, but NOT normalized by the
    mean -- we want an absolute speed in metres/frame here, not a relative
    growth rate."""
    recent = values[-window:]
    if len(recent) < 3:
        return None
    x = np.arange(len(recent))
    y = np.array(recent)
    slope, _ = np.polyfit(x, y, 1)
    return slope


def clip_motion_features(csv_path: Path, video_path: Path):
    frame_width, frame_height = get_frame_size(video_path)
    focal_px = estimate_focal_px(frame_width)

    by_track, boxes_by_frame = load_tracks_with_class(csv_path)
    tracks = {tid: h for tid, h in by_track.items() if len(h) >= MIN_HISTORY}
    if not tracks:
        return None

    n_frames = max(f for frames in by_track.values() for f, *_ in frames) + 1
    scales = background_scale_factors(video_path, boxes_by_frame, n_frames)
    mean_ego_correction = float(np.mean([abs(s - 1.0) for s in scales])) if scales else 0.0

    max_closing_speed = 0.0
    min_real_ttc = None

    for track_id, history in tracks.items():
        ema_height = None
        depth_history = []  # ego-compensated depth, derived from the SMOOTHED height

        for frame, (x1, y1, x2, y2), conf, class_id in history:
            box_height = y2 - y1
            # Smooth the raw pixel measurement BEFORE inverting it to depth --
            # depth = k / height is highly nonlinear, so noise smoothed after
            # inversion is already amplified; smoothing the height first keeps
            # that amplification from happening in the first place.
            ema_height = box_height if ema_height is None else EMA_ALPHA * box_height + (1 - EMA_ALPHA) * ema_height

            close_enough = (ema_height / frame_height) >= MIN_PROXIMITY_FRACTION
            if not close_enough:
                depth_history.append(None)
                continue

            raw_depth = estimate_depth_m(ema_height, class_id, focal_px)
            if raw_depth is None:
                depth_history.append(None)
                continue

            corrected_depth = raw_depth * cumulative_scale(scales, 0, frame)
            depth_history.append(corrected_depth)

            recent = [d for d in depth_history[-WINDOW:] if d is not None]
            if len(recent) >= WINDOW:
                slope_m_per_frame = windowed_slope_real(recent, WINDOW)
                if slope_m_per_frame is not None and slope_m_per_frame < 0:
                    closing_speed_mps = -slope_m_per_frame * FPS
                    if closing_speed_mps > MAX_PLAUSIBLE_CLOSING_SPEED_MPS:
                        continue  # numerical artifact, not a real reading -- discard
                    current_depth = corrected_depth
                    ttc_s = current_depth / closing_speed_mps if closing_speed_mps > 0.01 else None

                    max_closing_speed = max(max_closing_speed, closing_speed_mps)
                    if ttc_s is not None and (min_real_ttc is None or ttc_s < min_real_ttc):
                        min_real_ttc = ttc_s

    return {
        "max_closing_speed_mps": round(max_closing_speed, 4),
        "min_real_ttc_s": round(min_real_ttc, 3) if min_real_ttc is not None else "",
        "mean_ego_correction": round(mean_ego_correction, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Build v2 (ego-compensated, physically-grounded) risk features.")
    parser.add_argument("--tracks-dir", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output", default="risk_dataset_v2.csv")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N clips (for a quick sanity check).")
    args = parser.parse_args()

    tracks_dir = Path(args.tracks_dir)
    videos_dir = Path(args.videos_dir)
    video_index = {p.stem: p for p in videos_dir.rglob("*.mp4")}

    fieldnames = ["clip", "split", "label", "max_closing_speed_mps", "min_real_ttc_s", "mean_ego_correction"]
    rows = []
    csv_paths = sorted(tracks_dir.rglob("*_tracks.csv"))
    if args.limit:
        csv_paths = csv_paths[: args.limit]

    for i, csv_path in enumerate(csv_paths):
        clip_name = csv_path.stem.replace("_tracks", "")
        video_path = video_index.get(clip_name)
        if video_path is None:
            continue

        path_str = str(csv_path)
        label = "crash" if "positive" in path_str else "normal"
        split = "train" if "training" in path_str else "test"

        feats = clip_motion_features(csv_path, video_path)
        if feats is None:
            continue

        row = {"clip": clip_name, "split": split, "label": label, **feats}
        rows.append(row)
        print(f"  [{i+1}/{len(csv_paths)}] {clip_name} ({label}): "
              f"max_closing_speed={row['max_closing_speed_mps']} m/s, "
              f"min_real_ttc={row['min_real_ttc_s']} s, "
              f"mean_ego_correction={row['mean_ego_correction']}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
