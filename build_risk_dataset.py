"""
Stockade -- Phase 2 (learned classifier): build a labeled, per-clip feature
table from the Phase 1 trajectory CSVs.

DAD only labels whole clips as crash/normal (via the positive/negative
folder name) -- there's no ground truth for which track or frame is "the"
risky one. So this builds one row per CLIP, aggregating raw per-track
signals across the whole clip, rather than one row per tracked vehicle.
That matches the label granularity DAD actually provides.

Unlike score_risk.py's peak_risk (already collapsed through fixed
thresholds into one scalar), this keeps the underlying raw signals
separate so a learned classifier has something to combine:
  - max_growth: strongest windowed closing-speed slope seen, independent
    of whether it passed the proximity/persistence gates
  - max_proximity: largest box-height fraction of frame (closest approach)
  - max_lateral_var: strongest lateral-weaving variance
  - max_persistence: longest run of consecutive "closing" frames
  - mean_confidence, num_qualifying_tracks, n_frames: scene/detection quality

The train/test split column mirrors DAD's own videos/training vs
videos/testing folders, so a classifier trained here can be evaluated
without leakage.

Usage:
    python build_risk_dataset.py --tracks-dir outputs_dad_full --videos-dir data/DAD --output risk_dataset.csv
"""

import argparse
import csv
from pathlib import Path
from statistics import variance

from score_risk import (
    MIN_HISTORY,
    MIN_PROXIMITY_FRACTION,
    WINDOW,
    build_smoothed,
    get_frame_size,
    load_tracks,
    windowed_slope,
)


def clip_features(csv_path: Path, video_path: Path):
    frame_width, frame_height = get_frame_size(video_path)
    by_track = load_tracks(csv_path)
    tracks = {tid: h for tid, h in by_track.items() if len(h) >= MIN_HISTORY}

    if not tracks:
        return None

    max_growth = 0.0
    max_proximity = 0.0
    max_lateral_var = 0.0
    max_persistence = 0
    confidences = []
    n_frames_total = 0

    for track_id, history in tracks.items():
        smoothed = build_smoothed(history)
        confidences.extend(conf for _, _, conf in history)
        n_frames_total = max(n_frames_total, max(f for f, *_ in history) + 1)

        area_history = []
        lateral_positions = []
        consecutive_close = 0

        for frame, area, cx, h in smoothed:
            area_history.append(area)
            lateral_positions.append(cx / frame_width)

            proximity = h / frame_height
            max_proximity = max(max_proximity, proximity)

            own_growth = windowed_slope(area_history, WINDOW) if len(area_history) >= WINDOW else None
            if own_growth is not None:
                max_growth = max(max_growth, own_growth)
                is_close = proximity >= MIN_PROXIMITY_FRACTION and own_growth > 0.001
                consecutive_close = consecutive_close + 1 if is_close else 0
                max_persistence = max(max_persistence, consecutive_close)

            if len(lateral_positions) >= MIN_HISTORY:
                lat_var = variance(lateral_positions[-MIN_HISTORY:])
                max_lateral_var = max(max_lateral_var, lat_var)

    return {
        "max_growth": round(max_growth, 6),
        "max_proximity": round(max_proximity, 4),
        "max_lateral_var": round(max_lateral_var, 6),
        "max_persistence": max_persistence,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "num_qualifying_tracks": len(tracks),
        "n_frames": n_frames_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Build a labeled per-clip feature table for the Phase 2 learned classifier.")
    parser.add_argument("--tracks-dir", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--output", default="risk_dataset.csv")
    args = parser.parse_args()

    tracks_dir = Path(args.tracks_dir)
    videos_dir = Path(args.videos_dir)
    video_index = {p.stem: p for p in videos_dir.rglob("*.mp4")}

    fieldnames = [
        "clip", "split", "label",
        "max_growth", "max_proximity", "max_lateral_var", "max_persistence",
        "mean_confidence", "num_qualifying_tracks", "n_frames",
    ]
    rows = []
    skipped_no_tracks = 0
    skipped_no_video = 0

    for csv_path in sorted(tracks_dir.rglob("*_tracks.csv")):
        clip_name = csv_path.stem.replace("_tracks", "")
        video_path = video_index.get(clip_name)
        if video_path is None:
            skipped_no_video += 1
            continue

        path_str = str(csv_path)
        label = "crash" if "positive" in path_str else "normal"
        split = "train" if "training" in path_str else "test"

        feats = clip_features(csv_path, video_path)
        if feats is None:
            skipped_no_tracks += 1
            continue

        rows.append({"clip": clip_name, "split": split, "label": label, **feats})

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")
    n_crash = sum(1 for r in rows if r["label"] == "crash")
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  train: {n_train}, test: {n_test}")
    print(f"  crash: {n_crash}, normal: {len(rows) - n_crash}")
    print(f"  skipped (no tracks met MIN_HISTORY): {skipped_no_tracks}")
    print(f"  skipped (no matching video): {skipped_no_video}")


if __name__ == "__main__":
    main()
