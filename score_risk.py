"""
Stockade — Phase 2: baseline risk scoring (v3)

Reads the trajectory CSVs that track_vehicles.py produces and turns each
tracked vehicle's frame-by-frame boxes into a rule-based risk score, using
only geometry — no ML model yet. This is the transparent baseline every
later (learned) model has to beat.

v2 changes from the first pass, after the first evaluation showed very low
recall (most real crashes scored near zero):

  1. Closing-speed is now a WINDOWED LINEAR-REGRESSION SLOPE of box area
     over the last N frames, not a single frame-to-frame diff. A two-point
     diff is about as noisy a signal as exists; fitting a line over a
     window is far more robust to one jittery detection.
  2. Box AREA (width x height) instead of just height -- more robust to a
     vehicle being partially cropped or occluded on one side.
  3. Low-confidence detections are dropped before any feature is computed
     -- a marginal, flickering detection shouldn't be allowed to define
     "closing fast."
  4. A PROXIMITY GATE: a vehicle has to already take up a meaningful
     fraction of the frame height before its closing-speed is scored at
     all. A distant, small car's growth-rate estimate is almost pure
     noise; only a car that's already close enough to matter counts.

v3 changes, in direct response to v2's real problem (52% recall but a very
high false-alarm rate):

  1. BUG FIX: the persistence gate (require N consecutive "closing" frames
     before it counts) had an off-by-design reset -- it zeroed its own
     counter on every frame that hadn't *yet* reached N, which made it
     mathematically impossible for the counter to ever reach N. This
     silently disabled almost every "closing fast" detection and was
     invisible in testing because whatever else changed (TTC threshold,
     peer checks) had no candidates left to affect. Fixed.
  2. TRIED AND DROPPED: a "peer-relative" check (only count a vehicle as
     closing if it's closing faster than the median of every other
     tracked vehicle in the same frame, to filter out whole-scene effects
     like the ego car braking into traffic). Implemented and tested at
     several margins -- it never changed a single clip's classification,
     because classification uses each clip's PEAK risk frame, and the
     peer check only ever suppressed non-peak frames. A real idea, but
     inert for this scoring method, so removed rather than kept as dead
     complexity.
  3. TUNED for precision (this was the actual lever): raised
     PERSISTENCE_FRAMES 3 -> 4 and MIN_PROXIMITY_FRACTION 0.12 -> 0.15.
     Grid-searched against the 46-clip labeled set. Net effect at
     threshold 0.5: precision 57% -> 60%, false-alarm rate 39% -> 26%,
     at the cost of recall 52% -> 39%. A genuine tradeoff, not a free
     win -- fewer false alarms, but some real closing vehicles now score
     below threshold too.

Usage:
    python score_risk.py --tracks-dir outputs --videos-dir data/DAD
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import variance

import cv2
import numpy as np

# Tunable thresholds -- deliberately simple and inspectable, per the
# "transparent baseline" approach in the project outline.
MIN_CONFIDENCE = 0.35            # drop marginal detections before scoring
MIN_HISTORY = 8                  # frames needed before scoring a track at all
WINDOW = 5                       # frames used for the regression-slope estimate
MIN_PROXIMITY_FRACTION = 0.15    # box height must be >= 15% of frame height to be "close enough to matter"
TTC_THRESHOLD_FRAMES = 15        # "danger" if proxy TTC < 1.5s at 10fps
PERSISTENCE_FRAMES = 4           # require the signal to hold before it counts
LATERAL_VARIANCE_THRESHOLD = 0.003
EMA_ALPHA = 0.4


def get_frame_size(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    cap.release()
    return width, height


def load_tracks(csv_path: Path):
    """track_id -> sorted list of (frame, x1,y1,x2,y2, confidence), low-confidence boxes dropped."""
    by_track = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            conf = float(row["confidence"])
            if conf < MIN_CONFIDENCE:
                continue
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            by_track[int(row["track_id"])].append((int(row["frame"]), box, conf))
    for tid in by_track:
        by_track[tid].sort(key=lambda r: r[0])
    return by_track


def windowed_slope(values, window):
    """Relative per-frame growth rate of `values` over the trailing `window` frames,
    via a linear regression slope normalized by the window's mean value."""
    recent = values[-window:]
    if len(recent) < 3:
        return None
    x = np.arange(len(recent))
    y = np.array(recent)
    mean_y = y.mean()
    if mean_y <= 0:
        return None
    slope, _ = np.polyfit(x, y, 1)
    return slope / mean_y  # relative growth per frame


def build_smoothed(history):
    """EMA-smoothed (frame, area, cx, h) series for one track."""
    smoothed = []
    ema_area = ema_cx = None
    for frame, (x1, y1, x2, y2), conf in history:
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        h = y2 - y1
        cx = (x1 + x2) / 2
        ema_area = area if ema_area is None else EMA_ALPHA * area + (1 - EMA_ALPHA) * ema_area
        ema_cx = cx if ema_cx is None else EMA_ALPHA * cx + (1 - EMA_ALPHA) * ema_cx
        smoothed.append((frame, ema_area, ema_cx, h))
    return smoothed


def score_clip(csv_path: Path, video_path: Path):
    frame_width, frame_height = get_frame_size(video_path)
    by_track = load_tracks(csv_path)
    tracks = {tid: h for tid, h in by_track.items() if len(h) >= MIN_HISTORY}

    per_frame_risk = defaultdict(float)
    per_frame_reason = {}

    for track_id, history in tracks.items():
        smoothed = build_smoothed(history)
        lateral_positions = []
        area_history = []
        consecutive_close = 0

        for frame, area, cx, h in smoothed:
            area_history.append(area)
            lateral_positions.append(cx / frame_width)

            risk = 0.0
            reason = None

            close_enough = (h / frame_height) >= MIN_PROXIMITY_FRACTION
            own_growth = windowed_slope(area_history, WINDOW) if len(area_history) >= WINDOW else None

            if close_enough and own_growth is not None:
                ttc_proxy = (1.0 / own_growth) if own_growth > 0.001 else None
                is_close = ttc_proxy is not None and 0 < ttc_proxy < TTC_THRESHOLD_FRAMES
                # consecutive_close is updated ONCE here and never reset again below --
                # a previous version reset it back to 0 whenever it hadn't yet reached
                # PERSISTENCE_FRAMES, which made the gate mathematically impossible to pass
                # (frame N needs N-1 to have already accumulated, but N-1's own check reset
                # it first). That silently zeroed out every "closing fast" detection.
                consecutive_close = consecutive_close + 1 if is_close else 0

                if is_close and consecutive_close >= PERSISTENCE_FRAMES:
                    risk = 1.0 - (ttc_proxy / TTC_THRESHOLD_FRAMES)
                    reason = (
                        f"track {track_id} closing fast (TTC proxy {ttc_proxy:.1f}f, held {consecutive_close}f, "
                        f"box {h/frame_height:.0%} of frame, growth {own_growth:.3f}/frame)"
                    )
            else:
                consecutive_close = 0

            if len(lateral_positions) >= MIN_HISTORY:
                recent = lateral_positions[-MIN_HISTORY:]
                lat_var = variance(recent)
                if lat_var > LATERAL_VARIANCE_THRESHOLD:
                    weave_risk = min(1.0, lat_var / (LATERAL_VARIANCE_THRESHOLD * 4))
                    if weave_risk > risk:
                        risk = weave_risk
                        reason = f"track {track_id} weaving (lateral var {lat_var:.4f})"

            if risk > per_frame_risk[frame]:
                per_frame_risk[frame] = risk
                per_frame_reason[frame] = reason

    if not per_frame_risk:
        return None

    max_frame = max(per_frame_risk, key=per_frame_risk.get)
    return {
        "n_frames_scored": len(per_frame_risk),
        "peak_risk": per_frame_risk[max_frame],
        "peak_frame": max_frame,
        "peak_reason": per_frame_reason.get(max_frame),
        "mean_risk": sum(per_frame_risk.values()) / len(per_frame_risk),
        "final_5_frames_mean": sum(
            v for f, v in per_frame_risk.items() if f >= max(per_frame_risk) - 5
        ) / max(1, len([f for f in per_frame_risk if f >= max(per_frame_risk) - 5])),
    }


def main():
    parser = argparse.ArgumentParser(description="Stockade Phase 2: rule-based risk scoring over Phase 1 trajectories.")
    parser.add_argument("--tracks-dir", required=True, help="Folder containing *_tracks.csv files (searched recursively).")
    parser.add_argument("--videos-dir", required=True, help="Folder containing the matching source clips (searched recursively).")
    args = parser.parse_args()

    tracks_dir = Path(args.tracks_dir)
    videos_dir = Path(args.videos_dir)

    video_index = {p.stem: p for p in videos_dir.rglob("*.mp4")}

    rows = []
    print(f"{'clip':<10} {'label':<10} {'peak_risk':>9} {'mean_risk':>9} {'end_risk':>9}  reason")
    print("-" * 100)
    for csv_path in sorted(tracks_dir.rglob("*_tracks.csv")):
        clip_name = csv_path.stem.replace("_tracks", "")
        video_path = video_index.get(clip_name)
        if video_path is None:
            print(f"{clip_name:<10} -- source video not found, skipping")
            continue

        # crude label from folder name, since DAD splits clips into
        # videos/{training,testing}/{positive,negative}/
        label = "crash" if "positive" in str(csv_path) else "normal"

        result = score_clip(csv_path, video_path)
        if result is None:
            print(f"{clip_name:<10} {label:<10}  no tracks met the minimum history")
            rows.append((label, 0.0))
            continue

        print(
            f"{clip_name:<10} {label:<10} {result['peak_risk']:>9.2f} "
            f"{result['mean_risk']:>9.2f} {result['final_5_frames_mean']:>9.2f}  {result['peak_reason']}"
        )
        rows.append((label, result["peak_risk"]))

    print("\n" + "=" * 60)
    print("Classification metrics (predicting 'crash' when peak_risk >= threshold)")
    print("=" * 60)
    for threshold in (0.3, 0.5, 0.7):
        tp = sum(1 for label, score in rows if label == "crash" and score >= threshold)
        fn = sum(1 for label, score in rows if label == "crash" and score < threshold)
        fp = sum(1 for label, score in rows if label == "normal" and score >= threshold)
        tn = sum(1 for label, score in rows if label == "normal" and score < threshold)
        n_crash = tp + fn
        n_normal = fp + tn
        recall = tp / n_crash if n_crash else 0.0
        false_alarm_rate = fp / n_normal if n_normal else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        print(
            f"threshold {threshold:.1f}:  recall {recall:.0%} ({tp}/{n_crash} crashes caught)   "
            f"false-alarm rate {false_alarm_rate:.0%} ({fp}/{n_normal} normal clips flagged)   "
            f"precision {precision:.0%}"
        )


if __name__ == "__main__":
    main()
