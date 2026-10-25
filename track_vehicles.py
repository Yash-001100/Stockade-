"""
Stockade — Phase 1: offline detection + tracking

What this does:
  1. Finds every .mp4 clip under an input folder (recursively).
  2. Runs a pretrained YOLO detector + ByteTrack over each clip.
  3. Saves an annotated video (boxes + persistent track IDs) so you can
     visually check whether tracking holds up across the whole clip.
  4. Saves a per-clip CSV of every tracked box, frame by frame — this is
     the trajectory data Phase 2 (risk scoring) will read from.

Usage (from a terminal, inside your activated virtual environment):

    python track_vehicles.py --input data/CCD --output outputs

Run with --help to see all options.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

# COCO class IDs for the vehicle types we care about.
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]


def find_clips(input_dir: Path):
    """All .mp4 files under input_dir, sorted for reproducible runs."""
    return sorted(input_dir.rglob("*.mp4"))


def process_clip(model: YOLO, clip_path: Path, input_root: Path, output_root: Path):
    """Run detection + tracking on one clip, save annotated video + trajectory CSV."""
    # Mirror the clip's folder structure under output_root, e.g.
    # data/CCD/videos/Crash-1500/000001.mp4 -> outputs/videos/Crash-1500/000001.mp4
    rel_path = clip_path.relative_to(input_root)
    clip_output_dir = output_root / rel_path.parent
    clip_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = clip_output_dir / f"{clip_path.stem}_tracks.csv"
    if csv_path.exists():
        print(f"  {clip_path.name}: already processed, skipping")
        return
    rows = []

    # stream=True processes frame-by-frame instead of loading the whole
    # clip into memory first — matters once you're running this over
    # thousands of clips rather than just a handful.
    results = model.track(
        source=str(clip_path),
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        persist=False,  # each clip is independent — don't carry track IDs over from the previous clip
        stream=True,
        verbose=False,
        project=str(clip_output_dir),
        name=clip_path.stem,
        exist_ok=True,
        save=True,  # writes the annotated video into clip_output_dir/<clip_stem>/
    )

    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue  # no tracked vehicles this frame
        for box, track_id, cls, conf in zip(
            boxes.xyxy.tolist(), boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
        ):
            x1, y1, x2, y2 = box
            rows.append(
                {
                    "frame": frame_idx,
                    "track_id": int(track_id),
                    "class_id": int(cls),
                    "confidence": round(float(conf), 4),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["frame", "track_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        writer.writerows(rows)

    n_tracks = len({r["track_id"] for r in rows})
    print(f"  {clip_path.name}: {len(rows)} boxes across {n_tracks} tracked vehicle(s) -> {csv_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Stockade Phase 1: detection + tracking over benchmark clips.")
    parser.add_argument("--input", required=True, help="Folder containing .mp4 clips (searched recursively).")
    parser.add_argument("--output", default="outputs", help="Folder to write annotated videos + trajectory CSVs.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights to use (downloaded automatically the first run).")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N clips (useful for a quick test).")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clips = find_clips(input_root)
    if args.limit:
        clips = clips[: args.limit]

    if not clips:
        print(f"No .mp4 files found under {input_root} — check the --input path.")
        return

    print(f"Found {len(clips)} clip(s) under {input_root}")
    model = YOLO(args.model)

    failures = []
    for clip_path in clips:
        try:
            process_clip(model, clip_path, input_root, output_root)
        except Exception as exc:
            # One truncated/corrupt clip shouldn't take down a run over
            # thousands of files -- log it and keep going.
            print(f"  {clip_path.name}: FAILED ({exc}) -- skipping")
            failures.append(clip_path)

    print(f"\nDone. Annotated videos + trajectory CSVs are under {output_root}")
    if failures:
        print(f"{len(failures)} clip(s) failed and were skipped:")
        for f in failures:
            print(f"  {f}")


if __name__ == "__main__":
    main()
