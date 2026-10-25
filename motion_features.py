"""
Stockade -- Phase 2 feature engineering: monocular depth approximation and
ego-motion compensation.

Neither DAD nor the detection CSVs carry camera calibration, so this is
explicitly an approximation, not ground truth:
  - focal length is assumed as focal_px = frame_width (~53 deg horizontal
    FOV -- a reasonable middle estimate for a dashcam, not a measured value)
  - each vehicle class is assigned a typical real-world height

Depth per frame comes from the standard pinhole relation:
    depth_m = (real_height_m * focal_px) / box_height_px

Ego-motion compensation: optical flow on background points (outside every
detected vehicle's box) estimates the camera's own frame-to-frame motion as
a similarity-transform scale factor. A tracked vehicle's depth is corrected
by the cumulative background scale since the track's first frame, so its
closing speed reflects motion relative to the road, not relative to the
ego camera (e.g. the ego car braking shouldn't itself count as "closing").
"""

import cv2
import numpy as np

# COCO class id -> typical real-world height (metres). Matches
# track_vehicles.py's VEHICLE_CLASSES = [2, 3, 5, 7].
REAL_HEIGHT_M = {
    2: 1.5,   # car
    3: 1.3,   # motorcycle
    5: 3.2,   # bus
    7: 2.8,   # truck
}

MAX_CORNERS = 200
MIN_MATCHED_POINTS = 15  # below this, the background-scale estimate is too noisy to trust


def estimate_focal_px(frame_width: int) -> float:
    """Assumed focal length in pixels. No calibration data exists for DAD,
    so this uses the common approximation focal_px ~= frame_width."""
    return float(frame_width)


def estimate_depth_m(box_height_px: float, class_id: int, focal_px: float) -> float:
    real_height = REAL_HEIGHT_M.get(class_id, 1.5)
    if box_height_px <= 0:
        return None
    return (real_height * focal_px) / box_height_px


def dilated_exclusion_mask(frame_shape, boxes_this_frame, dilate_px=10):
    """Boolean mask, True where a background feature point is allowed
    (i.e. outside every vehicle box, with a small margin)."""
    h, w = frame_shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for (x1, y1, x2, y2) in boxes_this_frame:
        x1 = max(0, int(x1) - dilate_px)
        y1 = max(0, int(y1) - dilate_px)
        x2 = min(w, int(x2) + dilate_px)
        y2 = min(h, int(y2) + dilate_px)
        mask[y1:y2, x1:x2] = 0
    return mask


def background_scale_factors(video_path, boxes_by_frame, n_frames):
    """Per-frame-pair background scale factor from optical flow, isolating
    ego-camera motion from the tracked vehicles' own motion.

    Returns a list of length n_frames where entry i is the scale factor
    from frame i-1 to frame i (entry 0 is always 1.0, no prior frame).
    A value > 1 means the background appears to magnify (ego moving
    forward into the scene); < 1 means it recedes (ego braking/reversing).
    Falls back to 1.0 (no correction) for any frame pair with too few
    reliably matched background points.
    """
    cap = cv2.VideoCapture(str(video_path))
    scales = [1.0] * n_frames

    ok, prev_frame = cap.read()
    if not ok:
        cap.release()
        return scales
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 1
    while frame_idx < n_frames:
        ok, curr_frame = cap.read()
        if not ok:
            break
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        mask = dilated_exclusion_mask(prev_gray.shape, boxes_by_frame.get(frame_idx - 1, []))
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=MAX_CORNERS, qualityLevel=0.01, minDistance=8, mask=mask)

        scale = 1.0
        if prev_pts is not None and len(prev_pts) >= MIN_MATCHED_POINTS:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            status = status.reshape(-1).astype(bool)
            good_prev = prev_pts[status]
            good_curr = curr_pts[status]

            if len(good_prev) >= MIN_MATCHED_POINTS:
                transform, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
                if transform is not None:
                    a, b = transform[0, 0], transform[1, 0]
                    est_scale = float(np.sqrt(a * a + b * b))
                    # Sanity bound -- a single frame pair shouldn't imply the
                    # background doubled or halved; discard implausible fits.
                    if 0.5 < est_scale < 2.0:
                        scale = est_scale

        scales[frame_idx] = scale
        prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    # Pad any remaining (video shorter than n_frames) with no correction.
    return scales


def cumulative_scale(scales, start_frame, end_frame):
    """Cumulative background scale from start_frame to end_frame (inclusive
    of the transition into start_frame+1 .. end_frame)."""
    cum = 1.0
    for i in range(start_frame + 1, end_frame + 1):
        if i < len(scales):
            cum *= scales[i]
    return cum
