from pathlib import Path

import cv2
import numpy as np


def iter_sampled_frames(video_path, every_sec=1.0, skip_start=0.0, skip_end=0.0):
    """Yield (timestamp_sec, frame) about once every `every_sec` seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # seconds -> frames
    step = max(1, round(fps * every_sec))
    first = round(fps * skip_start)
    last = total - round(fps * skip_end)

    i = 0
    try:
        while i < last:
            ok = cap.grab()
            if not ok:
                break

            if i >= first and (i - first) % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    yield i / fps, frame
            i += 1
    finally:
        cap.release()


def thumbnail(frame, size=64):
    """Small grayscale float version of a frame, for cheap comparisons."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    return small.astype(np.float32)


def is_duplicate(thumb_a, thumb_b, threshold=4.0):
    """True if two thumbnails are nearly the same (mean abs pixel diff < threshold)."""
    return float(np.mean(np.abs(thumb_a - thumb_b))) < threshold


def extract_frames(video_path, out_dir, every_sec=1.0, skip_start=5.0,
                   skip_end=5.0, dup_threshold=4.0):
    """Save sampled, de-duplicated frames as JPGs. Returns how many were saved."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    last_thumb = None
    saved = 0
    for t, frame in iter_sampled_frames(video_path, every_sec, skip_start, skip_end):
        thumb = thumbnail(frame)
        if last_thumb is not None and is_duplicate(thumb, last_thumb, dup_threshold):
            continue
        
        name = f"{video_path.stem}_{int(t):05d}.jpg"
        
        cv2.imwrite(str(out_dir / name), frame)

        last_thumb = thumb
        saved += 1
    return saved
