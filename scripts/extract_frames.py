"""Extract frames from every match video.

Usage:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --src C:/brawl-data/raw --out C:/brawl-data/frames --every 5
"""

import argparse
from pathlib import Path

from brawl_vod.frames import extract_frames


def main():
    parser = argparse.ArgumentParser(description="Extract de-duplicated frames from match videos.")
    parser.add_argument("--src", default="C:/brawl-data/raw", help="folder with .mp4 videos")
    parser.add_argument("--out", default="C:/brawl-data/frames", help="where to save JPGs")
    parser.add_argument("--every", type=float, default=5.0, help="seconds between sampled frames")
    args = parser.parse_args()

    videos = sorted(Path(args.src).glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found in {args.src}")

    total = 0
    for video in videos:
        n = extract_frames(video, args.out, every_sec=args.every)
        print(f"{video.name}: {n} frames")
        total += n
    print(f"Done: {total} frames from {len(videos)} videos -> {args.out}")


if __name__ == "__main__":
    main()
