from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract uniformly sampled video frames to .npy files.")
    parser.add_argument("--video", required=True, type=Path, help="Path to the source video.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write .npy frames into.")
    parser.add_argument("--num-frames", type=int, default=4096, help="Number of frames to sample uniformly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has invalid frame count: {frame_count}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = np.linspace(0, frame_count - 1, num=args.num_frames, dtype=np.int64)
    last_index = None

    for i, frame_index in enumerate(sample_indices):
        if frame_index != last_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"Failed to read frame at index {frame_index}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            np.save(output_dir / f"{i:05d}.npy", frame_rgb)
            cached_frame = frame_rgb
            last_index = int(frame_index)
        else:
            np.save(output_dir / f"{i:05d}.npy", cached_frame)

    cap.release()


if __name__ == "__main__":
    main()
