#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2

# ============================================================
# USER SETTINGS
# ============================================================

FRAME_DIR = Path(
    r"\\Naskampa\lts\Team\Mick\2 Photon\Refinement_Training\2P_Refinement_Frames"
)

OUTPUT_VIDEO = Path(
    r"\\Naskampa\lts\Team\Mick\2 Photon\Refinement_Training\2P_refinement_video.avi"
)

FPS = 5


# ============================================================
# MAIN
# ============================================================

def main():

    frames = sorted(FRAME_DIR.glob("*.png"))

    if len(frames) == 0:
        raise RuntimeError(f"No PNG files found in:\n{FRAME_DIR}")

    print("=" * 70)
    print("BUILD 2P REFINEMENT VIDEO")
    print("=" * 70)
    print(f"Found {len(frames)} frames")

    first = cv2.imread(str(frames[0]))

    if first is None:
        raise RuntimeError(f"Could not read first frame:\n{frames[0]}")

    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"XVID"),
        FPS,
        (width, height),
    )

    written = 0

    for frame_path in frames:

        img = cv2.imread(str(frame_path))

        if img is None:
            print(f"[WARN] Could not read: {frame_path.name}")
            continue

        writer.write(img)
        written += 1

        print(f"[ADD] {frame_path.name}")

    writer.release()

    print("\nDONE")
    print(f"Wrote {written} frames")
    print(f"Video saved to:\n{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()