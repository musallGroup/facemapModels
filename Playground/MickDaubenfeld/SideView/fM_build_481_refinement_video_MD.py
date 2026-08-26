#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2


# ============================================================
# USER SETTINGS
# ============================================================

FRAME_DIR = Path(
    r"\\Naskampa\lts\Team\Mick\FM_Side_View\2 Photon\New_Dataset\Refinement_Training\Refinement_Frames"
)

OUTPUT_VIDEO = Path(
    r"\\Naskampa\lts\Team\Mick\FM_Side_View\2 Photon\New_Dataset\Refinement_Training\Refinement_Videos"
    r"\2P_newDataset_481_refinement_video.avi"
)

FPS = 5


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("BUILD 2P NEW DATASET 481 REFINEMENT VIDEO")
    print("=" * 70)

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(FRAME_DIR.glob("*.png"))

    if len(frames) == 0:
        raise RuntimeError(f"No PNG files found in:\n{FRAME_DIR}")

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

        if img.shape[:2] != (height, width):
            raise ValueError(
                f"Frame size mismatch:\n"
                f"{frame_path.name}: {img.shape[:2]} but expected {(height, width)}"
            )

        writer.write(img)
        written += 1
        print(f"[ADD] {frame_path.name}")

    writer.release()

    print("\nDONE.")
    print(f"Wrote {written} frames")
    print(f"Video saved to:\n{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
