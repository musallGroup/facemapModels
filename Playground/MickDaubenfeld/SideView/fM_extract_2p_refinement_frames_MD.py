#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2


# ============================================================
# USER SETTINGS
# ============================================================

VIDEO_ROOT = Path(r"\\naskampa\data\BpodBehavior\2pvideos")

OUTPUT_ROOT = Path(r"\\Naskampa\lts\Team\Mick\2 Photon")
FRAME_DIR = OUTPUT_ROOT / "Refinement_Training" / "2P_Refinement_Frames"

# Wir nehmen je ein Video von 4 Mäusen
VIDEO_PREFIXES = [
    "319_PuffyPenguin_20250203_102054",
    "320_PuffyPenguin_20250311_114428",
    "321_PuffyPenguin_20250425_122840",
    "322_PuffyPenguin_20250321_141521",
]

FRAME_POSITIONS = {
    "01_early": 0.20,
    "02_midearly": 0.35,
    "03_middle": 0.50,
    "04_midlate": 0.65,
    "05_late": 0.80,
}


# ============================================================
# FUNCTIONS
# ============================================================

def find_video_by_prefix(prefix: str) -> Path:
    matches = sorted(VIDEO_ROOT.glob(f"{prefix}*.avi"))

    if len(matches) == 0:
        raise FileNotFoundError(f"No video found for prefix: {prefix}")

    if len(matches) > 1:
        print(f"[WARN] Multiple matches for {prefix}, using first:")
        for m in matches:
            print(f"   {m.name}")

    return matches[0]


def extract_frame(video_path: Path, fraction: float):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = int(total_frames * fraction)
    frame_idx = max(0, min(frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    return frame, frame_idx, total_frames, fps


def main():
    print("=" * 70)
    print("EXTRACT 2P REFINEMENT FRAMES")
    print("=" * 70)

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    for prefix in VIDEO_PREFIXES:
        video_path = find_video_by_prefix(prefix)

        print("\n" + "-" * 70)
        print(f"Video: {video_path}")

        stem = video_path.stem
        mouse = stem.split("_")[0]

        for label, fraction in FRAME_POSITIONS.items():
            frame, frame_idx, total_frames, fps = extract_frame(video_path, fraction)

            time_sec = frame_idx / fps if fps else 0
            time_min = time_sec / 60

            out_name = (
                f"{mouse}_2P_refinement_frame_{label}"
                f"_{stem}_frame{frame_idx}_time{time_min:.2f}min.png"
            )

            out_path = FRAME_DIR / out_name
            cv2.imwrite(str(out_path), frame)

            print(f"[OK] Saved: {out_path.name}")
            print(f"     frame {frame_idx}/{total_frames}, time {time_min:.2f} min")

    print("\nDONE.")
    print(f"Frames saved to:\n{FRAME_DIR}")


if __name__ == "__main__":
    main()