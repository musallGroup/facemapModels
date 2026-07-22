#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import cv2


# ============================================================
# USER SETTINGS
# ============================================================

OUTPUT_ROOT = Path(
    r"\\Naskampa\lts\Team\Mick\2 Photon\New_Dataset"
)

FRAME_DIR = OUTPUT_ROOT / "Refinement_Training" / "Refinement_Frames"

VIDEOS = {
    "session1_20260429_143941": Path(
        r"\\Naskampa\lts\BpodBehavior\481\PuffyPenguin\Session Data\20260429_143941"
        r"\481_PuffyPenguin_20260429_143941_cam1_20260421.avi"
    ),
    "session2_20260509_114234": Path(
        r"\\Naskampa\lts\BpodBehavior\481\PuffyPenguin\Session Data\20260509_114234"
        r"\481_PuffyPenguin_20260509_114234_cam1_20260430.avi"
    ),
    "session3_20260520_115219": Path(
        r"\\Naskampa\lts\BpodBehavior\481\PuffyPenguin\Session Data\20260520_115219"
        r"\481_PuffyPenguin_20260520_115219_cam1_20260520.avi"
    ),
}

FRAME_POSITIONS = {
    "01_20pct": 0.20,
    "02_35pct": 0.35,
    "03_50pct": 0.50,
    "04_65pct": 0.65,
    "05_80pct": 0.80,
}


# ============================================================
# FUNCTIONS
# ============================================================

def extract_frame(video_path: Path, frame_fraction: float):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video:\n{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = int(total_frames * frame_fraction)
    frame_idx = max(0, min(frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from:\n{video_path}")

    time_sec = frame_idx / fps if fps else 0
    time_min = time_sec / 60

    return frame, frame_idx, total_frames, fps, time_min


def main():
    print("=" * 70)
    print("EXTRACT 2P NEW DATASET 481 REFINEMENT FRAMES")
    print("=" * 70)

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    for session_name, video_path in VIDEOS.items():
        print("\n" + "-" * 70)
        print(f"Session: {session_name}")
        print(f"Video:   {video_path}")

        if not video_path.exists():
            print(f"[ERROR] Video not found, skipping:\n{video_path}")
            continue

        video_stem = video_path.stem

        for frame_label, fraction in FRAME_POSITIONS.items():
            frame, frame_idx, total_frames, fps, time_min = extract_frame(
                video_path=video_path,
                frame_fraction=fraction,
            )

            out_name = (
                f"481_2P_newDataset_{session_name}_{frame_label}"
                f"_frame{frame_idx}_time{time_min:.2f}min.png"
            )

            out_path = FRAME_DIR / out_name

            ok = cv2.imwrite(str(out_path), frame)

            if not ok:
                print(f"[ERROR] Could not write frame:\n{out_path}")
                continue

            print(f"[OK] Saved: {out_path.name}")
            print(f"     frame {frame_idx}/{total_frames}, time {time_min:.2f} min")

    print("\nDONE.")
    print(f"Frames saved to:\n{FRAME_DIR}")


if __name__ == "__main__":
    main()