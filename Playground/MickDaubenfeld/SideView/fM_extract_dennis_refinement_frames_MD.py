#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil
import cv2


# ============================================================
# USER SETTINGS
# ============================================================

OUTPUT_ROOT = Path(
    r"\\Naskampa\lts\Team\Mick\FM_Side_View\FM_Dennis_Cohort"
)

COMBINED_FRAME_DIR = OUTPUT_ROOT / "Refinement_Training" / "Dennis12_Frames"

VIDEOS = {
    "458": Path(
        r"\\Naskampa\lts\BpodBehavior\458\PuffyPenguin\Session Data\20260430_115934"
        r"\458_PuffyPenguin_20260430_115934_cam1_20260429_152112.avi"
    ),
    "460": Path(
        r"\\Naskampa\lts\BpodBehavior\460\PuffyPenguin\Session Data\20260509_142300"
        r"\460_PuffyPenguin_20260509_142300_cam1_20260507_092721.avi"
    ),
    "461": Path(
        r"\\Naskampa\lts\BpodBehavior\461\UncertainUrchin\Session Data\20260406_151318"
        r"\461_UncertainUrchin_20260406_151318_cam1_20260406_151353.avi"
    ),
    "462": Path(
        r"\\Naskampa\lts\BpodBehavior\462\UncertainUrchin\Session Data\20260401_160217"
        r"\462_UncertainUrchin_20260401_160217_cam1_20260401_160943.avi"
    ),
}

FRAME_POSITIONS = {
    "01_early": 0.25,
    "02_middle": 0.50,
    "03_late": 0.75,
}


# ============================================================
# FUNCTIONS
# ============================================================

def extract_frame(video_path: Path, frame_fraction: float):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = int(total_frames * frame_fraction)
    frame_idx = max(0, min(frame_idx, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    return frame, frame_idx, total_frames, fps


def main():
    print("=" * 70)
    print("EXTRACT DENNIS MICE REFINEMENT FRAMES")
    print("=" * 70)

    COMBINED_FRAME_DIR.mkdir(parents=True, exist_ok=True)

    for mouse, video_path in VIDEOS.items():
        print("\n" + "-" * 70)
        print(f"Mouse {mouse}")
        print(f"Video: {video_path}")

        if not video_path.exists():
            print(f"[ERROR] Video not found: {video_path}")
            continue

        mouse_frame_dir = OUTPUT_ROOT / f"Mouse_{mouse}" / "Refinement_Frames"
        mouse_frame_dir.mkdir(parents=True, exist_ok=True)

        for label, fraction in FRAME_POSITIONS.items():
            frame, frame_idx, total_frames, fps = extract_frame(video_path, fraction)

            time_sec = frame_idx / fps if fps else 0
            time_min = time_sec / 60

            out_name = (
                f"{mouse}_refinement_frame_{label}"
                f"_frame{frame_idx}_time{time_min:.2f}min.png"
            )

            mouse_out_path = mouse_frame_dir / out_name
            combined_out_path = COMBINED_FRAME_DIR / out_name

            cv2.imwrite(str(mouse_out_path), frame)
            shutil.copy2(mouse_out_path, combined_out_path)

            print(f"[OK] Saved mouse copy:    {mouse_out_path}")
            print(f"[OK] Saved training copy: {combined_out_path}")
            print(f"     frame {frame_idx}/{total_frames}, time {time_min:.2f} min")

    print("\nDONE.")
    print(f"Mouse-specific frames saved under:")
    print(f"{OUTPUT_ROOT}\\Mouse_XXX\\Refinement_Frames")
    print("\nCombined training frames saved under:")
    print(COMBINED_FRAME_DIR)


if __name__ == "__main__":
    main()
