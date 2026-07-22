from pathlib import Path
import cv2
import random

PROJECT_ROOT = Path(r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon")

INPUT_VIDEO = PROJECT_ROOT / "Training_Data" / "Training_Videos" / "cam0_frontview_training_mix.avi"
OUTPUT_DIR = PROJECT_ROOT / "Training_Data" / "Extracted_Frames" / "Round_01"

N_FRAMES = 30
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(INPUT_VIDEO))
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video: {INPUT_VIDEO}")
print(f"Total frames: {total_frames}")
print(f"FPS: {fps:.2f}")

random.seed(RANDOM_SEED)
frame_indices = sorted(random.sample(range(total_frames), min(N_FRAMES, total_frames)))

for i, frame_idx in enumerate(frame_indices, start=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    if not ret:
        print(f"Could not read frame {frame_idx}")
        continue

    out_path = OUTPUT_DIR / f"frame_{i:03d}_source_{frame_idx:06d}.png"
    cv2.imwrite(str(out_path), frame)
    print(f"Saved: {out_path}")

cap.release()

print("\nDone.")
print(f"Frames saved to: {OUTPUT_DIR}")