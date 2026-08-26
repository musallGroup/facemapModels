import cv2
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

FRAME_DIR = Path(
    r"\\Naskampa\lts\Team\Mick\FM_Side_View\FM_Dennis_Cohort\Refinement_Training\Dennis12_Frames"
)

OUTPUT_VIDEO = Path(
    r"\\Naskampa\lts\Team\Mick\FM_Side_View\FM_Dennis_Cohort\Refinement_Training\Dennis12_refinement_video.avi"
)

FPS = 5

# ============================================================
# BUILD VIDEO
# ============================================================

frames = sorted(FRAME_DIR.glob("*.png"))

if len(frames) == 0:
    raise RuntimeError(f"No PNG files found in:\n{FRAME_DIR}")

print(f"Found {len(frames)} frames")

first_frame = cv2.imread(str(frames[0]))

height, width = first_frame.shape[:2]

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"XVID"),
    FPS,
    (width, height)
)

for frame_path in frames:

    img = cv2.imread(str(frame_path))

    if img is None:
        print(f"[WARNING] Could not read: {frame_path}")
        continue

    writer.write(img)

    print(f"[ADD] {frame_path.name}")

writer.release()

print("\nDONE")
print(f"Video saved to:\n{OUTPUT_VIDEO}")
