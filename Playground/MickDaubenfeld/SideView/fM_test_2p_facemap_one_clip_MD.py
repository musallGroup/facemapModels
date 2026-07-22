from pathlib import Path
import cv2
import subprocess
import sys

VIDEO_ROOT = Path(r"\\naskampa\data\BpodBehavior\2pvideos")
MODEL_PATH = Path(r"\\Naskampa\lts\Team\Mick\Facemap_Models\Current_Model\refined_model_with_Dennis12Frames.pt")

OUT_ROOT = Path(r"Z:\Team\Mick\2 Photon")
CLIP_SECONDS = 60
START_SECONDS = 0

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mj2", ".mov"]


def find_first_video(folder):
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.rglob(f"*{ext}"))
    if not videos:
        raise FileNotFoundError(f"No videos found in {folder}")
    return sorted(videos)[0]


def make_clip(video_path, out_path, start_sec=0, duration_sec=60):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    n_frames = int(duration_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    written = 0
    while written < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    print(f"Saved clip: {out_path}")
    print(f"Frames written: {written}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    clip_dir = OUT_ROOT / "QC_01_one_min_clip"
    clip_dir.mkdir(parents=True, exist_ok=True)

    video_path = find_first_video(VIDEO_ROOT)

    clip_path = clip_dir / f"{video_path.stem}_1min_test.mp4"

    print("Input video:")
    print(video_path)
    print("\nModel:")
    print(MODEL_PATH)
    print("\nOutput:")
    print(clip_path)

    make_clip(video_path, clip_path, START_SECONDS, CLIP_SECONDS)

    print("\nNext step:")
    print("Open Facemap GUI in your facemap environment.")
    print("Load this clip:")
    print(clip_path)
    print("Set output folder to:")
    print(clip_dir)
    print("Select Pose model:")
    print(MODEL_PATH)
    print("Check Keypoints and click Process.")
    print("\nFacemap should save a *_FacemapPose.h5 file in the output folder.")


if __name__ == "__main__":
    main()