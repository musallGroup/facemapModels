from pathlib import Path
import cv2
import numpy as np
from facemap.pose.pose import Pose

CLIP_DIR = Path(r"Z:\Team\Mick\2 Photon\QC_01_one_min_clip")
MODEL_PATH = Path(r"Z:\Team\Mick\Facemap_Models\Current_Model\refined_model_with_Dennis12Frames.pt")

OUT_DIR = Path(r"Z:\Team\Mick\2 Photon\QC_02_facemap_overlay")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_LABELS = {"paw", "nose(r)"}
VIDEO_EXTENSIONS = [".mp4", ".avi", ".mj2", ".mov"]


def find_clip(folder):
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(folder.glob(f"*{ext}"))
    if not videos:
        raise FileNotFoundError(f"No video found in {folder}")
    return sorted(videos)[0]


def get_full_frame_bbox(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    return [[0, height, 0, width]]


def make_overlay_video(video_path, keypoints, labels, out_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints):
            break

        pts = keypoints[frame_idx]

        for i, label in enumerate(labels):
            if label in EXCLUDE_LABELS:
                continue
            if i >= pts.shape[0]:
                continue

            x, y = pts[i, 0], pts[i, 1]
            likelihood = pts[i, 2]

            if np.isnan(x) or np.isnan(y):
                continue

            x = int(round(x))
            y = int(round(y))

            if x < 0 or y < 0 or x >= width or y >= height:
                continue

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(
                frame,
                label,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print("Saved overlay video:")
    print(out_path)


def main():
    clip_path = find_clip(CLIP_DIR)

    print("Clip:")
    print(clip_path)

    print("\nModel:")
    print(MODEL_PATH)

    bbox = get_full_frame_bbox(clip_path)

    print("\nUsing full-frame bbox:")
    print(bbox)

    pose = Pose(
        filenames=[[str(clip_path)]],
        bbox=bbox,
        bbox_set=True,
        resize=True,
        add_padding=True,
        model_name=str(MODEL_PATH),
    )

    print("\nLoading Facemap model...")
    pose.load_model()

    if pose.net is None:
        raise RuntimeError("Facemap model loading failed: pose.net is still None.")

    print("Model loaded successfully.")

    print("\nRunning prediction...")
    pred_data, metadata = pose.predict_landmarks(video_id=0)

    keypoints = pred_data.cpu().numpy()
    labels = metadata["bodyparts"]

    print("\nPrediction output:")
    print("Keypoints shape:", keypoints.shape)
    print("Labels:", labels)

    overlay_path = OUT_DIR / f"{clip_path.stem}_OVERLAY_no_paw_no_noser.mp4"

    print("\nCreating overlay...")
    make_overlay_video(clip_path, keypoints, labels, overlay_path)

    print("\nDONE")
    print("Overlay video:")
    print(overlay_path)


if __name__ == "__main__":
    main()