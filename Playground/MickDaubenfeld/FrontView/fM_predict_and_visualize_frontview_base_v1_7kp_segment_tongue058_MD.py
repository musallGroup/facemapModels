"""
predict_and_visualize_frontview_base_v1_7kp_segment.py

Runs Frontview_Base_v1_7kp.pt on a selected short Cam0 segment and saves:
- HDF5 predictions
- JSON metadata
- MP4 QC overlay

Default duration: 60 seconds.

QC thresholds:
- all established keypoints: 0.80
- tongue_tip: 0.58

The output filenames include:
    TONGUE-THRESH-058

Facemap target: 1.0.8
"""

from __future__ import annotations

import json
from pathlib import Path
from tkinter import Tk, filedialog, simpledialog
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from facemap.pose.pose import Pose

PROJECT_ROOT = Path(r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon")
MODEL_PATH = PROJECT_ROOT / "Current_Model" / "Frontview_Base_v1_7kp.pt"
OUTPUT_DIR = PROJECT_ROOT / "Model_QC"

KEYPOINT_NAMES = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
    "tongue_tip",
]

DEFAULT_START_MINUTES = 14.0
DEFAULT_DURATION_SECONDS = 60.0
BATCH_SIZE = 1
GENERAL_LIKELIHOOD_THRESHOLD = 0.80
TONGUE_LIKELIHOOD_THRESHOLD = 0.58
POINT_RADIUS = 6
TONGUE_RADIUS = 4
FONT_SCALE = 0.50
LINE_THICKNESS = 2


def select_video() -> Path:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Select the original Cam0 video",
        filetypes=[
            ("Video files", "*.avi *.mp4 *.mov *.mkv"),
            ("AVI files", "*.avi"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise RuntimeError("No video was selected.")
    path = Path(selected)
    if not path.exists():
        raise FileNotFoundError(f"Selected video does not exist:\n{path}")
    return path


def ask_segment_settings(video_duration_seconds: float) -> tuple[float, float]:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    latest_start_minutes = max(
        0.0,
        (video_duration_seconds - DEFAULT_DURATION_SECONDS) / 60.0,
    )

    start_minutes = simpledialog.askfloat(
        title="Segment start",
        prompt=(
            "Start time in minutes:\n\n"
            f"Video duration: {video_duration_seconds / 60.0:.2f} min\n\n"
            "For the known 428 tongue sequence, 14.0 min is useful."
        ),
        initialvalue=DEFAULT_START_MINUTES,
        minvalue=0.0,
        maxvalue=latest_start_minutes,
        parent=root,
    )
    if start_minutes is None:
        root.destroy()
        raise RuntimeError("Segment selection was cancelled.")

    duration_seconds = simpledialog.askfloat(
        title="Segment duration",
        prompt="Duration in seconds (60 recommended):",
        initialvalue=DEFAULT_DURATION_SECONDS,
        minvalue=1.0,
        maxvalue=max(1.0, video_duration_seconds - start_minutes * 60.0),
        parent=root,
    )
    root.destroy()
    if duration_seconds is None:
        raise RuntimeError("Segment duration selection was cancelled.")

    return float(start_minutes), float(duration_seconds)


def validate_paths() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model does not exist:\n{MODEL_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_model_file() -> None:
    print("\n========================================")
    print("MODEL FILE CHECK")
    print("========================================")
    print(f"Model: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    required_heads = [
        "Conv2_1x1.conv0.weight",
        "Conv2_1x1.conv1.weight",
        "Conv2_1x1.conv2.weight",
    ]

    for head_name in required_heads:
        if head_name not in state_dict:
            raise KeyError(f"Missing model head:\n{head_name}")
        shape = tuple(state_dict[head_name].shape)
        print(f"  {head_name}: {shape}")
        if shape[0] != len(KEYPOINT_NAMES):
            raise RuntimeError(
                f"{head_name} has {shape[0]} outputs, expected {len(KEYPOINT_NAMES)}."
            )


def read_video_information(video_path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video:\n{video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("The video reports invalid metadata.")

    return fps, total_frames, width, height


def create_frame_indices(
    fps: float,
    total_frames: int,
    start_minutes: float,
    duration_seconds: float,
) -> np.ndarray:
    start_frame = int(round(start_minutes * 60.0 * fps))
    start_frame = max(0, start_frame)
    if start_frame >= total_frames:
        raise RuntimeError("The selected start time lies outside the video.")

    requested_frames = int(round(duration_seconds * fps))
    end_frame_exclusive = min(start_frame + requested_frames, total_frames)
    if end_frame_exclusive <= start_frame:
        raise RuntimeError("The selected segment contains no frames.")

    return np.arange(start_frame, end_frame_exclusive, dtype=np.int64)


def create_pose_object(video_path: Path) -> Pose:
    print("\n========================================")
    print("CREATING FACEMAP POSE OBJECT")
    print("========================================")

    pose = Pose(
        filenames=[[str(video_path)]],
        bbox=[],
        bbox_set=False,
        resize=False,
        add_padding=False,
        gui=None,
        GUIobject=None,
        net=None,
        model_name=str(MODEL_PATH),
    )
    pose.bodyparts = KEYPOINT_NAMES.copy()
    pose.batch_size = BATCH_SIZE

    print(f"Bodyparts: {pose.bodyparts}")
    print(f"Keypoints: {len(pose.bodyparts)}")
    print(f"Batch size: {pose.batch_size}")

    print("\n========================================")
    print("LOADING CUSTOM FACEMAP MODEL")
    print("========================================")
    pose.pose_prediction_setup()
    return pose


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_prediction_output(prediction_output: Any) -> tuple[np.ndarray, dict]:
    metadata: dict = {}
    if isinstance(prediction_output, tuple):
        if not prediction_output:
            raise RuntimeError("Facemap returned an empty prediction tuple.")
        predictions = to_numpy(prediction_output[0])
        if len(prediction_output) > 1 and isinstance(prediction_output[1], dict):
            metadata = prediction_output[1]
    else:
        predictions = to_numpy(prediction_output)
    return predictions, metadata


def validate_prediction_shape(
    predictions: np.ndarray,
    number_of_frames: int,
) -> np.ndarray:
    expected_shape = (number_of_frames, len(KEYPOINT_NAMES), 3)
    print(f"\nRaw prediction shape: {predictions.shape}")
    print(f"Expected shape:       {expected_shape}")

    if predictions.shape == expected_shape:
        return predictions

    alternative_shape = (len(KEYPOINT_NAMES), number_of_frames, 3)
    if predictions.shape == alternative_shape:
        return np.transpose(predictions, (1, 0, 2))

    raise RuntimeError(
        f"Unexpected prediction shape: {predictions.shape}; expected {expected_shape}"
    )


def save_predictions_h5(
    output_path: Path,
    video_path: Path,
    frame_indices: np.ndarray,
    predictions: np.ndarray,
) -> None:
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["video_path"] = str(video_path)
        h5_file.attrs["model_path"] = str(MODEL_PATH)
        h5_file.attrs["number_of_keypoints"] = len(KEYPOINT_NAMES)
        h5_file.create_dataset("frame_index", data=frame_indices, compression="gzip")

        facemap_group = h5_file.create_group("Facemap")
        for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
            keypoint_group = facemap_group.create_group(keypoint_name)
            keypoint_group.create_dataset(
                "x", data=predictions[:, keypoint_index, 0], compression="gzip"
            )
            keypoint_group.create_dataset(
                "y", data=predictions[:, keypoint_index, 1], compression="gzip"
            )
            keypoint_group.create_dataset(
                "likelihood",
                data=predictions[:, keypoint_index, 2],
                compression="gzip",
            )


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_metadata_json(
    output_path: Path,
    video_path: Path,
    frame_indices: np.ndarray,
    pose: Pose,
    prediction_metadata: dict,
) -> None:
    metadata = {
        "video_path": str(video_path),
        "model_path": str(MODEL_PATH),
        "facemap_version_target": "1.0.8",
        "keypoints": KEYPOINT_NAMES,
        "number_of_keypoints": len(KEYPOINT_NAMES),
        "processed_frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "batch_size": BATCH_SIZE,
        "device": str(pose.device),
        "facemap_prediction_metadata": prediction_metadata,
    }

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(make_json_serializable(metadata), json_file, indent=4)


def create_colors(number_of_keypoints: int) -> list[tuple[int, int, int]]:
    hues = np.linspace(0, 179, number_of_keypoints, endpoint=False, dtype=np.uint8)
    colors: list[tuple[int, int, int]] = []
    for hue in hues:
        hsv_pixel = np.uint8([[[hue, 220, 255]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(int(value) for value in bgr_pixel))
    return colors


def draw_cross(
    frame: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    size = radius + 2
    cv2.line(
        frame,
        (x - size, y - size),
        (x + size, y + size),
        color,
        LINE_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        (x - size, y + size),
        (x + size, y - size),
        color,
        LINE_THICKNESS,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame: np.ndarray,
    prediction_index: int,
    source_frame_index: int,
    predictions: np.ndarray,
    colors: list[tuple[int, int, int]],
    source_fps: float,
) -> np.ndarray:
    overlay = frame.copy()
    frame_height, frame_width = overlay.shape[:2]

    for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
        x_float, y_float, likelihood = predictions[prediction_index, keypoint_index]

        if not (
            np.isfinite(x_float)
            and np.isfinite(y_float)
            and np.isfinite(likelihood)
        ):
            continue

        x = int(round(float(x_float)))
        y = int(round(float(y_float)))
        if not (0 <= x < frame_width and 0 <= y < frame_height):
            continue

        color = colors[keypoint_index]
        radius = TONGUE_RADIUS if keypoint_name == "tongue_tip" else POINT_RADIUS

        threshold = (
            TONGUE_LIKELIHOOD_THRESHOLD
            if keypoint_name == "tongue_tip"
            else GENERAL_LIKELIHOOD_THRESHOLD
        )

        if likelihood >= threshold:
            cv2.circle(
                overlay,
                (x, y),
                radius,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                (x, y),
                radius + 2,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        else:
            draw_cross(overlay, x, y, color, radius)

        label = f"{keypoint_name} {likelihood:.2f}"
        text_x = min(x + 9, max(0, frame_width - 220))
        text_y = max(20, y - 8)
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    session_seconds = source_frame_index / source_fps
    header = (
        f"Frame: {source_frame_index}   "
        f"Session time: {session_seconds / 60.0:.3f} min   "
        f"General threshold: {GENERAL_LIKELIHOOD_THRESHOLD:.2f}   "
        f"Tongue threshold: {TONGUE_LIKELIHOOD_THRESHOLD:.2f}"
    )
    cv2.putText(
        overlay,
        header,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        thickness=4,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        header,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return overlay


def create_qc_video(
    output_path: Path,
    video_path: Path,
    frame_indices: np.ndarray,
    predictions: np.ndarray,
    source_fps: float,
    frame_width: int,
    frame_height: int,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video:\n{video_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video:\n{output_path}")

    colors = create_colors(len(KEYPOINT_NAMES))

    print("\n========================================")
    print("CREATING QC VIDEO")
    print("========================================")

    try:
        for prediction_index, source_frame_index in enumerate(frame_indices):
            source_frame_index = int(source_frame_index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not read source frame {source_frame_index}.")

            overlay = draw_overlay(
                frame,
                prediction_index,
                source_frame_index,
                predictions,
                colors,
                source_fps,
            )
            writer.write(overlay)

            processed = prediction_index + 1
            total = len(frame_indices)
            if processed == 1 or processed % 250 == 0 or processed == total:
                print(f"Processed {processed}/{total} frames")
    finally:
        capture.release()
        writer.release()


def print_likelihood_summary(predictions: np.ndarray) -> None:
    print("\n========================================")
    print("LIKELIHOOD SUMMARY")
    print("========================================")

    for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
        likelihood = predictions[:, keypoint_index, 2]
        print(
            f"{keypoint_name:<20} "
            f"mean={np.nanmean(likelihood):.4f}  "
            f"min={np.nanmin(likelihood):.4f}  "
            f"max={np.nanmax(likelihood):.4f}"
        )


def main() -> None:
    print("\n========================================")
    print("FRONTVIEW BASE V1 — SEGMENT QC")
    print("7 KEYPOINTS")
    print("========================================")

    validate_paths()
    inspect_model_file()

    print("\nSelect the original Cam0 video.")
    video_path = select_video()

    source_fps, total_frames, frame_width, frame_height = read_video_information(
        video_path
    )
    video_duration_seconds = total_frames / source_fps

    print("\nSelected video:")
    print(video_path)
    print(f"\nVideo duration: {video_duration_seconds / 60.0:.2f} min")

    start_minutes, duration_seconds = ask_segment_settings(video_duration_seconds)
    frame_indices = create_frame_indices(
        source_fps,
        total_frames,
        start_minutes,
        duration_seconds,
    )

    actual_start_seconds = frame_indices[0] / source_fps
    actual_end_seconds = frame_indices[-1] / source_fps

    print("\n========================================")
    print("SEGMENT SETTINGS")
    print("========================================")
    print(f"Start time:       {actual_start_seconds / 60.0:.3f} min")
    print(f"End time:         {actual_end_seconds / 60.0:.3f} min")
    print(f"Duration:         {len(frame_indices) / source_fps:.2f} s")
    print(f"Frames processed: {len(frame_indices)}")
    print(f"First frame:      {frame_indices[0]}")
    print(f"Last frame:       {frame_indices[-1]}")

    pose = create_pose_object(video_path)

    print("\n========================================")
    print("STARTING FACEMAP PREDICTION")
    print("========================================")

    prediction_output = pose.predict_landmarks(
        video_id=0,
        frame_ind=frame_indices,
    )
    predictions, prediction_metadata = normalize_prediction_output(prediction_output)
    predictions = validate_prediction_shape(predictions, len(frame_indices))

    output_stem = (
        f"{video_path.stem}_"
        f"{MODEL_PATH.stem}_"
        f"{actual_start_seconds / 60.0:.2f}-"
        f"{actual_end_seconds / 60.0:.2f}min_"
        f"TONGUE-THRESH-058"
    )

    h5_output_path = OUTPUT_DIR / f"{output_stem}.h5"
    json_output_path = OUTPUT_DIR / f"{output_stem}_metadata.json"
    video_output_path = OUTPUT_DIR / f"{output_stem}_QC.mp4"

    print("\nSaving HDF5 predictions...")
    save_predictions_h5(
        h5_output_path,
        video_path,
        frame_indices,
        predictions,
    )
    save_metadata_json(
        json_output_path,
        video_path,
        frame_indices,
        pose,
        prediction_metadata,
    )
    create_qc_video(
        video_output_path,
        video_path,
        frame_indices,
        predictions,
        source_fps,
        frame_width,
        frame_height,
    )
    print_likelihood_summary(predictions)

    print("\n========================================")
    print("SEGMENT QC COMPLETE")
    print("========================================")
    print("\nPredictions:")
    print(h5_output_path)
    print("\nMetadata:")
    print(json_output_path)
    print("\nQC video:")
    print(video_output_path)


if __name__ == "__main__":
    main()