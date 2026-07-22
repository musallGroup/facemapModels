"""
predict_frontview_full_video.py

Runs the custom 6-keypoint Facemap model on every frame of a selected
Cam0 video and saves the predictions as HDF5 plus metadata as JSON.

Facemap version:
    1.0.8

Current model:
    Frontview_v1_6kp.pt

Later, MODEL_PATH can be changed to:
    Frontview_Base_v2_6kp.pt
"""

from __future__ import annotations

import json
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any

import h5py
import numpy as np
import torch

from facemap.pose.pose import Pose


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Current_Model\Frontview_v1_6kp.pt"
)

OUTPUT_DIR = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Model_QC"
)

KEYPOINT_NAMES = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
]

# None means: process the complete video.
MAX_FRAMES: int | None = None

# Batch size 1 is the safest setting with the current CPU workflow.
BATCH_SIZE = 1


# ============================================================
# FILE SELECTION
# ============================================================

def select_video() -> Path:
    """Open a Windows dialog for selecting a Cam0 video."""

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_file = filedialog.askopenfilename(
        title="Select a complete Cam0 video for Facemap prediction",
        filetypes=[
            ("Video files", "*.avi *.mp4 *.mov *.mkv"),
            ("AVI files", "*.avi"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not selected_file:
        raise RuntimeError("No video was selected.")

    video_path = Path(selected_file)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Selected video does not exist:\n{video_path}"
        )

    return video_path


# ============================================================
# VALIDATION
# ============================================================

def validate_paths() -> None:
    """Check the model and create the output directory."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model does not exist:\n{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def inspect_model_file() -> None:
    """Confirm that the model contains six-output prediction heads."""

    print("\n========================================")
    print("MODEL FILE CHECK")
    print("========================================")
    print(f"Model: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "The model file does not contain a readable state dictionary."
        )

    required_heads = [
        "Conv2_1x1.conv0.weight",
        "Conv2_1x1.conv1.weight",
        "Conv2_1x1.conv2.weight",
    ]

    for head_name in required_heads:
        if head_name not in state_dict:
            raise KeyError(
                f"Required output head missing from model:\n{head_name}"
            )

        shape = tuple(state_dict[head_name].shape)

        print(f"  {head_name}: {shape}")

        if shape[0] != len(KEYPOINT_NAMES):
            raise RuntimeError(
                f"{head_name} has {shape[0]} outputs, "
                f"but {len(KEYPOINT_NAMES)} were expected."
            )


# ============================================================
# FACEMAP
# ============================================================

def create_pose_object(video_path: Path) -> Pose:
    """Create Facemap Pose and load the custom six-keypoint model."""

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

    # Must be assigned before pose_prediction_setup().
    pose.bodyparts = KEYPOINT_NAMES.copy()
    pose.batch_size = BATCH_SIZE

    print(f"Bodyparts:          {pose.bodyparts}")
    print(f"Number of outputs:  {len(pose.bodyparts)}")
    print(f"Batch size:         {pose.batch_size}")

    print("\n========================================")
    print("LOADING CUSTOM FACEMAP MODEL")
    print("========================================")

    pose.pose_prediction_setup()

    return pose


def determine_frame_indices(pose: Pose) -> np.ndarray:
    """Create the list of all video frame indices."""

    total_frames = int(pose.cumframes[-1])

    if total_frames <= 0:
        raise RuntimeError(
            "Facemap reports that the selected video contains no frames."
        )

    if MAX_FRAMES is None:
        frames_to_process = total_frames
    else:
        frames_to_process = min(
            int(MAX_FRAMES),
            total_frames,
        )

    frame_indices = np.arange(
        frames_to_process,
        dtype=np.int64,
    )

    print("\n========================================")
    print("FULL-VIDEO PREDICTION SETTINGS")
    print("========================================")
    print(f"Total video frames: {total_frames}")
    print(f"Frames processed:   {frames_to_process}")
    print(f"First frame:        {frame_indices[0]}")
    print(f"Last frame:         {frame_indices[-1]}")
    print(f"Keypoints:          {len(KEYPOINT_NAMES)}")
    print(f"Device:             {pose.device}")

    return frame_indices


# ============================================================
# OUTPUT NORMALIZATION
# ============================================================

def to_numpy(value: Any) -> np.ndarray:
    """Convert Torch tensors or array-like objects to NumPy."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def normalize_prediction_output(
    prediction_output: Any,
) -> tuple[np.ndarray, dict]:
    """Normalize Facemap's prediction return value."""

    metadata: dict = {}

    if isinstance(prediction_output, tuple):
        if not prediction_output:
            raise RuntimeError(
                "Facemap returned an empty prediction tuple."
            )

        predictions = to_numpy(
            prediction_output[0]
        )

        if (
            len(prediction_output) > 1
            and isinstance(prediction_output[1], dict)
        ):
            metadata = prediction_output[1]
    else:
        predictions = to_numpy(
            prediction_output
        )

    return predictions, metadata


def validate_prediction_shape(
    predictions: np.ndarray,
    number_of_frames: int,
) -> np.ndarray:
    """Require frames × keypoints × [x, y, likelihood]."""

    expected_shape = (
        number_of_frames,
        len(KEYPOINT_NAMES),
        3,
    )

    print("\nRaw prediction shape:", predictions.shape)
    print("Expected shape:      ", expected_shape)

    if predictions.shape == expected_shape:
        return predictions

    alternative_shape = (
        len(KEYPOINT_NAMES),
        number_of_frames,
        3,
    )

    if predictions.shape == alternative_shape:
        print("Transposing prediction axes.")

        return np.transpose(
            predictions,
            (1, 0, 2),
        )

    raise RuntimeError(
        "\nUnexpected prediction shape.\n"
        f"Received: {predictions.shape}\n"
        f"Expected: {expected_shape}"
    )


# ============================================================
# SAVING
# ============================================================

def save_predictions_hdf5(
    predictions: np.ndarray,
    frame_indices: np.ndarray,
    video_path: Path,
    output_path: Path,
) -> None:
    """Save x, y and likelihood values for all six keypoints."""

    print("\nSaving HDF5 predictions...")

    with h5py.File(
        output_path,
        "w",
    ) as h5_file:

        h5_file.attrs["video_path"] = str(video_path)
        h5_file.attrs["model_path"] = str(MODEL_PATH)
        h5_file.attrs["number_of_keypoints"] = len(KEYPOINT_NAMES)

        h5_file.create_dataset(
            "frame_index",
            data=frame_indices,
            compression="gzip",
        )

        facemap_group = h5_file.create_group(
            "Facemap"
        )

        for keypoint_index, keypoint_name in enumerate(
            KEYPOINT_NAMES
        ):
            keypoint_group = facemap_group.create_group(
                keypoint_name
            )

            keypoint_group.create_dataset(
                "x",
                data=predictions[:, keypoint_index, 0],
                compression="gzip",
            )

            keypoint_group.create_dataset(
                "y",
                data=predictions[:, keypoint_index, 1],
                compression="gzip",
            )

            keypoint_group.create_dataset(
                "likelihood",
                data=predictions[:, keypoint_index, 2],
                compression="gzip",
            )


def make_json_serializable(value: Any) -> Any:
    """Convert NumPy and Torch values for JSON storage."""

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
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def save_metadata_json(
    pose: Pose,
    metadata: dict,
    frame_indices: np.ndarray,
    video_path: Path,
    output_path: Path,
) -> None:
    """Save model and prediction metadata."""

    output_metadata = {
        "video_path": str(video_path),
        "model_path": str(MODEL_PATH),
        "facemap_version_target": "1.0.8",
        "bodyparts": KEYPOINT_NAMES,
        "number_of_keypoints": len(KEYPOINT_NAMES),
        "processed_frames": int(len(frame_indices)),
        "first_frame": int(frame_indices[0]),
        "last_frame": int(frame_indices[-1]),
        "batch_size": int(BATCH_SIZE),
        "device": str(pose.device),
        "bbox": getattr(pose, "bbox", None),
        "resize": getattr(pose, "resize", None),
        "add_padding": getattr(pose, "add_padding", None),
        "facemap_prediction_metadata": metadata,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            make_json_serializable(output_metadata),
            json_file,
            indent=4,
        )


def print_likelihood_summary(
    predictions: np.ndarray,
) -> None:
    """Print confidence statistics after inference."""

    print("\n========================================")
    print("LIKELIHOOD SUMMARY")
    print("========================================")

    for keypoint_index, keypoint_name in enumerate(
        KEYPOINT_NAMES
    ):
        likelihood = predictions[
            :,
            keypoint_index,
            2,
        ]

        print(
            f"{keypoint_name:<20} "
            f"mean={np.nanmean(likelihood):.4f}  "
            f"min={np.nanmin(likelihood):.4f}  "
            f"max={np.nanmax(likelihood):.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run complete-video Facemap inference."""

    print("\n========================================")
    print("FRONTVIEW FACEMAP — FULL VIDEO")
    print("========================================")

    validate_paths()
    inspect_model_file()

    print("\nSelect the Cam0 video to process.")
    video_path = select_video()

    print("\nSelected video:")
    print(video_path)

    pose = create_pose_object(
        video_path
    )

    frame_indices = determine_frame_indices(
        pose
    )

    print("\n========================================")
    print("STARTING FULL-VIDEO PREDICTION")
    print("========================================")
    print(
        "This may take many hours on CPU. "
        "Do not close the terminal."
    )

    prediction_output = pose.predict_landmarks(
        video_id=0,
        frame_ind=frame_indices,
    )

    predictions, metadata = normalize_prediction_output(
        prediction_output
    )

    predictions = validate_prediction_shape(
        predictions=predictions,
        number_of_frames=len(frame_indices),
    )

    output_stem = (
        f"{video_path.stem}_"
        f"{MODEL_PATH.stem}_FULL"
    )

    h5_output_path = (
        OUTPUT_DIR / f"{output_stem}.h5"
    )

    metadata_output_path = (
        OUTPUT_DIR / f"{output_stem}_metadata.json"
    )

    save_predictions_hdf5(
        predictions=predictions,
        frame_indices=frame_indices,
        video_path=video_path,
        output_path=h5_output_path,
    )

    save_metadata_json(
        pose=pose,
        metadata=metadata,
        frame_indices=frame_indices,
        video_path=video_path,
        output_path=metadata_output_path,
    )

    print_likelihood_summary(
        predictions
    )

    print("\n========================================")
    print("FULL-VIDEO PREDICTION COMPLETE")
    print("========================================")
    print("\nPredictions saved to:")
    print(h5_output_path)
    print("\nMetadata saved to:")
    print(metadata_output_path)


if __name__ == "__main__":
    main()