"""
predict_frontview_v1.py

Facemap inference for the custom Frontview_v1_6kp model.

Workflow:
1. Select a Cam0 video using a file dialog.
2. Load the custom Facemap model.
3. Predict six facial keypoints for the first MAX_FRAMES frames.
4. Save x, y and likelihood values as HDF5.
5. Save metadata as JSON.

Target Facemap version:
    facemap 1.0.8
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from tkinter import Tk, filedialog

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

# For the first test, only process 500 frames.
# Change to None later to process the complete video.
MAX_FRAMES: int | None = 500

BATCH_SIZE = 1


# ============================================================
# FILE SELECTION
# ============================================================

def select_video() -> Path:
    """Open a Windows file dialog and return the selected video."""

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_file = filedialog.askopenfilename(
        title="Select a Cam0 video for Facemap prediction",
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

def validate_model_path() -> None:
    """Check that the trained model exists."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model does not exist:\n{MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_model_file_information() -> None:
    """
    Print basic information about the saved model.

    This does not modify or load the model into Facemap. It only confirms
    that the file can be opened by PyTorch.
    """

    print("\n========================================")
    print("MODEL FILE CHECK")
    print("========================================")
    print(f"Model: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    print(f"Loaded object type: {type(checkpoint).__name__}")

    if isinstance(checkpoint, dict):
        print(f"Top-level entries: {len(checkpoint)}")

        possible_state_dicts = []

        if "state_dict" in checkpoint:
            possible_state_dicts.append(checkpoint["state_dict"])

        if "model_state_dict" in checkpoint:
            possible_state_dicts.append(checkpoint["model_state_dict"])

        possible_state_dicts.append(checkpoint)

        head_found = False

        for possible_state_dict in possible_state_dicts:
            if not isinstance(possible_state_dict, dict):
                continue

            for key, value in possible_state_dict.items():
                if (
                    "Conv2_1x1" in str(key)
                    and str(key).endswith("weight")
                    and hasattr(value, "shape")
                ):
                    print(f"  {key}: {tuple(value.shape)}")
                    head_found = True

        if not head_found:
            print(
                "No output-head tensors were identified automatically.\n"
                "This is not necessarily an error; Facemap will attempt "
                "to load the model next."
            )


# ============================================================
# FACEMAP SETUP
# ============================================================

def create_pose_object(video_path: Path) -> Pose:
    """
    Create the Facemap Pose object and load the custom model.

    The six bodypart names are assigned before Facemap constructs the
    prediction network.
    """

    filenames = [[str(video_path)]]

    print("\n========================================")
    print("CREATING FACEMAP POSE OBJECT")
    print("========================================")

    pose = Pose(
        filenames=filenames,
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
    print(f"Number of keypoints: {len(pose.bodyparts)}")
    print(f"Batch size: {pose.batch_size}")

    print("\n========================================")
    print("LOADING CUSTOM FACEMAP MODEL")
    print("========================================")

    pose.pose_prediction_setup()

    return pose


# ============================================================
# FRAME SELECTION
# ============================================================

def determine_frame_indices(pose: Pose) -> np.ndarray:
    """Return the frame indices that will be processed."""

    total_frames = int(pose.cumframes[-1])

    if total_frames <= 0:
        raise RuntimeError("Facemap reports that the video has no frames.")

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
    print("PREDICTION SETTINGS")
    print("========================================")
    print(f"Total video frames: {total_frames}")
    print(f"Frames processed:   {frames_to_process}")
    print(f"First frame:        {frame_indices[0]}")
    print(f"Last frame:         {frame_indices[-1]}")
    print(f"Keypoints:          {len(KEYPOINT_NAMES)}")
    print(f"Device:             {pose.device}")

    return frame_indices


# ============================================================
# PREDICTION CONVERSION
# ============================================================

def to_numpy(value: Any) -> np.ndarray:
    """Convert a PyTorch tensor or array-like object to NumPy."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def normalize_prediction_output(
    prediction_output: Any,
) -> tuple[np.ndarray, dict]:
    """
    Normalize Facemap's prediction return value.

    Expected Facemap output:
        prediction tensor/array
        optional metadata dictionary
    """

    metadata: dict = {}

    if isinstance(prediction_output, tuple):
        if len(prediction_output) == 0:
            raise RuntimeError(
                "Facemap returned an empty prediction tuple."
            )

        predictions = to_numpy(prediction_output[0])

        if len(prediction_output) > 1:
            if isinstance(prediction_output[1], dict):
                metadata = prediction_output[1]
            else:
                metadata["additional_output"] = str(
                    type(prediction_output[1]).__name__
                )
    else:
        predictions = to_numpy(prediction_output)

    return predictions, metadata


def validate_prediction_shape(
    predictions: np.ndarray,
    number_of_frames: int,
) -> np.ndarray:
    """
    Validate and, where unambiguous, correct the prediction dimensions.

    Required final shape:
        frames x keypoints x 3

    The final dimension contains:
        x, y, likelihood
    """

    print("\nRaw prediction shape:", predictions.shape)

    expected_shape = (
        number_of_frames,
        len(KEYPOINT_NAMES),
        3,
    )

    if predictions.shape == expected_shape:
        return predictions

    # Some implementations may return keypoints x frames x 3.
    alternative_shape = (
        len(KEYPOINT_NAMES),
        number_of_frames,
        3,
    )

    if predictions.shape == alternative_shape:
        predictions = np.transpose(
            predictions,
            (1, 0, 2),
        )

        print(
            "Prediction axes were transposed to "
            "frames x keypoints x values."
        )

        return predictions

    raise RuntimeError(
        "\nUnexpected prediction shape.\n"
        f"Received: {predictions.shape}\n"
        f"Expected: {expected_shape}\n\n"
        "Facemap inference ran, but the returned data format differs "
        "from the expected frames x keypoints x 3 format."
    )


# ============================================================
# OUTPUT
# ============================================================

def save_predictions_hdf5(
    predictions: np.ndarray,
    frame_indices: np.ndarray,
    video_path: Path,
    output_path: Path,
) -> None:
    """Save x, y and likelihood for each keypoint."""

    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["video_path"] = str(video_path)
        h5_file.attrs["model_path"] = str(MODEL_PATH)
        h5_file.attrs["number_of_keypoints"] = len(KEYPOINT_NAMES)

        h5_file.create_dataset(
            "frame_index",
            data=frame_indices,
        )

        facemap_group = h5_file.create_group("Facemap")

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
    """Convert common NumPy and PyTorch values for JSON output."""

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
    """Save prediction settings and Facemap metadata."""

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
    """Print likelihood statistics for every keypoint."""

    print("\n========================================")
    print("LIKELIHOOD SUMMARY")
    print("========================================")

    for keypoint_index, keypoint_name in enumerate(
        KEYPOINT_NAMES
    ):
        likelihood = predictions[:, keypoint_index, 2]

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
    """Run the complete Facemap prediction pipeline."""

    print("\n========================================")
    print("FRONTVIEW FACEMAP PREDICTION")
    print("========================================")

    validate_model_path()

    print("\nSelect a Cam0 video in the file window.")
    video_path = select_video()

    print(f"\nSelected video:\n{video_path}")

    print_model_file_information()

    pose = create_pose_object(video_path)
    frame_indices = determine_frame_indices(pose)

    print("\n========================================")
    print("STARTING FACEMAP PREDICTION")
    print("========================================")

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
        f"{video_path.stem}_Frontview_v1_6kp"
    )

    prediction_output_path = (
        OUTPUT_DIR / f"{output_stem}.h5"
    )

    metadata_output_path = (
        OUTPUT_DIR / f"{output_stem}_metadata.json"
    )

    save_predictions_hdf5(
        predictions=predictions,
        frame_indices=frame_indices,
        video_path=video_path,
        output_path=prediction_output_path,
    )

    save_metadata_json(
        pose=pose,
        metadata=metadata,
        frame_indices=frame_indices,
        video_path=video_path,
        output_path=metadata_output_path,
    )

    print_likelihood_summary(predictions)

    print("\n========================================")
    print("PREDICTION COMPLETE")
    print("========================================")
    print("\nPredictions saved to:")
    print(prediction_output_path)
    print("\nMetadata saved to:")
    print(metadata_output_path)


if __name__ == "__main__":
    main()