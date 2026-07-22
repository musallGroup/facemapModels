from pathlib import Path
import csv
import json
import random

import cv2
import numpy as np
import torch
import torch.nn as nn

from facemap.pose.pose import Pose
import facemap.pose.pose as pose_module
from facemap.pose import datasets
from facemap.pose import model_training


# ============================================================
# Fix für fehlende Imports in Facemap 1.0.8
# ============================================================

pose_module.datasets = datasets
pose_module.model_training = model_training


# ============================================================
# Pfade
# ============================================================

PROJECT_ROOT = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon"
)

FRAME_DIR = (
    PROJECT_ROOT
    / "Training_Data"
    / "Extracted_Frames"
    / "Round_01"
)

LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Round_01"
    / "frontview_labels_round01.csv"
)

TRAINING_VIDEO = (
    PROJECT_ROOT
    / "Training_Data"
    / "Training_Videos"
    / "cam0_frontview_training_mix.avi"
)

MODEL_DIR = PROJECT_ROOT / "Current_Model"

MODEL_STATE_OUT = MODEL_DIR / "Frontview_v1_6kp.pt"
MODEL_INFO_OUT = MODEL_DIR / "Frontview_v1_6kp_model_info.json"


# ============================================================
# Trainingskonfiguration
# ============================================================

TRAIN_LABELS = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
]

NUM_KEYPOINTS = len(TRAIN_LABELS)

NUM_EPOCHS = 80
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

RANDOM_SEED = 42


# ============================================================
# Hilfsfunktionen
# ============================================================

def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found:\n{csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_has_all_training_labels(row: dict) -> bool:
    if str(row.get("skipped", "0")).strip() == "1":
        return False

    for label in TRAIN_LABELS:
        visible = str(row.get(f"{label}_visible", "")).strip()
        x_value = str(row.get(f"{label}_x", "")).strip()
        y_value = str(row.get(f"{label}_y", "")).strip()

        if visible != "1":
            return False

        if x_value == "" or y_value == "":
            return False

    return True


def load_training_data(
    frame_dir: Path,
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = load_csv_rows(csv_path)

    images = []
    keypoints = []
    used_frames = []

    for row in rows:
        if not row_has_all_training_labels(row):
            continue

        frame_name = row["frame"]
        frame_path = frame_dir / frame_name

        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"WARNING: Could not read frame:\n{frame_path}")
            continue

        frame_keypoints = []

        for label in TRAIN_LABELS:
            x = float(row[f"{label}_x"])
            y = float(row[f"{label}_y"])
            frame_keypoints.append([x, y])

        images.append(image)
        keypoints.append(frame_keypoints)
        used_frames.append(frame_name)

    if not images:
        raise RuntimeError(
            "No valid training frames found. "
            "Check visibility flags and file paths."
        )

    image_data = np.stack(images, axis=0)
    keypoints_data = np.asarray(keypoints, dtype=np.float32)

    return image_data, keypoints_data, used_frames


def replace_facemap_output_heads(
    network: nn.Module,
    num_keypoints: int,
) -> None:
    """
    Ersetzt die drei Facemap-Heatmap-Heads:

        Conv2_1x1.conv0
        Conv2_1x1.conv1
        Conv2_1x1.conv2

    von 15 auf die gewünschte Anzahl Keypoints.
    """

    if not hasattr(network, "Conv2_1x1"):
        raise AttributeError(
            "Facemap network has no attribute 'Conv2_1x1'."
        )

    head_names = ["conv0", "conv1", "conv2"]

    for head_name in head_names:
        old_head = getattr(network.Conv2_1x1, head_name)

        if not isinstance(old_head, nn.Conv2d):
            raise TypeError(
                f"{head_name} is not a Conv2d layer: {type(old_head)}"
            )

        new_head = nn.Conv2d(
            in_channels=old_head.in_channels,
            out_channels=num_keypoints,
            kernel_size=old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding,
            dilation=old_head.dilation,
            groups=old_head.groups,
            bias=old_head.bias is not None,
            padding_mode=old_head.padding_mode,
        )

        # Sinnvolle Initialisierung für neu erzeugte Prediction-Heads
        nn.init.kaiming_normal_(
            new_head.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if new_head.bias is not None:
            nn.init.zeros_(new_head.bias)

        setattr(network.Conv2_1x1, head_name, new_head)

        print(
            f"Replaced {head_name}: "
            f"{old_head.out_channels} -> {new_head.out_channels} outputs"
        )


def verify_output_heads(
    network: nn.Module,
    expected_keypoints: int,
) -> None:
    for head_name in ["conv0", "conv1", "conv2"]:
        head = getattr(network.Conv2_1x1, head_name)

        if head.out_channels != expected_keypoints:
            raise RuntimeError(
                f"{head_name} still has {head.out_channels} outputs; "
                f"expected {expected_keypoints}."
            )

        print(
            f"{head_name}: "
            f"in={head.in_channels}, out={head.out_channels}"
        )


# ============================================================
# Hauptprogramm
# ============================================================

def main() -> None:
    set_random_seeds(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAINING_VIDEO.exists():
        raise FileNotFoundError(
            f"Training video not found:\n{TRAINING_VIDEO}"
        )

    image_data, keypoints_data, used_frames = load_training_data(
        FRAME_DIR,
        LABEL_CSV,
    )

    print("\n========================================")
    print("TRAINING DATA")
    print("========================================")
    print(f"Images shape:    {image_data.shape}")
    print(f"Keypoints shape: {keypoints_data.shape}")
    print(f"Frames used:     {len(used_frames)}")

    print("\nLabels:")
    for index, label in enumerate(TRAIN_LABELS):
        print(f"  {index}: {label}")

    if keypoints_data.shape[1] != NUM_KEYPOINTS:
        raise RuntimeError(
            "Unexpected number of keypoints in training data: "
            f"{keypoints_data.shape[1]}"
        )

    image_height = image_data.shape[1]
    image_width = image_data.shape[2]

    bbox = [[0, image_height, 0, image_width]]

    print(f"\nBounding box: {bbox}")

    print("\n========================================")
    print("LOADING FACEMAP BASE MODEL")
    print("========================================")

    pose = Pose(
        filenames=[[str(TRAINING_VIDEO)]],
        bbox=bbox,
        bbox_set=True,
        resize=True,
        add_padding=True,
        model_name=None,
    )

    pose.load_model()

    print("\nOriginal output heads:")
    for head_name in ["conv0", "conv1", "conv2"]:
        head = getattr(pose.net.Conv2_1x1, head_name)
        print(
            f"  {head_name}: "
            f"{head.in_channels} -> {head.out_channels}"
        )

    print("\n========================================")
    print("REPLACING OUTPUT HEADS")
    print("========================================")

    replace_facemap_output_heads(
        network=pose.net,
        num_keypoints=NUM_KEYPOINTS,
    )

    # Neue Layer auf dasselbe Gerät wie das restliche Modell verschieben
    pose.net = pose.net.to(pose.device)

    print("\nNew output heads:")
    verify_output_heads(
        network=pose.net,
        expected_keypoints=NUM_KEYPOINTS,
    )

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================")
    print(f"Epochs:        {NUM_EPOCHS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay:  {WEIGHT_DECAY}")
    print(f"Device:        {pose.device}")
    print()

    pose.train(
        image_data=image_data,
        keypoints_data=keypoints_data,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bbox=bbox,
    )

    print("\n========================================")
    print("SAVING MODEL")
    print("========================================")

    # Facemap-kompatibler State-Dict
    torch.save(
        pose.net.state_dict(),
        str(MODEL_STATE_OUT),
    )

    model_info = {
        "model_name": "Frontview_v1_6kp",
        "facemap_version": "1.0.8",
        "number_of_keypoints": NUM_KEYPOINTS,
        "labels": TRAIN_LABELS,
        "training_frames": len(used_frames),
        "training_frame_names": used_frames,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "bbox": bbox,
        "image_shape": list(image_data.shape),
        "keypoints_shape": list(keypoints_data.shape),
        "base_model": "facemap_model_state.pt",
        "note": (
            "Custom Facemap frontview model. "
            "Conv2_1x1 conv0/conv1/conv2 replaced with "
            "six-output heatmap heads before training."
        ),
    }

    with MODEL_INFO_OUT.open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)

    print(f"Model saved to:\n{MODEL_STATE_OUT}")
    print(f"\nModel information saved to:\n{MODEL_INFO_OUT}")
    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()