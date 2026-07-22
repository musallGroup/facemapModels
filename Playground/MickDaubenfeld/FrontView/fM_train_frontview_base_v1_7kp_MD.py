"""
train_frontview_base_v1_7kp.py

Trainiert ein universelles Frontview-Facemap-Basemodell mit 7 Keypoints
aus Round 01, Round 02 und Round 03 (Tongue).

Startmodell:
    Frontview_v1_6kp.pt

Neue Architektur:
    Die drei Facemap-Output-Heads werden von 6 auf 7 Kanäle erweitert.
    Kanäle 0-5 werden aus dem 6-Keypoint-Modell übernommen.
    Kanal 6 (tongue_tip) wird neu initialisiert.

Facemap:
    1.0.8
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from facemap.pose.pose import Pose
import facemap.pose.pose as pose_module
from facemap.pose import datasets
from facemap.pose import model_training


# ============================================================
# FACEMAP PATCHES
# ============================================================

pose_module.datasets = datasets
pose_module.model_training = model_training


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon"
)

ROUND_01_IMAGE_DIR = (
    PROJECT_ROOT
    / "Training_Data"
    / "Extracted_Frames"
    / "Round_01"
)

ROUND_01_LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Round_01"
    / "frontview_labels_round01.csv"
)

ROUND_02_IMAGE_DIR = (
    PROJECT_ROOT
    / "Training_Data"
    / "Extracted_Frames"
    / "Round_02_8mice"
)

ROUND_02_LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Round_02_8mice_labels.csv"
)

ROUND_03_IMAGE_DIR = (
    PROJECT_ROOT
    / "Training_Data"
    / "Extracted_Frames"
    / "Round_03_Tongue"
)

ROUND_03_LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Round_03_Tongue_labels.csv"
)

START_MODEL_PATH = (
    PROJECT_ROOT
    / "Current_Model"
    / "Frontview_v1_6kp.pt"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT
    / "Current_Model"
    / "Frontview_Base_v1_7kp.pt"
)

OUTPUT_INFO_PATH = (
    PROJECT_ROOT
    / "Current_Model"
    / "Frontview_Base_v1_7kp_model_info.json"
)

OUTPUT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Training_Data"
    / "Frontview_Base_v1_7kp_training_manifest.csv"
)


# ============================================================
# KEYPOINTS
# ============================================================

OLD_KEYPOINTS = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
]

KEYPOINTS = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
    "tongue_tip",
]

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0
RANDOM_SEED = 20260715

ALLOW_MODEL_OVERWRITE = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class RoundSpec:
    name: str
    image_dir: Path
    label_csv: Path


@dataclass
class TrainingSample:
    round_name: str
    image_name: str
    image_path: Path
    label_csv: Path
    image: np.ndarray
    keypoints: np.ndarray


ROUND_SPECS = [
    RoundSpec(
        name="Round_01",
        image_dir=ROUND_01_IMAGE_DIR,
        label_csv=ROUND_01_LABEL_CSV,
    ),
    RoundSpec(
        name="Round_02_8mice",
        image_dir=ROUND_02_IMAGE_DIR,
        label_csv=ROUND_02_LABEL_CSV,
    ),
    RoundSpec(
        name="Round_03_Tongue",
        image_dir=ROUND_03_IMAGE_DIR,
        label_csv=ROUND_03_LABEL_CSV,
    ),
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# FILE SELECTION
# ============================================================

def select_dummy_video() -> Path:
    """
    Facemap Pose benötigt ein Video zur Initialisierung.
    Dieses Video wird nicht als Trainingsmaterial verwendet.
    """

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askopenfilename(
        title="Select any normal Cam0 video for Facemap initialization",
        filetypes=[
            ("Video files", "*.avi *.mp4 *.mov *.mkv"),
            ("AVI files", "*.avi"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not selected:
        raise RuntimeError(
            "No Cam0 initialization video was selected."
        )

    path = Path(selected)

    if not path.exists():
        raise FileNotFoundError(
            f"Selected video does not exist:\n{path}"
        )

    return path


# ============================================================
# VALIDATION
# ============================================================

def validate_paths() -> None:
    errors: list[str] = []

    for spec in ROUND_SPECS:
        if not spec.image_dir.exists():
            errors.append(
                f"{spec.name} image directory missing:\n"
                f"{spec.image_dir}"
            )

        if not spec.label_csv.exists():
            errors.append(
                f"{spec.name} label CSV missing:\n"
                f"{spec.label_csv}"
            )

    if not START_MODEL_PATH.exists():
        errors.append(
            f"Starting model missing:\n{START_MODEL_PATH}"
        )

    if errors:
        raise FileNotFoundError(
            "\n\n".join(errors)
        )

    if (
        OUTPUT_MODEL_PATH.exists()
        and not ALLOW_MODEL_OVERWRITE
    ):
        raise RuntimeError(
            "\nOutput model already exists:\n"
            f"{OUTPUT_MODEL_PATH}\n\n"
            "Nothing was overwritten. Rename/delete the existing file "
            "or intentionally set ALLOW_MODEL_OVERWRITE = True."
        )

    OUTPUT_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CSV HELPERS
# ============================================================

def safe_text(
    row: dict[str, Any],
    column: str,
) -> str:
    value = row.get(column, "")

    if value is None:
        return ""

    return str(value).strip()


def get_image_column(
    fieldnames: list[str],
) -> str:
    """
    Erkennt die Spalte mit dem Bilddateinamen.

    Round 01 verwendet 'frame'.
    Round 02 und Round 03 verwenden 'image'.
    """

    candidates = [
        "image",
        "frame",
        "filename",
        "file",
        "image_name",
        "output_png_name",
    ]

    lower_to_original = {
        field.lower(): field
        for field in fieldnames
    }

    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise KeyError(
        "Could not identify the image-name column.\n"
        f"CSV columns: {fieldnames}"
    )


def parse_float_or_nan(value: str) -> float:
    value = value.strip()

    if not value:
        return float("nan")

    try:
        result = float(value)
    except ValueError:
        return float("nan")

    if not math.isfinite(result):
        return float("nan")

    return result


def parse_visibility(
    row: dict[str, Any],
    keypoint: str,
) -> bool:
    state = safe_text(
        row,
        f"{keypoint}_state",
    ).lower()

    visible_text = safe_text(
        row,
        f"{keypoint}_visible",
    ).lower()

    if state == "visible":
        return True

    if state in {
        "not_visible",
        "invisible",
        "unset",
    }:
        return False

    return visible_text in {
        "1",
        "1.0",
        "true",
        "yes",
    }


def parse_keypoint(
    row: dict[str, Any],
    keypoint: str,
) -> tuple[float, float]:
    if not parse_visibility(row, keypoint):
        return float("nan"), float("nan")

    x_value = parse_float_or_nan(
        safe_text(row, f"{keypoint}_x")
    )

    y_value = parse_float_or_nan(
        safe_text(row, f"{keypoint}_y")
    )

    if not (
        math.isfinite(x_value)
        and math.isfinite(y_value)
    ):
        return float("nan"), float("nan")

    return x_value, y_value


# ============================================================
# IMAGE LOADING
# ============================================================

def find_images(
    image_dir: Path,
) -> dict[str, Path]:
    images = {
        path.name: path
        for path in image_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    }

    if not images:
        raise RuntimeError(
            f"No images found in:\n{image_dir}"
        )

    return images


def read_grayscale_image(
    image_path: Path,
) -> np.ndarray:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    if image.ndim != 2:
        raise RuntimeError(
            f"Expected grayscale image, received "
            f"{image.shape}:\n{image_path}"
        )

    return image


def load_round_samples(
    spec: RoundSpec,
) -> list[TrainingSample]:
    images_by_name = find_images(
        spec.image_dir
    )

    samples: list[TrainingSample] = []
    seen_names: set[str] = set()

    with spec.label_csv.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header:\n{spec.label_csv}"
            )

        image_column = get_image_column(
            reader.fieldnames
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            image_name = safe_text(
                row,
                image_column,
            )

            if not image_name:
                continue

            if image_name in seen_names:
                raise RuntimeError(
                    f"Duplicate image entry in "
                    f"{spec.label_csv.name}:\n{image_name}"
                )

            seen_names.add(image_name)

            image_path = images_by_name.get(
                image_name
            )

            if image_path is None:
                raise FileNotFoundError(
                    f"CSV row {row_number} references an image "
                    f"not found in:\n{spec.image_dir}\n\n"
                    f"Image: {image_name}"
                )

            keypoints = np.full(
                (len(KEYPOINTS), 2),
                np.nan,
                dtype=np.float32,
            )

            for keypoint_index, keypoint in enumerate(
                KEYPOINTS
            ):
                x_column = f"{keypoint}_x"
                y_column = f"{keypoint}_y"

                # Falls eine alte CSV tongue_tip noch nicht enthält,
                # bleibt tongue_tip NaN und wird maskiert.
                if (
                    x_column not in reader.fieldnames
                    or y_column not in reader.fieldnames
                ):
                    continue

                x_value, y_value = parse_keypoint(
                    row,
                    keypoint,
                )

                keypoints[
                    keypoint_index,
                    0,
                ] = x_value

                keypoints[
                    keypoint_index,
                    1,
                ] = y_value

            image = read_grayscale_image(
                image_path
            )

            samples.append(
                TrainingSample(
                    round_name=spec.name,
                    image_name=image_name,
                    image_path=image_path,
                    label_csv=spec.label_csv,
                    image=image,
                    keypoints=keypoints,
                )
            )

    if not samples:
        raise RuntimeError(
            f"No labeled samples loaded for {spec.name}."
        )

    return samples


# ============================================================
# DATASET CHECKS
# ============================================================

def validate_image_shapes(
    samples: list[TrainingSample],
) -> tuple[int, int]:
    shapes = {
        tuple(sample.image.shape)
        for sample in samples
    }

    if len(shapes) != 1:
        details: dict[
            tuple[int, int],
            int,
        ] = {}

        for sample in samples:
            shape = tuple(sample.image.shape)
            details[shape] = details.get(shape, 0) + 1

        lines = [
            "Training images do not all share one resolution:"
        ]

        for shape, count in details.items():
            lines.append(
                f"  {shape}: {count} images"
            )

        raise RuntimeError(
            "\n".join(lines)
        )

    height, width = next(iter(shapes))

    return int(height), int(width)


def validate_coordinates(
    samples: list[TrainingSample],
    height: int,
    width: int,
) -> None:
    errors: list[str] = []

    for sample in samples:
        for keypoint_index, keypoint in enumerate(
            KEYPOINTS
        ):
            x_value = sample.keypoints[
                keypoint_index,
                0,
            ]

            y_value = sample.keypoints[
                keypoint_index,
                1,
            ]

            if np.isnan(x_value) and np.isnan(y_value):
                continue

            if np.isnan(x_value) != np.isnan(y_value):
                errors.append(
                    f"{sample.round_name}/{sample.image_name}: "
                    f"{keypoint} has only one coordinate."
                )

                continue

            if not (
                0 <= x_value < width
                and 0 <= y_value < height
            ):
                errors.append(
                    f"{sample.round_name}/{sample.image_name}: "
                    f"{keypoint}=({x_value:.2f}, {y_value:.2f}) "
                    f"outside {width}x{height}."
                )

    if errors:
        raise RuntimeError(
            "Invalid keypoint coordinates found:\n"
            + "\n".join(errors[:20])
        )


def print_visibility_summary(
    samples: list[TrainingSample],
) -> dict[str, int]:
    counts: dict[str, int] = {}

    print("\n========================================")
    print("VISIBILITY SUMMARY")
    print("========================================")

    for keypoint_index, keypoint in enumerate(
        KEYPOINTS
    ):
        visible_count = sum(
            not np.isnan(
                sample.keypoints[
                    keypoint_index,
                    0,
                ]
            )
            for sample in samples
        )

        counts[keypoint] = visible_count

        print(
            f"{keypoint:<20} "
            f"visible={visible_count:>3}  "
            f"masked={len(samples) - visible_count:>3}"
        )

    if counts["tongue_tip"] == 0:
        raise RuntimeError(
            "No visible tongue_tip labels were found."
        )

    return counts


# ============================================================
# FACEMAP MODEL
# ============================================================

def create_six_keypoint_pose(
    dummy_video_path: Path,
) -> Pose:
    pose = Pose(
        filenames=[[str(dummy_video_path)]],
        bbox=[],
        bbox_set=False,
        resize=False,
        add_padding=False,
        gui=None,
        GUIobject=None,
        net=None,
        model_name=str(START_MODEL_PATH),
    )

    pose.bodyparts = OLD_KEYPOINTS.copy()
    pose.pose_prediction_setup()

    return pose


def expand_output_heads_to_seven(
    pose: Pose,
) -> None:
    print("\n========================================")
    print("EXPANDING OUTPUT HEADS: 6 -> 7")
    print("========================================")

    for head_name in [
        "conv0",
        "conv1",
        "conv2",
    ]:
        old_head = getattr(
            pose.net.Conv2_1x1,
            head_name,
        )

        if not isinstance(old_head, nn.Conv2d):
            raise TypeError(
                f"Unexpected layer type for {head_name}: "
                f"{type(old_head).__name__}"
            )

        if old_head.out_channels != len(OLD_KEYPOINTS):
            raise RuntimeError(
                f"{head_name} has {old_head.out_channels} outputs; "
                f"expected {len(OLD_KEYPOINTS)}."
            )

        new_head = nn.Conv2d(
            in_channels=old_head.in_channels,
            out_channels=len(KEYPOINTS),
            kernel_size=old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding,
            dilation=old_head.dilation,
            groups=old_head.groups,
            bias=old_head.bias is not None,
            padding_mode=old_head.padding_mode,
        ).to(
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )

        with torch.no_grad():
            new_head.weight[
                : len(OLD_KEYPOINTS)
            ].copy_(
                old_head.weight
            )

            if (
                old_head.bias is not None
                and new_head.bias is not None
            ):
                new_head.bias[
                    : len(OLD_KEYPOINTS)
                ].copy_(
                    old_head.bias
                )

        setattr(
            pose.net.Conv2_1x1,
            head_name,
            new_head,
        )

        print(
            f"{head_name}: "
            f"{old_head.out_channels} -> "
            f"{new_head.out_channels} outputs"
        )

    pose.bodyparts = KEYPOINTS.copy()

    if hasattr(pose.net, "labels_id"):
        pose.net.labels_id = KEYPOINTS.copy()

    if hasattr(pose.net, "output_ch"):
        pose.net.output_ch = len(KEYPOINTS)


# ============================================================
# OUTPUT FILES
# ============================================================

def write_training_manifest(
    samples: list[TrainingSample],
) -> None:
    fieldnames = [
        "dataset_index",
        "round",
        "image_name",
        "image_path",
        "label_csv",
    ]

    for keypoint in KEYPOINTS:
        fieldnames.extend(
            [
                f"{keypoint}_x",
                f"{keypoint}_y",
                f"{keypoint}_visible",
            ]
        )

    with OUTPUT_MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for dataset_index, sample in enumerate(
            samples
        ):
            row: dict[str, Any] = {
                "dataset_index": dataset_index,
                "round": sample.round_name,
                "image_name": sample.image_name,
                "image_path": str(sample.image_path),
                "label_csv": str(sample.label_csv),
            }

            for keypoint_index, keypoint in enumerate(
                KEYPOINTS
            ):
                x_value = sample.keypoints[
                    keypoint_index,
                    0,
                ]

                y_value = sample.keypoints[
                    keypoint_index,
                    1,
                ]

                visible = int(
                    not np.isnan(x_value)
                    and not np.isnan(y_value)
                )

                row[
                    f"{keypoint}_visible"
                ] = visible

                if visible:
                    row[
                        f"{keypoint}_x"
                    ] = float(x_value)

                    row[
                        f"{keypoint}_y"
                    ] = float(y_value)

                else:
                    row[f"{keypoint}_x"] = ""
                    row[f"{keypoint}_y"] = ""

            writer.writerow(row)


def make_json_serializable(
    value: Any,
) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

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

    return value


def write_model_info(
    samples: list[TrainingSample],
    visibility_counts: dict[str, int],
    image_height: int,
    image_width: int,
    bbox: np.ndarray,
    device: torch.device,
) -> None:
    round_counts: dict[str, int] = {}

    for sample in samples:
        round_counts[sample.round_name] = (
            round_counts.get(sample.round_name, 0)
            + 1
        )

    info = {
        "model_name": OUTPUT_MODEL_PATH.stem,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "facemap_version_target": "1.0.8",
        "model_type": "universal_frontview_base_model",
        "starting_model": str(START_MODEL_PATH),
        "output_model": str(OUTPUT_MODEL_PATH),
        "keypoints": KEYPOINTS,
        "number_of_keypoints": len(KEYPOINTS),
        "copied_channels": OLD_KEYPOINTS,
        "new_channel": "tongue_tip",
        "number_of_training_images": len(samples),
        "images_per_round": round_counts,
        "visibility_counts": visibility_counts,
        "image_shape": [
            image_height,
            image_width,
        ],
        "bbox": bbox.tolist(),
        "training_parameters": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "random_seed": RANDOM_SEED,
            "device": str(device),
        },
        "training_manifest": str(
            OUTPUT_MANIFEST_PATH
        ),
        "visibility_handling": (
            "Invisible/unset keypoints were converted to NaN "
            "and masked during Facemap training."
        ),
    }

    with OUTPUT_INFO_PATH.open(
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            make_json_serializable(info),
            json_file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n========================================")
    print("FRONTVIEW BASE MODEL TRAINING")
    print("7 KEYPOINTS")
    print("========================================")

    set_all_seeds(
        RANDOM_SEED
    )

    validate_paths()

    all_samples: list[TrainingSample] = []

    print("\n========================================")
    print("LOADING TRAINING ROUNDS")
    print("========================================")

    for spec in ROUND_SPECS:
        print(f"\n{spec.name}")
        print(f"  Images: {spec.image_dir}")
        print(f"  Labels: {spec.label_csv}")

        round_samples = load_round_samples(
            spec
        )

        print(
            f"  Loaded: {len(round_samples)} frames"
        )

        all_samples.extend(
            round_samples
        )

    image_height, image_width = validate_image_shapes(
        all_samples
    )

    validate_coordinates(
        samples=all_samples,
        height=image_height,
        width=image_width,
    )

    visibility_counts = print_visibility_summary(
        all_samples
    )

    print("\n========================================")
    print("COMBINED TRAINING DATA")
    print("========================================")
    print(
        f"Images shape:    "
        f"({len(all_samples)}, "
        f"{image_height}, {image_width})"
    )

    print(
        f"Keypoints shape: "
        f"({len(all_samples)}, "
        f"{len(KEYPOINTS)}, 2)"
    )

    print(
        f"Frames used:     {len(all_samples)}"
    )

    image_data = np.stack(
        [
            sample.image
            for sample in all_samples
        ],
        axis=0,
    )

    keypoints_data = np.stack(
        [
            sample.keypoints
            for sample in all_samples
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    bbox = np.asarray(
        [[
            0,
            image_height,
            0,
            image_width,
        ]],
        dtype=np.int64,
    )

    print(f"\nBounding box: {bbox.tolist()}")

    write_training_manifest(
        all_samples
    )

    print(
        "\nSelect any normal Cam0 video.\n"
        "Facemap needs it only to initialize Pose."
    )

    dummy_video_path = select_dummy_video()

    print("\n========================================")
    print("LOADING 6-KEYPOINT START MODEL")
    print("========================================")
    print(f"Video: {dummy_video_path}")
    print(f"Model: {START_MODEL_PATH}")

    pose = create_six_keypoint_pose(
        dummy_video_path
    )

    print("\nOriginal output heads:")

    for head_name in [
        "conv0",
        "conv1",
        "conv2",
    ]:
        head = getattr(
            pose.net.Conv2_1x1,
            head_name,
        )

        print(
            f"  {head_name}: "
            f"{head.in_channels} -> "
            f"{head.out_channels}"
        )

    expand_output_heads_to_seven(
        pose
    )

    print("\nNew output heads:")

    for head_name in [
        "conv0",
        "conv1",
        "conv2",
    ]:
        head = getattr(
            pose.net.Conv2_1x1,
            head_name,
        )

        print(
            f"  {head_name}: "
            f"{head.in_channels} -> "
            f"{head.out_channels}"
        )

    print("\n========================================")
    print("STARTING TRAINING")
    print("========================================")
    print(f"Epochs:        {NUM_EPOCHS}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay:  {WEIGHT_DECAY}")
    print(f"Device:        {pose.device}")

    trained_model = pose.train(
        image_data=image_data,
        keypoints_data=keypoints_data,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        bbox=bbox,
    )

    pose.net = trained_model

    print("\n========================================")
    print("SAVING MODEL")
    print("========================================")

    torch.save(
        pose.net.state_dict(),
        OUTPUT_MODEL_PATH,
    )

    print("Model saved to:")
    print(OUTPUT_MODEL_PATH)

    write_model_info(
        samples=all_samples,
        visibility_counts=visibility_counts,
        image_height=image_height,
        image_width=image_width,
        bbox=bbox,
        device=pose.device,
    )

    print("\nModel information saved to:")
    print(OUTPUT_INFO_PATH)

    print("\nTraining manifest saved to:")
    print(OUTPUT_MANIFEST_PATH)

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()