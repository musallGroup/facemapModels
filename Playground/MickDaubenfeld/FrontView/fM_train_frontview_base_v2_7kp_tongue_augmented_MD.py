"""
train_frontview_base_v2_7kp_tongue_augmented.py

Fine-tunes the existing universal 7-keypoint Frontview Facemap model using:

    Round 01
    Round 02
    Round 03 Tongue
    Tongue Candidates (56 newly labeled frames)

Start model:
    Frontview_Base_v1_7kp.pt

Output model:
    Frontview_Base_v2_7kp_TongueAug.pt

Important:
- The model already has seven output channels.
- No architecture expansion is performed.
- Invisible or unset keypoints are converted to NaN and masked.
- Images in nested candidate-session folders are supported.
- Existing models are not overwritten.
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

TONGUE_CANDIDATE_IMAGE_DIR = (
    PROJECT_ROOT
    / "Tongue_Candidates"
)

TONGUE_CANDIDATE_LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Tongue_Candidates_7kp_labels.csv"
)

START_MODEL_PATH = (
    PROJECT_ROOT
    / "Current_Model"
    / "Frontview_Base_v1_7kp.pt"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT
    / "Current_Model"
    / "Frontview_Base_v2_7kp_TongueAug.pt"
)

OUTPUT_INFO_PATH = (
    PROJECT_ROOT
    / "Current_Model"
    / "Frontview_Base_v2_7kp_TongueAug_model_info.json"
)

OUTPUT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Training_Data"
    / "Frontview_Base_v2_7kp_TongueAug_training_manifest.csv"
)


# ============================================================
# KEYPOINTS
# ============================================================

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
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0
RANDOM_SEED = 20260720

ALLOW_MODEL_OVERWRITE = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class RoundSpec:
    name: str
    image_dir: Path
    label_csv: Path
    recursive_images: bool = False


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
    RoundSpec(
        name="Tongue_Candidates_7kp",
        image_dir=TONGUE_CANDIDATE_IMAGE_DIR,
        label_csv=TONGUE_CANDIDATE_LABEL_CSV,
        recursive_images=True,
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
    Facemap Pose requires a video for initialization.
    This video is not used as training material.
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

    OUTPUT_MANIFEST_PATH.parent.mkdir(
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
    recursive: bool,
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    iterator = (
        image_dir.rglob("*")
        if recursive
        else image_dir.iterdir()
    )

    image_paths = [
        path
        for path in iterator
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            and "_LABEL_QC" not in path.stem
        )
    ]

    if not image_paths:
        raise RuntimeError(
            f"No images found in:\n{image_dir}"
        )

    by_full_path = {
        str(path).replace("/", "\\").lower(): path
        for path in image_paths
    }

    by_name: dict[str, list[Path]] = {}

    for path in image_paths:
        by_name.setdefault(
            path.name,
            [],
        ).append(path)

    return by_full_path, by_name


def resolve_image_path(
    row: dict[str, Any],
    image_name: str,
    images_by_full_path: dict[str, Path],
    images_by_name: dict[str, list[Path]],
    spec: RoundSpec,
    row_number: int,
) -> Path:
    csv_png_path = safe_text(
        row,
        "png_path",
    )

    if csv_png_path:
        normalized = (
            csv_png_path
            .replace("/", "\\")
            .lower()
        )

        exact_path = images_by_full_path.get(
            normalized
        )

        if exact_path is not None:
            return exact_path

        direct_path = Path(csv_png_path)

        if direct_path.exists():
            return direct_path

    matches = images_by_name.get(
        image_name,
        [],
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"CSV row {row_number} references an image not found below:\n"
            f"{spec.image_dir}\n\n"
            f"Image: {image_name}"
        )

    raise RuntimeError(
        f"CSV row {row_number} uses a duplicate image name:\n"
        f"{image_name}\n\n"
        "The CSV must contain a valid png_path column for this image."
    )


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
    (
        images_by_full_path,
        images_by_name,
    ) = find_images(
        image_dir=spec.image_dir,
        recursive=spec.recursive_images,
    )

    samples: list[TrainingSample] = []
    seen_paths: set[str] = set()

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

            image_path = resolve_image_path(
                row=row,
                image_name=image_name,
                images_by_full_path=images_by_full_path,
                images_by_name=images_by_name,
                spec=spec,
                row_number=row_number,
            )

            normalized_path = (
                str(image_path)
                .replace("/", "\\")
                .lower()
            )

            if normalized_path in seen_paths:
                raise RuntimeError(
                    f"Duplicate image entry in "
                    f"{spec.label_csv.name}:\n{image_path}"
                )

            seen_paths.add(
                normalized_path
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

        for shape, count in sorted(details.items()):
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

def create_seven_keypoint_pose(
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

    pose.bodyparts = KEYPOINTS.copy()
    pose.pose_prediction_setup()

    return pose


def validate_model_heads(
    pose: Pose,
) -> None:
    print("\n========================================")
    print("VALIDATING 7-KEYPOINT MODEL")
    print("========================================")

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
            f"{head_name}: "
            f"{head.in_channels} -> "
            f"{head.out_channels}"
        )

        if head.out_channels != len(KEYPOINTS):
            raise RuntimeError(
                f"{head_name} has {head.out_channels} outputs; "
                f"expected {len(KEYPOINTS)}."
            )


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
        "model_type": "universal_frontview_base_model_finetuned",
        "starting_model": str(START_MODEL_PATH),
        "output_model": str(OUTPUT_MODEL_PATH),
        "keypoints": KEYPOINTS,
        "number_of_keypoints": len(KEYPOINTS),
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
        "augmentation_note": (
            "Includes the manually labeled Tongue_Candidates_7kp round."
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
    print("FRONTVIEW BASE V2 TRAINING")
    print("7 KEYPOINTS + TONGUE CANDIDATES")
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
    print("LOADING 7-KEYPOINT START MODEL")
    print("========================================")
    print(f"Video: {dummy_video_path}")
    print(f"Model: {START_MODEL_PATH}")

    pose = create_seven_keypoint_pose(
        dummy_video_path
    )

    validate_model_heads(
        pose
    )

    print("\n========================================")
    print("STARTING FINE-TUNING")
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