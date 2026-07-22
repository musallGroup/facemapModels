r"""
train_frontview_base_v3_7kp_kept.py

Fine-tunes the existing Frontview V2 model with all previous training rounds
plus the newly labeled Kept frames.

New data:
    Images:
    \\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon\
    Tongue_Frame_Review\Tongue_Frame_Review_20260720_135251\Kept

    Labels:
    \\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon\
    Training_Data\Labels\Kept_7kp_labels.csv

Start model:
    Frontview_Base_v2_7kp_TongueAug.pt

Output model:
    Frontview_Base_v3_7kp_KeptTongue.pt

Facemap target: 1.0.8
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
from facemap.pose import datasets, model_training


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
    PROJECT_ROOT / "Training_Data" / "Extracted_Frames" / "Round_01"
)
ROUND_01_LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Round_01"
    / "frontview_labels_round01.csv"
)

ROUND_02_IMAGE_DIR = (
    PROJECT_ROOT / "Training_Data" / "Extracted_Frames" / "Round_02_8mice"
)
ROUND_02_LABEL_CSV = (
    PROJECT_ROOT / "Training_Data" / "Labels" / "Round_02_8mice_labels.csv"
)

ROUND_03_IMAGE_DIR = (
    PROJECT_ROOT / "Training_Data" / "Extracted_Frames" / "Round_03_Tongue"
)
ROUND_03_LABEL_CSV = (
    PROJECT_ROOT / "Training_Data" / "Labels" / "Round_03_Tongue_labels.csv"
)

TONGUE_CANDIDATE_IMAGE_DIR = PROJECT_ROOT / "Tongue_Candidates"
TONGUE_CANDIDATE_LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Tongue_Candidates_7kp_labels.csv"
)

KEPT_IMAGE_DIR = (
    PROJECT_ROOT
    / "Tongue_Frame_Review"
    / "Tongue_Frame_Review_20260720_135251"
    / "Kept"
)
KEPT_LABEL_CSV = (
    PROJECT_ROOT / "Training_Data" / "Labels" / "Kept_7kp_labels.csv"
)

START_MODEL_PATH = (
    PROJECT_ROOT / "Current_Model" / "Frontview_Base_v2_7kp_TongueAug.pt"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT / "Current_Model" / "Frontview_Base_v3_7kp_KeptTongue.pt"
)
OUTPUT_INFO_PATH = (
    PROJECT_ROOT
    / "Current_Model"
    / "Frontview_Base_v3_7kp_KeptTongue_model_info.json"
)
OUTPUT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "Training_Data"
    / "Frontview_Base_v3_7kp_KeptTongue_training_manifest.csv"
)


# ============================================================
# KEYPOINTS AND TRAINING SETTINGS
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
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

NUM_EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.0
RANDOM_SEED = 20260720

# Protection against accidentally overwriting a completed V3 model.
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
        "Round_01",
        ROUND_01_IMAGE_DIR,
        ROUND_01_LABEL_CSV,
    ),
    RoundSpec(
        "Round_02_8mice",
        ROUND_02_IMAGE_DIR,
        ROUND_02_LABEL_CSV,
    ),
    RoundSpec(
        "Round_03_Tongue",
        ROUND_03_IMAGE_DIR,
        ROUND_03_LABEL_CSV,
    ),
    RoundSpec(
        "Tongue_Candidates_7kp",
        TONGUE_CANDIDATE_IMAGE_DIR,
        TONGUE_CANDIDATE_LABEL_CSV,
        recursive_images=True,
    ),
    RoundSpec(
        "Kept_7kp",
        KEPT_IMAGE_DIR,
        KEPT_LABEL_CSV,
        recursive_images=True,
    ),
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_path(path: Path | str) -> str:
    return str(path).replace("/", "\\").lower()


def safe_text(row: dict[str, Any], column: str) -> str:
    value = row.get(column, "")
    return "" if value is None else str(value).strip()


def parse_float_or_nan(value: str) -> float:
    if not value:
        return float("nan")

    try:
        result = float(value)
    except ValueError:
        return float("nan")

    return result if math.isfinite(result) else float("nan")


def parse_visibility(row: dict[str, Any], keypoint: str) -> bool:
    state = safe_text(row, f"{keypoint}_state").lower()
    visible = safe_text(row, f"{keypoint}_visible").lower()

    if state == "visible":
        return True

    if state in {"not_visible", "invisible", "unset"}:
        return False

    return visible in {"1", "1.0", "true", "yes"}


def parse_keypoint(
    row: dict[str, Any],
    keypoint: str,
) -> tuple[float, float]:
    if not parse_visibility(row, keypoint):
        return float("nan"), float("nan")

    x = parse_float_or_nan(safe_text(row, f"{keypoint}_x"))
    y = parse_float_or_nan(safe_text(row, f"{keypoint}_y"))

    if not (math.isfinite(x) and math.isfinite(y)):
        return float("nan"), float("nan")

    return x, y


def get_image_column(fieldnames: list[str]) -> str:
    candidates = [
        "image",
        "frame",
        "filename",
        "file",
        "image_name",
        "output_png_name",
    ]

    lookup = {name.lower(): name for name in fieldnames}

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    raise KeyError(
        "Could not identify the image-name column.\n"
        f"CSV columns: {fieldnames}"
    )


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths() -> None:
    errors: list[str] = []

    for spec in ROUND_SPECS:
        if not spec.image_dir.exists():
            errors.append(
                f"{spec.name} image directory missing:\n{spec.image_dir}"
            )

        if not spec.label_csv.exists():
            errors.append(
                f"{spec.name} label CSV missing:\n{spec.label_csv}"
            )

    if not START_MODEL_PATH.exists():
        errors.append(
            f"Starting model missing:\n{START_MODEL_PATH}"
        )

    if errors:
        raise FileNotFoundError("\n\n".join(errors))

    if OUTPUT_MODEL_PATH.exists() and not ALLOW_MODEL_OVERWRITE:
        raise RuntimeError(
            f"Output model already exists:\n{OUTPUT_MODEL_PATH}\n\n"
            "Nothing was overwritten. Rename/delete the existing file or set "
            "ALLOW_MODEL_OVERWRITE = True intentionally."
        )

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE AND LABEL LOADING
# ============================================================

def find_images(
    image_dir: Path,
    recursive: bool,
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()

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
        raise RuntimeError(f"No images found in:\n{image_dir}")

    by_full_path = {
        normalized_path(path): path
        for path in image_paths
    }

    by_name: dict[str, list[Path]] = {}
    for path in image_paths:
        by_name.setdefault(path.name, []).append(path)

    return by_full_path, by_name


def resolve_image_path(
    row: dict[str, Any],
    image_name: str,
    images_by_full_path: dict[str, Path],
    images_by_name: dict[str, list[Path]],
    spec: RoundSpec,
    row_number: int,
) -> Path:
    # The generalized GUI stores the exact original PNG path.
    csv_png_path = safe_text(row, "png_path")

    if csv_png_path:
        exact = images_by_full_path.get(normalized_path(csv_png_path))
        if exact is not None:
            return exact

        direct_path = Path(csv_png_path)
        if direct_path.exists():
            return direct_path

    matches = images_by_name.get(image_name, [])

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"CSV row {row_number} references an image not found below:\n"
            f"{spec.image_dir}\n\nImage: {image_name}"
        )

    raise RuntimeError(
        f"CSV row {row_number} uses a duplicate image name:\n"
        f"{image_name}\n\n"
        "A valid png_path entry is required for duplicate filenames."
    )


def read_grayscale_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(f"Could not read image:\n{image_path}")

    if image.ndim != 2:
        raise RuntimeError(
            f"Expected grayscale image, received {image.shape}:\n{image_path}"
        )

    return image


def load_round_samples(spec: RoundSpec) -> list[TrainingSample]:
    images_by_full_path, images_by_name = find_images(
        spec.image_dir,
        spec.recursive_images,
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
            raise RuntimeError(f"CSV has no header:\n{spec.label_csv}")

        image_column = get_image_column(reader.fieldnames)

        for row_number, row in enumerate(reader, start=2):
            image_name = safe_text(row, image_column)

            if not image_name:
                continue

            image_path = resolve_image_path(
                row,
                image_name,
                images_by_full_path,
                images_by_name,
                spec,
                row_number,
            )

            path_key = normalized_path(image_path)

            if path_key in seen_paths:
                raise RuntimeError(
                    f"Duplicate image entry in {spec.label_csv.name}:\n"
                    f"{image_path}"
                )

            seen_paths.add(path_key)

            keypoints = np.full(
                (len(KEYPOINTS), 2),
                np.nan,
                dtype=np.float32,
            )

            for keypoint_index, keypoint in enumerate(KEYPOINTS):
                if (
                    f"{keypoint}_x" not in reader.fieldnames
                    or f"{keypoint}_y" not in reader.fieldnames
                ):
                    continue

                x, y = parse_keypoint(row, keypoint)
                keypoints[keypoint_index] = (x, y)

            samples.append(
                TrainingSample(
                    round_name=spec.name,
                    image_name=image_name,
                    image_path=image_path,
                    label_csv=spec.label_csv,
                    image=read_grayscale_image(image_path),
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
    shape_counts: dict[tuple[int, int], int] = {}

    for sample in samples:
        shape = tuple(sample.image.shape)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    if len(shape_counts) != 1:
        lines = ["Training images do not all share one resolution:"]
        for shape, count in sorted(shape_counts.items()):
            lines.append(f"  {shape}: {count} images")
        raise RuntimeError("\n".join(lines))

    height, width = next(iter(shape_counts))
    return int(height), int(width)


def validate_coordinates(
    samples: list[TrainingSample],
    height: int,
    width: int,
) -> None:
    errors: list[str] = []

    for sample in samples:
        for keypoint_index, keypoint in enumerate(KEYPOINTS):
            x, y = sample.keypoints[keypoint_index]

            if np.isnan(x) and np.isnan(y):
                continue

            if np.isnan(x) != np.isnan(y):
                errors.append(
                    f"{sample.round_name}/{sample.image_name}: "
                    f"{keypoint} has only one coordinate."
                )
                continue

            if not (0 <= x < width and 0 <= y < height):
                errors.append(
                    f"{sample.round_name}/{sample.image_name}: "
                    f"{keypoint}=({x:.2f}, {y:.2f}) outside "
                    f"{width}x{height}."
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

    for keypoint_index, keypoint in enumerate(KEYPOINTS):
        visible = sum(
            not np.isnan(sample.keypoints[keypoint_index, 0])
            for sample in samples
        )
        counts[keypoint] = visible

        print(
            f"{keypoint:<20} "
            f"visible={visible:>4}  "
            f"masked={len(samples) - visible:>4}"
        )

    if counts["tongue_tip"] == 0:
        raise RuntimeError("No visible tongue_tip labels were found.")

    return counts


# ============================================================
# FACEMAP MODEL
# ============================================================

def select_dummy_video() -> Path:
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
        raise RuntimeError("No Cam0 initialization video was selected.")

    path = Path(selected)

    if not path.exists():
        raise FileNotFoundError(
            f"Selected video does not exist:\n{path}"
        )

    return path


def create_seven_keypoint_pose(dummy_video_path: Path) -> Pose:
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


def validate_model_heads(pose: Pose) -> None:
    print("\n========================================")
    print("VALIDATING 7-KEYPOINT MODEL")
    print("========================================")

    for head_name in ["conv0", "conv1", "conv2"]:
        head = getattr(pose.net.Conv2_1x1, head_name)

        print(
            f"{head_name}: "
            f"{head.in_channels} -> {head.out_channels}"
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
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for dataset_index, sample in enumerate(samples):
            row: dict[str, Any] = {
                "dataset_index": dataset_index,
                "round": sample.round_name,
                "image_name": sample.image_name,
                "image_path": str(sample.image_path),
                "label_csv": str(sample.label_csv),
            }

            for keypoint_index, keypoint in enumerate(KEYPOINTS):
                x, y = sample.keypoints[keypoint_index]
                visible = int(not np.isnan(x) and not np.isnan(y))

                row[f"{keypoint}_visible"] = visible
                row[f"{keypoint}_x"] = float(x) if visible else ""
                row[f"{keypoint}_y"] = float(y) if visible else ""

            writer.writerow(row)


def make_json_serializable(value: Any) -> Any:
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
        return [make_json_serializable(item) for item in value]
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
            round_counts.get(sample.round_name, 0) + 1
        )

    info = {
        "model_name": OUTPUT_MODEL_PATH.stem,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "facemap_version_target": "1.0.8",
        "model_type": "universal_frontview_base_model_finetuned",
        "starting_model": str(START_MODEL_PATH),
        "output_model": str(OUTPUT_MODEL_PATH),
        "keypoints": KEYPOINTS,
        "number_of_keypoints": len(KEYPOINTS),
        "number_of_training_images": len(samples),
        "images_per_round": round_counts,
        "visibility_counts": visibility_counts,
        "image_shape": [image_height, image_width],
        "bbox": bbox.tolist(),
        "training_parameters": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "random_seed": RANDOM_SEED,
            "device": str(device),
        },
        "training_manifest": str(OUTPUT_MANIFEST_PATH),
        "visibility_handling": (
            "Invisible/unset keypoints were converted to NaN and masked."
        ),
        "augmentation_note": (
            "V3 includes all V2 training rounds plus Kept_7kp labeled frames."
        ),
    }

    with OUTPUT_INFO_PATH.open("w", encoding="utf-8") as json_file:
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
    print("FRONTVIEW BASE V3 TRAINING")
    print("7 KEYPOINTS + KEPT TONGUE FRAMES")
    print("========================================")

    set_all_seeds(RANDOM_SEED)
    validate_paths()

    all_samples: list[TrainingSample] = []

    print("\n========================================")
    print("LOADING TRAINING ROUNDS")
    print("========================================")

    for spec in ROUND_SPECS:
        print(f"\n{spec.name}")
        print(f"  Images: {spec.image_dir}")
        print(f"  Labels: {spec.label_csv}")

        round_samples = load_round_samples(spec)

        print(f"  Loaded: {len(round_samples)} frames")
        all_samples.extend(round_samples)

    image_height, image_width = validate_image_shapes(all_samples)

    validate_coordinates(
        all_samples,
        image_height,
        image_width,
    )

    visibility_counts = print_visibility_summary(all_samples)

    image_data = np.stack(
        [sample.image for sample in all_samples],
        axis=0,
    )

    keypoints_data = np.stack(
        [sample.keypoints for sample in all_samples],
        axis=0,
    ).astype(np.float32, copy=False)

    bbox = np.asarray(
        [[0, image_height, 0, image_width]],
        dtype=np.int64,
    )

    print("\n========================================")
    print("COMBINED TRAINING DATA")
    print("========================================")
    print(f"Images shape:    {image_data.shape}")
    print(f"Keypoints shape: {keypoints_data.shape}")
    print(f"Frames used:     {len(all_samples)}")
    print(f"Bounding box:    {bbox.tolist()}")

    write_training_manifest(all_samples)

    print(
        "\nSelect any normal Cam0 video.\n"
        "Facemap needs it only to initialize Pose."
    )
    dummy_video_path = select_dummy_video()

    print("\n========================================")
    print("LOADING V2 START MODEL")
    print("========================================")
    print(f"Video: {dummy_video_path}")
    print(f"Model: {START_MODEL_PATH}")

    pose = create_seven_keypoint_pose(dummy_video_path)
    validate_model_heads(pose)

    print("\n========================================")
    print("STARTING V3 FINE-TUNING")
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
    print("SAVING V3 MODEL")
    print("========================================")

    torch.save(
        pose.net.state_dict(),
        OUTPUT_MODEL_PATH,
    )

    write_model_info(
        samples=all_samples,
        visibility_counts=visibility_counts,
        image_height=image_height,
        image_width=image_width,
        bbox=bbox,
        device=pose.device,
    )

    print(f"Model saved to:\n{OUTPUT_MODEL_PATH}")
    print(f"\nModel information saved to:\n{OUTPUT_INFO_PATH}")
    print(f"\nTraining manifest saved to:\n{OUTPUT_MANIFEST_PATH}")

    print("\n========================================")
    print("TRAINING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()