"""
qc_label_overlays_round03_tongue.py

Creates QC overlay images for the manually labeled Round 03 tongue data.

Input:
    Training_Data/Extracted_Frames/Round_03_Tongue
    Training_Data/Labels/Round_03_Tongue_labels.csv

Output:
    Model_QC/Round_03_Tongue_Label_QC

Visible keypoints are drawn as colored circles with labels.
Explicitly invisible keypoints are listed in the top-left corner.

The script does not modify the original images or label CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Training_Data\Extracted_Frames"
    r"\Round_03_Tongue"
)

LABEL_CSV = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Training_Data\Labels"
    r"\Round_03_Tongue_labels.csv"
)

OUTPUT_DIR = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Model_QC"
    r"\Round_03_Tongue_Label_QC"
)


# ============================================================
# LABEL SETTINGS
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

KEYPOINT_COLORS = {
    "nose_tip": (48, 59, 255),
    "nose_bottom": (0, 149, 255),
    "mouth": (0, 204, 255),
    "lowerlip": (89, 199, 52),
    "whiskerpad_left": (190, 199, 0),
    "whiskerpad_right": (255, 122, 0),
    "tongue_tip": (222, 82, 175),
}

POINT_RADII = {
    "nose_tip": 7,
    "nose_bottom": 7,
    "mouth": 7,
    "lowerlip": 7,
    "whiskerpad_left": 7,
    "whiskerpad_right": 7,
    "tongue_tip": 5,
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
TEXT_THICKNESS = 1
OUTLINE_THICKNESS = 3

ALLOW_OVERWRITE = True


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_paths() -> None:
    """Validate required files and create the output directory."""

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{IMAGE_DIR}"
        )

    if not LABEL_CSV.exists():
        raise FileNotFoundError(
            f"Label CSV does not exist:\n{LABEL_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not ALLOW_OVERWRITE:
        existing_images = [
            path
            for path in OUTPUT_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_IMAGE_EXTENSIONS
            )
        ]

        if existing_images:
            raise RuntimeError(
                "\nQC output directory already contains images:\n"
                f"{OUTPUT_DIR}\n\n"
                "Nothing was overwritten."
            )


def find_images() -> list[Path]:
    """Find all source images."""

    images = [
        path
        for path in IMAGE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    ]

    return sorted(
        images,
        key=lambda path: path.name.lower(),
    )


# ============================================================
# LABEL LOADING
# ============================================================

def parse_visible_value(value: str | None) -> int:
    """Parse a CSV visibility value safely."""

    if value is None:
        return 0

    value = value.strip()

    if not value:
        return 0

    try:
        return int(float(value))
    except ValueError:
        return 0


def load_labels() -> dict[str, dict[str, dict]]:
    """
    Load labels from CSV.

    Returns
    -------
    labels_by_image:
        {
            image_name: {
                keypoint: {
                    x,
                    y,
                    visible,
                    state
                }
            }
        }
    """

    labels_by_image: dict[str, dict[str, dict]] = {}

    with LABEL_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "The label CSV has no header."
            )

        if "image" not in reader.fieldnames:
            raise KeyError(
                "The label CSV does not contain an 'image' column."
            )

        for row in reader:
            image_name = row.get("image", "").strip()

            if not image_name:
                continue

            frame_labels: dict[str, dict] = {}

            for keypoint in KEYPOINTS:
                x_text = row.get(
                    f"{keypoint}_x",
                    "",
                ).strip()

                y_text = row.get(
                    f"{keypoint}_y",
                    "",
                ).strip()

                visible = parse_visible_value(
                    row.get(
                        f"{keypoint}_visible",
                        "0",
                    )
                )

                state = row.get(
                    f"{keypoint}_state",
                    "",
                ).strip()

                x_value: float | None = None
                y_value: float | None = None

                if visible == 1 and x_text and y_text:
                    try:
                        x_value = float(x_text)
                        y_value = float(y_text)
                    except ValueError:
                        visible = 0
                        x_value = None
                        y_value = None

                if not state:
                    state = (
                        "visible"
                        if visible == 1
                        else "not_visible"
                    )

                frame_labels[keypoint] = {
                    "x": x_value,
                    "y": y_value,
                    "visible": visible,
                    "state": state,
                }

            labels_by_image[image_name] = frame_labels

    return labels_by_image


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float = FONT_SCALE,
) -> None:
    """Draw readable text with a black outline."""

    cv2.putText(
        image,
        text,
        position,
        FONT,
        font_scale,
        (0, 0, 0),
        OUTLINE_THICKNESS,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        FONT,
        font_scale,
        color,
        TEXT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_keypoint(
    image: np.ndarray,
    keypoint: str,
    x: float,
    y: float,
) -> None:
    """Draw one visible keypoint and its name."""

    height, width = image.shape[:2]

    x_int = int(round(x))
    y_int = int(round(y))

    if not (
        0 <= x_int < width
        and 0 <= y_int < height
    ):
        return

    color = KEYPOINT_COLORS[keypoint]
    radius = POINT_RADII[keypoint]

    cv2.circle(
        image,
        (x_int, y_int),
        radius + 2,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (x_int, y_int),
        radius,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    text_x = min(
        x_int + 10,
        max(0, width - 210),
    )

    text_y = max(
        20,
        y_int - 9,
    )

    draw_text_with_outline(
        image=image,
        text=keypoint,
        position=(text_x, text_y),
        color=color,
    )


def draw_header(
    image: np.ndarray,
    image_name: str,
    invisible_keypoints: list[str],
) -> None:
    """Draw image name and visibility summary."""

    draw_text_with_outline(
        image=image,
        text=image_name,
        position=(20, 32),
        color=(255, 255, 255),
        font_scale=0.70,
    )

    if invisible_keypoints:
        invisible_text = (
            "Not visible: "
            + ", ".join(invisible_keypoints)
        )
    else:
        invisible_text = "Not visible: none"

    draw_text_with_outline(
        image=image,
        text=invisible_text,
        position=(20, 62),
        color=(210, 210, 210),
        font_scale=0.52,
    )


# ============================================================
# QC CREATION
# ============================================================

def create_overlay(
    image_path: Path,
    frame_labels: dict[str, dict],
) -> np.ndarray:
    """Create a QC overlay for one source image."""

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    overlay = image.copy()

    invisible_keypoints: list[str] = []

    for keypoint in KEYPOINTS:
        label = frame_labels[keypoint]

        if (
            label["visible"] == 1
            and label["x"] is not None
            and label["y"] is not None
        ):
            draw_keypoint(
                image=overlay,
                keypoint=keypoint,
                x=float(label["x"]),
                y=float(label["y"]),
            )
        else:
            invisible_keypoints.append(
                keypoint
            )

    draw_header(
        image=overlay,
        image_name=image_path.name,
        invisible_keypoints=invisible_keypoints,
    )

    return overlay


def save_overlay(
    overlay: np.ndarray,
    image_path: Path,
) -> Path:
    """Save one QC overlay image."""

    output_name = (
        f"{image_path.stem}_LABEL_QC.png"
    )

    output_path = OUTPUT_DIR / output_name

    success = cv2.imwrite(
        str(output_path),
        overlay,
    )

    if not success:
        raise RuntimeError(
            f"Could not save QC image:\n{output_path}"
        )

    return output_path


# ============================================================
# REPORTING
# ============================================================

def summarize_labels(
    labels_by_image: dict[str, dict[str, dict]],
    image_paths: list[Path],
) -> None:
    """Print visibility statistics for Round 03."""

    print("\n========================================")
    print("VISIBILITY SUMMARY")
    print("========================================")

    for keypoint in KEYPOINTS:
        visible_count = 0
        invisible_count = 0
        missing_count = 0

        for image_path in image_paths:
            frame_labels = labels_by_image.get(
                image_path.name
            )

            if frame_labels is None:
                missing_count += 1
                continue

            label = frame_labels[keypoint]

            if (
                label["visible"] == 1
                and label["x"] is not None
                and label["y"] is not None
            ):
                visible_count += 1
            else:
                invisible_count += 1

        print(
            f"{keypoint:<20} "
            f"visible={visible_count:>2}  "
            f"not_visible={invisible_count:>2}  "
            f"missing={missing_count:>2}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Create all Round 03 label-QC overlays."""

    print("\n========================================")
    print("ROUND 03 TONGUE LABEL QC")
    print("========================================")
    print(f"Images:    {IMAGE_DIR}")
    print(f"Labels:    {LABEL_CSV}")
    print(f"Output:    {OUTPUT_DIR}")

    validate_paths()

    image_paths = find_images()

    if not image_paths:
        raise RuntimeError(
            f"No images were found in:\n{IMAGE_DIR}"
        )

    labels_by_image = load_labels()

    print(f"\nImages found:       {len(image_paths)}")
    print(f"CSV label entries:  {len(labels_by_image)}")

    missing_label_entries = [
        image_path.name
        for image_path in image_paths
        if image_path.name not in labels_by_image
    ]

    if missing_label_entries:
        raise RuntimeError(
            "\nSome source images have no CSV label entry:\n"
            + "\n".join(missing_label_entries)
        )

    print("\n========================================")
    print("CREATING LABEL OVERLAYS")
    print("========================================")

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        overlay = create_overlay(
            image_path=image_path,
            frame_labels=labels_by_image[
                image_path.name
            ],
        )

        output_path = save_overlay(
            overlay=overlay,
            image_path=image_path,
        )

        print(
            f"[{index:02d}/{len(image_paths):02d}] "
            f"{output_path.name}"
        )

    summarize_labels(
        labels_by_image=labels_by_image,
        image_paths=image_paths,
    )

    print("\n========================================")
    print("ROUND 03 LABEL QC COMPLETE")
    print("========================================")
    print("\nQC overlays saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()