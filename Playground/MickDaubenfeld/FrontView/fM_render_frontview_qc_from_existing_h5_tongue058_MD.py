"""
render_frontview_qc_from_existing_h5_tongue058.py

Creates a new QC video from an existing Facemap HDF5 prediction file.

No Facemap inference is performed.

Thresholds:
- nose_tip, nose_bottom, mouth, lowerlip,
  whiskerpad_left, whiskerpad_right: 0.80
- tongue_tip: 0.58

The output filename contains:
    TONGUE-THRESH-058

Inputs selected by the user:
1. Original Cam0 video
2. Existing segment H5 prediction file

Expected H5 structure:
    frame_index
    Facemap/<keypoint>/x
    Facemap/<keypoint>/y
    Facemap/<keypoint>/likelihood
"""

from __future__ import annotations

from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import h5py
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

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
    "tongue_tip",
]

GENERAL_LIKELIHOOD_THRESHOLD = 0.80
TONGUE_LIKELIHOOD_THRESHOLD = 0.58

POINT_RADIUS = 6
TONGUE_RADIUS = 4
FONT_SCALE = 0.50
LINE_THICKNESS = 2

# None preserves the source video's FPS.
OUTPUT_FPS: float | None = None


# ============================================================
# FILE SELECTION
# ============================================================

def select_file(
    title: str,
    filetypes: list[tuple[str, str]],
) -> Path:
    """Open a Windows file-selection dialog."""

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes,
    )

    root.destroy()

    if not selected:
        raise RuntimeError(
            f"No file selected: {title}"
        )

    path = Path(selected)

    if not path.exists():
        raise FileNotFoundError(
            f"Selected file does not exist:\n{path}"
        )

    return path


def select_video() -> Path:
    """Select the original Cam0 source video."""

    return select_file(
        title="Select the original Cam0 video",
        filetypes=[
            ("Video files", "*.avi *.mp4 *.mov *.mkv"),
            ("AVI files", "*.avi"),
            ("All files", "*.*"),
        ],
    )


def select_h5() -> Path:
    """Select the existing segment prediction H5 file."""

    return select_file(
        title="Select the existing Frontview segment H5 prediction file",
        filetypes=[
            ("HDF5 files", "*.h5 *.hdf5"),
            ("All files", "*.*"),
        ],
    )


# ============================================================
# LOADING
# ============================================================

def read_video_information(
    video_path: Path,
) -> tuple[float, int, int, int]:
    """Return FPS, frame count, width and height."""

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    capture.release()

    if fps <= 0:
        raise RuntimeError(
            "The source video reports an invalid FPS value."
        )

    if total_frames <= 0:
        raise RuntimeError(
            "The source video reports no frames."
        )

    return fps, total_frames, width, height


def load_predictions(
    h5_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load predictions from HDF5.

    Returns
    -------
    frame_indices:
        Shape (n_frames,)

    predictions:
        Shape (n_frames, n_keypoints, 3)
        Last dimension = x, y, likelihood
    """

    with h5py.File(
        h5_path,
        "r",
    ) as h5_file:

        if "frame_index" not in h5_file:
            raise KeyError(
                "The H5 file does not contain 'frame_index'."
            )

        if "Facemap" not in h5_file:
            raise KeyError(
                "The H5 file does not contain the 'Facemap' group."
            )

        frame_indices = np.asarray(
            h5_file["frame_index"],
            dtype=np.int64,
        )

        facemap_group = h5_file["Facemap"]

        predictions = np.full(
            (
                len(frame_indices),
                len(KEYPOINT_NAMES),
                3,
            ),
            np.nan,
            dtype=np.float32,
        )

        for keypoint_index, keypoint_name in enumerate(
            KEYPOINT_NAMES
        ):
            if keypoint_name not in facemap_group:
                raise KeyError(
                    f"Missing keypoint in H5: {keypoint_name}"
                )

            keypoint_group = facemap_group[
                keypoint_name
            ]

            for dataset_name in [
                "x",
                "y",
                "likelihood",
            ]:
                if dataset_name not in keypoint_group:
                    raise KeyError(
                        f"Missing dataset: "
                        f"Facemap/{keypoint_name}/{dataset_name}"
                    )

            x = np.asarray(
                keypoint_group["x"],
                dtype=np.float32,
            )

            y = np.asarray(
                keypoint_group["y"],
                dtype=np.float32,
            )

            likelihood = np.asarray(
                keypoint_group["likelihood"],
                dtype=np.float32,
            )

            expected_length = len(
                frame_indices
            )

            if not (
                len(x) == expected_length
                and len(y) == expected_length
                and len(likelihood) == expected_length
            ):
                raise RuntimeError(
                    f"Prediction-length mismatch for "
                    f"{keypoint_name}."
                )

            predictions[
                :,
                keypoint_index,
                0,
            ] = x

            predictions[
                :,
                keypoint_index,
                1,
            ] = y

            predictions[
                :,
                keypoint_index,
                2,
            ] = likelihood

    if len(frame_indices) == 0:
        raise RuntimeError(
            "The H5 prediction file contains no frames."
        )

    return frame_indices, predictions


# ============================================================
# DRAWING
# ============================================================

def create_colors(
    number_of_keypoints: int,
) -> list[tuple[int, int, int]]:
    """Create distinct OpenCV BGR colors."""

    hues = np.linspace(
        0,
        179,
        number_of_keypoints,
        endpoint=False,
        dtype=np.uint8,
    )

    colors: list[
        tuple[int, int, int]
    ] = []

    for hue in hues:
        hsv_pixel = np.uint8(
            [[[hue, 220, 255]]]
        )

        bgr_pixel = cv2.cvtColor(
            hsv_pixel,
            cv2.COLOR_HSV2BGR,
        )[0, 0]

        colors.append(
            tuple(
                int(value)
                for value in bgr_pixel
            )
        )

    return colors


def draw_cross(
    frame: np.ndarray,
    x: int,
    y: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    """Draw an X for a below-threshold prediction."""

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
    """Draw all seven keypoints on one source frame."""

    overlay = frame.copy()

    frame_height, frame_width = overlay.shape[:2]

    for keypoint_index, keypoint_name in enumerate(
        KEYPOINT_NAMES
    ):
        x_float, y_float, likelihood = predictions[
            prediction_index,
            keypoint_index,
            :,
        ]

        if not (
            np.isfinite(x_float)
            and np.isfinite(y_float)
            and np.isfinite(likelihood)
        ):
            continue

        x = int(
            round(float(x_float))
        )

        y = int(
            round(float(y_float))
        )

        if not (
            0 <= x < frame_width
            and 0 <= y < frame_height
        ):
            continue

        threshold = (
            TONGUE_LIKELIHOOD_THRESHOLD
            if keypoint_name == "tongue_tip"
            else GENERAL_LIKELIHOOD_THRESHOLD
        )

        radius = (
            TONGUE_RADIUS
            if keypoint_name == "tongue_tip"
            else POINT_RADIUS
        )

        color = colors[
            keypoint_index
        ]

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
            draw_cross(
                frame=overlay,
                x=x,
                y=y,
                color=color,
                radius=radius,
            )

        label = (
            f"{keypoint_name} "
            f"{likelihood:.2f}"
        )

        text_x = min(
            x + 9,
            max(
                0,
                frame_width - 220,
            ),
        )

        text_y = max(
            20,
            y - 8,
        )

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

    session_seconds = (
        source_frame_index
        / source_fps
    )

    header = (
        f"Frame: {source_frame_index}   "
        f"Session time: "
        f"{session_seconds / 60.0:.3f} min   "
        f"General threshold: "
        f"{GENERAL_LIKELIHOOD_THRESHOLD:.2f}   "
        f"Tongue threshold: "
        f"{TONGUE_LIKELIHOOD_THRESHOLD:.2f}"
    )

    cv2.putText(
        overlay,
        header,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 0),
        thickness=4,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        header,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )

    return overlay


# ============================================================
# VIDEO CREATION
# ============================================================

def create_qc_video(
    video_path: Path,
    h5_path: Path,
    frame_indices: np.ndarray,
    predictions: np.ndarray,
) -> Path:
    """Render a new QC video without running inference."""

    (
        source_fps,
        total_source_frames,
        frame_width,
        frame_height,
    ) = read_video_information(
        video_path
    )

    if frame_indices[0] < 0:
        raise RuntimeError(
            "The H5 contains a negative source-frame index."
        )

    if frame_indices[-1] >= total_source_frames:
        raise RuntimeError(
            "The H5 contains frame indices outside "
            "the selected source video."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_minutes = (
        float(frame_indices[0])
        / source_fps
        / 60.0
    )

    end_minutes = (
        float(frame_indices[-1])
        / source_fps
        / 60.0
    )

    output_path = OUTPUT_DIR / (
        f"{video_path.stem}_"
        f"Frontview_Base_v1_7kp_"
        f"{start_minutes:.2f}-"
        f"{end_minutes:.2f}min_"
        f"TONGUE-THRESH-058_"
        f"RENDERED-FROM-H5_QC.mp4"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open source video:\n{video_path}"
        )

    output_fps = (
        source_fps
        if OUTPUT_FPS is None
        else float(OUTPUT_FPS)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (
            frame_width,
            frame_height,
        ),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not create output video:\n"
            f"{output_path}"
        )

    colors = create_colors(
        len(KEYPOINT_NAMES)
    )

    print("\n========================================")
    print("RENDER SETTINGS")
    print("========================================")
    print(f"Original video:    {video_path}")
    print(f"Prediction H5:     {h5_path}")
    print(f"Frames rendered:   {len(frame_indices)}")
    print(f"Start time:        {start_minutes:.3f} min")
    print(f"End time:          {end_minutes:.3f} min")
    print(
        f"General threshold: "
        f"{GENERAL_LIKELIHOOD_THRESHOLD:.2f}"
    )
    print(
        f"Tongue threshold:  "
        f"{TONGUE_LIKELIHOOD_THRESHOLD:.2f}"
    )
    print(f"Output:            {output_path}")

    print("\n========================================")
    print("CREATING QC VIDEO FROM EXISTING H5")
    print("NO FACEMAP INFERENCE")
    print("========================================")

    try:
        for prediction_index, source_frame_index in enumerate(
            frame_indices
        ):
            source_frame_index = int(
                source_frame_index
            )

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                source_frame_index,
            )

            success, frame = capture.read()

            if not success:
                raise RuntimeError(
                    f"Could not read source frame "
                    f"{source_frame_index}."
                )

            overlay = draw_overlay(
                frame=frame,
                prediction_index=prediction_index,
                source_frame_index=source_frame_index,
                predictions=predictions,
                colors=colors,
                source_fps=source_fps,
            )

            writer.write(
                overlay
            )

            processed = (
                prediction_index + 1
            )

            total = len(
                frame_indices
            )

            if (
                processed == 1
                or processed % 250 == 0
                or processed == total
            ):
                print(
                    f"Processed "
                    f"{processed}/{total} frames"
                )

    finally:
        capture.release()
        writer.release()

    return output_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Render the new threshold-QC video."""

    print("\n========================================")
    print("FRONTVIEW QC RENDER FROM EXISTING H5")
    print("TONGUE THRESHOLD = 0.58")
    print("========================================")

    print("\nSelect the original Cam0 video.")
    video_path = select_video()

    print("\nSelect the existing 14.00-15.00 min H5 prediction file.")
    h5_path = select_h5()

    print("\nLoading predictions...")

    frame_indices, predictions = load_predictions(
        h5_path
    )

    print(
        f"Loaded {len(frame_indices)} "
        "existing prediction frames."
    )

    output_path = create_qc_video(
        video_path=video_path,
        h5_path=h5_path,
        frame_indices=frame_indices,
        predictions=predictions,
    )

    print("\n========================================")
    print("QC VIDEO COMPLETE")
    print("========================================")
    print("\nSaved to:")
    print(output_path)


if __name__ == "__main__":
    main()