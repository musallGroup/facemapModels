"""
extract_tongue_frames_round03.py

Extracts targeted tongue-training frames from manually identified
one-second video segments.

Selection
---------
- 3 Cam0 videos
- 3 manually identified tongue windows per video
- 4 tightly spaced frames around the center of each window
- 36 frames total

At 60 fps, the four frames are approximately:

    center - 3 frames
    center - 1 frame
    center + 1 frame
    center + 3 frames

Outputs
-------
- PNG frames
- CSV manifest with:
    mouse
    source video
    time window
    sample time
    source frame index
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np


# ============================================================
# OUTPUT SETTINGS
# ============================================================

OUTPUT_DIR = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Training_Data\Extracted_Frames\Round_03_Tongue"
)

MANIFEST_PATH = (
    OUTPUT_DIR / "round03_tongue_frame_manifest.csv"
)

FRAMES_PER_WINDOW = 4

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}

# False prevents accidental overwriting.
ALLOW_OVERWRITE = False


# ============================================================
# MANUALLY IDENTIFIED TONGUE WINDOWS
# ============================================================

@dataclass(frozen=True)
class VideoSelection:
    """Definition of one source video and its tongue windows."""

    mouse_id: str
    expected_video_stem: str
    windows: tuple[tuple[str, str], ...]


VIDEO_SELECTIONS = [
    VideoSelection(
        mouse_id="428",
        expected_video_stem=(
            "428_PuffyPenguin_20251202_115310_"
            "cam0_20251202_115856"
        ),
        windows=(
            ("00:14:52", "00:14:53"),
            ("00:15:01", "00:15:02"),
            ("00:31:53", "00:31:54"),
        ),
    ),
    VideoSelection(
        mouse_id="460",
        expected_video_stem=(
            "460_PuffyPenguin_20260529_104245_"
            "cam0_20260515_135805"
        ),
        windows=(
            ("00:21:57", "00:21:58"),
            ("00:30:08", "00:30:09"),
            ("00:42:40", "00:42:41"),
        ),
    ),
    VideoSelection(
        mouse_id="431",
        expected_video_stem=(
            "431_PuffyPenguin_20260330_094942_"
            "cam0_20260330_095937"
        ),
        windows=(
            ("00:07:54", "00:07:55"),
            ("00:11:11", "00:11:12"),
            ("00:35:59", "00:36:00"),
        ),
    ),
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def sanitize_filename(text: str) -> str:
    """Convert text into a Windows-safe filename component."""

    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("._")


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert HH:MM:SS or MM:SS into seconds.

    Examples
    --------
    00:14:52 -> 892 seconds
    14:52    -> 892 seconds
    """

    parts = timestamp.split(":")

    if len(parts) == 3:
        hours, minutes, seconds = parts

    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts

    else:
        raise ValueError(
            f"Unsupported timestamp format: {timestamp}"
        )

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


# ============================================================
# FILE SELECTION
# ============================================================

def select_video(
    expected_video_stem: str,
) -> Path:
    """
    Ask the user to select one specific expected video.

    The selected filename is checked against the expected stem.
    """

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_file = filedialog.askopenfilename(
        title=f"Select video: {expected_video_stem}",
        filetypes=[
            ("Video files", "*.avi *.mp4 *.mov *.mkv"),
            ("AVI files", "*.avi"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not selected_file:
        raise RuntimeError(
            f"No video selected for:\n{expected_video_stem}"
        )

    video_path = Path(selected_file)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Selected video does not exist:\n{video_path}"
        )

    if (
        video_path.suffix.lower()
        not in SUPPORTED_VIDEO_EXTENSIONS
    ):
        raise RuntimeError(
            f"Unsupported video format:\n{video_path}"
        )

    if video_path.stem != expected_video_stem:
        raise RuntimeError(
            "\nWrong video selected.\n\n"
            f"Expected:\n{expected_video_stem}\n\n"
            f"Selected:\n{video_path.stem}"
        )

    return video_path


# ============================================================
# VIDEO HANDLING
# ============================================================

def inspect_video(
    video_path: Path,
) -> dict[str, float | int]:
    """Read basic video metadata."""

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
            f"Invalid FPS reported by video:\n{video_path}"
        )

    if total_frames <= 0:
        raise RuntimeError(
            f"No frames reported by video:\n{video_path}"
        )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video dimensions:\n{video_path}"
        )

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": total_frames / fps,
    }


def generate_sample_frame_indices(
    start_seconds: float,
    end_seconds: float,
    fps: float,
    total_frames: int,
) -> list[int]:
    """
    Generate four tightly clustered frames around the window center.

    At 60 fps the offsets correspond approximately to:

        center - 3 frames
        center - 1 frame
        center + 1 frame
        center + 3 frames

    Frame-based offsets are used rather than fixed second offsets,
    so the behavior remains correct for videos with different FPS.
    """

    if end_seconds <= start_seconds:
        raise ValueError(
            "Time-window end must be after the start."
        )

    center_seconds = (
        start_seconds + end_seconds
    ) / 2.0

    center_frame = int(
        round(center_seconds * fps)
    )

    frame_offsets = [
        -3,
        -1,
        1,
        3,
    ]

    first_window_frame = int(
        np.ceil(start_seconds * fps)
    )

    last_window_frame = int(
        np.floor(end_seconds * fps)
    )

    first_window_frame = max(
        0,
        first_window_frame,
    )

    last_window_frame = min(
        total_frames - 1,
        last_window_frame,
    )

    if last_window_frame < first_window_frame:
        raise RuntimeError(
            "The selected time window contains no valid frames."
        )

    frame_indices: list[int] = []

    for offset in frame_offsets:
        frame_index = center_frame + offset

        frame_index = max(
            first_window_frame,
            min(frame_index, last_window_frame),
        )

        frame_index = max(
            0,
            min(frame_index, total_frames - 1),
        )

        frame_indices.append(
            frame_index
        )

    # Avoid duplicates if rounding or clipping produced the same frame.
    unique_indices: list[int] = []

    for frame_index in frame_indices:
        candidate = frame_index

        while (
            candidate in unique_indices
            and candidate < last_window_frame
        ):
            candidate += 1

        while (
            candidate in unique_indices
            and candidate > first_window_frame
        ):
            candidate -= 1

        if candidate in unique_indices:
            raise RuntimeError(
                "Could not generate four unique frame indices "
                "inside the selected time window."
            )

        unique_indices.append(
            candidate
        )

    return unique_indices


def extract_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
):
    """Read one specific frame from an open video."""

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = capture.read()

    return success, frame


# ============================================================
# OUTPUT VALIDATION
# ============================================================

def validate_output_directory() -> None:
    """Prevent accidental overwriting of existing PNG files."""

    if OUTPUT_DIR.exists():
        existing_pngs = list(
            OUTPUT_DIR.glob("*.png")
        )

        if existing_pngs and not ALLOW_OVERWRITE:
            raise RuntimeError(
                "\nOutput folder already contains PNG files:\n"
                f"{OUTPUT_DIR}\n\n"
                "Nothing was overwritten.\n\n"
                "Delete or rename the existing "
                "Round_03_Tongue folder before restarting."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MANIFEST
# ============================================================

def write_manifest(
    rows: list[dict],
) -> None:
    """Write extraction metadata to CSV."""

    fieldnames = [
        "round",
        "mouse_id",
        "source_video_name",
        "source_video_path",
        "window_number",
        "window_start",
        "window_end",
        "sample_number_in_window",
        "sample_time_seconds",
        "source_frame_index",
        "frame_offset_from_center",
        "video_fps",
        "video_total_frames",
        "video_width",
        "video_height",
        "output_png_name",
        "output_png_path",
    ]

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# EXTRACTION
# ============================================================

def process_video(
    selection: VideoSelection,
    global_output_number: int,
) -> tuple[list[dict], int]:
    """
    Select one video and extract the defined tongue windows.
    """

    print("\n========================================")
    print(f"MOUSE {selection.mouse_id}")
    print("========================================")
    print("Expected video:")
    print(selection.expected_video_stem)

    video_path = select_video(
        selection.expected_video_stem
    )

    video_info = inspect_video(
        video_path
    )

    fps = float(video_info["fps"])
    total_frames = int(
        video_info["total_frames"]
    )

    print("\nSelected:")
    print(video_path)
    print(f"FPS:              {fps:.3f}")
    print(f"Total frames:     {total_frames}")
    print(
        f"Duration:         "
        f"{float(video_info['duration_seconds']) / 60.0:.2f} min"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not reopen video:\n{video_path}"
        )

    rows: list[dict] = []

    frame_offsets = [
        -3,
        -1,
        1,
        3,
    ]

    try:
        for window_index, (
            start_timestamp,
            end_timestamp,
        ) in enumerate(
            selection.windows,
            start=1,
        ):
            start_seconds = timestamp_to_seconds(
                start_timestamp
            )

            end_seconds = timestamp_to_seconds(
                end_timestamp
            )

            if (
                end_seconds
                > float(video_info["duration_seconds"])
            ):
                raise RuntimeError(
                    "\nTime window exceeds video duration.\n"
                    f"Video: {video_path.name}\n"
                    f"Window: "
                    f"{start_timestamp} - {end_timestamp}\n"
                    f"Video duration: "
                    f"{float(video_info['duration_seconds']):.2f} s"
                )

            frame_indices = (
                generate_sample_frame_indices(
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    fps=fps,
                    total_frames=total_frames,
                )
            )

            print(
                f"\nWindow {window_index}: "
                f"{start_timestamp} - {end_timestamp}"
            )

            print(
                "  Sampling tightly around window center:"
            )

            for sample_number, (
                frame_index,
                frame_offset,
            ) in enumerate(
                zip(
                    frame_indices,
                    frame_offsets,
                ),
                start=1,
            ):
                sample_time = (
                    frame_index / fps
                )

                success, frame = extract_frame(
                    capture=capture,
                    frame_index=frame_index,
                )

                if not success or frame is None:
                    raise RuntimeError(
                        f"Could not read frame {frame_index} "
                        f"from:\n{video_path}"
                    )

                global_output_number += 1

                output_filename = (
                    f"R03_"
                    f"{global_output_number:03d}_"
                    f"mouse-{sanitize_filename(selection.mouse_id)}_"
                    f"window-{window_index:02d}_"
                    f"sample-{sample_number:02d}_"
                    f"frame-{frame_index:07d}.png"
                )

                output_path = (
                    OUTPUT_DIR / output_filename
                )

                write_success = cv2.imwrite(
                    str(output_path),
                    frame,
                )

                if not write_success:
                    raise RuntimeError(
                        f"Could not save PNG:\n{output_path}"
                    )

                rows.append(
                    {
                        "round": "Round_03_Tongue",
                        "mouse_id": selection.mouse_id,
                        "source_video_name": video_path.name,
                        "source_video_path": str(video_path),
                        "window_number": window_index,
                        "window_start": start_timestamp,
                        "window_end": end_timestamp,
                        "sample_number_in_window": sample_number,
                        "sample_time_seconds": round(
                            sample_time,
                            6,
                        ),
                        "source_frame_index": frame_index,
                        "frame_offset_from_center": frame_offset,
                        "video_fps": fps,
                        "video_total_frames": total_frames,
                        "video_width": int(
                            video_info["width"]
                        ),
                        "video_height": int(
                            video_info["height"]
                        ),
                        "output_png_name": output_filename,
                        "output_png_path": str(output_path),
                    }
                )

                print(
                    f"  [{sample_number}/{FRAMES_PER_WINDOW}] "
                    f"offset={frame_offset:+d} frames | "
                    f"time={sample_time:.4f}s | "
                    f"frame={frame_index} | "
                    f"{output_filename}"
                )

    finally:
        capture.release()

    return rows, global_output_number


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete targeted tongue-frame extraction."""

    windows_per_video = 3

    expected_total = (
        len(VIDEO_SELECTIONS)
        * windows_per_video
        * FRAMES_PER_WINDOW
    )

    print("\n========================================")
    print("TARGETED TONGUE FRAME EXTRACTION")
    print("ROUND 03 — CENTERED SAMPLING")
    print("========================================")
    print(f"Videos:             {len(VIDEO_SELECTIONS)}")
    print(f"Windows per video:  {windows_per_video}")
    print(f"Frames per window:  {FRAMES_PER_WINDOW}")
    print(f"Target total:       {expected_total}")
    print(
        "Frame offsets:      "
        "-3, -1, +1, +3 around center"
    )
    print(f"Output directory:   {OUTPUT_DIR}")

    validate_output_directory()

    all_rows: list[dict] = []
    global_output_number = 0

    for selection in VIDEO_SELECTIONS:
        video_rows, global_output_number = process_video(
            selection=selection,
            global_output_number=global_output_number,
        )

        all_rows.extend(
            video_rows
        )

    if not all_rows:
        raise RuntimeError(
            "No tongue frames were extracted."
        )

    write_manifest(
        all_rows
    )

    print("\n========================================")
    print("TONGUE EXTRACTION COMPLETE")
    print("========================================")
    print(f"Frames extracted: {len(all_rows)}")
    print(f"Expected frames:  {expected_total}")

    print("\nFrames saved to:")
    print(OUTPUT_DIR)

    print("\nManifest saved to:")
    print(MANIFEST_PATH)

    if len(all_rows) == expected_total:
        print(
            f"\nSUCCESS: Exactly {expected_total} "
            "targeted center frames were extracted."
        )

    else:
        print(
            "\nWARNING: Extracted frame count differs "
            "from the expected total."
        )


if __name__ == "__main__":
    main()