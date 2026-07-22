"""
extract_tongue_frames_round03.py

Extracts targeted tongue-training frames from manually identified
one-second video segments.

Selection:
- 3 Cam0 videos
- 3 time windows per video
- 4 equally spaced frames per time window
- 36 frames total

Outputs:
- PNG frames
- CSV manifest with mouse, video, time window and source frame index
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

MANIFEST_PATH = OUTPUT_DIR / "round03_tongue_frame_manifest.csv"

FRAMES_PER_WINDOW = 4

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}

ALLOW_OVERWRITE = False


# ============================================================
# MANUALLY IDENTIFIED TONGUE WINDOWS
# ============================================================

@dataclass(frozen=True)
class VideoSelection:
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
# HELPERS
# ============================================================

def sanitize_filename(text: str) -> str:
    """Convert text into a Windows-safe filename component."""

    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("._")


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert HH:MM:SS or MM:SS to seconds.
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


def select_video(expected_video_stem: str) -> Path:
    """
    Open a file dialog and ask for the expected Cam0 video.
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

    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
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


def inspect_video(video_path: Path) -> dict:
    """Read video metadata."""

    capture = cv2.VideoCapture(str(video_path))

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

    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": total_frames / fps,
    }


def validate_output_directory() -> None:
    """Prevent accidental overwriting."""

    if OUTPUT_DIR.exists():
        existing_pngs = list(
            OUTPUT_DIR.glob("*.png")
        )

        if existing_pngs and not ALLOW_OVERWRITE:
            raise RuntimeError(
                "\nOutput folder already contains PNG files:\n"
                f"{OUTPUT_DIR}\n\n"
                "Nothing was overwritten.\n"
                "Rename/delete the folder or intentionally set "
                "ALLOW_OVERWRITE = True."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def generate_sample_times(
    start_seconds: float,
    end_seconds: float,
) -> np.ndarray:
    """
    Generate equally spaced sample times inside the window.

    The exact boundaries are avoided slightly, so all samples lie
    comfortably inside the identified tongue interval.
    """

    if end_seconds <= start_seconds:
        raise ValueError(
            "Time-window end must be after the start."
        )

    window_duration = end_seconds - start_seconds

    margin = min(
        0.05,
        window_duration * 0.10,
    )

    first_time = start_seconds + margin
    last_time = end_seconds - margin

    if last_time <= first_time:
        first_time = start_seconds
        last_time = end_seconds

    return np.linspace(
        first_time,
        last_time,
        FRAMES_PER_WINDOW,
    )


def extract_frame(
    capture: cv2.VideoCapture,
    frame_index: int,
):
    """Read one frame from an already opened video."""

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = capture.read()

    return success, frame


def write_manifest(rows: list[dict]) -> None:
    """Save extraction metadata as CSV."""

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
    Select one video and extract frames from all defined windows.
    """

    print("\n========================================")
    print(f"MOUSE {selection.mouse_id}")
    print("========================================")
    print(f"Expected video:\n{selection.expected_video_stem}")

    video_path = select_video(
        selection.expected_video_stem
    )

    video_info = inspect_video(
        video_path
    )

    print("\nSelected:")
    print(video_path)
    print(f"FPS:              {video_info['fps']:.3f}")
    print(f"Total frames:     {video_info['total_frames']}")
    print(
        f"Duration:         "
        f"{video_info['duration_seconds'] / 60.0:.2f} min"
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not reopen video:\n{video_path}"
        )

    rows: list[dict] = []

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

            if end_seconds > video_info["duration_seconds"]:
                raise RuntimeError(
                    "\nTime window exceeds video duration.\n"
                    f"Video: {video_path.name}\n"
                    f"Window: {start_timestamp} - {end_timestamp}\n"
                    f"Duration: "
                    f"{video_info['duration_seconds']:.2f} s"
                )

            sample_times = generate_sample_times(
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )

            print(
                f"\nWindow {window_index}: "
                f"{start_timestamp} - {end_timestamp}"
            )

            used_frame_indices: set[int] = set()

            for sample_number, sample_time in enumerate(
                sample_times,
                start=1,
            ):
                frame_index = int(
                    round(sample_time * video_info["fps"])
                )

                frame_index = min(
                    max(frame_index, 0),
                    video_info["total_frames"] - 1,
                )

                # Avoid accidental duplicate frames due to FPS rounding.
                while (
                    frame_index in used_frame_indices
                    and frame_index
                    < video_info["total_frames"] - 1
                ):
                    frame_index += 1

                used_frame_indices.add(
                    frame_index
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

                output_path = OUTPUT_DIR / output_filename

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
                            float(sample_time),
                            6,
                        ),
                        "source_frame_index": frame_index,
                        "video_fps": video_info["fps"],
                        "video_total_frames": video_info[
                            "total_frames"
                        ],
                        "video_width": video_info["width"],
                        "video_height": video_info["height"],
                        "output_png_name": output_filename,
                        "output_png_path": str(output_path),
                    }
                )

                print(
                    f"  [{sample_number}/{FRAMES_PER_WINDOW}] "
                    f"time={sample_time:.3f}s | "
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
    """Run targeted tongue-frame extraction."""

    expected_total = (
        len(VIDEO_SELECTIONS)
        * 3
        * FRAMES_PER_WINDOW
    )

    print("\n========================================")
    print("TARGETED TONGUE FRAME EXTRACTION")
    print("ROUND 03")
    print("========================================")
    print(f"Videos:             {len(VIDEO_SELECTIONS)}")
    print("Windows per video:  3")
    print(f"Frames per window:  {FRAMES_PER_WINDOW}")
    print(f"Target total:       {expected_total}")
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
            "targeted frames were extracted."
        )
    else:
        print(
            "\nWARNING: Extracted frame count differs "
            "from the expected total."
        )


if __name__ == "__main__":
    main()