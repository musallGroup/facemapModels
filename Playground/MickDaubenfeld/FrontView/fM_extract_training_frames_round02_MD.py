"""
extract_training_frames_round02.py

Creates Round 02 of the Frontview Facemap training dataset.

Strategy
--------
- 12 mice
- 5 frames per mouse
- preferably 5 different sessions/videos per mouse
- only Cam0 videos
- one random frame per selected video
- 60 frames total

Outputs
-------
Training_Data/
    Extracted_Frames/
        Round_02/
            PNG files
            round02_frame_manifest.csv

Each PNG filename contains:
    mouse ID
    session
    source video
    source frame index

Target environment:
    fm_front
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import cv2


# ============================================================
# SETTINGS
# ============================================================

BPOD_ROOT = Path(
    r"\\Naskampa\lts\BpodBehavior"
)

OUTPUT_DIR = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Training_Data\Extracted_Frames\Round_02"
)

MANIFEST_PATH = OUTPUT_DIR / "round02_frame_manifest.csv"

MOUSE_IDS = [
    "426",
    "427",
    "428",
    "431",
    "458",
    "460",
    "461",
    "462",
    "479",
    "480",
    "481",
    "482",
]

FRAMES_PER_MOUSE = 5

# Makes the random selection reproducible.
# Running the script again with the same seed and unchanged files
# gives the same selections.
RANDOM_SEED = 20260616

# Avoid frames from the very beginning and end of each video.
FRAME_MARGIN_FRACTION = 0.05

# Videos shorter than this are ignored.
MINIMUM_VIDEO_FRAMES = 100

SUPPORTED_VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}

# Set to True only when you intentionally want to overwrite
# an already existing Round_02 extraction.
ALLOW_OVERWRITE = False


# ============================================================
# HELPERS
# ============================================================

def sanitize_filename(text: str) -> str:
    """Convert text into a Windows-safe filename component."""

    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("._")


def is_cam0_video(path: Path) -> bool:
    """Return True when a file appears to be a Cam0 video."""

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return False

    filename = path.name.lower()

    cam0_patterns = [
        "cam0",
        "cam_0",
        "camera0",
        "camera_0",
    ]

    return any(pattern in filename for pattern in cam0_patterns)


def find_cam0_videos(mouse_id: str) -> list[Path]:
    """Recursively find Cam0 videos below one mouse folder."""

    mouse_root = BPOD_ROOT / mouse_id

    if not mouse_root.exists():
        print(f"WARNING: Mouse folder not found: {mouse_root}")
        return []

    videos = [
        path
        for path in mouse_root.rglob("*")
        if path.is_file() and is_cam0_video(path)
    ]

    return sorted(videos)


def get_session_name(video_path: Path) -> str:
    """
    Determine a useful session name from the video path.

    For the expected structure:

        Mouse/
            Protocol/
                Session Data/
                    SESSION/
                        video.avi

    the direct parent of the video is normally the session folder.
    """

    return video_path.parent.name


def inspect_video(video_path: Path) -> dict | None:
    """
    Read basic video information.

    Returns None if the video cannot be opened or is too short.
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"  Could not open: {video_path}")
        return None

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    capture.release()

    if total_frames < MINIMUM_VIDEO_FRAMES:
        print(
            f"  Video ignored because it is too short "
            f"({total_frames} frames): {video_path}"
        )
        return None

    if width <= 0 or height <= 0:
        print(
            f"  Video ignored because dimensions are invalid: "
            f"{video_path}"
        )
        return None

    return {
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "fps": fps,
    }


def choose_random_frame_index(
    total_frames: int,
    random_generator: random.Random,
) -> int:
    """Choose a random frame while avoiding the video boundaries."""

    margin = int(
        total_frames * FRAME_MARGIN_FRACTION
    )

    first_allowed = max(0, margin)
    last_allowed = min(
        total_frames - 1,
        total_frames - margin - 1,
    )

    if last_allowed <= first_allowed:
        first_allowed = 0
        last_allowed = total_frames - 1

    return random_generator.randint(
        first_allowed,
        last_allowed,
    )


def extract_frame(
    video_path: Path,
    frame_index: int,
) -> tuple[bool, object]:
    """Read one specific frame from a video."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return False, None

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_index,
    )

    success, frame = capture.read()
    capture.release()

    return success, frame


def write_manifest(rows: list[dict]) -> None:
    """Write extraction metadata to CSV."""

    fieldnames = [
        "round",
        "mouse_id",
        "session",
        "source_video_name",
        "source_video_path",
        "source_frame_index",
        "video_total_frames",
        "video_width",
        "video_height",
        "video_fps",
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
# VALIDATION
# ============================================================

def validate_paths() -> None:
    """Validate input and output paths before extraction."""

    if not BPOD_ROOT.exists():
        raise FileNotFoundError(
            "The Bpod root folder does not exist:\n"
            f"{BPOD_ROOT}"
        )

    if OUTPUT_DIR.exists():
        existing_pngs = list(
            OUTPUT_DIR.glob("*.png")
        )

        if existing_pngs and not ALLOW_OVERWRITE:
            raise RuntimeError(
                "\nRound_02 already contains PNG files:\n"
                f"{OUTPUT_DIR}\n\n"
                "Nothing was overwritten.\n"
                "Move/delete the existing files or set "
                "ALLOW_OVERWRITE = True intentionally."
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# EXTRACTION
# ============================================================

def process_mouse(
    mouse_id: str,
    random_generator: random.Random,
    global_output_number: int,
) -> tuple[list[dict], int]:
    """Select and extract frames for one mouse."""

    print("\n----------------------------------------")
    print(f"MOUSE {mouse_id}")
    print("----------------------------------------")

    videos = find_cam0_videos(mouse_id)

    print(f"Cam0 videos found: {len(videos)}")

    if not videos:
        print("No usable Cam0 videos found.")
        return [], global_output_number

    # Group videos by session folder. This prevents selecting multiple
    # videos from the same session whenever enough sessions exist.
    videos_by_session: dict[str, list[Path]] = {}

    for video_path in videos:
        session_name = get_session_name(video_path)

        videos_by_session.setdefault(
            session_name,
            [],
        ).append(video_path)

    session_names = list(
        videos_by_session.keys()
    )

    random_generator.shuffle(session_names)

    selected_candidates: list[Path] = []

    # Initially select one random Cam0 video per different session.
    for session_name in session_names:
        session_videos = videos_by_session[
            session_name
        ]

        selected_video = random_generator.choice(
            session_videos
        )

        selected_candidates.append(
            selected_video
        )

        if len(selected_candidates) >= FRAMES_PER_MOUSE:
            break

    # Fallback when fewer than five distinct session folders exist.
    if len(selected_candidates) < FRAMES_PER_MOUSE:
        remaining_videos = [
            video
            for video in videos
            if video not in selected_candidates
        ]

        random_generator.shuffle(
            remaining_videos
        )

        selected_candidates.extend(
            remaining_videos[
                : FRAMES_PER_MOUSE
                - len(selected_candidates)
            ]
        )

    # Add additional candidates so corrupt videos can be replaced.
    backup_candidates = [
        video
        for video in videos
        if video not in selected_candidates
    ]

    random_generator.shuffle(
        backup_candidates
    )

    candidate_queue = (
        selected_candidates + backup_candidates
    )

    rows: list[dict] = []
    used_videos: set[Path] = set()

    for video_path in candidate_queue:
        if len(rows) >= FRAMES_PER_MOUSE:
            break

        if video_path in used_videos:
            continue

        used_videos.add(video_path)

        video_info = inspect_video(
            video_path
        )

        if video_info is None:
            continue

        frame_index = choose_random_frame_index(
            total_frames=video_info["total_frames"],
            random_generator=random_generator,
        )

        success, frame = extract_frame(
            video_path=video_path,
            frame_index=frame_index,
        )

        if not success or frame is None:
            print(
                f"  Could not read frame {frame_index}: "
                f"{video_path}"
            )
            continue

        session_name = get_session_name(
            video_path
        )

        global_output_number += 1

        output_filename = (
            f"R02_"
            f"{global_output_number:03d}_"
            f"mouse-{sanitize_filename(mouse_id)}_"
            f"session-{sanitize_filename(session_name)}_"
            f"frame-{frame_index:07d}.png"
        )

        output_path = OUTPUT_DIR / output_filename

        write_success = cv2.imwrite(
            str(output_path),
            frame,
        )

        if not write_success:
            print(
                f"  Failed to save PNG: {output_path}"
            )
            global_output_number -= 1
            continue

        row = {
            "round": "Round_02",
            "mouse_id": mouse_id,
            "session": session_name,
            "source_video_name": video_path.name,
            "source_video_path": str(video_path),
            "source_frame_index": frame_index,
            "video_total_frames": video_info[
                "total_frames"
            ],
            "video_width": video_info["width"],
            "video_height": video_info["height"],
            "video_fps": video_info["fps"],
            "output_png_name": output_filename,
            "output_png_path": str(output_path),
        }

        rows.append(row)

        print(
            f"  [{len(rows)}/{FRAMES_PER_MOUSE}] "
            f"{session_name} | "
            f"frame {frame_index} | "
            f"{output_filename}"
        )

    if len(rows) < FRAMES_PER_MOUSE:
        print(
            f"WARNING: Only {len(rows)} of "
            f"{FRAMES_PER_MOUSE} frames could be extracted "
            f"for mouse {mouse_id}."
        )

    return rows, global_output_number


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Create the complete Round 02 frame dataset."""

    print("\n========================================")
    print("FRONTVIEW TRAINING FRAME EXTRACTION")
    print("ROUND 02")
    print("========================================")
    print(f"Bpod root:        {BPOD_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Mice:             {len(MOUSE_IDS)}")
    print(f"Frames per mouse: {FRAMES_PER_MOUSE}")
    print(
        f"Target total:     "
        f"{len(MOUSE_IDS) * FRAMES_PER_MOUSE}"
    )
    print(f"Random seed:      {RANDOM_SEED}")

    validate_paths()

    random_generator = random.Random(
        RANDOM_SEED
    )

    all_rows: list[dict] = []
    global_output_number = 0

    for mouse_id in MOUSE_IDS:
        mouse_rows, global_output_number = process_mouse(
            mouse_id=mouse_id,
            random_generator=random_generator,
            global_output_number=global_output_number,
        )

        all_rows.extend(mouse_rows)

    if not all_rows:
        raise RuntimeError(
            "No frames were extracted."
        )

    write_manifest(all_rows)

    expected_total = (
        len(MOUSE_IDS) * FRAMES_PER_MOUSE
    )

    print("\n========================================")
    print("EXTRACTION COMPLETE")
    print("========================================")
    print(f"Frames extracted: {len(all_rows)}")
    print(f"Expected frames:  {expected_total}")
    print("\nFrames saved to:")
    print(OUTPUT_DIR)
    print("\nManifest saved to:")
    print(MANIFEST_PATH)

    print("\nFrames per mouse:")

    for mouse_id in MOUSE_IDS:
        count = sum(
            row["mouse_id"] == mouse_id
            for row in all_rows
        )

        print(f"  {mouse_id}: {count}")

    if len(all_rows) != expected_total:
        print(
            "\nWARNING:\n"
            "The target of 60 frames was not reached. "
            "Check the warnings above before labeling."
        )
    else:
        print(
            "\nSUCCESS: Round 02 contains exactly "
            f"{expected_total} frames."
        )


if __name__ == "__main__":
    main()