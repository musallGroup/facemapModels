"""
find_tongue_frames_v2.py

Batch-runs Frontview_Base_v1_7kp.pt on the middle five minutes of selected
BpodBehavior Cam0 videos and exports diverse tongue candidates.

Outputs per session:
- original PNG frames for manual labeling
- one CSV summary across all sessions

Important:
- Uses the same Facemap Pose setup as the working QC script.
- Does NOT use pandas.
- Does NOT create QC videos.
- Runs on CPU with the current fm_front environment.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from facemap.pose.pose import Pose

PROJECT_ROOT = Path(r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon")
MODEL_PATH = PROJECT_ROOT / "Current_Model" / "Frontview_Base_v1_7kp.pt"
OUTPUT_ROOT = PROJECT_ROOT / "Tongue_Candidates"
BPOD_ROOT = Path(r"\\NASKAMPA\lts\BpodBehavior")

SESSIONS = [
    ("426", "20260318_111731"),
    ("427", "20251120_105427"),
    ("428", "20251212_112811"),
    ("431", "20260325_102512"),
    ("460", "20260526_104742"),
    ("461", "20260601_112110"),
    ("462", "20260603_144921"),
]

KEYPOINT_NAMES = [
    "nose_tip", "nose_bottom", "mouth", "lowerlip",
    "whiskerpad_left", "whiskerpad_right", "tongue_tip",
]
TONGUE_INDEX = KEYPOINT_NAMES.index("tongue_tip")
SEGMENT_DURATION_SECONDS = 5 * 60.0
CANDIDATES_PER_SESSION = 8
MIN_SEPARATION_SECONDS = 1.5
PREFERRED_LIKELIHOOD = 0.50
BATCH_SIZE = 1
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


def validate_paths() -> None:
    if not BPOD_ROOT.exists():
        raise FileNotFoundError(f"Bpod root does not exist:\n{BPOD_ROOT}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model does not exist:\n{MODEL_PATH}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def natural_sort_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", str(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def is_cam0_video(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False
    name = path.stem.lower().replace("-", "_").replace(" ", "_")
    return any(token in name for token in ("cam0", "camera0", "camera_0", "cam_0"))


def candidate_session_directories(mouse: str, timestamp: str) -> list[Path]:
    candidates = [
        BPOD_ROOT / mouse / timestamp,
        BPOD_ROOT / mouse / f"{mouse}_{timestamp}",
        BPOD_ROOT / mouse / f"{mouse} {timestamp}",
        BPOD_ROOT / f"{mouse}_{timestamp}",
        BPOD_ROOT / f"{mouse} {timestamp}",
        BPOD_ROOT / timestamp,
    ]
    existing, seen = [], set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen and candidate.exists():
            existing.append(candidate)
            seen.add(key)
    return existing


def search_mouse_tree(mouse: str, timestamp: str) -> list[Path]:
    mouse_roots = []
    direct_mouse_root = BPOD_ROOT / mouse
    if direct_mouse_root.exists():
        mouse_roots.append(direct_mouse_root)
    try:
        for child in BPOD_ROOT.iterdir():
            if child.is_dir() and mouse in child.name and child not in mouse_roots:
                mouse_roots.append(child)
    except PermissionError:
        pass

    matches = []
    for root in mouse_roots:
        try:
            for path in root.rglob("*"):
                if path.is_dir() and timestamp in path.name:
                    matches.append(path)
        except (PermissionError, OSError):
            continue
    return sorted(set(matches), key=natural_sort_key)


def find_session_directory(mouse: str, timestamp: str) -> Path:
    direct_matches = candidate_session_directories(mouse, timestamp)
    if direct_matches:
        return direct_matches[0]
    fallback_matches = search_mouse_tree(mouse, timestamp)
    if fallback_matches:
        return fallback_matches[0]
    raise FileNotFoundError(
        f"Could not locate session directory for mouse {mouse}, timestamp {timestamp}\n"
        f"Search root: {BPOD_ROOT}"
    )


def find_cam0_video(session_directory: Path) -> Path:
    matches = []
    try:
        for path in session_directory.rglob("*"):
            if is_cam0_video(path):
                matches.append(path)
    except (PermissionError, OSError) as error:
        raise RuntimeError(f"Could not search:\n{session_directory}\n{error}") from error
    if not matches:
        raise FileNotFoundError(f"No Cam0 video found below:\n{session_directory}")
    matches.sort(key=natural_sort_key)
    try:
        matches.sort(key=lambda path: path.stat().st_size, reverse=True)
    except OSError:
        pass
    return matches[0]


def read_video_information(video_path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video:\n{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError(f"Invalid video metadata:\n{video_path}")
    return fps, total_frames, width, height


def create_middle_segment_indices(fps: float, total_frames: int) -> np.ndarray:
    video_duration_seconds = total_frames / fps
    duration_seconds = min(SEGMENT_DURATION_SECONDS, video_duration_seconds)
    start_seconds = max(0.0, (video_duration_seconds - duration_seconds) / 2.0)
    start_frame = int(round(start_seconds * fps))
    number_of_frames = int(round(duration_seconds * fps))
    end_frame_exclusive = min(start_frame + number_of_frames, total_frames)
    if end_frame_exclusive <= start_frame:
        raise RuntimeError("The selected middle segment contains no frames.")
    return np.arange(start_frame, end_frame_exclusive, dtype=np.int64)


def inspect_model_file() -> None:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    for head_name in (
        "Conv2_1x1.conv0.weight",
        "Conv2_1x1.conv1.weight",
        "Conv2_1x1.conv2.weight",
    ):
        if head_name not in state_dict:
            raise KeyError(f"Missing model head: {head_name}")
        if tuple(state_dict[head_name].shape)[0] != len(KEYPOINT_NAMES):
            raise RuntimeError(f"Wrong output count in {head_name}")


def create_pose_object(video_path: Path) -> Pose:
    pose = Pose(
        filenames=[[str(video_path)]], bbox=[], bbox_set=False,
        resize=False, add_padding=False, gui=None, GUIobject=None,
        net=None, model_name=str(MODEL_PATH),
    )
    pose.bodyparts = KEYPOINT_NAMES.copy()
    pose.batch_size = BATCH_SIZE
    pose.pose_prediction_setup()
    return pose


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_prediction_output(prediction_output: Any) -> np.ndarray:
    if isinstance(prediction_output, tuple):
        if not prediction_output:
            raise RuntimeError("Facemap returned an empty prediction tuple.")
        return to_numpy(prediction_output[0])
    return to_numpy(prediction_output)


def validate_prediction_shape(predictions: np.ndarray, number_of_frames: int) -> np.ndarray:
    expected = (number_of_frames, len(KEYPOINT_NAMES), 3)
    if predictions.shape == expected:
        return predictions
    alternative = (len(KEYPOINT_NAMES), number_of_frames, 3)
    if predictions.shape == alternative:
        return np.transpose(predictions, (1, 0, 2))
    raise RuntimeError(f"Unexpected prediction shape {predictions.shape}; expected {expected}.")


def rank_local_maxima(likelihood: np.ndarray) -> np.ndarray:
    safe = np.asarray(likelihood, dtype=np.float64).copy()
    safe[~np.isfinite(safe)] = -np.inf
    if len(safe) < 3:
        return np.argsort(safe)[::-1]
    local_maximum = np.zeros(len(safe), dtype=bool)
    local_maximum[1:-1] = (safe[1:-1] >= safe[:-2]) & (safe[1:-1] >= safe[2:])
    local_maximum[0] = safe[0] >= safe[1]
    local_maximum[-1] = safe[-1] >= safe[-2]
    local_indices = np.flatnonzero(local_maximum)
    local_indices = local_indices[np.argsort(safe[local_indices])[::-1]]
    all_ranked = np.argsort(safe)[::-1]
    remaining = all_ranked[~np.isin(all_ranked, local_indices)]
    return np.concatenate([local_indices, remaining])


def select_temporally_separated_candidates(likelihood: np.ndarray, fps: float) -> list[int]:
    ranked = rank_local_maxima(likelihood)
    minimum_distance_frames = max(1, int(round(MIN_SEPARATION_SECONDS * fps)))
    selected = []

    for index in ranked:
        index = int(index)
        score = float(likelihood[index])
        if not np.isfinite(score) or score < PREFERRED_LIKELIHOOD:
            continue
        if all(abs(index - previous) >= minimum_distance_frames for previous in selected):
            selected.append(index)
        if len(selected) >= CANDIDATES_PER_SESSION:
            break

    if len(selected) < CANDIDATES_PER_SESSION:
        for index in ranked:
            index = int(index)
            score = float(likelihood[index])
            if not np.isfinite(score) or index in selected:
                continue
            if all(abs(index - previous) >= minimum_distance_frames for previous in selected):
                selected.append(index)
            if len(selected) >= CANDIDATES_PER_SESSION:
                break
    return sorted(selected)


def export_candidate_frames(video_path: Path, session_output: Path, mouse: str,
                            timestamp: str, frame_indices: np.ndarray,
                            predictions: np.ndarray, selected_indices: list[int],
                            fps: float) -> list[dict[str, Any]]:
    session_output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video:\n{video_path}")
    rows = []
    try:
        for candidate_number, prediction_index in enumerate(selected_indices, start=1):
            source_frame = int(frame_indices[prediction_index])
            x, y, likelihood = predictions[prediction_index, TONGUE_INDEX]
            session_seconds = source_frame / fps
            file_name = (
                f"{mouse}_{timestamp}_candidate_{candidate_number:02d}_"
                f"frame_{source_frame}_t_{session_seconds:.3f}s_"
                f"likelihood_{float(likelihood):.3f}.png"
            )
            output_path = session_output / file_name
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not read source frame {source_frame}.")
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Could not save PNG:\n{output_path}")
            rows.append({
                "mouse": mouse,
                "session_timestamp": timestamp,
                "video_path": str(video_path),
                "candidate_number": candidate_number,
                "source_frame": source_frame,
                "session_time_seconds": round(session_seconds, 6),
                "session_time_minutes": round(session_seconds / 60.0, 6),
                "tongue_x": float(x),
                "tongue_y": float(y),
                "tongue_likelihood": float(likelihood),
                "above_preferred_threshold": float(likelihood) >= PREFERRED_LIKELIHOOD,
                "png_path": str(output_path),
            })
    finally:
        capture.release()
    return rows


def write_summary_csv(rows: list[dict[str, Any]]) -> Path:
    output_path = OUTPUT_ROOT / "tongue_candidates_summary.csv"
    fieldnames = [
        "mouse", "session_timestamp", "video_path", "candidate_number",
        "source_frame", "session_time_seconds", "session_time_minutes",
        "tongue_x", "tongue_y", "tongue_likelihood",
        "above_preferred_threshold", "png_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def process_session(mouse: str, timestamp: str) -> list[dict[str, Any]]:
    print("\n" + "=" * 72)
    print(f"SESSION: {mouse}  {timestamp}")
    print("=" * 72)
    session_directory = find_session_directory(mouse, timestamp)
    print(f"Session directory:\n{session_directory}")
    video_path = find_cam0_video(session_directory)
    print(f"\nCam0 video:\n{video_path}")
    fps, total_frames, width, height = read_video_information(video_path)
    print(f"\nFPS: {fps:.4f} | Frames: {total_frames} | Resolution: {width}x{height}")
    print(f"Duration: {total_frames / fps / 60.0:.3f} min")
    frame_indices = create_middle_segment_indices(fps, total_frames)
    print(f"Middle segment: {frame_indices[0] / fps / 60.0:.3f} to "
          f"{frame_indices[-1] / fps / 60.0:.3f} min ({len(frame_indices)} frames)")
    print("\nCreating Facemap Pose object...")
    pose = create_pose_object(video_path)
    print("Running prediction...")
    prediction_output = pose.predict_landmarks(video_id=0, frame_ind=frame_indices)
    predictions = normalize_prediction_output(prediction_output)
    predictions = validate_prediction_shape(predictions, len(frame_indices))
    tongue_likelihood = predictions[:, TONGUE_INDEX, 2]
    selected = select_temporally_separated_candidates(tongue_likelihood, fps)
    print("\nSelected candidates:")
    for number, prediction_index in enumerate(selected, start=1):
        source_frame = int(frame_indices[prediction_index])
        score = float(tongue_likelihood[prediction_index])
        print(f"  {number:02d}: frame={source_frame}, "
              f"time={source_frame / fps / 60.0:.3f} min, likelihood={score:.4f}")
    session_output = OUTPUT_ROOT / f"{mouse}_{timestamp}"
    rows = export_candidate_frames(
        video_path, session_output, mouse, timestamp,
        frame_indices, predictions, selected, fps,
    )
    print(f"\nSaved {len(rows)} PNG files to:\n{session_output}")
    return rows


def main() -> None:
    print("\n" + "=" * 72)
    print("FRONTVIEW TONGUE CANDIDATE FINDER V2")
    print("=" * 72)
    print(f"Model:\n{MODEL_PATH}")
    print(f"\nOutput root:\n{OUTPUT_ROOT}")
    print(f"\nSessions: {len(SESSIONS)}")
    print(f"Segment duration: {SEGMENT_DURATION_SECONDS / 60.0:.1f} min")
    print(f"Candidates per session: {CANDIDATES_PER_SESSION}")
    validate_paths()
    inspect_model_file()
    all_rows = []
    failures = []
    for mouse, timestamp in SESSIONS:
        try:
            all_rows.extend(process_session(mouse, timestamp))
        except Exception as error:
            failures.append((mouse, timestamp, str(error)))
            print("\n!!! SESSION FAILED !!!")
            print(f"{mouse}  {timestamp}\n{error}")
            print("Continuing with the next session.")
    summary_path = write_summary_csv(all_rows)
    print("\n" + "=" * 72)
    print("BATCH COMPLETE")
    print("=" * 72)
    print(f"Successful candidate rows: {len(all_rows)}")
    print(f"Failed sessions: {len(failures)}")
    print(f"\nSummary CSV:\n{summary_path}")
    print(f"\nCandidate folders:\n{OUTPUT_ROOT}")
    if failures:
        print("\nFAILED SESSIONS:")
        for mouse, timestamp, message in failures:
            print("-" * 72)
            print(f"{mouse}  {timestamp}\n{message}")
    print("\nReview the PNGs manually and keep only true visible-tongue frames.")


if __name__ == "__main__":
    main()