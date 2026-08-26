#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import pickle
import re
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch


# ============================================================
# USER SETTINGS
# ============================================================

MOUSE_ID = "481"

SESSIONS_ROOT = Path(
    r"\\Naskampa\lts\Projects\singleNeuronLearning\2p_PuffyPenguin\481"
)

MODEL_PATH = Path(
    r"\\Naskampa\lts\Team\Mick\FM_Side_View\00_LabelForge_Basemodels\SideView_Face_2P_v1"
    r"\SideView_Face_2P_v1.pt"
)

OUTPUT_ROOT = Path(
    r"\\Naskampa\lts\Team\Mick\FM_Side_View\2 Photon\New_Dataset\Cohort"
)

VIDEO_GLOB = "*_cam1_*.avi"

BBOX = [0, 1024, 0, 1280]
RESIZE = True
ADD_PADDING = True

RUN_POSE = True
SKIP_POSE_IF_H5_EXISTS = True

LIKELIHOOD_THRESHOLD = 0.80

SAVE_PER_FRAME_LONG = True
PREFER_PARQUET = True

EXCLUDED_LABELS = {
    "paw",
    "nose(r)",
}


# ============================================================
# PATHS
# ============================================================

@dataclass
class MousePaths:
    mouse: str
    base: Path
    raw_pose: Path
    per_frame: Path
    per_label_summary: Path
    global_summary: Path


def make_mouse_paths(mouse: str) -> MousePaths:
    base = OUTPUT_ROOT / f"Mouse_{mouse}"

    paths = MousePaths(
        mouse=mouse,
        base=base,
        raw_pose=base / "raw_pose",
        per_frame=base / "per_frame",
        per_label_summary=base / "per_label_summary",
        global_summary=base / "global_summary",
    )

    for p in [
        paths.base,
        paths.raw_pose,
        paths.per_frame,
        paths.per_label_summary,
        paths.global_summary,
    ]:
        p.mkdir(parents=True, exist_ok=True)

    return paths


def cohort_summary_dir() -> Path:
    out = OUTPUT_ROOT / "Cohort_Summary"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================
# HELPERS
# ============================================================

def is_session_folder(path: Path) -> bool:
    return bool(re.match(r"^\d{8}_\d{6}$", path.name))


def parse_video_info(video_path: Path) -> Dict[str, Optional[str]]:
    stem = video_path.stem
    parts = stem.split("_")

    info = {
        "video_stem": stem,
        "mouse": parts[0] if len(parts) > 0 else None,
        "animal": parts[1] if len(parts) > 1 else None,
        "session_date": parts[2] if len(parts) > 2 else None,
        "session_time": parts[3] if len(parts) > 3 else None,
        "camera": parts[4] if len(parts) > 4 else None,
        "video_date": parts[5] if len(parts) > 5 else None,
    }

    if info["mouse"] and info["animal"] and info["session_date"] and info["session_time"] and info["camera"]:
        info["session_id"] = (
            f'{info["mouse"]}_{info["animal"]}_'
            f'{info["session_date"]}_{info["session_time"]}_{info["camera"]}'
        )
    else:
        info["session_id"] = stem

    return info


def find_videos() -> List[Path]:
    if not SESSIONS_ROOT.exists():
        raise FileNotFoundError(f"Sessions root not found:\n{SESSIONS_ROOT}")

    videos = []

    for session_dir in sorted(SESSIONS_ROOT.iterdir()):
        if not session_dir.is_dir():
            continue

        if not is_session_folder(session_dir):
            continue

        session_videos = sorted(session_dir.glob(VIDEO_GLOB))

        if len(session_videos) == 0:
            print(f"[WARN] No cam1 video in: {session_dir.name}")
            continue

        if len(session_videos) > 1:
            print(f"[WARN] Multiple cam1 videos in {session_dir.name}, using first:")
            for v in session_videos:
                print(f"       {v.name}")

        videos.append(session_videos[0])

    print(f"[INFO] Found {len(videos)} cam1 videos for mouse {MOUSE_ID}")
    return videos


def expected_h5_name(video_path: Path) -> str:
    return f"{video_path.stem}_FacemapPose.h5"


def expected_pkl_name(video_path: Path) -> str:
    return f"{video_path.stem}_FacemapPose_metadata.pkl"


def save_table(df: pd.DataFrame, path_no_suffix: Path, prefer_parquet: bool = True) -> Path:
    if prefer_parquet:
        try:
            out = path_no_suffix.with_suffix(".parquet")
            df.to_parquet(out, index=False)
            return out
        except Exception:
            pass

    out = path_no_suffix.with_suffix(".csv.gz")
    df.to_csv(out, index=False, compression="gzip")
    return out


def is_real_facemap_h5(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        with h5py.File(path, "r") as f:
            if "Facemap" not in f:
                return False

            g = f["Facemap"]

            if len(g.keys()) == 0:
                return False

            for label in g.keys():
                if label in EXCLUDED_LABELS:
                    continue

                lg = g[label]

                if "x" in lg and "y" in lg:
                    return True

        return False

    except Exception:
        return False


def save_pose_result_manually(result_tensor, meta_dict, h5_path: Path, metadata_path: Path) -> None:
    if isinstance(result_tensor, torch.Tensor):
        arr = result_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(result_tensor)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    n_frames, n_bodyparts, n_coords = arr.shape

    if n_coords != 3:
        raise ValueError(f"Expected last dim = 3, got shape {arr.shape}")

    bodyparts = meta_dict.get("bodyparts", None)

    if bodyparts is None or len(bodyparts) != n_bodyparts:
        bodyparts = [f"label_{i}" for i in range(n_bodyparts)]

    with h5py.File(h5_path, "w") as f:
        g = f.create_group("Facemap")

        for i, bp in enumerate(bodyparts):
            bp_name = str(bp)
            bg = g.create_group(bp_name)

            bg.create_dataset("x", data=arr[:, i, 0], compression="gzip")
            bg.create_dataset("y", data=arr[:, i, 1], compression="gzip")
            bg.create_dataset("likelihood", data=arr[:, i, 2], compression="gzip")

    with open(metadata_path, "wb") as f:
        pickle.dump(meta_dict, f)


# ============================================================
# FACEMAP INFERENCE
# ============================================================

def run_facemap_pose_on_video(video_path: Path, out_raw_dir: Path) -> Tuple[Optional[Path], Optional[Path], str]:
    t0 = time.time()

    from facemap.pose.pose import Pose

    expected_h5 = out_raw_dir / expected_h5_name(video_path)
    expected_pkl = out_raw_dir / expected_pkl_name(video_path)

    if SKIP_POSE_IF_H5_EXISTS and is_real_facemap_h5(expected_h5):
        print(f"[SKIP POSE] Existing H5: {expected_h5.name}")
        return expected_h5, expected_pkl if expected_pkl.exists() else None, "SKIPPED_EXISTING"

    print(f"[POSE] {video_path}")

    pose = Pose(
        filenames=[[str(video_path)]],
        bbox=[BBOX],
        bbox_set=True,
        resize=RESIZE,
        add_padding=ADD_PADDING,
        model_name=str(MODEL_PATH),
    )

    pose.load_model()
    result = pose.predict_landmarks(video_id=0)

    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(f"Unexpected predict_landmarks output type: {type(result)}")

    tensor, meta = result

    print("[INFO] Saving pose output manually...")
    save_pose_result_manually(tensor, meta, expected_h5, expected_pkl)

    if not is_real_facemap_h5(expected_h5):
        print(f"[WARN] H5 not valid after save: {expected_h5.name}")
        return None, expected_pkl if expected_pkl.exists() else None, "FAILED_H5_INVALID"

    dt = time.time() - t0
    print(f"[OK] Saved H5: {expected_h5.name}")
    print(f"[TIME] Pose took {dt / 60:.2f} min")

    return expected_h5, expected_pkl if expected_pkl.exists() else None, "SUCCESS"


# ============================================================
# ANALYSIS
# ============================================================

def load_facemap_h5_as_long(path: Path) -> pd.DataFrame:
    rows = []

    with h5py.File(path, "r") as f:
        if "Facemap" not in f:
            raise ValueError(f"No 'Facemap' group found in H5: {path}")

        g = f["Facemap"]

        for label in g.keys():
            if label in EXCLUDED_LABELS:
                continue

            label_group = g[label]

            if "x" not in label_group or "y" not in label_group:
                continue

            x = np.asarray(label_group["x"][:], dtype=float).reshape(-1)
            y = np.asarray(label_group["y"][:], dtype=float).reshape(-1)

            if "likelihood" in label_group:
                likelihood = np.asarray(label_group["likelihood"][:], dtype=float).reshape(-1)
            else:
                likelihood = np.full(len(x), np.nan, dtype=float)

            n = min(len(x), len(y), len(likelihood))

            if n == 0:
                continue

            rows.append(
                pd.DataFrame(
                    {
                        "frame": np.arange(n, dtype=int),
                        "label": str(label),
                        "x": x[:n],
                        "y": y[:n],
                        "likelihood": likelihood[:n],
                    }
                )
            )

    if not rows:
        raise ValueError(f"No usable labels found in H5: {path}")

    return pd.concat(rows, ignore_index=True)


def add_movement_columns_from_long(per_frame_long: pd.DataFrame) -> pd.DataFrame:
    out = per_frame_long.copy()
    out = out.sort_values(["label", "frame"]).reset_index(drop=True)

    out["dx"] = out.groupby("label")["x"].diff()
    out["dy"] = out.groupby("label")["y"].diff()
    out["movement"] = np.sqrt(out["dx"] ** 2 + out["dy"] ** 2)

    out["valid_point"] = out["likelihood"] >= LIKELIHOOD_THRESHOLD
    prev_valid = out.groupby("label")["valid_point"].shift(1).fillna(False)
    out["valid_pair"] = out["valid_point"] & prev_valid

    out.loc[~out["valid_pair"], ["dx", "dy", "movement"]] = np.nan

    return out


def summarize_label_movement(per_frame_long: pd.DataFrame) -> pd.DataFrame:
    def safe_p95(x: pd.Series) -> float:
        x = x.dropna()
        return float(np.nan) if len(x) == 0 else float(np.percentile(x, 95))

    def high_motion_fraction(x: pd.Series) -> float:
        x = x.dropna()

        if len(x) == 0:
            return float(np.nan)

        threshold = np.median(x) + 2 * np.std(x)
        return float(np.mean(x > threshold))

    grp = per_frame_long.groupby("label", dropna=False)

    out = grp["movement"].agg(
        n_frames_used=lambda x: int(x.notna().sum()),
        mean_motion="mean",
        median_motion="median",
        std_motion="std",
        max_motion="max",
    ).reset_index()

    out["p95_motion"] = grp["movement"].apply(safe_p95).values
    out["high_motion_fraction"] = grp["movement"].apply(high_motion_fraction).values

    return out


def summarize_global_session(per_frame_long: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frame_global = per_frame_long.groupby("frame", dropna=False).agg(
        mean_motion=("movement", "mean"),
        median_motion=("movement", "median"),
        std_motion=("movement", "std"),
        n_labels_used=("movement", lambda x: int(x.notna().sum())),
    ).reset_index()

    clean = frame_global["mean_motion"].dropna().to_numpy(dtype=float)

    if len(clean) == 0:
        summary = {
            "n_frames_used": 0,
            "mean_motion": np.nan,
            "median_motion": np.nan,
            "std_motion": np.nan,
            "p95_motion": np.nan,
            "high_motion_fraction": np.nan,
        }
    else:
        threshold = np.median(clean) + 2 * np.std(clean)

        summary = {
            "n_frames_used": int(len(clean)),
            "mean_motion": float(np.mean(clean)),
            "median_motion": float(np.median(clean)),
            "std_motion": float(np.std(clean)),
            "p95_motion": float(np.percentile(clean, 95)),
            "high_motion_fraction": float(np.mean(clean > threshold)),
        }

    return frame_global, summary


def analyze_one_h5(h5_path: Path, mouse_paths: MousePaths) -> Tuple[Dict[str, object], pd.DataFrame]:
    t0 = time.time()

    stem_base = h5_path.name.replace("_FacemapPose.h5", "")
    info = parse_video_info(Path(stem_base))

    per_frame_base = mouse_paths.per_frame / f"{stem_base}_framewise"
    per_label_csv = mouse_paths.per_label_summary / f"{stem_base}_all_labels_summary.csv"
    frame_global_csv = mouse_paths.global_summary / f"{stem_base}_frame_global.csv"

    per_frame_long = load_facemap_h5_as_long(h5_path)
    per_frame_long = add_movement_columns_from_long(per_frame_long)

    if SAVE_PER_FRAME_LONG:
        save_table(per_frame_long, per_frame_base, prefer_parquet=PREFER_PARQUET)

    per_label = summarize_label_movement(per_frame_long)

    for key in ["video_stem", "mouse", "animal", "session_id", "session_date", "session_time", "camera"]:
        per_label.insert(0, key, info.get(key) if key != "video_stem" else stem_base)

    per_label.to_csv(per_label_csv, index=False)

    frame_global, session_summary = summarize_global_session(per_frame_long)

    for key in ["video_stem", "mouse", "animal", "session_id", "session_date", "session_time", "camera"]:
        frame_global.insert(0, key, info.get(key) if key != "video_stem" else stem_base)

    frame_global.to_csv(frame_global_csv, index=False)

    out = {
        "video_stem": stem_base,
        "mouse": info.get("mouse"),
        "animal": info.get("animal"),
        "session_id": info.get("session_id"),
        "session_date": info.get("session_date"),
        "session_time": info.get("session_time"),
        "camera": info.get("camera"),
        "n_labels": int(per_frame_long["label"].nunique()),
        "raw_h5": str(h5_path),
        **session_summary,
    }

    dt = time.time() - t0
    print(f"[TIME] Analysis for {h5_path.name} took {dt:.2f} s")

    return out, per_label


def build_mouse_outputs(mouse_paths: MousePaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    h5_files = sorted(mouse_paths.raw_pose.glob("*_FacemapPose.h5"))
    h5_files = [h5 for h5 in h5_files if is_real_facemap_h5(h5)]

    session_rows = []
    label_tables = []

    for h5 in h5_files:
        print(f"[ANALYZE] {h5.name}")

        try:
            session_row, per_label = analyze_one_h5(h5, mouse_paths)
            session_rows.append(session_row)
            label_tables.append(per_label)

        except Exception as e:
            print(f"[FAIL ANALYZE] {h5.name}")
            traceback.print_exc()

            stem_base = h5.name.replace("_FacemapPose.h5", "")
            info = parse_video_info(Path(stem_base))

            session_rows.append(
                {
                    "video_stem": stem_base,
                    "mouse": info.get("mouse"),
                    "animal": info.get("animal"),
                    "session_id": info.get("session_id"),
                    "session_date": info.get("session_date"),
                    "session_time": info.get("session_time"),
                    "error": str(e),
                }
            )

    session_summary = pd.DataFrame(session_rows)

    if not session_summary.empty:
        session_summary = session_summary.sort_values(
            ["mouse", "session_date", "session_time", "video_stem"],
            na_position="last",
        ).reset_index(drop=True)

        session_summary["session_index"] = np.arange(1, len(session_summary) + 1)
        session_summary.to_csv(mouse_paths.global_summary / "session_summary.csv", index=False)

    all_labels = pd.concat(label_tables, ignore_index=True) if label_tables else pd.DataFrame()

    if not all_labels.empty:
        all_labels.to_csv(mouse_paths.global_summary / "all_sessions_all_labels.csv", index=False)

    return session_summary, all_labels


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    total_t0 = time.time()

    print("=" * 70)
    print("2P NEW DATASET MOUSE 481 FACEMAP COHORT EXTRACTION")
    print("=" * 70)
    print(f"Sessions root : {SESSIONS_ROOT}")
    print(f"Model path    : {MODEL_PATH}")
    print(f"Output root   : {OUTPUT_ROOT}")
    print(f"Mouse         : {MOUSE_ID}")
    print(f"Video glob    : {VIDEO_GLOB}")
    print(f"Likelihood threshold: {LIKELIHOOD_THRESHOLD}")
    print(f"Excluded labels: {EXCLUDED_LABELS}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n{MODEL_PATH}")

    mouse_paths = make_mouse_paths(MOUSE_ID)
    videos = find_videos()

    processing_log = []

    if RUN_POSE:
        for i, video in enumerate(videos, start=1):
            print("\n" + "-" * 70)
            print(f"[VIDEO {i}/{len(videos)}] {video.name}")
            print("-" * 70)

            info = parse_video_info(video)

            try:
                h5, pkl, status = run_facemap_pose_on_video(video, mouse_paths.raw_pose)

                processing_log.append(
                    {
                        **info,
                        "video_path": str(video),
                        "status": status,
                        "raw_h5": str(h5) if h5 else "",
                    }
                )

            except Exception as e:
                print(f"[FAIL POSE] {video}")
                traceback.print_exc()

                processing_log.append(
                    {
                        **info,
                        "video_path": str(video),
                        "status": "FAILED_POSE",
                        "error": str(e),
                    }
                )

    session_summary, all_labels = build_mouse_outputs(mouse_paths)

    out_cohort = cohort_summary_dir()

    if not session_summary.empty:
        session_summary.to_csv(
            out_cohort / "all_sessions_global_summary.csv",
            index=False,
        )

    if not all_labels.empty:
        all_labels.to_csv(
            out_cohort / "all_sessions_all_labels.csv",
            index=False,
        )

    if processing_log:
        pd.DataFrame(processing_log).to_csv(
            out_cohort / "processing_log.csv",
            index=False,
        )

    total_dt = time.time() - total_t0

    print("\nDONE.")
    print(f"[TOTAL TIME] Full runtime: {total_dt / 60:.2f} min")
    print(f"[OUTPUT] {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
