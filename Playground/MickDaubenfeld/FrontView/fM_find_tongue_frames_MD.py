# -*- coding: utf-8 -*-
"""
Batch-Suche nach Zungenframes in mehreren Frontview-Sessions.

Pro Session:
1) cam0-Video finden
2) mittlere 5 Minuten auswählen
3) Facemap-Pose nur auf diesen Frames ausführen
4) tongue_tip-Likelihood aus der H5 lesen
5) Lick-Events gruppieren
6) pro Event den besten Frame als PNG exportieren
7) CSV schreiben
"""

from __future__ import annotations

import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import pandas as pd
from facemap.pose.pose import Pose


MODEL_PATH = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon\Current_Model"
    r"\Frontview_Base_v1_7kp.pt"
)

OUTPUT_ROOT = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon"
    r"\Training_Data\Round_04_Tongue_Candidates"
)

SESSION_DIRS = [
    Path(r"\\Naskampa\lts\BpodBehavior\426\PuffyPenguin\Session Data\20260318_111731"),
    Path(r"\\Naskampa\lts\BpodBehavior\427\PuffyPenguin\Session Data\20251120_105427"),
    Path(r"\\Naskampa\lts\BpodBehavior\428\PuffyPenguin\Session Data\20251212_112811"),
    Path(r"\\Naskampa\lts\BpodBehavior\431\PuffyPenguin\Session Data\20260325_102512"),
    Path(r"\\Naskampa\lts\BpodBehavior\460\PuffyPenguin\Session Data\20260526_104742"),
    Path(r"\\Naskampa\lts\BpodBehavior\461\PuffyPenguin\Session Data\20260601_112110"),
    Path(r"\\Naskampa\lts\BpodBehavior\462\PuffyPenguin\Session Data\20260603_144921"),
]

SEGMENT_MINUTES = 5.0
MAX_CANDIDATES_PER_SESSION = 8
TONGUE_THRESHOLD = 0.55
MAX_EVENT_GAP_FRAMES = 3
MIN_SECONDS_BETWEEN_EXPORTED_EVENTS = 1.0
USE_TOP_PEAK_FALLBACK = True
BBOX = None


def find_cam0_video(session_dir: Path) -> Path:
    if not session_dir.exists():
        raise FileNotFoundError(f"Session-Ordner nicht gefunden:\n{session_dir}")

    candidates = []
    for pattern in ("*.avi", "*.AVI", "*.mp4", "*.MP4"):
        candidates.extend(session_dir.rglob(pattern))

    cam0 = [
        p for p in candidates
        if "cam0" in p.name.lower()
        and not any(x in p.name.lower() for x in ("qc", "render", "labeled", "preview"))
    ]

    if not cam0:
        raise FileNotFoundError(f"Keine cam0-Videodatei gefunden in:\n{session_dir}")

    cam0.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cam0[0]


def get_video_info(video_path: Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden:\n{video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            f"Ungültige Videometadaten: fps={fps}, frames={frame_count}\n{video_path}"
        )

    return fps, frame_count, frame_count / fps


def middle_segment_indices(
    fps: float,
    frame_count: int,
    segment_minutes: float,
) -> tuple[int, int, np.ndarray]:
    segment_frames = int(round(segment_minutes * 60 * fps))
    segment_frames = min(segment_frames, frame_count)
    start = max(0, (frame_count - segment_frames) // 2)
    end = min(frame_count, start + segment_frames)
    return start, end, np.arange(start, end, dtype=np.int64)


def snapshot_h5_files(folders: Iterable[Path]) -> dict[Path, tuple[int, int]]:
    state = {}
    for folder in folders:
        if not folder.exists():
            continue
        for pattern in ("*.h5", "*.hdf5"):
            for p in folder.rglob(pattern):
                try:
                    st = p.stat()
                    state[p.resolve()] = (st.st_size, st.st_mtime_ns)
                except OSError:
                    pass
    return state


def locate_new_or_changed_h5(
    before: dict[Path, tuple[int, int]],
    folders: Iterable[Path],
    started_ns: int,
) -> Path:
    candidates = []
    for folder in folders:
        if not folder.exists():
            continue
        for pattern in ("*.h5", "*.hdf5"):
            for p in folder.rglob(pattern):
                try:
                    rp = p.resolve()
                    st = p.stat()
                    previous = before.get(rp)
                    changed = previous is None or previous != (st.st_size, st.st_mtime_ns)
                    recent = st.st_mtime_ns >= started_ns
                    if changed or recent:
                        candidates.append(p)
                except OSError:
                    pass

    if not candidates:
        raise FileNotFoundError(
            "Facemap lief durch, aber es wurde keine neue/geänderte H5 gefunden."
        )

    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return candidates[0]


def instantiate_pose(video_path: Path) -> Pose:
    signature = inspect.signature(Pose)
    supported = set(signature.parameters)

    candidate_kwargs = {
        "filenames": [[str(video_path)]],
        "model_name": str(MODEL_PATH),
        "bbox": BBOX,
        "bbox_set": BBOX is not None,
    }
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}

    try:
        return Pose(**kwargs)
    except TypeError as exc:
        raise TypeError(
            "Pose konnte nicht initialisiert werden. Vergleiche die Parameter "
            "mit eurem funktionierenden Prediction-Script.\n"
            f"Erkannte Signatur: {signature}\n"
            f"Übergebene Parameter: {kwargs}"
        ) from exc


def run_facemap_segment(video_path: Path, frame_indices: np.ndarray) -> Path:
    search_folders = [video_path.parent, Path.cwd(), OUTPUT_ROOT]
    before = snapshot_h5_files(search_folders)
    started_ns = time.time_ns()

    pose = instantiate_pose(video_path)
    print(
        f"  Facemap-Prediction: {len(frame_indices):,} Frames "
        f"({frame_indices[0]:,} bis {frame_indices[-1]:,})"
    )
    pose.predict_landmarks(video_id=0, frame_ind=frame_indices)

    return locate_new_or_changed_h5(before, search_folders, started_ns)


def model_keypoint_names() -> list[str] | None:
    json_candidates = [
        MODEL_PATH.with_name(MODEL_PATH.stem + "_model_info.json"),
        MODEL_PATH.with_suffix(".json"),
    ]

    for path in json_candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for key in ("keypoints", "bodyparts", "keypoint_names", "labels"):
            value = data.get(key)
            if isinstance(value, list) and value:
                return [str(x) for x in value]

        for parent_key in ("config", "model", "metadata"):
            value = data.get(parent_key)
            if isinstance(value, dict):
                for key in ("keypoints", "bodyparts", "keypoint_names", "labels"):
                    names = value.get(key)
                    if isinstance(names, list) and names:
                        return [str(x) for x in names]
    return None


def read_tongue_likelihood_pandas(h5_path: Path) -> np.ndarray | None:
    try:
        with pd.HDFStore(h5_path, mode="r") as store:
            keys = store.keys()
    except Exception:
        return None

    for key in keys:
        try:
            obj = pd.read_hdf(h5_path, key=key)
        except Exception:
            continue
        if not isinstance(obj, pd.DataFrame):
            continue

        for col in obj.columns:
            joined = " ".join(str(x).lower() for x in col) if isinstance(col, tuple) else str(col).lower()
            if "tongue" in joined and any(x in joined for x in ("likelihood", "confidence", "prob")):
                return pd.to_numeric(obj[col], errors="coerce").to_numpy(dtype=float)
    return None


def read_tongue_likelihood_h5py(h5_path: Path) -> np.ndarray | None:
    datasets = []
    with h5py.File(h5_path, "r") as h5:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    datasets.append((name, np.asarray(obj)))
                except Exception:
                    pass
        h5.visititems(visitor)

    for name, arr in datasets:
        low = name.lower()
        if "tongue" in low and any(x in low for x in ("likelihood", "confidence", "prob")):
            return np.asarray(arr, dtype=float).squeeze()

    names = model_keypoint_names()
    tongue_index = None
    if names:
        for i, name in enumerate(names):
            if "tongue" in name.lower():
                tongue_index = i
                break

    if tongue_index is not None:
        for name, arr in datasets:
            low = name.lower()
            if any(x in low for x in ("likelihood", "confidence", "prob")) and arr.ndim == 2:
                if arr.shape[1] > tongue_index:
                    return np.asarray(arr[:, tongue_index], dtype=float)
                if arr.shape[0] > tongue_index:
                    return np.asarray(arr[tongue_index, :], dtype=float)

    print("  Konnte tongue_tip-Likelihood nicht automatisch finden.")
    print("  Vorhandene H5-Datasets:")
    for name, arr in datasets:
        print(f"    {name}: shape={arr.shape}, dtype={arr.dtype}")
    return None


def read_tongue_likelihood(h5_path: Path) -> np.ndarray:
    values = read_tongue_likelihood_pandas(h5_path)
    if values is None:
        values = read_tongue_likelihood_h5py(h5_path)
    if values is None:
        raise KeyError("tongue_tip-Likelihood konnte in der H5 nicht gefunden werden.")
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError("Leeres Likelihood-Array.")
    return values


def group_threshold_events(
    likelihood: np.ndarray,
    threshold: float,
    max_gap_frames: int,
) -> list[tuple[int, int, int, float]]:
    above = np.flatnonzero(np.isfinite(likelihood) & (likelihood >= threshold))
    if len(above) == 0:
        return []

    groups = [[int(above[0])]]
    for idx in above[1:]:
        idx = int(idx)
        if idx - groups[-1][-1] <= max_gap_frames + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    events = []
    for group in groups:
        start, end = group[0], group[-1]
        peak = start + int(np.nanargmax(likelihood[start:end + 1]))
        events.append((start, end, peak, float(likelihood[peak])))
    return events


def local_maxima(likelihood: np.ndarray) -> list[int]:
    if len(likelihood) < 3:
        return list(range(len(likelihood)))
    finite = np.nan_to_num(likelihood, nan=-np.inf)
    mask = (finite[1:-1] >= finite[:-2]) & (finite[1:-1] >= finite[2:])
    return (np.flatnonzero(mask) + 1).tolist()


def select_candidates(
    likelihood: np.ndarray,
    fps: float,
    max_candidates: int,
) -> list[tuple[int, float, str]]:
    events = group_threshold_events(
        likelihood,
        threshold=TONGUE_THRESHOLD,
        max_gap_frames=MAX_EVENT_GAP_FRAMES,
    )

    ranked = sorted(
        [(peak, score, "threshold_event") for _, _, peak, score in events],
        key=lambda x: x[1],
        reverse=True,
    )

    if USE_TOP_PEAK_FALLBACK and len(ranked) < max_candidates:
        existing = {x[0] for x in ranked}
        extras = sorted(
            [
                (idx, float(likelihood[idx]), "top_peak_fallback")
                for idx in local_maxima(likelihood)
                if idx not in existing and np.isfinite(likelihood[idx])
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        ranked.extend(extras)

    min_distance = max(1, int(round(MIN_SECONDS_BETWEEN_EXPORTED_EVENTS * fps)))
    selected = []
    for item in ranked:
        idx = item[0]
        if all(abs(idx - prev[0]) >= min_distance for prev in selected):
            selected.append(item)
        if len(selected) >= max_candidates:
            break

    return sorted(selected, key=lambda x: x[0])


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}h{m:02d}m{s:02d}s{ms:03d}ms"


def process_session(session_dir: Path) -> list[dict]:
    mouse = session_dir.parts[-4] if len(session_dir.parts) >= 4 else "unknown_mouse"
    session = session_dir.name

    print("\n" + "=" * 90)
    print(f"Maus {mouse} | Session {session}")

    video_path = find_cam0_video(session_dir)
    print(f"  Video: {video_path}")

    fps, frame_count, duration_s = get_video_info(video_path)
    start_frame, end_frame, frame_indices = middle_segment_indices(
        fps, frame_count, SEGMENT_MINUTES
    )

    print(
        f"  FPS: {fps:.3f} | Frames: {frame_count:,} | "
        f"Dauer: {duration_s / 60:.2f} min"
    )
    print(
        f"  Segment: {start_frame / fps / 60:.2f} bis "
        f"{end_frame / fps / 60:.2f} min"
    )

    h5_path = run_facemap_segment(video_path, frame_indices)
    print(f"  H5 gefunden: {h5_path}")

    session_out = OUTPUT_ROOT / f"mouse-{mouse}_session-{session}"
    session_out.mkdir(parents=True, exist_ok=True)

    copied_h5 = session_out / h5_path.name
    if h5_path.resolve() != copied_h5.resolve():
        shutil.copy2(h5_path, copied_h5)

    likelihood = read_tongue_likelihood(h5_path)

    if len(likelihood) == len(frame_indices):
        local_likelihood = likelihood
    elif len(likelihood) >= end_frame:
        local_likelihood = likelihood[start_frame:end_frame]
    else:
        raise ValueError(
            f"Unerwartete H5-Länge: {len(likelihood):,}. "
            f"Erwartet: {len(frame_indices):,} Segmentframes "
            f"oder mindestens {end_frame:,} Gesamtframes."
        )

    selected = select_candidates(
        local_likelihood,
        fps=fps,
        max_candidates=MAX_CANDIDATES_PER_SESSION,
    )

    event_count = len(group_threshold_events(
        local_likelihood, TONGUE_THRESHOLD, MAX_EVENT_GAP_FRAMES
    ))
    print(f"  Gefundene Schwellen-Events: {event_count}")
    print(f"  Exportiere {len(selected)} Kandidaten.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht erneut geöffnet werden:\n{video_path}")

    rows = []
    try:
        for rank, (local_idx, score, reason) in enumerate(selected, start=1):
            global_frame = start_frame + int(local_idx)
            timestamp_s = global_frame / fps

            cap.set(cv2.CAP_PROP_POS_FRAMES, global_frame)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Frame {global_frame} konnte nicht gelesen werden.")

            timestamp_text = format_timestamp(timestamp_s)
            filename = (
                f"mouse-{mouse}_session-{session}"
                f"_rank-{rank:02d}"
                f"_time-{timestamp_text}"
                f"_frame-{global_frame:09d}"
                f"_tongueLik-{score:.3f}.png"
            )
            image_path = session_out / filename

            if not cv2.imwrite(str(image_path), frame):
                raise IOError(f"PNG konnte nicht gespeichert werden:\n{image_path}")

            rows.append({
                "mouse": mouse,
                "session": session,
                "video_path": str(video_path),
                "segment_start_frame": start_frame,
                "segment_end_frame": end_frame,
                "global_frame": global_frame,
                "timestamp_seconds": timestamp_s,
                "timestamp": timestamp_text,
                "tongue_likelihood": score,
                "selection_reason": reason,
                "image_path": str(image_path),
                "h5_path": str(copied_h5),
            })
    finally:
        cap.release()

    pd.DataFrame(rows).to_csv(
        session_out / "tongue_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return rows


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden:\n{MODEL_PATH}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    failures = []
    start_time = time.time()

    for session_dir in SESSION_DIRS:
        try:
            all_rows.extend(process_session(session_dir))
        except Exception as exc:
            print(f"\nFEHLER bei {session_dir}:\n{type(exc).__name__}: {exc}")
            failures.append({
                "session_dir": str(session_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            OUTPUT_ROOT / "ALL_tongue_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if failures:
        pd.DataFrame(failures).to_csv(
            OUTPUT_ROOT / "FAILED_sessions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    elapsed_min = (time.time() - start_time) / 60
    print("\n" + "=" * 90)
    print("FERTIG")
    print(f"Exportierte Kandidaten insgesamt: {len(all_rows)}")
    print(f"Fehlgeschlagene Sessions: {len(failures)}")
    print(f"Gesamtdauer: {elapsed_min:.1f} Minuten")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()