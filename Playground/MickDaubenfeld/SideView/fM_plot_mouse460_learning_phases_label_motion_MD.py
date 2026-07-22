#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

MOUSE_ID = "460"

raw_pose_dir = Path(
    r"Z:\Team\Mick\FM_Dennis_Cohort\Mouse_460\raw_pose"
)

task_csv = Path(
    r"C:\Users\daubenfeld\Desktop\Widefield_Imaging_FZJ - 458_460.csv"
)

out_png = Path(
    r"C:\Users\daubenfeld\Desktop\mouse_460_learning_plus_labelmotion_median_style.png"
)

out_csv = Path(
    r"C:\Users\daubenfeld\Desktop\mouse_460_learning_plus_labelmotion_median_style.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LIKELIHOOD_THRESHOLD = 0.90
SMOOTH_WINDOW = 7

LABEL_LOWERLIP = "lowerlip"
LABEL_NOSE = "nose(top)"
LABEL_WHISKERS = ["whisker(I)", "whisker(II)", "whisker(III)"]

COLOR_VISUAL = "#e83e8c"   # pink
COLOR_TACTILE = "#f39c12"  # orange
COLOR_MULTI = "#2ec4c7"    # turquoise
COLOR_DISC = "#7a3db8"     # purple

COLOR_LIP = "#1f77b4"
COLOR_NOSE = "#ff7f0e"
COLOR_WHISK = "#2ca02c"

VALID_TASKS = ["Tactile", "Visual", "Multi", "Disc"]

TASK_COLORS = {
    "Visual": COLOR_VISUAL,
    "Tactile": COLOR_TACTILE,
    "Multi": COLOR_MULTI,
    "Disc": COLOR_DISC,
}

PERFORMANCE_MODALITIES = {
    "Visual": {
        "col": "perfVisual",
        "color": COLOR_VISUAL,
        "label": "Visual",
    },
    "Tactile": {
        "col": "perfTactile",
        "color": COLOR_TACTILE,
        "label": "Tactile",
    },
    "Multi": {
        "col": "perfMulti",
        "color": COLOR_MULTI,
        "label": "Multi",
    },
}


# ============================================================
# HELPERS
# ============================================================

def parse_session_date_from_name(filename: str) -> str:
    parts = filename.split("_")
    return parts[2] if len(parts) >= 3 else ""


def compute_label_motion_median(label_group, likelihood_threshold=0.90) -> float:
    x = np.asarray(label_group["x"][:], dtype=float)
    y = np.asarray(label_group["y"][:], dtype=float)

    if "likelihood" in label_group:
        likelihood = np.asarray(label_group["likelihood"][:], dtype=float)
    else:
        likelihood = np.full_like(x, np.nan)

    dx = np.diff(x)
    dy = np.diff(y)
    movement = np.sqrt(dx**2 + dy**2)

    valid = (likelihood[1:] >= likelihood_threshold) & (
        likelihood[:-1] >= likelihood_threshold
    )
    movement[~valid] = np.nan

    if np.all(np.isnan(movement)):
        return np.nan

    return float(np.nanmedian(movement))


def zscore_series(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=0)

    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - mu) / sd


def forward_smooth_from_first_valid(s: pd.Series, window: int = 7) -> pd.Series:
    out = s[::-1].rolling(window=window, min_periods=1).mean()[::-1]
    valid_idx = np.flatnonzero(~s.isna().to_numpy())

    if len(valid_idx) == 0:
        return pd.Series(np.nan, index=s.index)

    out.iloc[:valid_idx[0]] = np.nan
    return out


# ============================================================
# LOAD TASK / PERFORMANCE INFO FOR MOUSE 460
# ============================================================

raw = pd.read_csv(task_csv, header=None)

# CSV layout:
# column 0 = DATE
# column 7 = Mouse 460 Task
# column 8 = Mouse 460 Performance
# column 10 = Mouse 460 Trials

task_df = raw.iloc[2:, [0, 7, 8, 10]].copy()
task_df.columns = ["date_text", "Task", "Performance", "Trials"]

task_df["Task"] = task_df["Task"].astype(str).str.strip()
task_df["Performance"] = pd.to_numeric(task_df["Performance"], errors="coerce")
task_df["Trials"] = pd.to_numeric(task_df["Trials"], errors="coerce")

task_df["session_date"] = pd.to_datetime(
    task_df["date_text"],
    dayfirst=True,
    errors="coerce"
).dt.strftime("%Y%m%d")

task_df = task_df.dropna(subset=["session_date"]).copy()
task_df = task_df[task_df["Task"].isin(VALID_TASKS)].copy()

if task_df.empty:
    raise ValueError("No valid Tactile/Visual/Multi/Disc rows found in task CSV.")

# Convert percent to fraction if needed
if task_df["Performance"].max(skipna=True) > 1.5:
    task_df["Performance"] = task_df["Performance"] / 100.0

task_df = task_df[["session_date", "Task", "Performance", "Trials"]]
task_df = task_df.drop_duplicates(subset=["session_date"], keep="first")


# ============================================================
# LOAD FACEMAP LABEL MOTION FROM H5
# ============================================================

rows = []

h5_files = sorted(raw_pose_dir.glob("*_FacemapPose.h5"))

if not h5_files:
    raise FileNotFoundError(f"No FacemapPose H5 files found in: {raw_pose_dir}")

for h5_file in h5_files:
    session_date = parse_session_date_from_name(h5_file.name)

    try:
        with h5py.File(h5_file, "r") as f:
            if "Facemap" not in f:
                continue

            g = f["Facemap"]

            row = {
                "session_date": session_date,
                "h5_file": h5_file.name,
            }

            row["lowerlip"] = (
                compute_label_motion_median(g[LABEL_LOWERLIP], LIKELIHOOD_THRESHOLD)
                if LABEL_LOWERLIP in g else np.nan
            )

            row["nose_top"] = (
                compute_label_motion_median(g[LABEL_NOSE], LIKELIHOOD_THRESHOLD)
                if LABEL_NOSE in g else np.nan
            )

            whisker_vals = []

            for w in LABEL_WHISKERS:
                if w in g:
                    value = compute_label_motion_median(g[w], LIKELIHOOD_THRESHOLD)

                    if not np.isnan(value):
                        whisker_vals.append(value)

            row["whisker_median"] = (
                float(np.median(whisker_vals)) if len(whisker_vals) > 0 else np.nan
            )

            rows.append(row)

    except Exception as e:
        print(f"FAILED: {h5_file.name} -> {e}")

label_df = pd.DataFrame(rows)

if label_df.empty:
    raise ValueError("No usable Facemap label data found.")

label_df = (
    label_df
    .groupby("session_date", as_index=False)[["lowerlip", "nose_top", "whisker_median"]]
    .mean()
    .sort_values("session_date")
    .reset_index(drop=True)
)


# ============================================================
# MERGE TASK/PERFORMANCE + LABEL MOTION
# ============================================================

merged = task_df.merge(label_df, on="session_date", how="inner")
merged = merged.sort_values("session_date").reset_index(drop=True)

if merged.empty:
    raise ValueError("No overlap between task CSV dates and Facemap H5 dates.")

merged["session_day_index"] = range(1, len(merged) + 1)


# ============================================================
# CREATE MODALITY-SPECIFIC PERFORMANCE COLUMNS
# Important: color is tied to modality, NOT to task order.
# Therefore Tactile is orange even if it is the first phase.
# ============================================================

for task, spec in PERFORMANCE_MODALITIES.items():
    col = spec["col"]

    merged[col] = np.where(
        merged["Task"] == task,
        merged["Performance"],
        np.nan
    )

    merged[f"{col}_smooth"] = forward_smooth_from_first_valid(
        pd.Series(merged[col]),
        SMOOTH_WINDOW
    )


# ============================================================
# Z-SCORE LABEL MOTION LIKE OLD 427/428 FIGURE
# ============================================================

for col in ["lowerlip", "nose_top", "whisker_median"]:
    merged[f"{col}_z"] = zscore_series(merged[col])
    merged[f"{col}_z_smooth"] = (
        merged[f"{col}_z"]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
        .mean()
    )


# ============================================================
# TASK SWITCHES
# ============================================================

switches = []
prev_task = None

for _, row in merged.iterrows():
    task = row["Task"]
    x = row["session_day_index"]

    if prev_task is None or task != prev_task:
        switches.append((x, task))

    prev_task = task


# ============================================================
# SAVE MERGED CSV
# ============================================================

merged.to_csv(out_csv, index=False)
print(f"Saved merged CSV to: {out_csv}")


# ============================================================
# PLOT: OLD 427/428 STYLE
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(14, 8.2),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1.15]}
)

# Panel letters
ax1.text(
    -0.05, 1.08, "A",
    transform=ax1.transAxes,
    fontsize=18,
    fontweight="bold",
    va="top",
    ha="left"
)

ax2.text(
    -0.05, 1.08, "B",
    transform=ax2.transAxes,
    fontsize=18,
    fontweight="bold",
    va="top",
    ha="left"
)


# ---------------- A: PERFORMANCE ----------------

for task, spec in PERFORMANCE_MODALITIES.items():
    label = spec["label"]
    col = spec["col"]
    color = spec["color"]

    ax1.scatter(
        merged["session_day_index"],
        merged[col],
        color=color,
        alpha=0.22,
        s=40,
        edgecolors="none"
    )

    ax1.plot(
        merged["session_day_index"],
        merged[f"{col}_smooth"],
        color=color,
        linewidth=3.2,
        label=f"{label} (w={SMOOTH_WINDOW})"
    )

ax1.axhline(
    0.5,
    linestyle="--",
    linewidth=1.1,
    color="gray",
    alpha=0.8
)

ax1.set_ylabel("Performance")
ax1.set_ylim(0.4, 1.02)
ax1.set_title(f"Mouse {MOUSE_ID}: learning and label-specific facial movement")
ax1.legend(loc="lower right", frameon=True, fontsize=11)


# ---------------- B: LABEL MOTION ----------------

motion_specs = [
    ("lowerlip", "Lower lip", COLOR_LIP),
    ("nose_top", "Nose (top)", COLOR_NOSE),
    ("whisker_median", "Whisker (median)", COLOR_WHISK),
]

for col, label, color in motion_specs:
    ax2.plot(
        merged["session_day_index"],
        merged[f"{col}_z"],
        color=color,
        linewidth=1.0,
        alpha=0.35,
        label="_nolegend_"
    )

for col, label, color in motion_specs:
    ax2.plot(
        merged["session_day_index"],
        merged[f"{col}_z_smooth"],
        color=color,
        linewidth=3.2,
        label=label
    )

ax2.axhline(
    0,
    linestyle="--",
    linewidth=1.1,
    color="gray",
    alpha=0.7
)

ax2.set_xlabel("Day-level overlap session")
ax2.set_ylabel("Z-scored label movement")
ax2.legend(loc="lower right", frameon=True, fontsize=11)


# ---------------- TASK SWITCH LINES ----------------

for x, task in switches:
    color = TASK_COLORS.get(task, "black")

    ax1.axvline(
        x,
        linestyle="--",
        linewidth=1.5,
        color=color,
        alpha=0.9
    )

    ax2.axvline(
        x,
        linestyle="--",
        linewidth=1.5,
        color=color,
        alpha=0.9
    )


# Task labels on bottom panel
y_top = ax2.get_ylim()[1]

for x, task in switches:
    color = TASK_COLORS.get(task, "black")

    ax2.text(
        x + 0.2,
        y_top,
        task,
        rotation=90,
        va="top",
        ha="left",
        fontsize=10,
        color=color
    )


# ---------------- STYLE ----------------

for ax in [ax1, ax2]:
    ax.grid(False)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.show()

print(f"Saved figure to: {out_png}")