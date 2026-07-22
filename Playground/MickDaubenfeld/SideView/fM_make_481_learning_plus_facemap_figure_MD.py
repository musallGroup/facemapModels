#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

MOUSE_ID = "481"

PERFORMANCE_CSV = Path(
    r"Z:\Team\Mick\2 Photon\New_Dataset\Cohort_Summary\Performance\performance_481.csv"
)

TASK_CSV = Path(
    r"C:\Users\daubenfeld\Desktop\PPC_Mice - Tabellenblatt3.csv"
)

FACEMAP_CSV = Path(
    r"Z:\Team\Mick\2 Photon\New_Dataset\Cohort\Cohort_Summary\all_sessions_all_labels.csv"
)

OUT_DIR = Path(
    r"Z:\Team\Mick\2 Photon\New_Dataset\Figures"
)

OUT_PNG = OUT_DIR / "mouse_481_learning_plus_facemap_motion.png"
OUT_PDF = OUT_DIR / "mouse_481_learning_plus_facemap_motion.pdf"
OUT_CSV = OUT_DIR / "mouse_481_learning_plus_facemap_motion_merged.csv"


# ============================================================
# SETTINGS
# ============================================================

SMOOTH_WINDOW = 7
MIN_TRIALS = 20

LABEL_NOSE = "nose(tip)"
LABEL_MOUTH = "mouth"
LABEL_WHISKERS = ["whisker(I)", "whisker(II)", "whisker(III)"]

COLOR_AUDIO = "#1f77b4"
COLOR_FIRST_INNATE = "#008000"
COLOR_AUDIO_NO_DELAY = "#1b6f8a"
COLOR_AUDIO_DELAY = "#ff4f9a"
COLOR_DISCRIMINATION = "#b084f5"
COLOR_SECOND_INNATE = "#00c853"

COLOR_NOSE = "#0072B2"
COLOR_MOUTH = "#D55E00"
COLOR_WHISKER = "#009E73"

TASK_COLORS = {
    "First Innate": COLOR_FIRST_INNATE,
    "Audio Task Without Delay": COLOR_AUDIO_NO_DELAY,
    "Audio Task With Delay": COLOR_AUDIO_DELAY,
    "Discriminitation": COLOR_DISCRIMINATION,
    "Discrimination": COLOR_DISCRIMINATION,
    "Second Innate": COLOR_SECOND_INNATE,
}


# ============================================================
# HELPERS
# ============================================================

def clean_date(x):
    return (
        str(x)
        .replace(".0", "")
        .strip()
    )


def smooth_series(s, window=7):
    return s.rolling(window=window, center=True, min_periods=1).mean()


def zscore_series(s):
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def normalize_task_name(task):
    task = str(task).strip()
    if task == "Discriminitation":
        return "Discrimination"
    return task


# ============================================================
# LOAD PERFORMANCE
# ============================================================

perf = pd.read_csv(PERFORMANCE_CSV)

perf["session_date"] = perf["session_date"].apply(clean_date)
perf["perfAudio"] = pd.to_numeric(perf["perfAudio"], errors="coerce")
perf["nAudio"] = pd.to_numeric(perf["nAudio"], errors="coerce")

perf.loc[perf["nAudio"] < MIN_TRIALS, "perfAudio"] = np.nan

perf = (
    perf
    .sort_values(["session_date", "session_time"])
    .drop_duplicates("session_date", keep="first")
    .reset_index(drop=True)
)


# ============================================================
# LOAD TASK CSV
# ============================================================

task_raw = pd.read_csv(TASK_CSV)

# expected columns: Animal, ExperimentDate, Task, ...
task = task_raw.copy()

task = task[task["Animal"].astype(str).str.strip() == MOUSE_ID].copy()

task["session_date"] = task["ExperimentDate"].apply(clean_date)
task["Task"] = task["Task"].apply(normalize_task_name)

task = (
    task[["session_date", "Task"]]
    .dropna()
    .drop_duplicates("session_date", keep="first")
    .sort_values("session_date")
    .reset_index(drop=True)
)


# ============================================================
# LOAD FACEMAP SUMMARY
# ============================================================

fm = pd.read_csv(FACEMAP_CSV)

fm["session_date"] = fm["session_date"].apply(clean_date)
fm["median_motion"] = pd.to_numeric(fm["median_motion"], errors="coerce")

# Pivot: one row per session_date, one column per label
pivot = fm.pivot_table(
    index="session_date",
    columns="label",
    values="median_motion",
    aggfunc="mean"
).reset_index()

needed = [LABEL_NOSE, LABEL_MOUTH]
for label in needed:
    if label not in pivot.columns:
        raise ValueError(f"Missing label in Facemap summary: {label}")

available_whiskers = [w for w in LABEL_WHISKERS if w in pivot.columns]

if len(available_whiskers) == 0:
    raise ValueError("No whisker labels found in Facemap summary.")

pivot["nose_tip"] = pivot[LABEL_NOSE]
pivot["mouth"] = pivot[LABEL_MOUTH]
pivot["whisker_median"] = pivot[available_whiskers].median(axis=1)

facial = pivot[["session_date", "nose_tip", "mouth", "whisker_median"]].copy()


# ============================================================
# MERGE
# ============================================================

merged = perf.merge(task, on="session_date", how="left")
merged = merged.merge(facial, on="session_date", how="inner")

merged = merged.sort_values("session_date").reset_index(drop=True)
merged["session_index"] = np.arange(1, len(merged) + 1)

if merged.empty:
    raise ValueError("No overlap between performance and Facemap data.")


# ============================================================
# SMOOTH + ZSCORE
# ============================================================

merged["perfAudio_smooth"] = smooth_series(merged["perfAudio"], SMOOTH_WINDOW)

for col in ["nose_tip", "mouth", "whisker_median"]:
    merged[f"{col}_z"] = zscore_series(merged[col])
    merged[f"{col}_z_smooth"] = smooth_series(merged[f"{col}_z"], SMOOTH_WINDOW)


# ============================================================
# TASK SWITCHES
# ============================================================

switches = []
prev_task = None

for _, row in merged.iterrows():
    task_name = row["Task"]
    x = row["session_index"]

    if pd.notna(task_name):
        if prev_task is None or task_name != prev_task:
            switches.append((x, task_name))
        prev_task = task_name


# ============================================================
# SAVE MERGED CSV
# ============================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_CSV, index=False)
print(f"Saved merged data: {OUT_CSV}")


# ============================================================
# PLOT
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(14, 8.2),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1.15]}
)

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

ax1.scatter(
    merged["session_index"],
    merged["perfAudio"],
    color=COLOR_AUDIO,
    alpha=0.25,
    s=40,
    edgecolors="none"
)

ax1.plot(
    merged["session_index"],
    merged["perfAudio_smooth"],
    color=COLOR_AUDIO,
    linewidth=3.2,
    label=f"Audio (w={SMOOTH_WINDOW})"
)

ax1.axhline(
    0.5,
    linestyle="--",
    linewidth=1.1,
    color="gray",
    alpha=0.8
)

ax1.set_ylabel("Performance")
ax1.set_ylim(0.35, 1.02)
ax1.set_title(f"Mouse {MOUSE_ID}: learning and label-specific facial movement")
ax1.legend(loc="lower right", frameon=True, fontsize=11)


# ---------------- B: FACEMAP MOTION ----------------

motion_specs = [
    ("mouth", "Mouth", COLOR_MOUTH),
    ("nose_tip", "Nose tip", COLOR_NOSE),
    ("whisker_median", "Whisker median", COLOR_WHISKER),
]

for col, label, color in motion_specs:
    ax2.plot(
        merged["session_index"],
        merged[f"{col}_z"],
        color=color,
        linewidth=1.0,
        alpha=0.35,
        label="_nolegend_"
    )

for col, label, color in motion_specs:
    ax2.plot(
        merged["session_index"],
        merged[f"{col}_z_smooth"],
        color=color,
        linewidth=3.2,
        label=f"{label} (w={SMOOTH_WINDOW})"
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

for x, task_name in switches:
    color = TASK_COLORS.get(task_name, "black")

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

# Task labels on bottom plot
y_top = ax2.get_ylim()[1]

for x, task_name in switches:
    color = TASK_COLORS.get(task_name, "black")

    ax2.text(
        x + 0.2,
        y_top,
        task_name,
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

ax1.set_xlim(0, merged["session_index"].max() + 1)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF)
plt.show()

print(f"Saved PNG: {OUT_PNG}")
print(f"Saved PDF: {OUT_PDF}")