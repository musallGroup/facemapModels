from pathlib import Path
import csv
import cv2

PROJECT_ROOT = Path(r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon")

FRAME_DIR = PROJECT_ROOT / "Training_Data" / "Extracted_Frames" / "Round_01"
LABEL_CSV = PROJECT_ROOT / "Training_Data" / "Labels" / "Round_01" / "frontview_labels_round01.csv"

OUT_DIR = PROJECT_ROOT / "Model_QC" / "Round_01_Label_Overlay"
OUT_IMG_DIR = OUT_DIR / "Overlay_Images"
OUT_VIDEO = OUT_DIR / "round01_label_overlay_video.mp4"

LABELS = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
    "tongue_tip",
]

COLORS = {
    "nose_tip": (0, 0, 255),
    "nose_bottom": (0, 165, 255),
    "mouth": (255, 255, 0),
    "lowerlip": (255, 0, 0),
    "whiskerpad_left": (0, 255, 0),
    "whiskerpad_right": (255, 0, 255),
    "tongue_tip": (0, 255, 255),
}

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

with open(LABEL_CSV, "r", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} label rows")

overlay_paths = []

visibility_counts = {label: 0 for label in LABELS}
skipped_count = 0

for idx, row in enumerate(rows, start=1):
    frame_name = row["frame"]
    skipped = row.get("skipped", "0") == "1"

    frame_path = FRAME_DIR / frame_name
    img = cv2.imread(str(frame_path))

    if img is None:
        print(f"Could not read: {frame_path}")
        continue

    if skipped:
        skipped_count += 1
        cv2.putText(
            img,
            "SKIPPED",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )
    else:
        for label in LABELS:
            visible = row.get(f"{label}_visible", "") == "1"

            if not visible:
                continue

            x = float(row[f"{label}_x"])
            y = float(row[f"{label}_y"])
            visibility_counts[label] += 1

            color = COLORS[label]
            cv2.circle(img, (int(x), int(y)), 7, color, -1)
            cv2.circle(img, (int(x), int(y)), 9, (0, 0, 0), 2)
            cv2.putText(
                img,
                label,
                (int(x) + 10, int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    cv2.putText(
        img,
        f"{idx}/{len(rows)}  {frame_name}",
        (20, img.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    out_path = OUT_IMG_DIR / f"overlay_{idx:03d}_{frame_name}"
    cv2.imwrite(str(out_path), img)
    overlay_paths.append(out_path)

print("\nVisibility counts:")
for label, count in visibility_counts.items():
    print(f"{label:18s}: {count}/{len(rows)}")

print(f"Skipped frames      : {skipped_count}/{len(rows)}")

if overlay_paths:
    first = cv2.imread(str(overlay_paths[0]))
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(OUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2,
        (width, height),
    )

    for p in overlay_paths:
        frame = cv2.imread(str(p))
        writer.write(frame)

    writer.release()

print("\nDone.")
print(f"Overlay images saved to:\n{OUT_IMG_DIR}")
print(f"Overlay video saved to:\n{OUT_VIDEO}")