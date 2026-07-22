from pathlib import Path
import csv
import cv2
import random
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon")

FRAME_DIR = PROJECT_ROOT / "Training_Data" / "Extracted_Frames" / "Round_01"
LABEL_DIR = PROJECT_ROOT / "Training_Data" / "Labels" / "Round_01"
LABEL_CSV = LABEL_DIR / "frontview_labels_round01.csv"

LABELS = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
    "tongue_tip",
]

LABEL_COLORS = {
    "nose_tip": "red",
    "nose_bottom": "orange",
    "mouth": "cyan",
    "lowerlip": "blue",
    "whiskerpad_left": "lime",
    "whiskerpad_right": "magenta",
    "tongue_tip": "yellow",
}

RANDOMIZE_FRAME_ORDER = True
RANDOM_SEED = 42

LABEL_DIR.mkdir(parents=True, exist_ok=True)


def load_existing_rows(csv_path):
    if not csv_path.exists():
        return {}

    rows = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["frame"]] = row
    return rows


def save_all_rows(csv_path, rows_by_frame):
    fieldnames = ["frame", "skipped"]
    for label in LABELS:
        fieldnames.extend([
            f"{label}_x",
            f"{label}_y",
            f"{label}_visible",
        ])

    rows = [rows_by_frame[k] for k in sorted(rows_by_frame.keys())]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_empty_row(frame_name, skipped=0):
    row = {"frame": frame_name, "skipped": skipped}
    for label in LABELS:
        row[f"{label}_x"] = ""
        row[f"{label}_y"] = ""
        row[f"{label}_visible"] = ""
    return row


def row_is_completed(row):
    if row.get("skipped", "0") == "1":
        return True

    for label in LABELS:
        if row.get(f"{label}_visible", "") == "":
            return False

    return True


def annotate_frame(frame_path, existing_row=None, frame_number=1, total_frames=1):
    img_bgr = cv2.imread(str(frame_path))
    if img_bgr is None:
        print(f"Could not read image: {frame_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    state = {
        "current_idx": 0,
        "done": False,
        "skip": False,
        "points": [],
    }

    if existing_row is not None and row_is_completed(existing_row):
        return existing_row

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.subplots_adjust(right=0.72)

    ax.imshow(img_rgb)
    ax.axis("off")

    artists = []

    def clear_artists():
        for artist in artists:
            try:
                artist.remove()
            except Exception:
                pass
        artists.clear()

    def draw_state():
        clear_artists()

        ax.set_title(
            f"Frame {frame_number}/{total_frames}: {frame_path.name}\n"
            "Click = set point | N = not visible | Backspace = undo | "
            "Enter = save | S = skip frame | Esc = abort",
            fontsize=10,
        )

        for item in state["points"]:
            label = item["label"]
            visible = item["visible"]
            x = item.get("x")
            y = item.get("y")

            if visible:
                color = LABEL_COLORS.get(label, "white")
                sc = ax.scatter([x], [y], s=45, color=color)
                txt = ax.text(
                    x + 5,
                    y + 5,
                    label,
                    fontsize=8,
                    color=color,
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
                )
                artists.extend([sc, txt])

        checklist_lines = []
        for i, label in enumerate(LABELS):
            found = next((p for p in state["points"] if p["label"] == label), None)

            if found is None:
                marker = "→" if i == state["current_idx"] else "○"
                checklist_lines.append(f"{marker} {label}")
            else:
                if found["visible"]:
                    checklist_lines.append(f"✓ {label}")
                else:
                    checklist_lines.append(f"N {label}")

        if len(state["points"]) == len(LABELS):
            checklist_lines.append("")
            checklist_lines.append("ENTER = save frame")

        checklist = "\n".join(checklist_lines)

        textbox = fig.text(
            0.74,
            0.92,
            checklist,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        artists.append(textbox)

        fig.canvas.draw_idle()

    def add_visible_point(x, y):
        if state["current_idx"] >= len(LABELS):
            return

        label = LABELS[state["current_idx"]]
        state["points"].append({
            "label": label,
            "x": float(x),
            "y": float(y),
            "visible": 1,
        })
        state["current_idx"] += 1
        draw_state()

    def add_not_visible():
        if state["current_idx"] >= len(LABELS):
            return

        label = LABELS[state["current_idx"]]
        state["points"].append({
            "label": label,
            "x": "",
            "y": "",
            "visible": 0,
        })
        state["current_idx"] += 1
        draw_state()

    def undo_last():
        if not state["points"]:
            return

        state["points"].pop()
        state["current_idx"] = max(0, state["current_idx"] - 1)
        draw_state()

    def on_click(event):
        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if state["current_idx"] >= len(LABELS):
            return

        add_visible_point(event.xdata, event.ydata)

    def on_key(event):
        key = event.key

        if key == "n":
            add_not_visible()

        elif key == "backspace":
            undo_last()

        elif key == "enter":
            if len(state["points"]) == len(LABELS):
                state["done"] = True
                plt.close(fig)
            else:
                print("Frame not complete yet. Finish all labels or press S to skip.")

        elif key == "s":
            state["skip"] = True
            state["done"] = True
            plt.close(fig)

        elif key == "escape":
            print("Abort requested.")
            plt.close(fig)
            raise KeyboardInterrupt

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    draw_state()
    plt.show()

    if state["skip"]:
        return make_empty_row(frame_path.name, skipped=1)

    if not state["done"]:
        print(f"Not saved: {frame_path.name}")
        return None

    row = make_empty_row(frame_path.name, skipped=0)

    for item in state["points"]:
        label = item["label"]
        row[f"{label}_visible"] = str(item["visible"])

        if item["visible"]:
            row[f"{label}_x"] = f"{item['x']:.3f}"
            row[f"{label}_y"] = f"{item['y']:.3f}"

    return row


def main():
    frames = sorted(FRAME_DIR.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No PNG frames found in: {FRAME_DIR}")

    existing_rows = load_existing_rows(LABEL_CSV)

    todo_frames = [
        f for f in frames
        if f.name not in existing_rows or not row_is_completed(existing_rows[f.name])
    ]

    if RANDOMIZE_FRAME_ORDER:
        random.seed(RANDOM_SEED)
        random.shuffle(todo_frames)

    print(f"Total frames: {len(frames)}")
    print(f"Already completed: {len(frames) - len(todo_frames)}")
    print(f"Remaining: {len(todo_frames)}")
    print(f"CSV: {LABEL_CSV}\n")

    try:
        for idx, frame_path in enumerate(todo_frames, start=1):
            row = annotate_frame(
                frame_path,
                existing_row=existing_rows.get(frame_path.name),
                frame_number=idx,
                total_frames=len(todo_frames),
            )

            if row is None:
                continue

            existing_rows[frame_path.name] = row
            save_all_rows(LABEL_CSV, existing_rows)
            print(f"Saved: {frame_path.name}")

    except KeyboardInterrupt:
        save_all_rows(LABEL_CSV, existing_rows)
        print("\nStopped. Progress saved.")

    print("\nDone.")
    print(f"Saved CSV:\n{LABEL_CSV}")


if __name__ == "__main__":
    main()