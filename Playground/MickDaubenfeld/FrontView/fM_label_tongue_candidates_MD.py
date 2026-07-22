"""
label_tongue_candidates_full_7kp.py

Stable full-keypoint labeling GUI for the PNGs created by
find_tongue_frames_v2.py.

The interface intentionally stays close to the proven Round 03 Tongue GUI:
- large image canvas on the left
- fixed control panel on the right
- seven color-coded keypoint buttons
- explicit visible / not-visible decisions
- automatic resume from an existing CSV
- coordinates saved in original image resolution

Input:
    Tongue_Candidates/<session folders>/*.png

Optional metadata:
    Tongue_Candidates/tongue_candidates_summary.csv

Output:
    Training_Data/Labels/Tongue_Candidates_7kp_labels.csv

Controls:
    Left click       Place selected keypoint
    Right click      Mark selected keypoint as not visible
    N                Mark selected keypoint as not visible
    Backspace        Clear selected keypoint
    1-7              Select keypoint
    D / Right arrow  Save and move to next frame
    A / Left arrow   Save and move to previous frame
    S                Save all labels
    R                Reset current frame
    Escape           Save and close

Important visibility rules:
- Never estimate a hidden point.
- If tongue_tip is visible, place it at the distal tongue tip.
- If no tongue is visible, mark tongue_tip as not visible.
- If the tongue covers lowerlip, mark lowerlip as not visible.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View\2_Photon"
)

IMAGE_ROOT = PROJECT_ROOT / "Tongue_Candidates"

CANDIDATE_SUMMARY_CSV = (
    IMAGE_ROOT / "tongue_candidates_summary.csv"
)

LABEL_CSV = (
    PROJECT_ROOT
    / "Training_Data"
    / "Labels"
    / "Tongue_Candidates_7kp_labels.csv"
)


# ============================================================
# KEYPOINT SETTINGS
# ============================================================

KEYPOINTS = [
    "nose_tip",
    "nose_bottom",
    "mouth",
    "lowerlip",
    "whiskerpad_left",
    "whiskerpad_right",
    "tongue_tip",
]

KEYPOINT_COLORS = {
    "nose_tip": "#ff3b30",
    "nose_bottom": "#ff9500",
    "mouth": "#ffcc00",
    "lowerlip": "#34c759",
    "whiskerpad_left": "#00c7be",
    "whiskerpad_right": "#007aff",
    "tongue_tip": "#af52de",
}

KEYPOINT_RADII = {
    "nose_tip": 6,
    "nose_bottom": 6,
    "mouth": 6,
    "lowerlip": 6,
    "whiskerpad_left": 6,
    "whiskerpad_right": 6,
    "tongue_tip": 4,
}

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

CANVAS_BACKGROUND = "#202020"

# Slightly stronger / darker pink than before.
RULE_TEXT_COLOR = "#a02078"

STATE_UNSET = "unset"
STATE_VISIBLE = "visible"
STATE_NOT_VISIBLE = "not_visible"


def make_empty_label() -> dict:
    return {
        "x": None,
        "y": None,
        "visible": 0,
        "state": STATE_UNSET,
    }


def make_empty_frame_labels() -> dict[str, dict]:
    return {
        keypoint: make_empty_label()
        for keypoint in KEYPOINTS
    }


# ============================================================
# GUI
# ============================================================

class TongueCandidateFullLabelGUI:
    """Stable seven-keypoint labeling interface."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        self.root.title(
            "Frontview Facemap Labeling — Tongue Candidates"
        )

        self.root.geometry("1550x980")
        self.root.minsize(1100, 750)

        self.image_paths = self.find_images()

        if not self.image_paths:
            raise RuntimeError(
                f"No candidate images were found below:\n{IMAGE_ROOT}"
            )

        self.metadata_by_path = self.load_candidate_metadata()

        LABEL_CSV.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.all_labels = self.load_existing_labels()

        self.current_image_index = self.find_first_unfinished_image()
        self.current_keypoint_index = 0

        self.original_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        self.current_frame_labels = make_empty_frame_labels()

        self.display_scale = 1.0
        self.image_offset_x = 0.0
        self.image_offset_y = 0.0

        self.build_interface()
        self.bind_controls()
        self.load_frame(self.current_image_index)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_program,
        )

    # ========================================================
    # FILES
    # ========================================================

    @staticmethod
    def normalize_path(path: Path | str) -> str:
        return str(Path(path)).replace("/", "\\").lower()

    def find_images(self) -> list[Path]:
        if not IMAGE_ROOT.exists():
            raise FileNotFoundError(
                f"Candidate image directory does not exist:\n{IMAGE_ROOT}"
            )

        images = [
            path
            for path in IMAGE_ROOT.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
                and "_LABEL_QC" not in path.stem
            )
        ]

        return sorted(
            images,
            key=lambda path: (
                path.parent.name.lower(),
                path.name.lower(),
            ),
        )

    def load_candidate_metadata(self) -> dict[str, dict[str, str]]:
        metadata: dict[str, dict[str, str]] = {}

        if not CANDIDATE_SUMMARY_CSV.exists():
            return metadata

        with CANDIDATE_SUMMARY_CSV.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            for row in reader:
                png_path = row.get("png_path", "").strip()

                if not png_path:
                    continue

                metadata[
                    self.normalize_path(png_path)
                ] = row

        return metadata

    def load_existing_labels(self) -> dict[str, dict]:
        labels_by_path: dict[str, dict] = {}

        if not LABEL_CSV.exists():
            print("No existing tongue-candidate label CSV found.")
            return labels_by_path

        with LABEL_CSV.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            for row in reader:
                png_path = row.get("png_path", "").strip()

                if not png_path:
                    continue

                frame_labels = make_empty_frame_labels()

                for keypoint in KEYPOINTS:
                    x_text = row.get(
                        f"{keypoint}_x",
                        "",
                    ).strip()

                    y_text = row.get(
                        f"{keypoint}_y",
                        "",
                    ).strip()

                    visible_text = row.get(
                        f"{keypoint}_visible",
                        "",
                    ).strip()

                    state_text = row.get(
                        f"{keypoint}_state",
                        "",
                    ).strip()

                    try:
                        visible = int(float(visible_text))
                    except (TypeError, ValueError):
                        visible = 0

                    if (
                        visible == 1
                        and x_text
                        and y_text
                    ):
                        try:
                            x_value = float(x_text)
                            y_value = float(y_text)
                        except ValueError:
                            x_value = None
                            y_value = None
                            visible = 0

                        if x_value is not None and y_value is not None:
                            frame_labels[keypoint] = {
                                "x": x_value,
                                "y": y_value,
                                "visible": 1,
                                "state": STATE_VISIBLE,
                            }
                            continue

                    if state_text == STATE_NOT_VISIBLE:
                        state = STATE_NOT_VISIBLE

                    elif visible_text:
                        state = STATE_NOT_VISIBLE

                    else:
                        state = STATE_UNSET

                    frame_labels[keypoint] = {
                        "x": None,
                        "y": None,
                        "visible": 0,
                        "state": state,
                    }

                labels_by_path[
                    self.normalize_path(png_path)
                ] = frame_labels

        print(
            f"Loaded labels for {len(labels_by_path)} images from:\n"
            f"{LABEL_CSV}"
        )

        return labels_by_path

    def save_all_labels(self) -> None:
        self.store_current_frame()

        fieldnames = [
            "png_path",
            "image",
            "mouse",
            "session_timestamp",
            "source_frame",
            "session_time_seconds",
            "model_likelihood",
        ]

        for keypoint in KEYPOINTS:
            fieldnames.extend(
                [
                    f"{keypoint}_x",
                    f"{keypoint}_y",
                    f"{keypoint}_visible",
                    f"{keypoint}_state",
                ]
            )

        with LABEL_CSV.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for image_path in self.image_paths:
                path_key = self.normalize_path(image_path)

                frame_labels = self.all_labels.get(
                    path_key,
                    make_empty_frame_labels(),
                )

                metadata = self.metadata_by_path.get(
                    path_key,
                    {},
                )

                row: dict[str, str | float | int] = {
                    "png_path": str(image_path),
                    "image": image_path.name,
                    "mouse": metadata.get("mouse", ""),
                    "session_timestamp": metadata.get(
                        "session_timestamp",
                        "",
                    ),
                    "source_frame": metadata.get(
                        "source_frame",
                        "",
                    ),
                    "session_time_seconds": metadata.get(
                        "session_time_seconds",
                        "",
                    ),
                    "model_likelihood": metadata.get(
                        "tongue_likelihood",
                        "",
                    ),
                }

                for keypoint in KEYPOINTS:
                    label = frame_labels[keypoint]

                    if (
                        label["state"] == STATE_VISIBLE
                        and label["x"] is not None
                        and label["y"] is not None
                    ):
                        row[f"{keypoint}_x"] = round(
                            float(label["x"]),
                            3,
                        )
                        row[f"{keypoint}_y"] = round(
                            float(label["y"]),
                            3,
                        )
                        row[f"{keypoint}_visible"] = 1
                        row[f"{keypoint}_state"] = STATE_VISIBLE

                    else:
                        row[f"{keypoint}_x"] = ""
                        row[f"{keypoint}_y"] = ""
                        row[f"{keypoint}_visible"] = 0
                        row[f"{keypoint}_state"] = label["state"]

                writer.writerow(row)

        self.status_var.set(
            f"Saved: {LABEL_CSV.name}"
        )

        print(f"Labels saved to:\n{LABEL_CSV}")

    def find_first_unfinished_image(self) -> int:
        for index, image_path in enumerate(self.image_paths):
            frame_labels = self.all_labels.get(
                self.normalize_path(image_path)
            )

            if frame_labels is None:
                return index

            if any(
                label["state"] == STATE_UNSET
                for label in frame_labels.values()
            ):
                return index

        return 0

    # ========================================================
    # INTERFACE
    # ========================================================

    def build_interface(self) -> None:
        main_frame = tk.Frame(self.root)

        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.canvas = tk.Canvas(
            main_frame,
            background=CANVAS_BACKGROUND,
            highlightthickness=0,
            cursor="crosshair",
        )

        self.canvas.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )

        side_panel = tk.Frame(
            main_frame,
            width=285,
            padx=12,
            pady=12,
        )

        side_panel.pack(
            side=tk.RIGHT,
            fill=tk.Y,
        )

        side_panel.pack_propagate(False)

        tk.Label(
            side_panel,
            text="Tongue Candidates",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=(0, 6))

        self.frame_counter_var = tk.StringVar()
        self.filename_var = tk.StringVar()
        self.session_var = tk.StringVar()
        self.selected_keypoint_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.status_var = tk.StringVar()

        tk.Label(
            side_panel,
            textvariable=self.frame_counter_var,
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=3)

        tk.Label(
            side_panel,
            textvariable=self.filename_var,
            font=("Segoe UI", 9),
            wraplength=255,
            justify=tk.CENTER,
        ).pack(pady=(0, 4))

        tk.Label(
            side_panel,
            textvariable=self.session_var,
            font=("Segoe UI", 8),
            wraplength=255,
            justify=tk.CENTER,
            foreground="#555555",
        ).pack(pady=(0, 10))

        tk.Label(
            side_panel,
            text="Selected keypoint:",
            font=("Segoe UI", 10),
        ).pack()

        self.selected_keypoint_label = tk.Label(
            side_panel,
            textvariable=self.selected_keypoint_var,
            font=("Segoe UI", 13, "bold"),
            padx=8,
            pady=6,
        )

        self.selected_keypoint_label.pack(
            fill=tk.X,
            pady=(3, 12),
        )

        self.keypoint_buttons: list[tk.Button] = []

        for index, keypoint in enumerate(KEYPOINTS):
            button = tk.Button(
                side_panel,
                command=lambda idx=index: self.select_keypoint(idx),
                anchor="w",
            )

            button.pack(
                fill=tk.X,
                pady=2,
            )

            self.keypoint_buttons.append(button)

        tk.Frame(
            side_panel,
            height=2,
            background="#aaaaaa",
        ).pack(
            fill=tk.X,
            pady=12,
        )

        tk.Button(
            side_panel,
            text="Mark not visible (N)",
            command=self.mark_current_not_visible,
        ).pack(
            fill=tk.X,
            pady=3,
        )

        tk.Button(
            side_panel,
            text="Clear selected point (Backspace)",
            command=self.clear_current_keypoint,
        ).pack(
            fill=tk.X,
            pady=3,
        )

        tk.Button(
            side_panel,
            text="Reset complete frame (R)",
            command=self.reset_current_frame,
        ).pack(
            fill=tk.X,
            pady=3,
        )

        tk.Button(
            side_panel,
            text="Save labels (S)",
            command=self.save_all_labels,
            font=("Segoe UI", 10, "bold"),
        ).pack(
            fill=tk.X,
            pady=(12, 3),
        )

        navigation_frame = tk.Frame(side_panel)

        navigation_frame.pack(
            fill=tk.X,
            pady=8,
        )

        tk.Button(
            navigation_frame,
            text="← Previous",
            command=self.previous_frame,
        ).pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
            padx=(0, 3),
        )

        tk.Button(
            navigation_frame,
            text="Next →",
            command=self.next_frame,
        ).pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
            padx=(3, 0),
        )

        tk.Label(
            side_panel,
            textvariable=self.progress_var,
            font=("Segoe UI", 10, "bold"),
            justify=tk.LEFT,
        ).pack(
            anchor="w",
            pady=(10, 5),
        )

        rule_text = (
            "Tongue rules:\n"
            "• tongue visible → place tongue_tip\n"
            "• no tongue → mark tongue_tip not visible\n"
            "• tongue covers lower lip → lowerlip not visible\n"
            "• never estimate a hidden point"
        )

        tk.Label(
            side_panel,
            text=rule_text,
            font=("Segoe UI", 9, "bold"),
            justify=tk.LEFT,
            wraplength=255,
            foreground=RULE_TEXT_COLOR,
        ).pack(
            anchor="w",
            pady=(8, 10),
        )

        help_text = (
            "Left click: place point\n"
            "Right click / N: not visible\n"
            "1–7: select keypoint\n"
            "A/D or arrows: previous/next\n"
            "Backspace: clear selected point\n"
            "R: reset frame\n"
            "S: save\n"
            "Esc: save and close"
        )

        tk.Label(
            side_panel,
            text=help_text,
            font=("Segoe UI", 9),
            justify=tk.LEFT,
        ).pack(
            anchor="w",
        )

        tk.Label(
            side_panel,
            textvariable=self.status_var,
            font=("Segoe UI", 8),
            wraplength=255,
            justify=tk.LEFT,
            foreground="#555555",
        ).pack(
            anchor="w",
            pady=(12, 0),
        )

    def bind_controls(self) -> None:
        self.canvas.bind(
            "<Button-1>",
            self.place_keypoint,
        )

        self.canvas.bind(
            "<Button-3>",
            lambda event: self.mark_current_not_visible(),
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self.redraw_canvas(),
        )

        self.root.bind(
            "<KeyPress-a>",
            lambda event: self.previous_frame(),
        )

        self.root.bind(
            "<KeyPress-d>",
            lambda event: self.next_frame(),
        )

        self.root.bind(
            "<Left>",
            lambda event: self.previous_frame(),
        )

        self.root.bind(
            "<Right>",
            lambda event: self.next_frame(),
        )

        self.root.bind(
            "<KeyPress-n>",
            lambda event: self.mark_current_not_visible(),
        )

        self.root.bind(
            "<KeyPress-s>",
            lambda event: self.save_all_labels(),
        )

        self.root.bind(
            "<KeyPress-r>",
            lambda event: self.reset_current_frame(),
        )

        self.root.bind(
            "<BackSpace>",
            lambda event: self.clear_current_keypoint(),
        )

        self.root.bind(
            "<Escape>",
            lambda event: self.close_program(),
        )

        for index in range(len(KEYPOINTS)):
            self.root.bind(
                str(index + 1),
                lambda event, idx=index: self.select_keypoint(idx),
            )

    # ========================================================
    # FRAME NAVIGATION
    # ========================================================

    def load_frame(self, image_index: int) -> None:
        image_index = max(
            0,
            min(image_index, len(self.image_paths) - 1),
        )

        self.current_image_index = image_index

        image_path = self.image_paths[image_index]

        self.original_image = Image.open(
            image_path
        ).convert("RGB")

        existing = self.all_labels.get(
            self.normalize_path(image_path)
        )

        if existing is None:
            self.current_frame_labels = make_empty_frame_labels()
        else:
            self.current_frame_labels = {
                keypoint: dict(existing[keypoint])
                for keypoint in KEYPOINTS
            }

        self.current_keypoint_index = (
            self.find_first_unfinished_keypoint()
        )

        self.update_interface()
        self.redraw_canvas()

    def store_current_frame(self) -> None:
        image_path = self.image_paths[
            self.current_image_index
        ]

        self.all_labels[
            self.normalize_path(image_path)
        ] = {
            keypoint: dict(label)
            for keypoint, label
            in self.current_frame_labels.items()
        }

    def next_frame(self) -> None:
        if not self.current_frame_is_complete():
            proceed = messagebox.askyesno(
                "Incomplete frame",
                "Some keypoints have not been marked.\n\n"
                "Move to the next frame anyway?",
            )

            if not proceed:
                return

        self.save_all_labels()

        if self.current_image_index < len(self.image_paths) - 1:
            self.load_frame(
                self.current_image_index + 1
            )
        else:
            messagebox.showinfo(
                "Round complete",
                "You are on the final frame.\n\n"
                "Press S to save and Escape to close.",
            )

    def previous_frame(self) -> None:
        self.save_all_labels()

        if self.current_image_index > 0:
            self.load_frame(
                self.current_image_index - 1
            )
        else:
            self.status_var.set(
                "This is the first frame."
            )

    # ========================================================
    # COORDINATES
    # ========================================================

    def canvas_to_original(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> tuple[float, float] | None:
        if self.original_image is None:
            return None

        local_x = canvas_x - self.image_offset_x
        local_y = canvas_y - self.image_offset_y

        displayed_width = (
            self.original_image.width * self.display_scale
        )

        displayed_height = (
            self.original_image.height * self.display_scale
        )

        if not (
            0 <= local_x < displayed_width
            and 0 <= local_y < displayed_height
        ):
            return None

        return (
            local_x / self.display_scale,
            local_y / self.display_scale,
        )

    def original_to_canvas(
        self,
        original_x: float,
        original_y: float,
    ) -> tuple[float, float]:
        return (
            original_x * self.display_scale + self.image_offset_x,
            original_y * self.display_scale + self.image_offset_y,
        )

    # ========================================================
    # LABELING
    # ========================================================

    def select_keypoint(self, index: int) -> None:
        self.current_keypoint_index = max(
            0,
            min(index, len(KEYPOINTS) - 1),
        )

        self.update_interface()
        self.redraw_canvas()

    def place_keypoint(self, event: tk.Event) -> None:
        coordinates = self.canvas_to_original(
            event.x,
            event.y,
        )

        if coordinates is None:
            return

        x_value, y_value = coordinates

        keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.current_frame_labels[keypoint] = {
            "x": float(x_value),
            "y": float(y_value),
            "visible": 1,
            "state": STATE_VISIBLE,
        }

        self.advance_keypoint()
        self.update_interface()
        self.redraw_canvas()

    def mark_current_not_visible(self) -> None:
        keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.current_frame_labels[keypoint] = {
            "x": None,
            "y": None,
            "visible": 0,
            "state": STATE_NOT_VISIBLE,
        }

        self.status_var.set(
            f"{keypoint}: not visible"
        )

        self.advance_keypoint()
        self.update_interface()
        self.redraw_canvas()

    def clear_current_keypoint(self) -> None:
        keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.current_frame_labels[keypoint] = make_empty_label()

        self.status_var.set(
            f"{keypoint}: cleared"
        )

        self.update_interface()
        self.redraw_canvas()

    def reset_current_frame(self) -> None:
        confirmed = messagebox.askyesno(
            "Reset frame",
            "Delete all labels and visibility decisions "
            "on this frame?",
        )

        if not confirmed:
            return

        self.current_frame_labels = make_empty_frame_labels()
        self.current_keypoint_index = 0

        self.status_var.set(
            "Current frame reset."
        )

        self.update_interface()
        self.redraw_canvas()

    def advance_keypoint(self) -> None:
        for offset in range(1, len(KEYPOINTS) + 1):
            next_index = (
                self.current_keypoint_index + offset
            ) % len(KEYPOINTS)

            next_keypoint = KEYPOINTS[next_index]

            if (
                self.current_frame_labels[next_keypoint]["state"]
                == STATE_UNSET
            ):
                self.current_keypoint_index = next_index
                return

        self.status_var.set(
            "All 7 keypoints reviewed. Press D for next frame."
        )

    def find_first_unfinished_keypoint(self) -> int:
        for index, keypoint in enumerate(KEYPOINTS):
            if (
                self.current_frame_labels[keypoint]["state"]
                == STATE_UNSET
            ):
                return index

        return 0

    # ========================================================
    # DRAWING
    # ========================================================

    def redraw_canvas(self) -> None:
        if self.original_image is None:
            return

        canvas_width = max(
            self.canvas.winfo_width(),
            1,
        )

        canvas_height = max(
            self.canvas.winfo_height(),
            1,
        )

        scale_x = canvas_width / self.original_image.width
        scale_y = canvas_height / self.original_image.height

        self.display_scale = min(
            scale_x,
            scale_y,
        )

        displayed_width = max(
            1,
            int(round(
                self.original_image.width * self.display_scale
            )),
        )

        displayed_height = max(
            1,
            int(round(
                self.original_image.height * self.display_scale
            )),
        )

        self.image_offset_x = (
            canvas_width - displayed_width
        ) / 2.0

        self.image_offset_y = (
            canvas_height - displayed_height
        ) / 2.0

        resized = self.original_image.resize(
            (displayed_width, displayed_height),
            Image.Resampling.LANCZOS,
        )

        self.tk_image = ImageTk.PhotoImage(
            resized
        )

        self.canvas.delete("all")

        self.canvas.create_image(
            self.image_offset_x,
            self.image_offset_y,
            anchor=tk.NW,
            image=self.tk_image,
        )

        for index, keypoint in enumerate(KEYPOINTS):
            label = self.current_frame_labels[keypoint]

            if (
                label["state"] != STATE_VISIBLE
                or label["x"] is None
                or label["y"] is None
            ):
                continue

            canvas_x, canvas_y = self.original_to_canvas(
                float(label["x"]),
                float(label["y"]),
            )

            radius = KEYPOINT_RADII[keypoint]
            color = KEYPOINT_COLORS[keypoint]

            if index == self.current_keypoint_index:
                self.canvas.create_oval(
                    canvas_x - radius - 3,
                    canvas_y - radius - 3,
                    canvas_x + radius + 3,
                    canvas_y + radius + 3,
                    outline="white",
                    width=2,
                )

            self.canvas.create_oval(
                canvas_x - radius,
                canvas_y - radius,
                canvas_x + radius,
                canvas_y + radius,
                fill=color,
                outline="black",
                width=2,
            )

            self.canvas.create_text(
                canvas_x + 9,
                canvas_y - 8,
                text=keypoint,
                anchor=tk.SW,
                fill=color,
                font=("Segoe UI", 10, "bold"),
            )

    # ========================================================
    # STATUS
    # ========================================================

    def current_frame_is_complete(self) -> bool:
        return all(
            label["state"] != STATE_UNSET
            for label in self.current_frame_labels.values()
        )

    def count_current_states(self) -> tuple[int, int, int]:
        visible = sum(
            label["state"] == STATE_VISIBLE
            for label in self.current_frame_labels.values()
        )

        not_visible = sum(
            label["state"] == STATE_NOT_VISIBLE
            for label in self.current_frame_labels.values()
        )

        unfinished = sum(
            label["state"] == STATE_UNSET
            for label in self.current_frame_labels.values()
        )

        return visible, not_visible, unfinished

    def count_completed_images(self) -> int:
        completed = 0
        current_path = self.image_paths[
            self.current_image_index
        ]
        current_key = self.normalize_path(current_path)

        for image_path in self.image_paths:
            path_key = self.normalize_path(image_path)

            if path_key == current_key:
                frame_labels = self.current_frame_labels
            else:
                frame_labels = self.all_labels.get(path_key)

            if frame_labels is None:
                continue

            if all(
                label["state"] != STATE_UNSET
                for label in frame_labels.values()
            ):
                completed += 1

        return completed

    def update_interface(self) -> None:
        image_path = self.image_paths[
            self.current_image_index
        ]

        selected_keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        metadata = self.metadata_by_path.get(
            self.normalize_path(image_path),
            {},
        )

        self.frame_counter_var.set(
            f"Frame {self.current_image_index + 1} "
            f"of {len(self.image_paths)}"
        )

        self.filename_var.set(
            image_path.name
        )

        mouse = metadata.get(
            "mouse",
            image_path.parent.name.split("_")[0],
        )

        session = metadata.get(
            "session_timestamp",
            image_path.parent.name,
        )

        self.session_var.set(
            f"Mouse {mouse} | Session {session}"
        )

        self.selected_keypoint_var.set(
            selected_keypoint
        )

        selected_color = KEYPOINT_COLORS[
            selected_keypoint
        ]

        self.selected_keypoint_label.configure(
            background=selected_color,
            foreground="white",
        )

        visible, not_visible, unfinished = (
            self.count_current_states()
        )

        completed_images = self.count_completed_images()

        self.progress_var.set(
            f"Visible:       {visible}/7\n"
            f"Not visible:   {not_visible}/7\n"
            f"Unfinished:    {unfinished}/7\n"
            f"Frames done:   {completed_images}/{len(self.image_paths)}"
        )

        for index, button in enumerate(
            self.keypoint_buttons
        ):
            keypoint = KEYPOINTS[index]
            label = self.current_frame_labels[keypoint]

            if label["state"] == STATE_VISIBLE:
                symbol = "✓"

            elif label["state"] == STATE_NOT_VISIBLE:
                symbol = "Ø"

            else:
                symbol = "·"

            button.configure(
                text=f"{index + 1}. {keypoint}   {symbol}",
                background=KEYPOINT_COLORS[keypoint],
                foreground="white",
                activebackground=KEYPOINT_COLORS[keypoint],
                activeforeground="white",
                relief=(
                    tk.SUNKEN
                    if index == self.current_keypoint_index
                    else tk.RAISED
                ),
                borderwidth=(
                    3
                    if index == self.current_keypoint_index
                    else 1
                ),
            )

    # ========================================================
    # CLOSING
    # ========================================================

    def close_program(self) -> None:
        try:
            self.save_all_labels()

        except Exception as error:
            close_anyway = messagebox.askyesno(
                "Save error",
                f"Labels could not be saved:\n\n{error}\n\n"
                "Close anyway?",
            )

            if not close_anyway:
                return

        self.root.destroy()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n========================================")
    print("FRONTVIEW LABELING GUI")
    print("TONGUE CANDIDATES — FULL 7 KEYPOINTS")
    print("========================================")
    print(f"Images:    {IMAGE_ROOT}")
    print(f"Label CSV: {LABEL_CSV}")
    print(f"Keypoints: {len(KEYPOINTS)}")

    root = tk.Tk()

    try:
        TongueCandidateFullLabelGUI(root)
        root.mainloop()

    except Exception:
        root.destroy()
        raise


if __name__ == "__main__":
    main()