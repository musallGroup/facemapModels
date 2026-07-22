"""
label_frontview_frames_round03_tongue.py

Interactive labeling GUI for the targeted tongue frames from Round 03.

Input:
    Training_Data/Extracted_Frames/Round_03_Tongue

Output:
    Training_Data/Labels/Round_03_Tongue_labels.csv

Keypoints:
    1. nose_tip
    2. nose_bottom
    3. mouth
    4. lowerlip
    5. whiskerpad_left
    6. whiskerpad_right
    7. tongue_tip

Important visibility rules:
- If tongue_tip is clearly visible:
      tongue_tip = visible
- If no tongue is visible:
      tongue_tip = not visible
- If the tongue covers the lower lip:
      lowerlip = not visible
- Never estimate a hidden point.

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

The CSV stores:
    image
    <keypoint>_x
    <keypoint>_y
    <keypoint>_visible
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

IMAGE_DIR = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Training_Data\Extracted_Frames"
    r"\Round_03_Tongue"
)

LABEL_CSV = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Training_Data\Labels"
    r"\Round_03_Tongue_labels.csv"
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

# Smaller marker for tongue_tip to allow more precise placement.
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


# ============================================================
# LABEL STATES
# ============================================================

STATE_UNSET = "unset"
STATE_VISIBLE = "visible"
STATE_NOT_VISIBLE = "not_visible"


def make_empty_label() -> dict:
    """Create one empty keypoint label."""

    return {
        "x": None,
        "y": None,
        "visible": 0,
        "state": STATE_UNSET,
    }


def make_empty_frame_labels() -> dict[str, dict]:
    """Create an empty label dictionary for one frame."""

    return {
        keypoint: make_empty_label()
        for keypoint in KEYPOINTS
    }


# ============================================================
# GUI
# ============================================================

class FrontviewTongueLabelGUI:
    """Tkinter interface for labeling Round 03 tongue frames."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        self.root.title(
            "Frontview Facemap Labeling — Round 03 Tongue"
        )

        self.root.geometry("1550x980")
        self.root.minsize(1100, 750)

        self.image_paths = self.find_images()

        if not self.image_paths:
            raise RuntimeError(
                f"No images were found in:\n{IMAGE_DIR}"
            )

        LABEL_CSV.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.all_labels = self.load_existing_labels()

        self.current_image_index = 0
        self.current_keypoint_index = 0

        self.original_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None

        self.current_frame_labels = make_empty_frame_labels()

        self.display_scale = 1.0
        self.image_offset_x = 0.0
        self.image_offset_y = 0.0

        self.build_interface()
        self.bind_controls()
        self.load_frame(0)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_program,
        )

    # ========================================================
    # FILES
    # ========================================================

    def find_images(self) -> list[Path]:
        """Find all images in Round_03_Tongue."""

        if not IMAGE_DIR.exists():
            raise FileNotFoundError(
                f"Image directory does not exist:\n{IMAGE_DIR}"
            )

        images = [
            path
            for path in IMAGE_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        ]

        return sorted(
            images,
            key=lambda path: path.name.lower(),
        )

    def load_existing_labels(self) -> dict[str, dict]:
        """Load an existing CSV so labeling can be resumed."""

        labels_by_image: dict[str, dict] = {}

        if not LABEL_CSV.exists():
            print("No existing Round 03 label CSV found.")
            return labels_by_image

        with LABEL_CSV.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            for row in reader:
                image_name = row.get("image", "").strip()

                if not image_name:
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

                    # New CSVs store explicit state.
                    if state_text == STATE_NOT_VISIBLE:
                        state = STATE_NOT_VISIBLE

                    # Compatibility with older CSV files:
                    # a row with visible=0 counts as not visible.
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

                labels_by_image[image_name] = frame_labels

        print(
            f"Loaded labels for {len(labels_by_image)} images from:\n"
            f"{LABEL_CSV}"
        )

        return labels_by_image

    def save_all_labels(self) -> None:
        """Save labels for every frame to CSV."""

        self.store_current_frame()

        fieldnames = ["image"]

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
                image_name = image_path.name

                frame_labels = self.all_labels.get(
                    image_name,
                    make_empty_frame_labels(),
                )

                row: dict[str, str | float | int] = {
                    "image": image_name,
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

    # ========================================================
    # INTERFACE
    # ========================================================

    def build_interface(self) -> None:
        """Build canvas and control panel."""

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
            text="Round 03 — Tongue",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=(0, 6))

        self.frame_counter_var = tk.StringVar()
        self.filename_var = tk.StringVar()
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
            foreground="#7a1f8a",
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
        """Bind mouse and keyboard controls."""

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
        """Load an image and its existing labels."""

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
            image_path.name
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
        """Store current labels in memory."""

        image_name = self.image_paths[
            self.current_image_index
        ].name

        self.all_labels[image_name] = {
            keypoint: dict(label)
            for keypoint, label
            in self.current_frame_labels.items()
        }

    def next_frame(self) -> None:
        """Save and move to the next image."""

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
        """Save and move to the previous image."""

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
        """Convert canvas coordinates to original image coordinates."""

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
        """Convert original image coordinates to canvas coordinates."""

        return (
            original_x * self.display_scale + self.image_offset_x,
            original_y * self.display_scale + self.image_offset_y,
        )

    # ========================================================
    # LABELING
    # ========================================================

    def select_keypoint(self, index: int) -> None:
        """Select a keypoint."""

        self.current_keypoint_index = max(
            0,
            min(index, len(KEYPOINTS) - 1),
        )

        self.update_interface()
        self.redraw_canvas()

    def place_keypoint(self, event: tk.Event) -> None:
        """Place the selected keypoint."""

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
        """Explicitly mark selected keypoint as not visible."""

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
        """Return selected keypoint to an unfinished state."""

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
        """Reset all labels on the current frame."""

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
        """Advance to the next unfinished keypoint."""

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
        """Return the first keypoint without a decision."""

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
        """Draw the image and visible keypoints."""

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

            # Current point gets a white outline, but not a larger marker.
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
        """Return True when all seven keypoints have a decision."""

        return all(
            label["state"] != STATE_UNSET
            for label in self.current_frame_labels.values()
        )

    def count_current_states(self) -> tuple[int, int, int]:
        """Count visible, not-visible and unfinished keypoints."""

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
        """Count frames with decisions for all seven keypoints."""

        completed = 0

        for image_path in self.image_paths:
            frame_labels = self.all_labels.get(
                image_path.name
            )

            if frame_labels is None:
                continue

            if all(
                label["state"] != STATE_UNSET
                for label in frame_labels.values()
            ):
                completed += 1

        current_name = self.image_paths[
            self.current_image_index
        ].name

        if (
            current_name not in self.all_labels
            and self.current_frame_is_complete()
        ):
            completed += 1

        return completed

    def update_interface(self) -> None:
        """Update labels, counters and button states."""

        image_path = self.image_paths[
            self.current_image_index
        ]

        selected_keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.frame_counter_var.set(
            f"Frame {self.current_image_index + 1} "
            f"of {len(self.image_paths)}"
        )

        self.filename_var.set(
            image_path.name
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
        """Save and close the GUI."""

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
    """Start the Round 03 labeling GUI."""

    print("\n========================================")
    print("FRONTVIEW LABELING GUI")
    print("ROUND 03 — TONGUE")
    print("========================================")
    print(f"Images:    {IMAGE_DIR}")
    print(f"Label CSV: {LABEL_CSV}")
    print(f"Keypoints: {len(KEYPOINTS)}")

    root = tk.Tk()

    try:
        FrontviewTongueLabelGUI(root)
        root.mainloop()

    except Exception:
        root.destroy()
        raise


if __name__ == "__main__":
    main()