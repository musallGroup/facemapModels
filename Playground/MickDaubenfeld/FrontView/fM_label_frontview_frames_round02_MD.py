"""
label_frontview_frames_round02.py

Interactive labeling GUI for Frontview Facemap training data.

Round:
    Round_02_8mice

Keypoints:
    nose_tip
    nose_bottom
    mouth
    lowerlip
    whiskerpad_left
    whiskerpad_right
    tongue_tip

Controls
--------
Left mouse click:
    Place the currently selected keypoint.

Right mouse click:
    Mark the currently selected keypoint as NOT visible.

S:
    Save current frame.

D or Right Arrow:
    Save and go to next frame.

A or Left Arrow:
    Save and go to previous frame.

N:
    Mark current keypoint as not visible.

R:
    Reset all labels on the current frame.

Backspace:
    Delete the currently selected keypoint.

1–7:
    Select a specific keypoint.

Escape:
    Save and close.

The CSV contains x, y and visibility values for every keypoint.
Coordinates are stored in the original image coordinate system.
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
    r"\2_Photon\Training_Data\Extracted_Frames\Round_02_8mice"
)

LABEL_CSV = Path(
    r"\\NASKAMPA\lts\Team\Mick\FM_Front_View"
    r"\2_Photon\Training_Data\Labels"
    r"\Round_02_8mice_labels.csv"
)


# ============================================================
# LABEL SETTINGS
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

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

POINT_RADIUS = 6
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 900


# ============================================================
# MAIN GUI
# ============================================================

class FrontviewLabelGUI:
    """Interactive GUI for labeling frontview mouse images."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root

        self.root.title(
            "Frontview Facemap Labeling — Round 02"
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
        self.display_image: ImageTk.PhotoImage | None = None

        self.scale = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0

        self.current_frame_labels: dict[
            str,
            dict[str, float | int | None],
        ] = {}

        self.unsaved_changes = False

        self.build_interface()
        self.bind_controls()
        self.load_frame(0)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_program,
        )

    # ========================================================
    # FILE HANDLING
    # ========================================================

    def find_images(self) -> list[Path]:
        """Find all supported images in the image directory."""

        if not IMAGE_DIR.exists():
            raise FileNotFoundError(
                f"Image directory does not exist:\n{IMAGE_DIR}"
            )

        image_paths = [
            path
            for path in IMAGE_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        ]

        return sorted(
            image_paths,
            key=lambda path: path.name.lower(),
        )

    def empty_frame_labels(
        self,
    ) -> dict[str, dict[str, float | int | None]]:
        """Create an empty label structure for one frame."""

        return {
            keypoint: {
                "x": None,
                "y": None,
                "visible": 0,
            }
            for keypoint in KEYPOINTS
        }

    def load_existing_labels(
        self,
    ) -> dict[
        str,
        dict[str, dict[str, float | int | None]],
    ]:
        """Load an existing CSV so labeling can be resumed."""

        all_labels: dict[
            str,
            dict[str, dict[str, float | int | None]],
        ] = {}

        if not LABEL_CSV.exists():
            return all_labels

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

                frame_labels = self.empty_frame_labels()

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
                        "0",
                    ).strip()

                    try:
                        visible = int(float(visible_text))
                    except ValueError:
                        visible = 0

                    if visible == 1 and x_text and y_text:
                        try:
                            x_value = float(x_text)
                            y_value = float(y_text)
                        except ValueError:
                            x_value = None
                            y_value = None
                            visible = 0
                    else:
                        x_value = None
                        y_value = None
                        visible = 0

                    frame_labels[keypoint] = {
                        "x": x_value,
                        "y": y_value,
                        "visible": visible,
                    }

                all_labels[image_name] = frame_labels

        print(
            f"Loaded labels for {len(all_labels)} images "
            f"from:\n{LABEL_CSV}"
        )

        return all_labels

    def save_all_labels(self) -> None:
        """Write all labels to CSV."""

        self.store_current_frame_in_memory()

        fieldnames = ["image"]

        for keypoint in KEYPOINTS:
            fieldnames.extend(
                [
                    f"{keypoint}_x",
                    f"{keypoint}_y",
                    f"{keypoint}_visible",
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
                    self.empty_frame_labels(),
                )

                row: dict[str, str | int | float] = {
                    "image": image_name,
                }

                for keypoint in KEYPOINTS:
                    label = frame_labels[keypoint]
                    visible = int(label["visible"] or 0)

                    if (
                        visible == 1
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

                    else:
                        row[f"{keypoint}_x"] = ""
                        row[f"{keypoint}_y"] = ""
                        row[f"{keypoint}_visible"] = 0

                writer.writerow(row)

        self.unsaved_changes = False

        self.status_message_var.set(
            f"Saved labels to: {LABEL_CSV.name}"
        )

        print(f"Labels saved to:\n{LABEL_CSV}")

    # ========================================================
    # INTERFACE
    # ========================================================

    def build_interface(self) -> None:
        """Create the complete Tkinter interface."""

        main_frame = tk.Frame(self.root)
        main_frame.pack(
            fill=tk.BOTH,
            expand=True,
        )

        self.canvas = tk.Canvas(
            main_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            background="#202020",
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
            width=260,
            padx=12,
            pady=12,
        )

        side_panel.pack(
            side=tk.RIGHT,
            fill=tk.Y,
        )

        side_panel.pack_propagate(False)

        title_label = tk.Label(
            side_panel,
            text="Frontview Labels",
            font=("Segoe UI", 16, "bold"),
        )

        title_label.pack(
            pady=(0, 8),
        )

        self.frame_counter_var = tk.StringVar()
        self.filename_var = tk.StringVar()
        self.current_keypoint_var = tk.StringVar()
        self.progress_var = tk.StringVar()
        self.status_message_var = tk.StringVar()

        tk.Label(
            side_panel,
            textvariable=self.frame_counter_var,
            font=("Segoe UI", 11, "bold"),
        ).pack(pady=3)

        tk.Label(
            side_panel,
            textvariable=self.filename_var,
            font=("Segoe UI", 9),
            wraplength=235,
            justify=tk.CENTER,
        ).pack(pady=(0, 12))

        tk.Label(
            side_panel,
            text="Current keypoint:",
            font=("Segoe UI", 10),
        ).pack()

        self.current_keypoint_label = tk.Label(
            side_panel,
            textvariable=self.current_keypoint_var,
            font=("Segoe UI", 13, "bold"),
            padx=8,
            pady=6,
        )

        self.current_keypoint_label.pack(
            fill=tk.X,
            pady=(3, 12),
        )

        tk.Label(
            side_panel,
            text="Keypoints",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        self.keypoint_buttons: list[tk.Button] = []

        for index, keypoint in enumerate(KEYPOINTS):
            button = tk.Button(
                side_panel,
                text=f"{index + 1}. {keypoint}",
                command=lambda idx=index: (
                    self.select_keypoint(idx)
                ),
                anchor="w",
                relief=tk.RAISED,
            )

            button.pack(
                fill=tk.X,
                pady=2,
            )

            self.keypoint_buttons.append(button)

        separator = tk.Frame(
            side_panel,
            height=2,
            background="#b0b0b0",
        )

        separator.pack(
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
            text="Delete current point",
            command=self.delete_current_keypoint,
        ).pack(
            fill=tk.X,
            pady=3,
        )

        tk.Button(
            side_panel,
            text="Reset current frame (R)",
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
            pady=(12, 4),
        )

        help_text = (
            "Left click: place point\n"
            "Right click: not visible\n"
            "A/D or arrows: previous/next\n"
            "1–7: select keypoint\n"
            "N: not visible\n"
            "Backspace: delete point\n"
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
            pady=(10, 4),
        )

        tk.Label(
            side_panel,
            textvariable=self.status_message_var,
            font=("Segoe UI", 8),
            wraplength=235,
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
            self.mark_not_visible_from_mouse,
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
            "<KeyPress-s>",
            lambda event: self.save_all_labels(),
        )

        self.root.bind(
            "<KeyPress-n>",
            lambda event: self.mark_current_not_visible(),
        )

        self.root.bind(
            "<KeyPress-r>",
            lambda event: self.reset_current_frame(),
        )

        self.root.bind(
            "<BackSpace>",
            lambda event: self.delete_current_keypoint(),
        )

        self.root.bind(
            "<Escape>",
            lambda event: self.close_program(),
        )

        for index in range(len(KEYPOINTS)):
            self.root.bind(
                str(index + 1),
                lambda event, idx=index: (
                    self.select_keypoint(idx)
                ),
            )

        self.canvas.bind(
            "<Configure>",
            self.redraw_after_resize,
        )

    # ========================================================
    # FRAME MANAGEMENT
    # ========================================================

    def load_frame(self, image_index: int) -> None:
        """Load an image and its existing labels."""

        image_index = max(
            0,
            min(image_index, len(self.image_paths) - 1),
        )

        self.current_image_index = image_index

        image_path = self.image_paths[
            self.current_image_index
        ]

        self.original_image = Image.open(
            image_path
        ).convert("RGB")

        saved_labels = self.all_labels.get(
            image_path.name
        )

        if saved_labels is None:
            self.current_frame_labels = (
                self.empty_frame_labels()
            )
        else:
            self.current_frame_labels = {
                keypoint: dict(values)
                for keypoint, values in saved_labels.items()
            }

        self.current_keypoint_index = (
            self.find_first_unfinished_keypoint()
        )

        self.unsaved_changes = False

        self.update_interface_text()
        self.redraw_canvas()

    def store_current_frame_in_memory(self) -> None:
        """Store current frame labels before navigation or saving."""

        image_name = self.image_paths[
            self.current_image_index
        ].name

        self.all_labels[image_name] = {
            keypoint: dict(values)
            for keypoint, values
            in self.current_frame_labels.items()
        }

    def next_frame(self) -> None:
        """Save and move to the next image."""

        self.save_all_labels()

        if self.current_image_index < len(self.image_paths) - 1:
            self.load_frame(
                self.current_image_index + 1
            )
        else:
            messagebox.showinfo(
                "Round complete",
                "You are already on the final frame.",
            )

    def previous_frame(self) -> None:
        """Save and move to the previous image."""

        self.save_all_labels()

        if self.current_image_index > 0:
            self.load_frame(
                self.current_image_index - 1
            )
        else:
            self.status_message_var.set(
                "This is the first frame."
            )

    def find_first_unfinished_keypoint(self) -> int:
        """
        Select the first keypoint without a completed decision.

        A keypoint counts as finished when it has either:
        - visible = 1 with coordinates
        - visible = 0 after explicit labeling

        Since an untouched point and an invisible point both use visible=0
        in the final CSV, the GUI starts at the first point without
        coordinates. Previously labeled images can still be reviewed
        using number keys.
        """

        for index, keypoint in enumerate(KEYPOINTS):
            label = self.current_frame_labels[keypoint]

            if (
                label["visible"] == 1
                and label["x"] is not None
                and label["y"] is not None
            ):
                continue

            return index

        return 0

    # ========================================================
    # COORDINATE CONVERSION
    # ========================================================

    def canvas_to_original(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> tuple[float, float] | None:
        """Convert canvas coordinates to original image coordinates."""

        if self.original_image is None:
            return None

        display_x = canvas_x - self.image_offset_x
        display_y = canvas_y - self.image_offset_y

        displayed_width = (
            self.original_image.width * self.scale
        )

        displayed_height = (
            self.original_image.height * self.scale
        )

        if not (
            0 <= display_x < displayed_width
            and 0 <= display_y < displayed_height
        ):
            return None

        original_x = display_x / self.scale
        original_y = display_y / self.scale

        return original_x, original_y

    def original_to_canvas(
        self,
        original_x: float,
        original_y: float,
    ) -> tuple[float, float]:
        """Convert original image coordinates to canvas coordinates."""

        canvas_x = (
            original_x * self.scale
            + self.image_offset_x
        )

        canvas_y = (
            original_y * self.scale
            + self.image_offset_y
        )

        return canvas_x, canvas_y

    # ========================================================
    # LABELING
    # ========================================================

    def select_keypoint(self, index: int) -> None:
        """Select one keypoint for labeling."""

        self.current_keypoint_index = max(
            0,
            min(index, len(KEYPOINTS) - 1),
        )

        self.update_interface_text()
        self.redraw_canvas()

    def place_keypoint(
        self,
        event: tk.Event,
    ) -> None:
        """Place the selected keypoint with a left click."""

        coordinates = self.canvas_to_original(
            event.x,
            event.y,
        )

        if coordinates is None:
            return

        original_x, original_y = coordinates

        keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.current_frame_labels[keypoint] = {
            "x": float(original_x),
            "y": float(original_y),
            "visible": 1,
        }

        self.unsaved_changes = True

        self.advance_to_next_keypoint()
        self.update_interface_text()
        self.redraw_canvas()

    def mark_not_visible_from_mouse(
        self,
        event: tk.Event,
    ) -> None:
        """Right-click shortcut for invisible keypoint."""

        self.mark_current_not_visible()

    def mark_current_not_visible(self) -> None:
        """Mark the selected keypoint as not visible."""

        keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.current_frame_labels[keypoint] = {
            "x": None,
            "y": None,
            "visible": 0,
        }

        self.unsaved_changes = True

        self.advance_to_next_keypoint()
        self.update_interface_text()
        self.redraw_canvas()

    def delete_current_keypoint(self) -> None:
        """Delete the currently selected keypoint."""

        keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.current_frame_labels[keypoint] = {
            "x": None,
            "y": None,
            "visible": 0,
        }

        self.unsaved_changes = True

        self.status_message_var.set(
            f"Deleted: {keypoint}"
        )

        self.update_interface_text()
        self.redraw_canvas()

    def reset_current_frame(self) -> None:
        """Remove all labels from the current frame."""

        confirmed = messagebox.askyesno(
            "Reset frame",
            "Delete all labels on the current frame?",
        )

        if not confirmed:
            return

        self.current_frame_labels = (
            self.empty_frame_labels()
        )

        self.current_keypoint_index = 0
        self.unsaved_changes = True

        self.status_message_var.set(
            "Current frame was reset."
        )

        self.update_interface_text()
        self.redraw_canvas()

    def advance_to_next_keypoint(self) -> None:
        """Advance selection after placing or skipping a keypoint."""

        if self.current_keypoint_index < len(KEYPOINTS) - 1:
            self.current_keypoint_index += 1
        else:
            self.status_message_var.set(
                "All keypoints reviewed. Press D for next frame."
            )

    # ========================================================
    # DRAWING
    # ========================================================

    def redraw_after_resize(
        self,
        event: tk.Event,
    ) -> None:
        """Redraw the image when the canvas size changes."""

        self.redraw_canvas()

    def redraw_canvas(self) -> None:
        """Display image and all labels."""

        if self.original_image is None:
            return

        canvas_width = max(
            1,
            self.canvas.winfo_width(),
        )

        canvas_height = max(
            1,
            self.canvas.winfo_height(),
        )

        scale_x = canvas_width / self.original_image.width
        scale_y = canvas_height / self.original_image.height

        self.scale = min(
            scale_x,
            scale_y,
        )

        displayed_width = max(
            1,
            int(round(
                self.original_image.width * self.scale
            )),
        )

        displayed_height = max(
            1,
            int(round(
                self.original_image.height * self.scale
            )),
        )

        self.image_offset_x = (
            canvas_width - displayed_width
        ) / 2

        self.image_offset_y = (
            canvas_height - displayed_height
        ) / 2

        resized_image = self.original_image.resize(
            (displayed_width, displayed_height),
            Image.Resampling.LANCZOS,
        )

        self.display_image = ImageTk.PhotoImage(
            resized_image
        )

        self.canvas.delete("all")

        self.canvas.create_image(
            self.image_offset_x,
            self.image_offset_y,
            anchor=tk.NW,
            image=self.display_image,
        )

        for keypoint_index, keypoint in enumerate(KEYPOINTS):
            label = self.current_frame_labels[keypoint]

            if (
                label["visible"] != 1
                or label["x"] is None
                or label["y"] is None
            ):
                continue

            canvas_x, canvas_y = self.original_to_canvas(
                float(label["x"]),
                float(label["y"]),
            )

            color = KEYPOINT_COLORS[keypoint]

            radius = POINT_RADIUS

            if keypoint_index == self.current_keypoint_index:
                radius += 3

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
                canvas_x + 10,
                canvas_y - 10,
                text=keypoint,
                anchor=tk.SW,
                fill=color,
                font=("Segoe UI", 10, "bold"),
            )

    # ========================================================
    # STATUS
    # ========================================================

    def count_visible_keypoints(self) -> int:
        """Count visible keypoints on the current frame."""

        return sum(
            int(
                label["visible"] == 1
                and label["x"] is not None
                and label["y"] is not None
            )
            for label in self.current_frame_labels.values()
        )

    def count_labeled_images(self) -> int:
        """Count images containing at least one visible label."""

        count = 0

        for frame_labels in self.all_labels.values():
            has_visible_label = any(
                label["visible"] == 1
                for label in frame_labels.values()
            )

            if has_visible_label:
                count += 1

        current_image_name = self.image_paths[
            self.current_image_index
        ].name

        if current_image_name not in self.all_labels:
            if self.count_visible_keypoints() > 0:
                count += 1

        return count

    def update_interface_text(self) -> None:
        """Update all GUI text and button highlighting."""

        image_path = self.image_paths[
            self.current_image_index
        ]

        current_keypoint = KEYPOINTS[
            self.current_keypoint_index
        ]

        self.frame_counter_var.set(
            f"Frame {self.current_image_index + 1} "
            f"of {len(self.image_paths)}"
        )

        self.filename_var.set(
            image_path.name
        )

        self.current_keypoint_var.set(
            current_keypoint
        )

        current_color = KEYPOINT_COLORS[
            current_keypoint
        ]

        self.current_keypoint_label.configure(
            background=current_color,
            foreground="white",
        )

        visible_count = self.count_visible_keypoints()
        labeled_images = self.count_labeled_images()

        self.progress_var.set(
            f"Visible points: {visible_count}/{len(KEYPOINTS)}\n"
            f"Images started: {labeled_images}/{len(self.image_paths)}"
        )

        for index, button in enumerate(
            self.keypoint_buttons
        ):
            keypoint = KEYPOINTS[index]
            label = self.current_frame_labels[keypoint]

            if index == self.current_keypoint_index:
                relief = tk.SUNKEN
                border_width = 3
            else:
                relief = tk.RAISED
                border_width = 1

            if (
                label["visible"] == 1
                and label["x"] is not None
                and label["y"] is not None
            ):
                button_text = f"{index + 1}. {keypoint}  ✓"
            else:
                button_text = f"{index + 1}. {keypoint}"

            button.configure(
                text=button_text,
                background=KEYPOINT_COLORS[keypoint],
                foreground="white",
                activebackground=KEYPOINT_COLORS[keypoint],
                activeforeground="white",
                relief=relief,
                borderwidth=border_width,
            )

    # ========================================================
    # CLOSING
    # ========================================================

    def close_program(self) -> None:
        """Save labels and close the program."""

        try:
            self.save_all_labels()
        except Exception as error:
            should_close = messagebox.askyesno(
                "Save error",
                f"Labels could not be saved:\n\n{error}\n\n"
                "Close anyway?",
            )

            if not should_close:
                return

        self.root.destroy()


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    """Start the labeling GUI."""

    print("\n========================================")
    print("FRONTVIEW LABELING GUI")
    print("ROUND 02 — 8 MICE")
    print("========================================")
    print(f"Images: {IMAGE_DIR}")
    print(f"Labels: {LABEL_CSV}")
    print(f"Keypoints: {len(KEYPOINTS)}")

    root = tk.Tk()

    try:
        FrontviewLabelGUI(root)
        root.mainloop()

    except Exception:
        root.destroy()
        raise


if __name__ == "__main__":
    main()