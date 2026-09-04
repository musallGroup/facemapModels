from __future__ import annotations

import csv
import random
import re
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..common.image_viewer import ImageViewer
from ..common.dialogs import information, warning, confirm


VIDEO_FILTER = (
    "Video files (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;"
    "All files (*.*)"
)

MANIFEST_COLUMNS = [
    "saved_at",
    "video_path",
    "video_name",
    "frame_index",
    "total_frames",
    "timestamp_seconds",
    "timestamp_hh_mm_ss",
    "fps",
    "selection_mode",
    "saved_image",
]


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def frame_to_timestamp(frame_index: int, fps: float) -> str:
    seconds = frame_index / fps if fps > 0 else 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining:06.3f}"


class ResetIconButton(QPushButton):
    """Small, minimal reset button for one display parameter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("FrameSmallButton")
        self.setToolTip("Reset to 1.00")
        self.setFixedSize(32, 28)
        self.setText("~")
        self.setStyleSheet("""
            QPushButton#FrameSmallButton {
                background: #2A2E35;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 0;
                font-family: "Segoe UI";
                font-size: 15px;
                font-weight: 700;
            }

            QPushButton#FrameSmallButton:hover {
                background: #343942;
                color: #EAEAEA;
                border: 1px solid #D18B47;
            }

            QPushButton#FrameSmallButton:pressed {
                background: #1E2127;
                border: 1px solid #D18B47;
            }
        """)


class FramePickerPage(QWidget):
    previous_requested = Signal()
    next_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.video_paths: list[Path] = []
        self.video_index = -1

        self.capture: cv2.VideoCapture | None = None
        self.current_frame_bgr = None
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 0.0

        self.output_dir: Path | None = None
        self.manifest_path: Path | None = None

        self.saved_keys: set[tuple[str, int]] = set()
        self.saved_count_total = 0

        self.ignore_slider = False

        self.setStyleSheet("""
            QFrame#FramePickerPanel,
            QFrame#TransportPanel,
            QFrame#RandomPanel,
            QFrame#DisplayPanel {
                background: #1D2026;
                border: 1px solid #2D323A;
                border-radius: 14px;
            }

            QListWidget#VideoList {
                background: #14161A;
                color: #EAEAEA;
                border: 1px solid #2D323A;
                border-radius: 10px;
                padding: 6px;
            }

            QListWidget#VideoList::item {
                padding: 7px 8px;
                border-radius: 6px;
            }

            QListWidget#VideoList::item:selected {
                background: #2A2E35;
                color: #EAEAEA;
            }

            QListWidget#VideoList::item:hover {
                background: #22262D;
            }

            QLabel#FrameMeta {
                color: #AEB4BF;
                font-size: 12px;
            }

            QLabel#FrameStatus {
                color: #D18B47;
                font-size: 12px;
                font-weight: 600;
            }

            QPushButton#FrameSmallButton {
                background: #2A2E35;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }

            QPushButton#FrameSmallButton:hover {
                background: #343942;
                border: 1px solid #D18B47;
            }

            QPushButton#FrameSaveButton {
                background: #D18B47;
                color: #111111;
                border: none;
                border-radius: 9px;
                padding: 9px 18px;
                font-weight: 700;
            }

            QPushButton#FrameSaveButton:hover {
                background: #DFA15F;
            }

            QSpinBox#RandomSpin {
                background: #14161A;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 6px 8px;
                min-height: 22px;
            }

            QSpinBox#RandomSpin:focus {
                border: 1px solid #D18B47;
            }

            QSlider#FrameSlider::groove:horizontal,
            QSlider#DisplaySlider::groove:horizontal {
                height: 5px;
                background: #2B2F36;
                border-radius: 2px;
            }

            QSlider#FrameSlider::sub-page:horizontal,
            QSlider#DisplaySlider::sub-page:horizontal {
                background: #D18B47;
                border-radius: 2px;
            }

            QSlider#FrameSlider::handle:horizontal,
            QSlider#DisplaySlider::handle:horizontal {
                background: #EAEAEA;
                border: 1px solid #1E2127;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)

        self.build_ui()
        self.build_shortcuts()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(7)

        intro_row = QHBoxLayout()

        intro = QLabel(
            "Start by adding one or more source videos. Then pick frames manually "
            "or let LabelForge extract a random sample."
        )
        intro.setObjectName("PageSubtitle")
        intro.setWordWrap(True)

        intro_row.addWidget(intro, 1)
        root.addLayout(intro_row)

        content_row = QHBoxLayout()
        content_row.setSpacing(10)

        # LEFT: source videos --------------------------------------------
        videos_panel = QFrame()
        videos_panel.setObjectName("FramePickerPanel")
        videos_panel.setFixedWidth(230)

        videos_layout = QVBoxLayout(videos_panel)
        videos_layout.setContentsMargins(10, 10, 10, 10)
        videos_layout.setSpacing(8)

        video_title = QLabel("Source videos")
        video_title.setObjectName("FieldLabel")

        add_videos = QPushButton("+ Add Videos")
        add_videos.setObjectName("FrameSaveButton")
        add_videos.setToolTip("Add the source videos you want to sample frames from")
        add_videos.clicked.connect(self.add_videos)

        source_hint = QLabel(
            "Start here: add the videos you want to use for this dataset."
        )
        source_hint.setObjectName("FrameMeta")
        source_hint.setWordWrap(True)

        self.video_list = QListWidget()
        self.video_list.setObjectName("VideoList")
        self.video_list.currentRowChanged.connect(self.load_video)

        remove_video = QPushButton("Remove selected")
        remove_video.setObjectName("FrameSmallButton")
        remove_video.clicked.connect(self.remove_selected_video)

        self.video_count_label = QLabel("0 videos")
        self.video_count_label.setObjectName("FrameMeta")

        videos_layout.addWidget(video_title)
        videos_layout.addWidget(add_videos)
        videos_layout.addWidget(source_hint)
        videos_layout.addWidget(self.video_list, 1)
        videos_layout.addWidget(remove_video)
        videos_layout.addWidget(self.video_count_label)

        # RIGHT: extraction + display tools -------------------------------
        tools_column = QVBoxLayout()
        tools_column.setSpacing(10)

        random_panel = QFrame()
        random_panel.setObjectName("RandomPanel")
        random_panel.setFixedWidth(245)

        random_layout = QVBoxLayout(random_panel)
        random_layout.setContentsMargins(10, 10, 10, 10)
        random_layout.setSpacing(8)

        random_title = QLabel("Random extraction")
        random_title.setObjectName("FieldLabel")

        random_hint = QLabel(
            "Extract the same number of random frames from every loaded video."
        )
        random_hint.setObjectName("FrameMeta")
        random_hint.setWordWrap(True)

        count_row = QHBoxLayout()
        count_label = QLabel("Frames / video")
        count_label.setObjectName("FrameMeta")

        self.random_count = QSpinBox()
        self.random_count.setObjectName("RandomSpin")
        self.random_count.setRange(1, 1000)
        self.random_count.setValue(10)
        self.random_count.setFixedWidth(76)

        count_row.addWidget(count_label)
        count_row.addStretch(1)
        count_row.addWidget(self.random_count)

        random_button = QPushButton("Extract Random Frames")
        random_button.setObjectName("FrameSaveButton")
        random_button.clicked.connect(self.extract_random_frames)

        random_layout.addWidget(random_title)
        random_layout.addWidget(random_hint)
        random_layout.addLayout(count_row)
        random_layout.addWidget(random_button)

        display_panel = QFrame()
        display_panel.setObjectName("DisplayPanel")
        display_panel.setFixedWidth(245)

        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(10, 10, 10, 10)
        display_layout.setSpacing(7)

        display_title = QLabel("Display adjustment")
        display_title.setObjectName("FieldLabel")

        self.brightness_slider, self.brightness_value = self._make_display_slider(
            "Brightness",
            self.on_brightness_slider,
        )
        self.contrast_slider, self.contrast_value = self._make_display_slider(
            "Contrast",
            self.on_contrast_slider,
        )
        self.gamma_slider, self.gamma_value = self._make_display_slider(
            "Gamma",
            self.on_gamma_slider,
        )

        reset_display = QPushButton("Reset all")
        reset_display.setObjectName("FrameSmallButton")
        reset_display.clicked.connect(self.reset_display)

        display_layout.addWidget(display_title)
        display_layout.addLayout(self._display_slider_row(
            "Brightness",
            self.brightness_slider,
            self.brightness_value,
            lambda: self.reset_single_display("brightness"),
        ))
        display_layout.addLayout(self._display_slider_row(
            "Contrast",
            self.contrast_slider,
            self.contrast_value,
            lambda: self.reset_single_display("contrast"),
        ))
        display_layout.addLayout(self._display_slider_row(
            "Gamma",
            self.gamma_slider,
            self.gamma_value,
            lambda: self.reset_single_display("gamma"),
        ))
        display_layout.addWidget(reset_display)

        display_hint = QLabel(
            "Wheel: zoom · middle-drag: pan\n"
            "Double middle: reset view\n"
            "Shift/Ctrl/Alt + wheel: display"
        )
        display_hint.setObjectName("FrameMeta")
        display_hint.setWordWrap(True)
        display_layout.addWidget(display_hint)

        tools_column.addWidget(random_panel)
        tools_column.addWidget(display_panel)
        tools_column.addStretch(1)

        # MAIN VIEWER ----------------------------------------------------
        main_panel = QFrame()
        main_panel.setObjectName("FramePickerPanel")

        main_layout = QVBoxLayout(main_panel)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        top_meta = QHBoxLayout()

        self.video_name_label = QLabel("No video loaded")
        self.video_name_label.setObjectName("FieldLabel")

        self.saved_label = QLabel("Saved: 0")
        self.saved_label.setObjectName("FrameMeta")
        self.saved_label.setAlignment(Qt.AlignRight)

        top_meta.addWidget(self.video_name_label, 1)
        top_meta.addWidget(self.saved_label)

        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(560, 500)
        self.viewer.display_changed.connect(self.sync_display_controls)

        viewer_row = QHBoxLayout()
        viewer_row.setContentsMargins(0, 0, 0, 0)
        viewer_row.addWidget(self.viewer, 1)

        transport = QFrame()
        transport.setObjectName("TransportPanel")
        transport.setMaximumWidth(1520)

        transport_layout = QVBoxLayout(transport)
        transport_layout.setContentsMargins(12, 9, 12, 9)
        transport_layout.setSpacing(7)

        info_row = QHBoxLayout()

        self.frame_label = QLabel("Frame: —")
        self.frame_label.setObjectName("FrameMeta")

        self.time_label = QLabel("Time: —")
        self.time_label.setObjectName("FrameMeta")
        self.time_label.setAlignment(Qt.AlignRight)

        info_row.addWidget(self.frame_label)
        info_row.addStretch(1)
        info_row.addWidget(self.time_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setObjectName("FrameSlider")
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider_move)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        for label, amount in [
            ("≪ 10", -10),
            ("‹ 1", -1),
        ]:
            button = QPushButton(label)
            button.setObjectName("FrameSmallButton")
            button.clicked.connect(
                lambda checked=False, step=amount: self.move_frame(step)
            )
            controls.addWidget(button)

        save = QPushButton("Save Frame")
        save.setObjectName("FrameSaveButton")
        save.clicked.connect(self.save_current_frame)
        controls.addWidget(save)

        for label, amount in [
            ("1 ›", 1),
            ("10 ≫", 10),
        ]:
            button = QPushButton(label)
            button.setObjectName("FrameSmallButton")
            button.clicked.connect(
                lambda checked=False, step=amount: self.move_frame(step)
            )
            controls.addWidget(button)

        controls_container = QHBoxLayout()
        controls_container.addStretch(1)
        controls_container.addLayout(controls)
        controls_container.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("FrameStatus")
        self.status_label.setAlignment(Qt.AlignCenter)

        transport_layout.addLayout(info_row)
        transport_layout.addWidget(self.slider)
        transport_layout.addLayout(controls_container)
        transport_layout.addWidget(self.status_label)

        transport_row = QHBoxLayout()
        transport_row.setContentsMargins(0, 0, 0, 0)
        transport_row.addStretch(1)
        transport_row.addWidget(transport, 1)
        transport_row.addStretch(1)

        main_layout.addLayout(top_meta)
        main_layout.addLayout(viewer_row, 5)
        main_layout.addLayout(transport_row, 0)

        content_row.addWidget(videos_panel)
        content_row.addWidget(main_panel, 1)
        content_row.addLayout(tools_column)

        root.addLayout(content_row, 1)

        bottom = QHBoxLayout()

        previous = QPushButton("← Previous")
        previous.setObjectName("BackButton")
        previous.clicked.connect(self.previous_requested.emit)

        self.output_label = QLabel(
            "Frames will be stored inside the current LabelForge project."
        )
        self.output_label.setObjectName("FieldHint")
        self.output_label.setAlignment(Qt.AlignCenter)

        next_button = QPushButton("Continue →")
        next_button.setObjectName("PrimaryNextButton")
        next_button.clicked.connect(self.continue_to_labeling)

        bottom.addWidget(previous)
        bottom.addStretch(1)
        bottom.addWidget(self.output_label)
        bottom.addStretch(1)
        bottom.addWidget(next_button)

        root.addLayout(bottom)

    def _make_display_slider(self, name: str, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName("DisplaySlider")
        slider.setRange(20, 300)
        slider.setValue(100)
        slider.valueChanged.connect(callback)

        value = QLabel("1.00")
        value.setObjectName("FrameMeta")
        value.setFixedWidth(34)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        return slider, value

    def _display_slider_row(
        self,
        name: str,
        slider,
        value_label,
        reset_action,
    ):
        row = QHBoxLayout()
        row.setSpacing(5)

        label = QLabel(name)
        label.setObjectName("FrameMeta")
        label.setFixedWidth(58)

        reset_button = ResetIconButton()
        reset_button.setToolTip(f"Reset {name.lower()} to 1.00")
        reset_button.clicked.connect(reset_action)

        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        row.addWidget(reset_button)
        return row

    def build_shortcuts(self) -> None:
        shortcuts = [
            ("A", lambda: self.move_frame(-1)),
            ("Left", lambda: self.move_frame(-1)),
            ("D", lambda: self.move_frame(1)),
            ("Right", lambda: self.move_frame(1)),
            ("Shift+A", lambda: self.move_frame(-10)),
            ("Shift+Left", lambda: self.move_frame(-10)),
            ("Shift+D", lambda: self.move_frame(10)),
            ("Shift+Right", lambda: self.move_frame(10)),
            ("Space", self.save_current_frame),
        ]

        self.shortcuts = []
        for sequence, action in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)

    # ------------------------------------------------------------------
    # Display controls
    # ------------------------------------------------------------------

    def on_brightness_slider(self, value: int) -> None:
        self.viewer.set_brightness(value / 100.0)

    def on_contrast_slider(self, value: int) -> None:
        self.viewer.set_contrast(value / 100.0)

    def on_gamma_slider(self, value: int) -> None:
        self.viewer.set_gamma(value / 100.0)

    def sync_display_controls(
        self,
        brightness: float,
        contrast: float,
        gamma: float,
    ) -> None:
        sliders = [
            (self.brightness_slider, self.brightness_value, brightness),
            (self.contrast_slider, self.contrast_value, contrast),
            (self.gamma_slider, self.gamma_value, gamma),
        ]

        for slider, label, value in sliders:
            slider.blockSignals(True)
            slider.setValue(round(value * 100))
            slider.blockSignals(False)
            label.setText(f"{value:.2f}")

    def reset_single_display(self, control: str) -> None:
        if control == "brightness":
            self.viewer.set_brightness(1.0)
        elif control == "contrast":
            self.viewer.set_contrast(1.0)
        elif control == "gamma":
            self.viewer.set_gamma(1.0)

    def reset_display(self) -> None:
        self.viewer.reset_display()

    # ------------------------------------------------------------------
    # Project context
    # ------------------------------------------------------------------

    def set_project_context(self, project_draft: dict) -> None:
        location = project_draft.get("location", "").strip()
        name = project_draft.get("name", "").strip()

        if not location or not name:
            return

        safe_name = sanitize_name(name) or "LabelForge_Project"
        new_output_dir = Path(location) / safe_name / "frames"

        if new_output_dir == self.output_dir:
            return

        self.output_dir = new_output_dir
        self.manifest_path = self.output_dir / "extraction_manifest.csv"

        self.output_label.setText(f"Output: {self.output_dir}")
        self.load_existing_manifest_entries()

    # ------------------------------------------------------------------
    # Video list
    # ------------------------------------------------------------------

    def add_videos(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add source videos",
            "",
            VIDEO_FILTER,
        )

        if not files:
            return

        added = 0

        for filename in files:
            path = Path(filename)

            if path in self.video_paths:
                continue

            self.video_paths.append(path)

            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            self.video_list.addItem(item)
            added += 1

        self.update_video_count()

        if self.video_list.currentRow() < 0 and self.video_paths:
            self.video_list.setCurrentRow(0)

        if added:
            self.set_status(f"Added {added} video(s).")

    def remove_selected_video(self) -> None:
        row = self.video_list.currentRow()

        if row < 0 or row >= len(self.video_paths):
            return

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.video_paths.pop(row)
        self.video_list.takeItem(row)

        self.update_video_count()

        if self.video_paths:
            self.video_list.setCurrentRow(min(row, len(self.video_paths) - 1))
        else:
            self.reset_preview()

    def update_video_count(self) -> None:
        self.video_count_label.setText(f"{len(self.video_paths)} video(s)")

    # ------------------------------------------------------------------
    # Video handling
    # ------------------------------------------------------------------

    def load_video(self, index: int) -> None:
        if index < 0 or index >= len(self.video_paths):
            return

        if self.capture is not None:
            self.capture.release()

        self.video_index = index

        path = self.video_paths[index]
        self.capture = cv2.VideoCapture(str(path))

        if not self.capture.isOpened():
            warning(
                self,
                "Video error",
                f"Could not open video:\n{path}",
            )
            self.reset_preview()
            return

        self.total_frames = int(
            self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        self.fps = float(
            self.capture.get(cv2.CAP_PROP_FPS)
        )

        if self.fps <= 0:
            self.fps = 60.0

        if self.total_frames <= 0:
            warning(
                self,
                "Video error",
                f"No frames found in:\n{path}",
            )
            self.reset_preview()
            return

        self.ignore_slider = True
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.slider.setValue(0)
        self.ignore_slider = False

        self.video_name_label.setText(
            f"Video {index + 1}/{len(self.video_paths)} · {path.name}"
        )

        self.show_frame(0)
        self.set_status("Video loaded.")

    def read_frame(self, frame_index: int):
        if self.capture is None:
            return None

        frame_index = max(
            0,
            min(int(frame_index), self.total_frames - 1),
        )

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()

        if not ok:
            return None

        return frame

    def show_frame(self, frame_index: int) -> None:
        frame = self.read_frame(frame_index)

        if frame is None:
            self.set_status(f"Could not read frame {frame_index}.")
            return

        self.current_frame_index = int(frame_index)
        self.current_frame_bgr = frame.copy()

        self.ignore_slider = True
        self.slider.setValue(self.current_frame_index)
        self.ignore_slider = False

        self.viewer.set_bgr_image(frame)

        self.frame_label.setText(
            f"Frame: {self.current_frame_index:,} / {self.total_frames - 1:,}"
        )

        self.time_label.setText(
            f"Time: {frame_to_timestamp(self.current_frame_index, self.fps)}    "
            f"FPS: {self.fps:.3f}"
        )

        self.update_saved_label()

    def move_frame(self, amount: int) -> None:
        if self.total_frames <= 0:
            return

        target = max(
            0,
            min(
                self.current_frame_index + amount,
                self.total_frames - 1,
            ),
        )

        self.show_frame(target)

    def on_slider_move(self, value: int) -> None:
        if self.ignore_slider:
            return

        if self.total_frames <= 0:
            return

        self.show_frame(value)

    def reset_preview(self) -> None:
        self.current_frame_bgr = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.slider.setRange(0, 0)
        self.viewer.clear_image()
        self.video_name_label.setText("No video loaded")
        self.frame_label.setText("Frame: —")
        self.time_label.setText("Time: —")
        self.update_saved_label()

    # ------------------------------------------------------------------
    # Saving + manifest
    # ------------------------------------------------------------------

    def ensure_output(self) -> bool:
        if self.output_dir is None:
            information(
                self,
                "Project information required",
                "Please complete the Project step before saving frames.",
            )
            return False

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.manifest_path is None:
            self.manifest_path = self.output_dir / "extraction_manifest.csv"

        self.ensure_manifest_schema()
        return True

    def ensure_manifest_schema(self) -> None:
        if self.manifest_path is None:
            return

        if (
            not self.manifest_path.exists()
            or self.manifest_path.stat().st_size == 0
        ):
            with self.manifest_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=MANIFEST_COLUMNS,
                )
                writer.writeheader()
            return

        with self.manifest_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            reader = csv.DictReader(handle)
            current_columns = reader.fieldnames or []
            rows = list(reader)

        if current_columns == MANIFEST_COLUMNS:
            return

        migrated_rows = [
            {
                column: row.get(column, "")
                for column in MANIFEST_COLUMNS
            }
            for row in rows
        ]

        with self.manifest_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=MANIFEST_COLUMNS,
            )
            writer.writeheader()
            writer.writerows(migrated_rows)

    def save_frame_data(
        self,
        video_path: Path,
        frame_index: int,
        frame_bgr,
        fps: float,
        total_frames: int,
        selection_mode: str,
    ) -> bool:
        if not self.ensure_output():
            return False

        key = (
            str(video_path.resolve()),
            int(frame_index),
        )

        if key in self.saved_keys:
            return False

        safe_stem = sanitize_name(video_path.stem)
        output_name = f"{safe_stem}_frame_{frame_index:09d}.png"
        output_path = self.output_dir / output_name

        # IMPORTANT: Always save the ORIGINAL frame, never the display-adjusted view.
        ok = cv2.imwrite(str(output_path), frame_bgr)

        if not ok:
            return False

        timestamp_seconds = frame_index / fps if fps > 0 else 0.0

        with self.manifest_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=MANIFEST_COLUMNS,
            )
            writer.writerow(
                {
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "frame_index": frame_index,
                    "total_frames": total_frames,
                    "timestamp_seconds": f"{timestamp_seconds:.6f}",
                    "timestamp_hh_mm_ss": frame_to_timestamp(frame_index, fps),
                    "fps": f"{fps:.6f}",
                    "selection_mode": selection_mode,
                    "saved_image": str(output_path),
                }
            )

        self.saved_keys.add(key)
        self.saved_count_total += 1
        return True

    def save_current_frame(self) -> None:
        if self.current_frame_bgr is None:
            return

        path = self.video_paths[self.video_index]

        saved = self.save_frame_data(
            path,
            self.current_frame_index,
            self.current_frame_bgr,
            self.fps,
            self.total_frames,
            "manual",
        )

        if saved:
            self.set_status(f"Saved frame {self.current_frame_index:,}.")
        else:
            self.set_status(
                "This frame is already saved or could not be written."
            )

        self.update_saved_label()

    def extract_random_frames(self) -> None:
        if not self.video_paths:
            warning(
                self,
                "No videos",
                "Please add at least one source video first.",
            )
            return

        if not self.ensure_output():
            return

        count = self.random_count.value()
        total_saved_now = 0

        for video_path in self.video_paths:
            capture = cv2.VideoCapture(str(video_path))

            if not capture.isOpened():
                continue

            total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )

            if fps <= 0:
                fps = 60.0

            if total_frames <= 0:
                capture.release()
                continue

            existing_for_video = {
                frame_idx
                for path_str, frame_idx in self.saved_keys
                if path_str == str(video_path.resolve())
            }

            candidates = [
                i for i in range(total_frames)
                if i not in existing_for_video
            ]

            if not candidates:
                capture.release()
                continue

            sample_size = min(count, len(candidates))
            selected_indices = sorted(
                random.sample(candidates, sample_size)
            )

            for frame_index in selected_indices:
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index,
                )
                ok, frame = capture.read()

                if not ok:
                    continue

                if self.save_frame_data(
                    video_path,
                    frame_index,
                    frame,
                    fps,
                    total_frames,
                    "random",
                ):
                    total_saved_now += 1

            capture.release()

        self.update_saved_label()
        self.set_status(
            f"Random extraction complete · {total_saved_now} new frame(s) saved."
        )

    def load_existing_manifest_entries(self) -> None:
        self.saved_keys = set()
        self.saved_count_total = 0

        if self.manifest_path is None or not self.manifest_path.exists():
            self.update_saved_label()
            return

        try:
            with self.manifest_path.open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                reader = csv.DictReader(handle)

                for row in reader:
                    video_path = row.get("video_path", "")
                    frame_index = row.get("frame_index", "")

                    if video_path and frame_index:
                        self.saved_keys.add(
                            (
                                str(Path(video_path).resolve()),
                                int(frame_index),
                            )
                        )

            self.saved_count_total = len(self.saved_keys)

        except Exception:
            self.saved_keys = set()
            self.saved_count_total = 0

        self.update_saved_label()

    def update_saved_label(self) -> None:
        if 0 <= self.video_index < len(self.video_paths):
            active = str(self.video_paths[self.video_index].resolve())
            saved_this_video = sum(
                1
                for path_str, _ in self.saved_keys
                if path_str == active
            )
        else:
            saved_this_video = 0

        self.saved_label.setText(
            f"Saved this video: {saved_this_video}    "
            f"Total: {self.saved_count_total}"
        )

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def collect_data(self) -> dict:
        return {
            "videos": [str(path) for path in self.video_paths],
            "output_dir": str(self.output_dir) if self.output_dir else "",
            "manifest_path": str(self.manifest_path) if self.manifest_path else "",
            "saved_frames": self.saved_count_total,
            "random_frames_per_video": self.random_count.value(),
        }

    def continue_to_labeling(self) -> None:
        if not self.video_paths:
            warning(
                self,
                "No videos",
                "Please add at least one source video.",
            )
            return

        if self.saved_count_total == 0:
            if not confirm(
                self,
                "No frames saved",
                "No frames have been saved yet.\n\nContinue anyway?",
                confirm_text="Continue",
                cancel_text="Cancel",
            ):
                return

        self.next_requested.emit(self.collect_data())

    def close_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
