from __future__ import annotations

import csv
import re
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..common.dialogs import confirm, information, warning
from ..common.image_viewer import ImageViewer


STATE_UNSET = "unset"
STATE_VISIBLE = "visible"
STATE_NOT_VISIBLE = "not_visible"


def sanitize_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def empty_label() -> dict:
    return {
        "x": None,
        "y": None,
        "visible": 0,
        "state": STATE_UNSET,
    }


class KeypointGroupSection(QFrame):
    """Collapsible keypoint group used in the Label workspace."""

    def __init__(
        self,
        group_name: str,
        group_color: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("KeypointGroupSection")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.toggle = QToolButton()
        self.toggle.setObjectName("KeypointGroupHeader")
        self.toggle.setText(group_name or "Ungrouped")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setArrowType(Qt.DownArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setStyleSheet(
            f"QToolButton#KeypointGroupHeader {{ border-left: 4px solid {group_color}; }}"
        )

        self.content = QWidget()
        self.content.setObjectName("KeypointGroupContent")

        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(7, 2, 0, 4)
        self.content_layout.setSpacing(4)

        self.toggle.toggled.connect(self.set_expanded)

        layout.addWidget(self.toggle)
        layout.addWidget(self.content)

    def set_expanded(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )



class LabelingPage(QWidget):
    previous_requested = Signal()
    next_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.project_draft: dict = {}
        self.keypoint_draft: dict = {}
        self.frame_draft: dict = {}

        self.keypoints: list[dict] = []
        self.frame_paths: list[Path] = []
        self.current_frame_index = 0
        self.current_keypoint_index = 0

        # Labeling traversal:
        # "frame"    = finish all keypoints on one frame, then next frame
        # "keypoint" = finish one keypoint across all frames, then next keypoint
        self.labeling_mode = "frame"
        self._completion_announced = False

        self.labels_by_path: dict[str, dict[str, dict]] = {}
        self.current_labels: dict[str, dict] = {}

        self.labels_dir: Path | None = None
        self.labels_csv: Path | None = None

        self.shortcut_objects: list[QShortcut] = []
        self.keypoint_buttons: list[QPushButton] = []
        self.group_sections: list[KeypointGroupSection] = []

        self._building_context = False

        self.setStyleSheet("""
            QFrame#LabelPanel,
            QFrame#LabelToolPanel,
            QFrame#LabelTransportPanel {
                background: #1D2026;
                border: 1px solid #2D323A;
                border-radius: 14px;
            }

            QLabel#LabelMeta {
                color: #AEB4BF;
                font-size: 12px;
            }

            QLabel#LabelStatus {
                color: #D18B47;
                font-size: 12px;
                font-weight: 600;
            }

            QPushButton#KeypointButton {
                background: #14161A;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 8px 10px;
                text-align: left;
                font-weight: 600;
            }

            QPushButton#KeypointButton:hover {
                border: 1px solid #D18B47;
                background: #22262D;
            }

            QPushButton#KeypointButton:checked {
                border: 2px solid #D18B47;
                background: #2A2E35;
            }

            QFrame#KeypointGroupSection,
            QWidget#KeypointGroupContent,
            QWidget#KeypointContainer {
                background: transparent;
                border: none;
            }

            QToolButton#KeypointGroupHeader {
                background: #252930;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 7px 9px;
                font-weight: 700;
                text-align: left;
            }

            QToolButton#KeypointGroupHeader:hover {
                background: #2D323A;
                border: 1px solid #D18B47;
            }

            QFrame#ActiveKeypointCard {
                background: #14161A;
                border: 1px solid #3A4049;
                border-radius: 10px;
            }

            QLabel#ActiveKeypointEyebrow {
                color: #AEB4BF;
                font-size: 10px;
                font-weight: 700;
            }

            QLabel#ActiveKeypointName {
                color: #EAEAEA;
                font-size: 17px;
                font-weight: 700;
            }

            QPushButton#LabelSmallButton {
                background: #2A2E35;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }

            QPushButton#LabelSmallButton:hover {
                border: 1px solid #D18B47;
                background: #343942;
            }

            QPushButton#LabelPrimaryButton {
                background: #D18B47;
                color: #111111;
                border: none;
                border-radius: 9px;
                padding: 9px 18px;
                font-weight: 700;
            }

            QPushButton#LabelPrimaryButton:hover {
                background: #DFA15F;
            }

            QPushButton#ModeButton {
                background: #14161A;
                color: #AEB4BF;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
            }

            QPushButton#ModeButton:hover {
                color: #EAEAEA;
                border: 1px solid #D18B47;
            }

            QPushButton#ModeButton:checked {
                background: #2A2E35;
                color: #EAEAEA;
                border: 1px solid #D18B47;
            }

            QSlider#DisplaySlider::groove:horizontal {
                height: 5px;
                background: #2B2F36;
                border-radius: 2px;
            }

            QSlider#DisplaySlider::sub-page:horizontal {
                background: #D18B47;
                border-radius: 2px;
            }

            QSlider#DisplaySlider::handle:horizontal {
                background: #EAEAEA;
                border: 1px solid #1E2127;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)

        self.build_ui()
        self.build_global_shortcuts()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(7)

        # Dataset progress stays above the workspace.
        # Labeling mode itself lives with the other workspace controls on the right.
        progress_row = QHBoxLayout()
        self.progress_label = QLabel("No frames loaded")
        self.progress_label.setObjectName("LabelMeta")
        self.progress_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        progress_row.addStretch(1)
        progress_row.addWidget(self.progress_label)
        root.addLayout(progress_row)

        content = QHBoxLayout()
        content.setSpacing(10)

        # LEFT: keypoints ------------------------------------------------
        keypoint_panel = QFrame()
        keypoint_panel.setObjectName("LabelPanel")
        keypoint_panel.setFixedWidth(270)

        keypoint_layout = QVBoxLayout(keypoint_panel)
        keypoint_layout.setContentsMargins(10, 10, 10, 10)
        keypoint_layout.setSpacing(8)

        keypoint_title = QLabel("Keypoints")
        keypoint_title.setObjectName("FieldLabel")

        self.selected_label = QLabel("No keypoint selected")
        self.selected_label.setObjectName("LabelStatus")
        self.selected_label.setWordWrap(True)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        scroll.viewport().setStyleSheet("background: #1D2026;")

        self.keypoint_container = QWidget()
        self.keypoint_container.setObjectName("KeypointContainer")
        self.keypoint_container.setStyleSheet("background: #1D2026;")

        self.keypoint_list_layout = QVBoxLayout(self.keypoint_container)
        self.keypoint_list_layout.setContentsMargins(0, 0, 0, 0)
        self.keypoint_list_layout.setSpacing(6)
        self.keypoint_list_layout.addStretch(1)

        scroll.setWidget(self.keypoint_container)

        self.state_summary = QLabel("")
        self.state_summary.setObjectName("LabelMeta")
        self.state_summary.setWordWrap(True)

        not_visible = QPushButton("Mark not visible (N)")
        not_visible.setObjectName("LabelSmallButton")
        not_visible.clicked.connect(self.mark_current_not_visible)

        clear_point = QPushButton("Clear selected (Backspace)")
        clear_point.setObjectName("LabelSmallButton")
        clear_point.clicked.connect(self.clear_current_keypoint)

        keypoint_layout.addWidget(keypoint_title)
        keypoint_layout.addWidget(self.selected_label)
        keypoint_layout.addWidget(scroll, 1)
        keypoint_layout.addWidget(self.state_summary)
        keypoint_layout.addWidget(not_visible)
        keypoint_layout.addWidget(clear_point)

        # CENTER: viewer -------------------------------------------------
        main_panel = QFrame()
        main_panel.setObjectName("LabelPanel")

        main_layout = QVBoxLayout(main_panel)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        name_row = QHBoxLayout()

        self.frame_name_label = QLabel("No frame loaded")
        self.frame_name_label.setObjectName("FieldLabel")

        self.save_status = QLabel("")
        self.save_status.setObjectName("LabelMeta")
        self.save_status.setAlignment(Qt.AlignRight)

        name_row.addWidget(self.frame_name_label, 1)
        name_row.addWidget(self.save_status)

        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(620, 520)
        self.viewer.image_left_clicked.connect(self.place_current_keypoint)
        self.viewer.image_right_clicked.connect(
            lambda _x, _y: self.mark_current_not_visible()
        )
        self.viewer.display_changed.connect(self.sync_display_controls)

        transport = QFrame()
        transport.setObjectName("LabelTransportPanel")

        transport_layout = QHBoxLayout(transport)
        transport_layout.setContentsMargins(10, 8, 10, 8)
        transport_layout.setSpacing(8)

        previous = QPushButton("← Previous")
        previous.setObjectName("LabelSmallButton")
        previous.clicked.connect(self.previous_frame)

        save = QPushButton("Save labels")
        save.setObjectName("LabelPrimaryButton")
        save.clicked.connect(self.save_all_labels)

        next_button = QPushButton("Next →")
        next_button.setObjectName("LabelSmallButton")
        next_button.clicked.connect(self.next_frame)

        self.frame_counter = QLabel("Frame —")
        self.frame_counter.setObjectName("LabelMeta")
        self.frame_counter.setAlignment(Qt.AlignCenter)

        transport_layout.addWidget(previous)
        transport_layout.addStretch(1)
        transport_layout.addWidget(self.frame_counter)
        transport_layout.addStretch(1)
        transport_layout.addWidget(save)
        transport_layout.addWidget(next_button)

        self.active_card = QFrame()
        self.active_card.setObjectName("ActiveKeypointCard")

        active_layout = QHBoxLayout(self.active_card)
        active_layout.setContentsMargins(12, 8, 12, 8)
        active_layout.setSpacing(10)

        active_text = QVBoxLayout()
        active_text.setSpacing(0)

        active_eyebrow = QLabel("NOW LABELING")
        active_eyebrow.setObjectName("ActiveKeypointEyebrow")

        self.active_keypoint_name = QLabel("No keypoint selected")
        self.active_keypoint_name.setObjectName("ActiveKeypointName")

        self.active_group_label = QLabel("")
        self.active_group_label.setObjectName("LabelMeta")
        self.active_group_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        active_text.addWidget(active_eyebrow)
        active_text.addWidget(self.active_keypoint_name)

        active_layout.addLayout(active_text)
        active_layout.addStretch(1)
        active_layout.addWidget(self.active_group_label)

        main_layout.addLayout(name_row)
        main_layout.addWidget(self.active_card)
        main_layout.addWidget(self.viewer, 1)
        main_layout.addWidget(transport)

        # RIGHT: display + help -----------------------------------------
        tools = QVBoxLayout()
        tools.setSpacing(10)

        display_panel = QFrame()
        display_panel.setObjectName("LabelToolPanel")
        display_panel.setFixedWidth(255)

        display_layout = QVBoxLayout(display_panel)
        display_layout.setContentsMargins(10, 10, 10, 10)
        display_layout.setSpacing(7)

        display_title = QLabel("Display adjustment")
        display_title.setObjectName("FieldLabel")

        self.brightness_slider, self.brightness_value = self._make_slider(
            self.on_brightness_slider
        )
        self.contrast_slider, self.contrast_value = self._make_slider(
            self.on_contrast_slider
        )
        self.gamma_slider, self.gamma_value = self._make_slider(
            self.on_gamma_slider
        )

        display_layout.addWidget(display_title)
        display_layout.addLayout(
            self._display_row(
                "Brightness",
                self.brightness_slider,
                self.brightness_value,
                lambda: self.viewer.set_brightness(1.0),
            )
        )
        display_layout.addLayout(
            self._display_row(
                "Contrast",
                self.contrast_slider,
                self.contrast_value,
                lambda: self.viewer.set_contrast(1.0),
            )
        )
        display_layout.addLayout(
            self._display_row(
                "Gamma",
                self.gamma_slider,
                self.gamma_value,
                lambda: self.viewer.set_gamma(1.0),
            )
        )

        reset_all = QPushButton("Reset all")
        reset_all.setObjectName("LabelSmallButton")
        reset_all.clicked.connect(self.viewer.reset_display)
        display_layout.addWidget(reset_all)

        mode_panel = QFrame()
        mode_panel.setObjectName("LabelToolPanel")
        mode_panel.setFixedWidth(255)

        mode_layout = QVBoxLayout(mode_panel)
        mode_layout.setContentsMargins(10, 10, 10, 10)
        mode_layout.setSpacing(8)

        mode_title = QLabel("Labeling mode")
        mode_title.setObjectName("FieldLabel")

        mode_buttons = QHBoxLayout()
        mode_buttons.setSpacing(7)

        self.frame_mode_button = QPushButton("Frame")
        self.frame_mode_button.setObjectName("ModeButton")
        self.frame_mode_button.setCheckable(True)
        self.frame_mode_button.setChecked(True)
        self.frame_mode_button.setToolTip(
            "Finish the keypoints on one frame before automatically continuing."
        )

        self.keypoint_mode_button = QPushButton("Keypoint")
        self.keypoint_mode_button.setObjectName("ModeButton")
        self.keypoint_mode_button.setCheckable(True)
        self.keypoint_mode_button.setToolTip(
            "Keep the current keypoint and label it across the frames."
        )

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.frame_mode_button)
        self.mode_group.addButton(self.keypoint_mode_button)

        self.frame_mode_button.clicked.connect(
            lambda checked=False: self.set_labeling_mode("frame")
        )
        self.keypoint_mode_button.clicked.connect(
            lambda checked=False: self.set_labeling_mode("keypoint")
        )

        mode_buttons.addWidget(self.frame_mode_button, 1)
        mode_buttons.addWidget(self.keypoint_mode_button, 1)

        self.mode_hint = QLabel(
            "Complete the keypoints on the current frame."
        )
        self.mode_hint.setObjectName("LabelMeta")
        self.mode_hint.setWordWrap(True)

        mode_layout.addWidget(mode_title)
        mode_layout.addLayout(mode_buttons)
        mode_layout.addWidget(self.mode_hint)

        help_panel = QFrame()
        help_panel.setObjectName("LabelToolPanel")
        help_panel.setFixedWidth(255)

        help_layout = QVBoxLayout(help_panel)
        help_layout.setContentsMargins(10, 10, 10, 10)

        help_title = QLabel("Controls")
        help_title.setObjectName("FieldLabel")

        help_text = QLabel(
            "Left click: place point\n"
            "Right click / N: not visible\n"
            "Mouse wheel: zoom\n"
            "Middle-drag: pan\n"
            "Double middle: reset view\n"
            "↑ / ↓: previous / next keypoint\n"
            "A / D or ← / →: previous / next frame\n"
            "Backspace: clear selected\n"
            "S: save labels"
        )
        help_text.setObjectName("LabelMeta")
        help_text.setWordWrap(True)

        help_layout.addWidget(help_title)
        help_layout.addWidget(help_text)

        tools.addWidget(display_panel)
        tools.addWidget(mode_panel)
        tools.addWidget(help_panel)
        tools.addStretch(1)

        content.addWidget(keypoint_panel)
        content.addWidget(main_panel, 1)
        content.addLayout(tools)

        root.addLayout(content, 1)

        bottom = QHBoxLayout()

        back_step = QPushButton("← Previous step")
        back_step.setObjectName("BackButton")
        back_step.clicked.connect(self.previous_requested.emit)

        self.output_label = QLabel("Labels will be saved inside the current LabelForge project.")
        self.output_label.setObjectName("FieldHint")
        self.output_label.setAlignment(Qt.AlignCenter)

        continue_button = QPushButton("Continue →")
        continue_button.setObjectName("PrimaryNextButton")
        continue_button.clicked.connect(self.continue_to_review)

        bottom.addWidget(back_step)
        bottom.addStretch(1)
        bottom.addWidget(self.output_label)
        bottom.addStretch(1)
        bottom.addWidget(continue_button)

        root.addLayout(bottom)

    def _make_slider(self, callback):
        slider = QSlider(Qt.Horizontal)
        slider.setObjectName("DisplaySlider")
        slider.setRange(20, 300)
        slider.setValue(100)
        slider.valueChanged.connect(callback)

        value = QLabel("1.00")
        value.setObjectName("LabelMeta")
        value.setFixedWidth(34)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return slider, value

    def _display_row(self, name, slider, value_label, reset_action):
        row = QHBoxLayout()
        row.setSpacing(5)

        label = QLabel(name)
        label.setObjectName("LabelMeta")
        label.setFixedWidth(58)

        reset = QPushButton("~")
        reset.setObjectName("LabelSmallButton")
        reset.setToolTip(f"Reset {name.lower()} to 1.00")
        reset.setFixedSize(32, 28)
        reset.clicked.connect(reset_action)

        row.addWidget(label)
        row.addWidget(slider, 1)
        row.addWidget(value_label)
        row.addWidget(reset)
        return row

    # ------------------------------------------------------------------
    # Context / data loading
    # ------------------------------------------------------------------

    def set_context(
        self,
        project_draft: dict,
        keypoint_draft: dict,
        frame_draft: dict,
    ) -> None:
        if self._building_context:
            return

        self._building_context = True
        try:
            self.project_draft = dict(project_draft)
            self.keypoint_draft = dict(keypoint_draft)
            self.frame_draft = dict(frame_draft)

            self.keypoints = self._flatten_keypoints(keypoint_draft)
            self.frame_paths = self._load_frame_paths(frame_draft)

            self._configure_output_paths()
            self._build_keypoint_buttons()
            self._build_keypoint_shortcuts()
            self._load_existing_labels()

            if self.frame_paths:
                self.current_frame_index = min(
                    self.current_frame_index,
                    len(self.frame_paths) - 1,
                )
                self.load_frame(self.current_frame_index)
            else:
                self.viewer.clear_image()
                self.frame_name_label.setText("No saved frames found")
                self.frame_counter.setText("Frame —")
                self.progress_label.setText("0 frames")
        finally:
            self._building_context = False

    def _flatten_keypoints(self, keypoint_draft: dict) -> list[dict]:
        flattened = []

        for group in keypoint_draft.get("groups", []):
            for keypoint in group.get("keypoints", []):
                flattened.append(
                    {
                        "name": keypoint.get("name", ""),
                        "shortcut": keypoint.get("shortcut", ""),
                        "color": keypoint.get("color", "#D18B47"),
                        "group": group.get("name", ""),
                    }
                )

        return [kp for kp in flattened if kp["name"]]

    def _load_frame_paths(self, frame_draft: dict) -> list[Path]:
        manifest = frame_draft.get("manifest_path", "")

        if manifest and Path(manifest).exists():
            paths = []
            with Path(manifest).open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                for row in csv.DictReader(handle):
                    saved_image = row.get("saved_image", "").strip()
                    if saved_image and Path(saved_image).exists():
                        paths.append(Path(saved_image))

            # manifest order is useful and reproducible
            return paths

        output_dir = frame_draft.get("output_dir", "")
        if output_dir and Path(output_dir).exists():
            return sorted(Path(output_dir).glob("*.png"))

        return []

    def _configure_output_paths(self) -> None:
        location = self.project_draft.get("location", "").strip()
        name = self.project_draft.get("name", "").strip()

        if not location or not name:
            self.labels_dir = None
            self.labels_csv = None
            return

        safe_name = sanitize_name(name) or "LabelForge_Project"
        project_root = Path(location) / safe_name

        self.labels_dir = project_root / "labels"
        self.labels_csv = self.labels_dir / "labels.csv"

        self.output_label.setText(f"Output: {self.labels_csv}")

    def _build_keypoint_buttons(self) -> None:
        for section in self.group_sections:
            section.setParent(None)
            section.deleteLater()

        self.group_sections = []
        self.keypoint_buttons = []

        while self.keypoint_list_layout.count() > 1:
            item = self.keypoint_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        grouped: dict[str, list[tuple[int, dict]]] = {}

        for index, kp in enumerate(self.keypoints):
            group_name = kp.get("group", "") or "Ungrouped"
            grouped.setdefault(group_name, []).append((index, kp))

        for group_name, members in grouped.items():
            group_color = members[0][1].get("color", "#D18B47")

            section = KeypointGroupSection(
                group_name,
                group_color,
            )
            self.group_sections.append(section)

            for index, kp in members:
                button = QPushButton()
                button.setObjectName("KeypointButton")
                button.setCheckable(True)
                button.clicked.connect(
                    lambda checked=False, idx=index: self.select_keypoint(idx)
                )

                # Preserve global index lookup while visually grouping.
                button._keypoint_index = index

                self.keypoint_buttons.append(button)
                section.content_layout.addWidget(button)

            self.keypoint_list_layout.insertWidget(
                self.keypoint_list_layout.count() - 1,
                section,
            )

        if self.keypoints:
            self.current_keypoint_index = min(
                self.current_keypoint_index,
                len(self.keypoints) - 1,
            )

        self.update_keypoint_panel()

    def _build_keypoint_shortcuts(self) -> None:
        for shortcut in self.shortcut_objects:
            shortcut.setParent(None)
            shortcut.deleteLater()

        self.shortcut_objects = []

        for index, kp in enumerate(self.keypoints):
            sequence = kp.get("shortcut", "").strip()

            if not sequence:
                continue

            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda idx=index: self.select_keypoint(idx)
            )
            self.shortcut_objects.append(shortcut)

    def _load_existing_labels(self) -> None:
        self.labels_by_path = {}

        if self.labels_csv is None or not self.labels_csv.exists():
            return

        with self.labels_csv.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            for row in csv.DictReader(handle):
                path_text = row.get("png_path", "").strip()

                if not path_text:
                    continue

                frame_labels = {
                    kp["name"]: empty_label()
                    for kp in self.keypoints
                }

                for kp in self.keypoints:
                    name = kp["name"]
                    state = row.get(f"{name}_state", "").strip()

                    if state == STATE_VISIBLE:
                        try:
                            x = float(row.get(f"{name}_x", ""))
                            y = float(row.get(f"{name}_y", ""))
                        except ValueError:
                            continue

                        frame_labels[name] = {
                            "x": x,
                            "y": y,
                            "visible": 1,
                            "state": STATE_VISIBLE,
                        }

                    elif state == STATE_NOT_VISIBLE:
                        frame_labels[name] = {
                            "x": None,
                            "y": None,
                            "visible": 0,
                            "state": STATE_NOT_VISIBLE,
                        }

                self.labels_by_path[self._normalize_path(path_text)] = frame_labels

    # ------------------------------------------------------------------
    # Labeling mode
    # ------------------------------------------------------------------

    def set_labeling_mode(self, mode: str) -> None:
        if mode not in {"frame", "keypoint"}:
            return

        self.store_current_frame()
        self.labeling_mode = mode

        self.frame_mode_button.setChecked(mode == "frame")
        self.keypoint_mode_button.setChecked(mode == "keypoint")

        if mode == "frame":
            self.mode_hint.setText(
                "Complete the keypoints on the current frame."
            )
        else:
            self.mode_hint.setText(
                "Label the current keypoint across the frames."
            )

        self.save_status.setText(
            "Frame mode" if mode == "frame" else "Keypoint mode"
        )

    def select_previous_keypoint(self) -> None:
        if not self.keypoints:
            return

        self.current_keypoint_index = (
            self.current_keypoint_index - 1
        ) % len(self.keypoints)

        self.update_keypoint_panel()
        self.update_overlays()

    def select_next_keypoint(self) -> None:
        if not self.keypoints:
            return

        self.current_keypoint_index = (
            self.current_keypoint_index + 1
        ) % len(self.keypoints)

        self.update_keypoint_panel()
        self.update_overlays()

    # ------------------------------------------------------------------
    # Label state
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        return str(Path(path)).replace("/", "\\").lower()

    def _new_frame_labels(self) -> dict[str, dict]:
        return {
            kp["name"]: empty_label()
            for kp in self.keypoints
        }

    def _complete_frame_labels(
        self,
        labels: dict[str, dict] | None,
    ) -> dict[str, dict]:
        """
        Return a full label dictionary for the CURRENT keypoint schema.

        Older/partial in-memory states may contain only the keypoints that
        were already touched (e.g. only eye_top). Every missing keypoint is
        therefore filled as STATE_UNSET instead of causing a KeyError.
        """
        labels = labels or {}
        completed: dict[str, dict] = {}

        for kp in self.keypoints:
            name = kp["name"]
            source = labels.get(name)

            if not isinstance(source, dict):
                completed[name] = empty_label()
                continue

            completed[name] = {
                "x": source.get("x"),
                "y": source.get("y"),
                "visible": int(source.get("visible", 0) or 0),
                "state": source.get("state", STATE_UNSET) or STATE_UNSET,
            }

        return completed

    def store_current_frame(self) -> None:
        if not self.frame_paths:
            return

        path = self.frame_paths[self.current_frame_index]
        completed = self._complete_frame_labels(self.current_labels)

        self.labels_by_path[self._normalize_path(path)] = {
            name: dict(label)
            for name, label in completed.items()
        }

    def load_frame(self, index: int) -> None:
        if not self.frame_paths:
            return

        self.store_current_frame()

        index = max(0, min(index, len(self.frame_paths) - 1))
        self.current_frame_index = index
        path = self.frame_paths[index]

        frame = cv2.imread(str(path))

        if frame is None:
            warning(
                self,
                "Frame error",
                f"Could not open frame:\n{path}",
            )
            return

        existing = self.labels_by_path.get(
            self._normalize_path(path)
        )

        self.current_labels = self._complete_frame_labels(existing)

        self.current_keypoint_index = self._first_unfinished_keypoint()
        self.viewer.set_bgr_image(
            frame,
            preserve_view=(self.labeling_mode == "keypoint"),
        )

        self.frame_name_label.setText(path.name)
        self.frame_counter.setText(
            f"Frame {index + 1} of {len(self.frame_paths)}"
        )
        completed_annotations = self.count_completed_annotations()
        total_annotations = len(self.frame_paths) * len(self.keypoints)

        self.progress_label.setText(
            f"{completed_annotations} / {total_annotations} annotations"
        )

        self.update_keypoint_panel()
        self.update_overlays()

    def select_keypoint(self, index: int) -> None:
        if not self.keypoints:
            return

        self.current_keypoint_index = max(
            0,
            min(index, len(self.keypoints) - 1),
        )
        self.update_keypoint_panel()
        self.update_overlays()

    def place_current_keypoint(self, x: float, y: float) -> None:
        if not self.keypoints:
            return

        name = self.keypoints[self.current_keypoint_index]["name"]

        self.current_labels[name] = {
            "x": float(x),
            "y": float(y),
            "visible": 1,
            "state": STATE_VISIBLE,
        }

        self.save_status.setText(f"Placed {name}")
        self.after_annotation()

    def mark_current_not_visible(self) -> None:
        if not self.keypoints:
            return

        name = self.keypoints[self.current_keypoint_index]["name"]

        self.current_labels[name] = {
            "x": None,
            "y": None,
            "visible": 0,
            "state": STATE_NOT_VISIBLE,
        }

        self.save_status.setText(f"{name}: not visible")
        self.after_annotation()

    def clear_current_keypoint(self) -> None:
        if not self.keypoints:
            return

        name = self.keypoints[self.current_keypoint_index]["name"]
        self.current_labels[name] = empty_label()
        self._completion_announced = False

        self.update_keypoint_panel()
        self.update_overlays()

    def after_annotation(self) -> None:
        """
        Continue automatically according to the selected labeling mode.
        A keypoint is complete when it has coordinates OR is marked not visible.
        """
        self.store_current_frame()
        self.update_keypoint_panel()
        self.update_overlays()

        if self.dataset_complete():
            self.save_all_labels()
            self.announce_completion()
            return

        if self.labeling_mode == "keypoint":
            self.advance_keypoint_mode()
        else:
            self.advance_frame_mode()

    def advance_frame_mode(self) -> None:
        """Finish all keypoints on this frame, then move to the next unfinished frame."""
        next_keypoint = self.find_unset_keypoint_on_current_frame(
            start_after=self.current_keypoint_index
        )

        if next_keypoint is not None:
            self.current_keypoint_index = next_keypoint
            self.update_keypoint_panel()
            self.update_overlays()
            return

        # Current frame is complete. Persist before moving.
        self.save_all_labels()

        next_frame = self.find_next_incomplete_frame(
            start_after=self.current_frame_index
        )

        if next_frame is None:
            if self.dataset_complete():
                self.announce_completion()
            return

        self.load_frame(next_frame)
        self.current_keypoint_index = self._first_unfinished_keypoint()
        self.update_keypoint_panel()
        self.update_overlays()

    def advance_keypoint_mode(self) -> None:
        """
        Keep the same keypoint while traversing frames. Once this keypoint is
        complete on every frame, return to the first unfinished frame of the
        next keypoint.
        """
        active_keypoint = self.current_keypoint_index

        next_frame = self.find_next_frame_for_keypoint(
            active_keypoint,
            start_after=self.current_frame_index,
        )

        if next_frame is not None:
            self.save_all_labels()
            self.load_frame(next_frame)
            self.current_keypoint_index = active_keypoint
            self.update_keypoint_panel()
            self.update_overlays()
            return

        # This keypoint is complete across all frames. Find the next keypoint
        # that still has at least one unfinished frame.
        next_keypoint = self.find_next_incomplete_keypoint(
            start_after=active_keypoint
        )

        if next_keypoint is None:
            self.save_all_labels()
            self.announce_completion()
            return

        first_frame = self.find_first_frame_for_keypoint(next_keypoint)

        if first_frame is None:
            return

        self.save_all_labels()
        self.load_frame(first_frame)
        self.current_keypoint_index = next_keypoint
        self.update_keypoint_panel()
        self.update_overlays()

    def find_unset_keypoint_on_current_frame(
        self,
        *,
        start_after: int,
    ) -> int | None:
        if not self.keypoints:
            return None

        for offset in range(1, len(self.keypoints) + 1):
            index = (start_after + offset) % len(self.keypoints)
            name = self.keypoints[index]["name"]
            label = self.current_labels.get(name, empty_label())

            if label.get("state", STATE_UNSET) == STATE_UNSET:
                return index

        return None

    def find_next_incomplete_frame(
        self,
        *,
        start_after: int,
    ) -> int | None:
        if not self.frame_paths:
            return None

        self.store_current_frame()

        for offset in range(1, len(self.frame_paths) + 1):
            index = (start_after + offset) % len(self.frame_paths)
            path = self.frame_paths[index]
            labels = self._complete_frame_labels(
                self.labels_by_path.get(
                    self._normalize_path(path)
                )
            )

            if any(
                labels.get(kp["name"], empty_label()).get(
                    "state",
                    STATE_UNSET,
                ) == STATE_UNSET
                for kp in self.keypoints
            ):
                return index

        return None

    def find_next_frame_for_keypoint(
        self,
        keypoint_index: int,
        *,
        start_after: int,
    ) -> int | None:
        if not self.frame_paths or not self.keypoints:
            return None

        self.store_current_frame()
        name = self.keypoints[keypoint_index]["name"]

        for offset in range(1, len(self.frame_paths) + 1):
            index = (start_after + offset) % len(self.frame_paths)
            path = self.frame_paths[index]
            labels = self._complete_frame_labels(
                self.labels_by_path.get(
                    self._normalize_path(path)
                )
            )

            if labels.get(name, empty_label()).get(
                "state",
                STATE_UNSET,
            ) == STATE_UNSET:
                return index

        return None

    def find_next_incomplete_keypoint(
        self,
        *,
        start_after: int,
    ) -> int | None:
        if not self.keypoints:
            return None

        for offset in range(1, len(self.keypoints) + 1):
            index = (start_after + offset) % len(self.keypoints)

            if self.find_first_frame_for_keypoint(index) is not None:
                return index

        return None

    def find_first_frame_for_keypoint(
        self,
        keypoint_index: int,
    ) -> int | None:
        if not self.keypoints:
            return None

        self.store_current_frame()
        name = self.keypoints[keypoint_index]["name"]

        for index, path in enumerate(self.frame_paths):
            labels = self._complete_frame_labels(
                self.labels_by_path.get(
                    self._normalize_path(path)
                )
            )

            if labels.get(name, empty_label()).get(
                "state",
                STATE_UNSET,
            ) == STATE_UNSET:
                return index

        return None

    def dataset_complete(self) -> bool:
        if not self.frame_paths or not self.keypoints:
            return False

        self.store_current_frame()

        for path in self.frame_paths:
            labels = self._complete_frame_labels(
                self.labels_by_path.get(
                    self._normalize_path(path)
                )
            )

            for kp in self.keypoints:
                if labels.get(
                    kp["name"],
                    empty_label(),
                ).get("state", STATE_UNSET) == STATE_UNSET:
                    return False

        return True

    def announce_completion(self) -> None:
        if self._completion_announced:
            return

        self._completion_announced = True

        total_annotations = (
            len(self.frame_paths) * len(self.keypoints)
        )

        information(
            self,
            "Labeling complete 🔥",
            "Poah, was ein Macher. Du bist endlich durch! 🔥",
        )

    def advance_keypoint(self) -> None:
        if not self.keypoints:
            return

        for offset in range(1, len(self.keypoints) + 1):
            index = (
                self.current_keypoint_index + offset
            ) % len(self.keypoints)

            name = self.keypoints[index]["name"]
            label = self.current_labels.get(name)

            if label is None or label.get("state", STATE_UNSET) == STATE_UNSET:
                self.current_keypoint_index = index
                return

    def _first_unfinished_keypoint(self) -> int:
        for index, kp in enumerate(self.keypoints):
            label = self.current_labels.get(
                kp["name"],
                empty_label(),
            )
            if label["state"] == STATE_UNSET:
                return index
        return 0

    def update_keypoint_panel(self) -> None:
        if not self.keypoints:
            self.selected_label.setText("No keypoints configured")
            self.state_summary.setText("")
            return

        selected = self.keypoints[self.current_keypoint_index]
        self.selected_label.setText(
            f'Active: {selected["name"]}'
        )
        self.active_keypoint_name.setText(selected["name"])

        shortcut = selected.get("shortcut", "") or "—"
        group_name = selected.get("group", "") or "Ungrouped"
        self.active_group_label.setText(
            f'{group_name}   ·   shortcut {shortcut}'
        )

        selected_color = selected.get("color", "#D18B47")
        self.active_card.setStyleSheet(
            f"QFrame#ActiveKeypointCard {{ "
            f"background: #14161A; "
            f"border: 1px solid #3A4049; "
            f"border-left: 6px solid {selected_color}; "
            f"border-radius: 10px; }}"
        )

        visible = 0
        not_visible = 0
        unset = 0

        for index, kp in enumerate(self.keypoints):
            label = self.current_labels.get(
                kp["name"],
                empty_label(),
            )

            if label["state"] == STATE_VISIBLE:
                symbol = "✓"
                visible += 1
            elif label["state"] == STATE_NOT_VISIBLE:
                symbol = "Ø"
                not_visible += 1
            else:
                symbol = "·"
                unset += 1

            shortcut = kp["shortcut"] or "—"
            button = next(
                (
                    btn for btn in self.keypoint_buttons
                    if getattr(btn, "_keypoint_index", None) == index
                ),
                None,
            )

            if button is None:
                continue

            button.setText(
                f'{shortcut}   {kp["name"]}   {symbol}'
            )
            button.setChecked(index == self.current_keypoint_index)

            # Keep the selected/keypoint color visible without filling the whole row.
            button.setStyleSheet(
                f'QPushButton#KeypointButton {{ border-left: 5px solid {kp["color"]}; }}'
            )

        self.state_summary.setText(
            f"Visible: {visible}   ·   Not visible: {not_visible}\n"
            f"Unfinished: {unset}"
        )

    def update_overlays(self) -> None:
        overlays = []

        for index, kp in enumerate(self.keypoints):
            label = self.current_labels.get(kp["name"])

            if (
                label
                and label["state"] == STATE_VISIBLE
                and label["x"] is not None
                and label["y"] is not None
            ):
                overlays.append(
                    {
                        "x": label["x"],
                        "y": label["y"],
                        "color": kp["color"],
                        "name": kp["name"],
                        "selected": index == self.current_keypoint_index,
                    }
                )

        self.viewer.set_overlays(overlays)

    # ------------------------------------------------------------------
    # Navigation / saving
    # ------------------------------------------------------------------

    def current_frame_complete(self) -> bool:
        if not self.keypoints:
            return False

        labels = self._complete_frame_labels(self.current_labels)

        return all(
            labels[kp["name"]].get("state", STATE_UNSET)
            != STATE_UNSET
            for kp in self.keypoints
        )

    def next_frame(self) -> None:
        """
        Manual navigation is always free.

        Moving forward from the last frame wraps around to the first frame.
        Incomplete annotations are only validated when the user tries to
        continue to Review.
        """
        if not self.frame_paths:
            return

        self.save_all_labels()

        next_index = (
            self.current_frame_index + 1
        ) % len(self.frame_paths)

        self.load_frame(next_index)

    def previous_frame(self) -> None:
        """
        Manual navigation is always free.

        Moving backward from the first frame wraps around to the last frame.
        """
        if not self.frame_paths:
            return

        self.save_all_labels()

        previous_index = (
            self.current_frame_index - 1
        ) % len(self.frame_paths)

        self.load_frame(previous_index)

    def count_completed_annotations(self) -> int:
        if not self.frame_paths or not self.keypoints:
            return 0

        self.store_current_frame()
        count = 0

        for path in self.frame_paths:
            labels = self._complete_frame_labels(
                self.labels_by_path.get(
                    self._normalize_path(path)
                )
            )

            for kp in self.keypoints:
                if labels.get(
                    kp["name"],
                    empty_label(),
                ).get("state", STATE_UNSET) != STATE_UNSET:
                    count += 1

        return count

    def count_completed_frames(self) -> int:
        completed = 0

        if self.frame_paths:
            self.store_current_frame()

        for path in self.frame_paths:
            labels = self.labels_by_path.get(
                self._normalize_path(path)
            )

            if labels is None:
                continue

            if all(
                labels.get(kp["name"], empty_label())["state"]
                != STATE_UNSET
                for kp in self.keypoints
            ):
                completed += 1

        return completed

    def save_all_labels(self) -> None:
        if self.labels_csv is None:
            warning(
                self,
                "Project information required",
                "Complete the Project step before saving labels.",
            )
            return

        if not self.keypoints:
            warning(
                self,
                "No keypoints",
                "Configure at least one keypoint before labeling.",
            )
            return

        self.store_current_frame()

        self.labels_dir.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "png_path",
            "image",
            "image_folder",
        ]

        for kp in self.keypoints:
            name = kp["name"]
            fieldnames.extend(
                [
                    f"{name}_x",
                    f"{name}_y",
                    f"{name}_visible",
                    f"{name}_state",
                ]
            )

        with self.labels_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()

            for path in self.frame_paths:
                labels = self._complete_frame_labels(
                    self.labels_by_path.get(
                        self._normalize_path(path)
                    )
                )

                row = {
                    "png_path": str(path),
                    "image": path.name,
                    "image_folder": str(path.parent),
                }

                for kp in self.keypoints:
                    name = kp["name"]
                    label = labels.get(name, empty_label())

                    if (
                        label["state"] == STATE_VISIBLE
                        and label["x"] is not None
                        and label["y"] is not None
                    ):
                        row[f"{name}_x"] = round(float(label["x"]), 3)
                        row[f"{name}_y"] = round(float(label["y"]), 3)
                        row[f"{name}_visible"] = 1
                        row[f"{name}_state"] = STATE_VISIBLE
                    else:
                        row[f"{name}_x"] = ""
                        row[f"{name}_y"] = ""
                        row[f"{name}_visible"] = 0
                        row[f"{name}_state"] = label["state"]

                writer.writerow(row)

        self.save_status.setText("Saved")
        completed_annotations = self.count_completed_annotations()
        total_annotations = len(self.frame_paths) * len(self.keypoints)

        self.progress_label.setText(
            f"{completed_annotations} / {total_annotations} annotations"
        )

    def missing_annotations(self) -> list[tuple[int, Path, str]]:
        """
        Return every unfinished Frame × Keypoint assignment.

        A finished assignment has either coordinates (visible) or an explicit
        not-visible state.
        """
        self.store_current_frame()
        missing: list[tuple[int, Path, str]] = []

        for frame_index, path in enumerate(self.frame_paths):
            labels = self._complete_frame_labels(
                self.labels_by_path.get(
                    self._normalize_path(path)
                )
            )

            for kp in self.keypoints:
                name = kp["name"]
                state = labels.get(
                    name,
                    empty_label(),
                ).get("state", STATE_UNSET)

                if state == STATE_UNSET:
                    missing.append(
                        (frame_index, path, name)
                    )

        return missing

    def format_missing_annotations(
        self,
        missing: list[tuple[int, Path, str]],
        *,
        max_lines: int = 28,
    ) -> str:
        if not missing:
            return ""

        lines = [
            "Some annotations are still missing:",
            "",
        ]

        for frame_index, path, keypoint_name in missing[:max_lines]:
            lines.append(
                f"Frame {frame_index + 1} · {path.name}\\n"
                f"    → {keypoint_name}"
            )

        remaining = len(missing) - max_lines

        if remaining > 0:
            lines.extend(
                [
                    "",
                    f"…and {remaining} more unfinished annotation(s).",
                ]
            )

        lines.extend(
            [
                "",
                "Every keypoint must either have coordinates or be marked not visible.",
            ]
        )

        return "\\n".join(lines)

    def collect_data(self) -> dict:
        """Return the current Label step state for the workflow."""
        self.store_current_frame()

        return {
            "labels_csv": str(self.labels_csv) if self.labels_csv else "",
            "frame_count": len(self.frame_paths),
            "completed_frames": self.count_completed_frames(),
            "completed_annotations": self.count_completed_annotations(),
            "total_annotations": len(self.frame_paths) * len(self.keypoints),
            "missing_annotations": len(self.missing_annotations()),
        }

    def jump_to_annotation(
        self,
        frame_index: int,
        keypoint_name: str,
    ) -> None:
        """
        Jump to a specific Frame × Keypoint assignment.
        Used by Review when the user wants to fix an issue.
        """
        if not self.frame_paths or not self.keypoints:
            return

        frame_index = max(
            0,
            min(int(frame_index), len(self.frame_paths) - 1),
        )

        keypoint_index = next(
            (
                index
                for index, kp in enumerate(self.keypoints)
                if kp["name"] == keypoint_name
            ),
            None,
        )

        self.load_frame(frame_index)

        if keypoint_index is not None:
            self.current_keypoint_index = keypoint_index
            self.update_keypoint_panel()
            self.update_overlays()

        self.save_status.setText(
            f"Review issue · Frame {frame_index + 1} · {keypoint_name}"
        )

    def continue_to_review(self) -> None:
        if not self.frame_paths:
            warning(
                self,
                "No frames",
                "There are no saved frames to label.",
            )
            return

        self.save_all_labels()

        missing = self.missing_annotations()

        if missing:
            warning(
                self,
                "Labeling incomplete",
                self.format_missing_annotations(missing),
            )
            return

        # Dataset is complete. The automatic completion message may already
        # have been shown during labeling, so do not force another one here.
        self.next_requested.emit(
            {
                "labels_csv": str(self.labels_csv) if self.labels_csv else "",
                "frame_count": len(self.frame_paths),
                "completed_frames": self.count_completed_frames(),
            }
        )

    # ------------------------------------------------------------------
    # Display
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
        for slider, label, value in [
            (self.brightness_slider, self.brightness_value, brightness),
            (self.contrast_slider, self.contrast_value, contrast),
            (self.gamma_slider, self.gamma_value, gamma),
        ]:
            slider.blockSignals(True)
            slider.setValue(round(value * 100))
            slider.blockSignals(False)
            label.setText(f"{value:.2f}")

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------

    def build_global_shortcuts(self) -> None:
        mapping = [
            ("N", self.mark_current_not_visible),
            ("Backspace", self.clear_current_keypoint),
            ("S", self.save_all_labels),
            ("Up", self.select_previous_keypoint),
            ("Down", self.select_next_keypoint),
            ("A", self.previous_frame),
            ("Left", self.previous_frame),
            ("D", self.next_frame),
            ("Right", self.next_frame),
        ]

        self.global_shortcuts = []

        for sequence, action in mapping:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(action)
            self.global_shortcuts.append(shortcut)
