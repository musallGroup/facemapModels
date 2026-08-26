from __future__ import annotations

from pathlib import Path
import json

from PySide6.QtCore import Qt, Signal
from .keypoint_setup import KeypointSetupPage
from .frame_picker import FramePickerPage
from .labeling import LabelingPage
from .review import ReviewPage
from .export import ExportPage
from ..common.dialogs import warning

from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class StepNavigation(QWidget):
    step_requested = Signal(int)

    def __init__(self, steps: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.steps = steps
        self.buttons: list[QPushButton] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        for index, name in enumerate(steps):
            button = QPushButton(name)
            button.setObjectName("WizardStepTab")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(
                lambda checked=False, idx=index: self.step_requested.emit(idx)
            )

            layout.addWidget(button, 1)
            self.buttons.append(button)

        self.set_active_step(0)

    def set_active_step(self, index: int) -> None:
        for i, button in enumerate(self.buttons):
            button.setChecked(i == index)


class CreateBaseModelWorkflow(QWidget):
    back_to_home_requested = Signal()

    STEPS = [
        "Project",
        "Keypoints",
        "Frames",
        "Label",
        "Review",
        "Export",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.project_draft = {
            "name": "",
            "description": "",
            "location": "",
        }
        self.keypoint_draft = {"groups": []}
        self.frame_draft = {}
        self.label_draft = {}

        self.current_step = 0

        self.outer = QVBoxLayout(self)
        self.outer.setContentsMargins(34, 16, 34, 20)
        self.outer.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setSpacing(14)

        self.back_home_button = QPushButton("← Back")
        self.back_home_button.setObjectName("BackButton")
        self.back_home_button.setFixedWidth(96)
        self.back_home_button.clicked.connect(self.back_to_home_requested.emit)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)

        self.title = QLabel("Create Base Model")
        self.title.setObjectName("PageTitle")

        self.subtitle = QLabel()
        self.subtitle.setObjectName("PageSubtitle")
        self.subtitle.setWordWrap(True)

        title_block.addWidget(self.title)
        title_block.addWidget(self.subtitle)

        self.step_counter = QLabel()
        self.step_counter.setObjectName("PageSubtitle")
        self.step_counter.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.step_counter.setFixedWidth(90)

        header_row.addWidget(self.back_home_button, 0, Qt.AlignTop)
        header_row.addLayout(title_block, 1)
        header_row.addWidget(self.step_counter, 0, Qt.AlignTop)

        self.step_navigation = StepNavigation(self.STEPS)
        self.step_navigation.step_requested.connect(self.go_to_step)

        self.stack = QStackedWidget()

        self.project_page = self.build_project_page()
        self.keypoints_page = KeypointSetupPage()
        self.keypoints_page.previous_requested.connect(lambda: self.go_to_step(0))
        self.keypoints_page.next_requested.connect(self.complete_keypoint_step)
        self.frames_page = FramePickerPage()
        self.frames_page.previous_requested.connect(lambda: self.go_to_step(1))
        self.frames_page.next_requested.connect(self.complete_frame_step)
        self.label_page = LabelingPage()
        self.label_page.previous_requested.connect(lambda: self.go_to_step(2))
        self.label_page.next_requested.connect(self.complete_label_step)
        self.review_page = ReviewPage()
        self.review_page.previous_requested.connect(lambda: self.go_to_step(3))
        self.review_page.export_requested.connect(lambda: self.go_to_step(5))
        self.review_page.fix_requested.connect(self.open_review_issue)
        self.export_page = ExportPage()
        self.export_page.previous_requested.connect(lambda: self.go_to_step(4))

        for page in [
            self.project_page,
            self.keypoints_page,
            self.frames_page,
            self.label_page,
            self.review_page,
            self.export_page,
        ]:
            self.stack.addWidget(page)

        self.outer.addLayout(header_row)
        self.outer.addSpacing(4)
        self.outer.addWidget(self.step_navigation)
        self.outer.addSpacing(4)
        self.outer.addWidget(self.stack, 1)

        self.go_to_step(0)

    def build_project_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        panel = QFrame()
        panel.setObjectName("WizardPanel")

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(30, 28, 30, 28)
        panel_layout.setSpacing(10)

        name_label = QLabel("Base model name")
        name_label.setObjectName("FieldLabel")

        name_hint = QLabel(
            "A short, readable name for the model family, e.g. “Pupil Base” or “Frontview Face”."
        )
        name_hint.setObjectName("FieldHint")
        name_hint.setWordWrap(True)

        self.name_input = QLineEdit()
        self.name_input.setObjectName("TextInput")
        self.name_input.setPlaceholderText("e.g. Pupil Base")

        description_label = QLabel("Description")
        description_label.setObjectName("FieldLabel")

        description_hint = QLabel(
            "Optional. Briefly describe what this base model is intended to track."
        )
        description_hint.setObjectName("FieldHint")
        description_hint.setWordWrap(True)

        self.description_input = QTextEdit()
        self.description_input.setObjectName("TextInput")
        self.description_input.setPlaceholderText(
            "e.g. General sideview pupil and facial landmarks for mouse behavior videos."
        )
        self.description_input.setFixedHeight(105)

        location_label = QLabel("Project location")
        location_label.setObjectName("FieldLabel")

        location_hint = QLabel(
            "Choose where this LabelForge project should be stored."
        )
        location_hint.setObjectName("FieldHint")
        location_hint.setWordWrap(True)

        location_row = QHBoxLayout()

        self.location_input = QLineEdit()
        self.location_input.setObjectName("TextInput")
        self.location_input.setPlaceholderText("Choose a folder...")

        browse = QPushButton("Browse…")
        browse.setObjectName("BrowseButton")
        browse.clicked.connect(self.choose_location)

        location_row.addWidget(self.location_input, 1)
        location_row.addWidget(browse)

        panel_layout.addWidget(name_label)
        panel_layout.addWidget(name_hint)
        panel_layout.addWidget(self.name_input)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(description_label)
        panel_layout.addWidget(description_hint)
        panel_layout.addWidget(self.description_input)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(location_label)
        panel_layout.addWidget(location_hint)
        panel_layout.addLayout(location_row)

        nav_row = QHBoxLayout()
        nav_row.addStretch(1)

        next_button = QPushButton("Continue →")
        next_button.setObjectName("PrimaryNextButton")
        next_button.clicked.connect(self.continue_from_project)
        nav_row.addWidget(next_button)

        layout.addWidget(panel)
        layout.addLayout(nav_row)
        layout.addStretch(1)

        return page

    def build_placeholder_page(self, title_text: str, body_text: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        panel = QFrame()
        panel.setObjectName("WizardPanel")

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(30, 28, 30, 28)
        panel_layout.setSpacing(12)

        title = QLabel(title_text)
        title.setObjectName("CardTitle")

        body = QLabel(body_text)
        body.setObjectName("PageSubtitle")
        body.setWordWrap(True)

        panel_layout.addWidget(title)
        panel_layout.addWidget(body)
        panel_layout.addStretch(1)

        nav_row = QHBoxLayout()

        previous_button = QPushButton("← Previous")
        previous_button.setObjectName("BackButton")
        previous_button.clicked.connect(self.go_to_previous_step)

        next_button = QPushButton("Next →")
        next_button.setObjectName("PrimaryNextButton")
        next_button.clicked.connect(self.go_to_next_step)

        nav_row.addWidget(previous_button)
        nav_row.addStretch(1)
        nav_row.addWidget(next_button)

        layout.addWidget(panel, 1)
        layout.addLayout(nav_row)

        return page

    def choose_location(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose LabelForge project location",
            self.location_input.text().strip(),
        )
        if selected:
            self.location_input.setText(selected)

    def save_project_draft(self) -> None:
        self.project_draft = {
            "name": self.name_input.text().strip(),
            "description": self.description_input.toPlainText().strip(),
            "location": self.location_input.text().strip(),
        }

    def validate_project_step(self) -> bool:
        self.save_project_draft()

        if not self.project_draft["name"]:
            warning(
                self,
                "Model name required",
                "Please enter a name for the new base model.",
            )
            self.name_input.setFocus()
            return False

        if not self.project_draft["location"]:
            warning(
                self,
                "Project location required",
                "Please choose where the LabelForge project should be stored.",
            )
            return False

        return True

    def continue_from_project(self) -> None:
        if self.validate_project_step():
            self.go_to_step(1)

    def complete_keypoint_step(self, data: dict) -> None:
        self.keypoint_draft = data
        self.go_to_step(2)

    def complete_frame_step(self, data: dict) -> None:
        self.frame_draft = data
        self.go_to_step(3)

    def complete_label_step(self, data: dict) -> None:
        self.label_draft = data
        self.go_to_step(4)

    def save_frame_draft(self) -> None:
        self.frame_draft = self.frames_page.collect_data()

    def save_keypoint_draft(self) -> None:
        """Save the current Keypoint editor state without validating it."""
        self.keypoint_draft = self.keypoints_page.collect_data()

    def save_label_draft(self) -> None:
        """
        Persist labels before leaving the Label step.

        This makes the round-trip safe:
        Label → Frames → add more frames → Label
        keeps all existing annotations and only introduces new unset frames.
        """
        self.label_page.save_all_labels()
        self.label_draft = self.label_page.collect_data()

    def open_review_issue(
        self,
        frame_index: int,
        keypoint_name: str,
    ) -> None:
        self.go_to_step(3)
        self.label_page.jump_to_annotation(
            frame_index,
            keypoint_name,
        )

    def save_label_dev_context(self) -> None:
        """
        Development helper: remember the most recent Label context so the
        standalone labeler can be opened without repeating Project/Keypoints/Frames.
        """
        app_root = Path(__file__).resolve().parents[3]
        dev_dir = app_root / ".dev"
        dev_dir.mkdir(parents=True, exist_ok=True)

        context_path = dev_dir / "label_context.json"

        payload = {
            "project_draft": self.project_draft,
            "keypoint_draft": self.keypoint_draft,
            "frame_draft": self.frame_draft,
        }

        context_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def go_to_step(self, index: int) -> None:
        index = max(0, min(index, len(self.STEPS) - 1))

        # Save whatever is currently being edited before changing tabs.
        if self.current_step == 0:
            self.save_project_draft()
        elif self.current_step == 1:
            self.save_keypoint_draft()
        elif self.current_step == 2:
            self.save_frame_draft()
        elif self.current_step == 3:
            self.save_label_draft()

        self.current_step = index
        self.stack.setCurrentIndex(index)
        self.step_navigation.set_active_step(index)
        self.step_counter.setText(f"Step {index + 1} of {len(self.STEPS)}")

        # Repopulate the Keypoints editor from its workflow draft when returning.
        if index == 1 and self.keypoint_draft.get("groups"):
            self.keypoints_page.load_data(self.keypoint_draft)

        if index == 2:
            self.frames_page.set_project_context(self.project_draft)

        if index == 3:
            self.save_label_dev_context()
            self.label_page.set_context(
                self.project_draft,
                self.keypoint_draft,
                self.frame_draft,
            )

        if index == 4:
            self.review_page.set_context(
                self.project_draft,
                self.keypoint_draft,
                self.frame_draft,
                self.label_draft,
            )

        if index == 5:
            self.export_page.set_context(
                self.project_draft,
                self.keypoint_draft,
                self.frame_draft,
                self.label_draft,
            )

        subtitles = [
            "Start with the basic project information. Keypoints, groups and colors are configured in the next step.",
            "Define which landmarks LabelForge should label and how they are organized.",
            "Select representative frames from your source videos.",
            "Annotate the selected frames.",
            "Check the dataset before export.",
            "Create training-ready files for the backend you want to use.",
        ]
        self.subtitle.setText(subtitles[index])

    def go_to_previous_step(self) -> None:
        self.go_to_step(self.current_step - 1)

    def go_to_next_step(self) -> None:
        if self.current_step < len(self.STEPS) - 1:
            self.go_to_step(self.current_step + 1)
