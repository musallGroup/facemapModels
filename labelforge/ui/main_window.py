from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .create_base.workflow import CreateBaseModelWorkflow
from .refine_model.workflow import RefineModelWorkflow
from .training_workspace import TrainingWorkspace


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("LabelForge")
        self.resize(1280, 820)
        self.setMinimumSize(950, 650)

        self.setStyleSheet("""
            QMainWindow {
                background: #14161a;
            }

            QWidget {
                color: #eaeaea;
                font-family: "Segoe UI";
                font-size: 14px;
            }

            #TopBar {
                background: #1c1f24;
                border-bottom: 1px solid #2b2f36;
            }

            #Brand {
                font-size: 24px;
                font-weight: 700;
            }

            QPushButton#WorkspaceTab {
                background: transparent;
                border: none;
                padding: 16px 24px;
                color: #aeb4bf;
                font-weight: 600;
            }

            QPushButton#WorkspaceTab:checked {
                color: #eaeaea;
                border-bottom: 3px solid #d18b47;
            }

            QPushButton#WorkspaceTab:hover {
                color: #eaeaea;
            }

            QPushButton#HelpButton {
                background: #2a2e35;
                border: 1px solid #3b4049;
                border-radius: 9px;
                padding: 9px 15px;
                font-weight: 600;
            }

            QPushButton#HelpButton:hover {
                background: #343942;
            }

            #PageTitle {
                font-size: 30px;
                font-weight: 700;
            }

            #PageSubtitle {
                color: #aeb4bf;
                font-size: 15px;
            }

            QFrame#ActionCard {
                background: #1d2026;
                border: 1px solid #2d323a;
                border-radius: 16px;
            }

            QFrame#ActionCard:hover {
                border: 1px solid #d18b47;
                background: #22262d;
            }

            QLabel#CardTitle {
                font-size: 20px;
                font-weight: 700;
            }

            QLabel#CardText {
                color: #aeb4bf;
                font-size: 14px;
            }

            QPushButton#CardButton {
                background: #d18b47;
                color: #111111;
                border: none;
                border-radius: 9px;
                padding: 10px 16px;
                font-weight: 700;
            }

            QPushButton#CardButton:hover {
                background: #dfa15f;
            }

            QPushButton#BackButton {
                background: transparent;
                border: 1px solid #3a4049;
                border-radius: 8px;
                padding: 8px 14px;
            }

            QPushButton#BackButton:hover {
                background: #292e36;
            }

            #PlaceholderBox {
                background: #1d2026;
                border: 1px solid #2d323a;
                border-radius: 14px;
            }

            #WizardPanel {
                background: #1d2026;
                border: 1px solid #2d323a;
                border-radius: 16px;
            }

            QLabel#FieldLabel {
                color: #eaeaea;
                font-size: 14px;
                font-weight: 600;
            }

            QLabel#FieldHint {
                color: #aeb4bf;
                font-size: 12px;
            }

            QLineEdit#TextInput,
            QTextEdit#TextInput {
                background: #14161a;
                color: #eaeaea;
                border: 1px solid #3a4049;
                border-radius: 9px;
                padding: 10px 12px;
                selection-background-color: #d18b47;
                selection-color: #111111;
            }

            QLineEdit#TextInput:focus,
            QTextEdit#TextInput:focus {
                border: 1px solid #d18b47;
            }

            QPushButton#BrowseButton {
                background: #2a2e35;
                border: 1px solid #3a4049;
                border-radius: 9px;
                padding: 10px 14px;
                font-weight: 600;
            }

            QPushButton#BrowseButton:hover {
                background: #343942;
                border: 1px solid #d18b47;
            }

            QPushButton#PrimaryNextButton {
                background: #d18b47;
                color: #111111;
                border: none;
                border-radius: 9px;
                padding: 11px 20px;
                font-weight: 700;
            }

            QPushButton#PrimaryNextButton:hover {
                background: #dfa15f;
            }

            QPushButton#WizardStepTab {
                background: transparent;
                color: #737984;
                border: none;
                border-bottom: 3px solid #2b2f36;
                padding: 8px 6px 9px 6px;
                font-size: 12px;
                font-weight: 600;
            }

            QPushButton#WizardStepTab:hover {
                color: #eaeaea;
                border-bottom: 3px solid #8f6640;
            }

            QPushButton#WizardStepTab:checked {
                color: #d18b47;
                border-bottom: 3px solid #d18b47;
                font-weight: 700;
            }

            QFrame#GroupCard {
                background: #181b20;
                border: 1px solid #2d323a;
                border-radius: 12px;
            }

            QFrame#KeypointRow {
                background: #14161a;
                border: 1px solid #2b2f36;
                border-radius: 9px;
            }

            QLabel#GroupNumber {
                color: #d18b47;
                font-size: 12px;
                font-weight: 700;
            }

            QLineEdit#CompactInput,
            QComboBox#CompactCombo {
                background: #14161a;
                color: #eaeaea;
                border: 1px solid #3a4049;
                border-radius: 8px;
                padding: 7px 9px;
                min-height: 20px;
            }

            QLineEdit#CompactInput:focus,
            QComboBox#CompactCombo:focus {
                border: 1px solid #d18b47;
            }

            QComboBox {
                background: #14161a;
                color: #eaeaea;
                border: 1px solid #3a4049;
                border-radius: 8px;
                padding: 7px 34px 7px 10px;
                min-height: 24px;
                selection-background-color: #d18b47;
                selection-color: #111111;
            }

            QComboBox:hover,
            QComboBox:focus,
            QComboBox:on {
                border: 1px solid #d18b47;
                background: #181b20;
            }

            QComboBox:disabled {
                background: #1b1e23;
                color: #737984;
                border-color: #2d323a;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                background: #2a2e35;
                border: none;
                border-left: 1px solid #3a4049;
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
            }

            QComboBox::drop-down:hover {
                background: #343942;
            }

            QComboBox::down-arrow {
                width: 8px;
                height: 8px;
            }

            QComboBox QAbstractItemView {
                background: #1d2026;
                color: #eaeaea;
                border: 1px solid #3a4049;
                outline: none;
                padding: 4px;
                selection-background-color: #d18b47;
                selection-color: #111111;
            }

            QSpinBox,
            QDoubleSpinBox {
                background: #14161a;
                color: #eaeaea;
                border: 1px solid #3a4049;
                border-radius: 7px;
                padding: 5px 8px;
                min-height: 22px;
                selection-background-color: #d18b47;
                selection-color: #111111;
            }

            QSpinBox:focus,
            QDoubleSpinBox:focus {
                border: 1px solid #d18b47;
            }

            QSpinBox::up-button,
            QSpinBox::down-button,
            QDoubleSpinBox::up-button,
            QDoubleSpinBox::down-button {
                background: #2a2e35;
                border: none;
                border-left: 1px solid #3a4049;
                width: 20px;
            }

            QSpinBox::up-button,
            QDoubleSpinBox::up-button {
                border-top-right-radius: 6px;
            }

            QSpinBox::down-button,
            QDoubleSpinBox::down-button {
                border-bottom-right-radius: 6px;
            }

            QAbstractSpinBox:disabled {
                background: #1b1e23;
                color: #737984;
            }

            QScrollArea,
            QScrollArea > QWidget,
            QScrollArea > QWidget > QWidget {
                background: #14161a;
            }

            QPushButton#SecondaryActionButton {
                background: #2a2e35;
                color: #eaeaea;
                border: 1px solid #3a4049;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }

            QPushButton#SecondaryActionButton:hover {
                border: 1px solid #d18b47;
                background: #343942;
            }

            QPushButton#ModeCard {
                background: #181b20;
                color: #eaeaea;
                border: 1px solid #3a4049;
                border-radius: 12px;
                padding: 14px 18px;
                text-align: left;
                font-size: 15px;
                font-weight: 700;
            }

            QPushButton#ModeCard:hover {
                background: #22262d;
                border-color: #8f6640;
            }

            QPushButton#ModeCard:checked {
                background: #2a2119;
                color: #f4c48f;
                border: 2px solid #d18b47;
            }

            QPushButton#AdvancedToggle {
                background: transparent;
                color: #c9a77f;
                border: none;
                padding: 8px 2px;
                text-align: left;
                font-weight: 600;
            }

            QPushButton#AdvancedToggle:hover { color: #f4c48f; }

            QFrame#AdvancedPanel {
                background: #17191e;
                border: 1px solid #30353d;
                border-radius: 9px;
                padding: 8px;
            }

            QLabel#ReadinessChecklist {
                background: #17191e;
                border: 1px solid #30353d;
                border-radius: 9px;
                padding: 13px;
            }

            QPushButton#DangerGhostButton {
                background: transparent;
                color: #aeb4bf;
                border: 1px solid #3a4049;
                border-radius: 7px;
                padding: 6px 9px;
            }

            QPushButton#DangerGhostButton:hover {
                color: #eaeaea;
                border: 1px solid #d18b47;
            }

            QScrollArea#KeypointScroll {
                background: transparent;
                border: none;
            }

            QScrollArea#KeypointScroll > QWidget > QWidget {
                background: transparent;
            }

            QLabel#StepActive {
                color: #d18b47;
                font-size: 12px;
                font-weight: 700;
            }

            QLabel#StepInactive {
                color: #737984;
                font-size: 12px;
                font-weight: 600;
            }

            QFrame#StepLineActive {
                background: #d18b47;
                min-height: 3px;
                max-height: 3px;
                border: none;
            }

            QFrame#StepLineInactive {
                background: #2b2f36;
                min-height: 3px;
                max-height: 3px;
                border: none;
            }
        """)

        self.stack = QStackedWidget()

        self.main_container = QWidget()
        self.main_layout = QVBoxLayout(self.main_container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.top_bar = self.build_top_bar()
        self.main_layout.addWidget(self.top_bar)
        self.main_layout.addWidget(self.stack, 1)

        self.setCentralWidget(self.main_container)

        self.home_page = self.build_home_page()
        self.create_page = CreateBaseModelWorkflow()
        self.create_page.back_to_home_requested.connect(self.show_label_workspace)
        self.library_refine_page = RefineModelWorkflow()
        self.library_refine_page.back_to_home_requested.connect(self.show_label_workspace)
        self.library_specialize_page = self.build_placeholder_page(
            "Model Library — Specialize",
            "Hier wird später ein Basismodell ausgewählt, von dem ein spezialisierter Zweig erstellt wird."
        )
        self.training_page = TrainingWorkspace()

        for page in [
            self.home_page,
            self.create_page,
            self.library_refine_page,
            self.library_specialize_page,
            self.training_page,
        ]:
            self.stack.addWidget(page)

        self.show_label_workspace()

    def build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 20, 0)
        layout.setSpacing(4)

        brand_label = QLabel()

        brand_path = (
            Path(__file__).resolve().parent.parent
            / "assets"
            / "labelforge_extended.png"
        )

        if brand_path.exists():
            brand_pixmap = QPixmap(str(brand_path))
            if not brand_pixmap.isNull():
                brand_label.setPixmap(
                    brand_pixmap.scaled(
                        250,
                        54,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )

        brand_label.setFixedSize(270, 58)
        layout.addWidget(brand_label)

        layout.addStretch(1)

        self.label_tab = QPushButton("Label Workspace")
        self.label_tab.setObjectName("WorkspaceTab")
        self.label_tab.setCheckable(True)
        self.label_tab.clicked.connect(self.show_label_workspace)

        self.training_tab = QPushButton("Training Workspace")
        self.training_tab.setObjectName("WorkspaceTab")
        self.training_tab.setCheckable(True)
        self.training_tab.clicked.connect(self.show_training_workspace)

        self.workspace_group = QButtonGroup(bar)
        self.workspace_group.setExclusive(True)
        self.workspace_group.addButton(self.label_tab)
        self.workspace_group.addButton(self.training_tab)

        layout.addWidget(self.label_tab)
        layout.addWidget(self.training_tab)

        help_button = QPushButton("?  Help")
        help_button.setObjectName("HelpButton")
        help_button.clicked.connect(self.show_help)
        layout.addSpacing(16)
        layout.addWidget(help_button)

        return bar

    def build_home_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(54, 48, 54, 48)
        content_layout.setSpacing(12)

        title = QLabel("What would you like to do?")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "Create a new labeling project, refine an existing model line, "
            "or specialize a base model for a specific setup."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        content_layout.addWidget(title)
        content_layout.addWidget(subtitle)
        content_layout.addSpacing(26)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(22)

        cards_row.addWidget(
            self.make_action_card(
                "Create Base Model",
                "Start from scratch and define a new keypoint schema, groups and color palettes.",
                "Create",
                self.show_create_base_model,
            )
        )
        cards_row.addWidget(
            self.make_action_card(
                "Refine Existing Model",
                "Continue an existing base-model line and prepare the next version.",
                "Open Library",
                lambda: self.stack.setCurrentWidget(self.library_refine_page),
            )
        )
        cards_row.addWidget(
            self.make_action_card(
                "Specialize Base Model",
                "Branch from a base model and create a setup-specific specialization.",
                "Open Library",
                lambda: self.stack.setCurrentWidget(self.library_specialize_page),
            )
        )

        content_layout.addLayout(cards_row)
        content_layout.addStretch(1)

        outer.addWidget(content, 1)
        return page

    def make_action_card(
        self,
        title: str,
        text: str,
        button_text: str,
        action,
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("ActionCard")
        card.setMinimumHeight(285)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(26, 28, 26, 26)
        layout.setSpacing(14)

        card_title = QLabel(title)
        card_title.setObjectName("CardTitle")

        card_text = QLabel(text)
        card_text.setObjectName("CardText")
        card_text.setWordWrap(True)
        card_text.setAlignment(Qt.AlignTop)

        button = QPushButton(button_text)
        button.setObjectName("CardButton")
        button.clicked.connect(action)

        layout.addWidget(card_title)
        layout.addWidget(card_text)
        layout.addStretch(1)
        layout.addWidget(button)

        return card

    def show_create_base_model(self) -> None:
        self.label_tab.setChecked(True)
        self.stack.setCurrentWidget(self.create_page)

    def build_placeholder_page(self, title_text: str, body_text: str) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(54, 42, 54, 42)
        layout.setSpacing(16)

        back = QPushButton("← Back")
        back.setObjectName("BackButton")
        back.clicked.connect(self.show_label_workspace)
        back.setFixedWidth(110)

        title = QLabel(title_text)
        title.setObjectName("PageTitle")

        box = QFrame()
        box.setObjectName("PlaceholderBox")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(30, 30, 30, 30)

        text = QLabel(body_text)
        text.setObjectName("PageSubtitle")
        text.setWordWrap(True)

        box_layout.addWidget(text)
        box_layout.addStretch(1)

        layout.addWidget(back)
        layout.addSpacing(8)
        layout.addWidget(title)
        layout.addWidget(box, 1)

        outer.addWidget(content, 1)
        return page

    def build_training_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(54, 48, 54, 48)

        title = QLabel("Training Workspace")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "We have deliberately left this area open for now. "
            "Next we will decide together which training-related tasks actually belong here."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)

        box = QFrame()
        box.setObjectName("PlaceholderBox")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(30, 30, 30, 30)

        message = QLabel(
            "Training itself will stay in Facemap or DeepLabCut.\n\n"
            "Possible future functions:\n"
            "• environment setup\n"
            "• training-package preparation\n"
            "• step-by-step backend guide\n"
            "• model import\n"
            "• QC handoff"
        )
        message.setObjectName("PageSubtitle")
        message.setWordWrap(True)

        box_layout.addWidget(message)
        box_layout.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(24)
        layout.addWidget(box, 1)

        outer.addWidget(content, 1)
        return page

    def show_label_workspace(self) -> None:
        self.label_tab.setChecked(True)
        self.stack.setCurrentWidget(self.home_page)

    def show_training_workspace(self) -> None:
        self.training_tab.setChecked(True)
        self.stack.setCurrentWidget(self.training_page)

    def show_help(self) -> None:
        QMessageBox.information(
            self,
            "LabelForge Help",
            "This is the first LabelForge prototype.\n\n"
            "For now, use it to test the basic navigation and overall feel.\n\n"
            "Next we will build the real project-creation flow and model library."
        )
