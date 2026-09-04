from __future__ import annotations

import json
import re
import shutil
import subprocess
import base64
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QProcess, QProcessEnvironment, QRect, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QInputDialog, QLabel, QLayout, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QProgressDialog, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from .bundle import (
    TrainingBundleConfig, create_bundle, discover_backend_environments,
    find_conda, safe_name, validate_config,
)
from .remote import (
    RemoteProfile, fetch_commands, preflight_command, start_command, status_command, sync_commands,
)
from .naming import ensure_v1, next_refinement_name
from ..common.dialogs import confirm
from ...model_metadata import read_facemap_labels_csv


class ClearComboBox(QComboBox):
    """Theme-independent combo box with an unmistakable dropdown marker."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent); self.setMinimumHeight(44)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QColor("#eaeaea") if self.isEnabled() else QColor("#737984"))
        painter.drawText(self.width() - 29, 0, 28, self.height(), Qt.AlignCenter, "▼")

    def wheelEvent(self, event) -> None:
        """Page scrolling must never change a closed dropdown by accident."""
        event.ignore()


class BundleWorker(QObject):
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, config: TrainingBundleConfig) -> None:
        super().__init__(); self.config = config

    def run(self) -> None:
        try: self.finished.emit(str(create_bundle(self.config)))
        except Exception as exc: self.failed.emit(str(exc))


class TrainingWorkspace(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.conda_path = find_conda()
        self.backend_environments: dict[str, list[str]] = {"facemap": [], "deeplabcut": []}
        self._process: QProcess | None = None
        self._command_queue: list[tuple[list[str], str | None]] = []
        self._command_output = ""
        self._operation = ""
        self._last_bundle: Path | None = None
        self._last_job_id = ""
        self._job_submit_time: float = 0.0
        self._elapsed_timer: QTimer | None = None
        self._validated = False
        self._remote_test_passed = False
        self._pending_totp = ""
        self._bundle_thread: QThread | None = None
        self._bundle_worker: BundleWorker | None = None
        self._help_topics: dict[QObject, tuple[str, str]] = {}
        self._build_ui()
        self._refresh_software_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        self._workspace_scroll = scroll
        body = QWidget(); body_layout = QHBoxLayout(body); body_layout.setContentsMargins(28, 30, 28, 42)
        self.left_rail = self._help_rail(); self.right_rail = self._route_rail()
        content = QWidget(); content.setMaximumWidth(1260); self._training_content = content
        layout = QVBoxLayout(content); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(16)
        title = QLabel("Training Workspace"); title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Create one reproducible training package, then run it locally, on a remote GPU workstation, "
            "or through Slurm on an HPC system."
        )
        subtitle.setObjectName("PageSubtitle"); subtitle.setWordWrap(True)
        layout.addWidget(title); layout.addWidget(subtitle)
        self.mode_card = self._mode_card(); self.software_card = self._software_card()
        self.run_card = self._run_card(); self.execution_card = self._execution_card(); self.actions_card = self._actions_card()
        layout.addWidget(self.mode_card); layout.addWidget(self.software_card); layout.addWidget(self.run_card)
        layout.addWidget(self.execution_card); layout.addWidget(self.actions_card)
        layout.addStretch(1)
        body_layout.addWidget(self.left_rail, 0, Qt.AlignTop)
        body_layout.addWidget(content, 1, Qt.AlignHCenter)
        body_layout.addWidget(self.right_rail, 0, Qt.AlignTop)
        self._workspace_body = body
        scroll.setWidget(body); outer.addWidget(scroll)
        self._setup_context_help()
        scroll.verticalScrollBar().valueChanged.connect(self._position_side_rails)
        self.run_card.setVisible(False); self.execution_card.setVisible(False); self.actions_card.setVisible(False)
        QTimer.singleShot(0, self._update_side_rails)

    def _help_rail(self) -> QFrame:
        rail = QFrame(); rail.setObjectName("SideRail"); rail.setFixedWidth(260)
        layout = QVBoxLayout(rail); layout.setContentsMargins(20, 24, 20, 24); layout.setSpacing(14)
        brand = QLabel('<span style="color:#f1f2f4">Label</span><span style="color:#d9944d">Forge</span>')
        brand.setObjectName("ContextBrand"); brand.setTextFormat(Qt.RichText)
        eyebrow = QLabel("CONTEXT GUIDE"); eyebrow.setObjectName("RailEyebrow")
        self.help_title = QLabel("Hover over anything"); self.help_title.setObjectName("HelpBubbleTitle"); self.help_title.setWordWrap(True); self.help_title.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.help_body = QLabel(
            "Move the pointer over a field or button. This panel explains what it expects, why it matters, and what happens next."
        )
        self.help_body.setObjectName("HelpBubbleBody"); self.help_body.setWordWrap(True); self.help_body.setAlignment(Qt.AlignTop)
        bubble = QFrame(); bubble.setObjectName("HelpBubble"); self.help_bubble = bubble
        bubble_layout = QVBoxLayout(bubble); bubble_layout.setContentsMargins(15, 15, 15, 17); bubble_layout.setSpacing(8)
        bubble_layout.addWidget(self.help_title); bubble_layout.addWidget(self.help_body)
        layout.addWidget(brand); layout.addWidget(eyebrow); layout.addWidget(bubble)
        return rail

    def _route_rail(self) -> QFrame:
        rail = QFrame(); rail.setObjectName("RouteRail"); rail.setFixedWidth(280)
        layout = QVBoxLayout(rail); layout.setContentsMargins(22, 26, 22, 26); layout.setSpacing(13)
        title = QLabel("YOUR TRAINING ROUTE"); title.setObjectName("RailEyebrow"); layout.addWidget(title)
        self.route_labels = []
        for text in ["01   CHOOSE GOAL", "02   CHECK TOOLS", "03   ADD MATERIAL", "04   CHOOSE COMPUTER", "05   LAUNCH RUN"]:
            label = QLabel(text); label.setObjectName("RouteStep"); label.setProperty("state", "locked")
            layout.addWidget(label); self.route_labels.append(label)
        note = QLabel("Follow the route from top to bottom. Each completed checkpoint unlocks the next one — your source files stay untouched.")
        note.setObjectName("RailCaption"); note.setWordWrap(True); layout.addSpacing(8); layout.addWidget(note); layout.addStretch(1)
        return rail

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event); self._update_side_rails()

    def _update_side_rails(self) -> None:
        if not hasattr(self, "left_rail"): return
        wide = self.width() >= 1500
        self.left_rail.setVisible(wide); self.right_rail.setVisible(wide)
        self._training_content.setMaximumWidth(1260 if wide else 1650)
        QTimer.singleShot(0, self._position_side_rails)

    def _position_side_rails(self, _value: int = 0) -> None:
        """Keep both guides visible while the central wizard scrolls."""
        if not hasattr(self, "_workspace_scroll") or not self.left_rail.isVisible(): return
        scroll_y = self._workspace_scroll.verticalScrollBar().value()
        maximum_y = max(30, self._workspace_body.height() - max(self.left_rail.height(), self.right_rail.height()) - 42)
        pinned_y = min(scroll_y + 30, maximum_y)
        self.left_rail.move(self.left_rail.x(), pinned_y)
        self.right_rail.move(self.right_rail.x(), pinned_y)
        self.left_rail.raise_(); self.right_rail.raise_()

    def _resize_help_bubble(self) -> None:
        """Fit the contextual card to the current explanation without clipping."""
        if not hasattr(self, "help_bubble"): return
        text_width = self.left_rail.width() - 70
        self.help_title.setFixedWidth(text_width); self.help_body.setFixedWidth(text_width)
        flags = int(Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop)
        title_height = self.help_title.fontMetrics().boundingRect(QRect(0, 0, text_width, 10_000), flags, self.help_title.text()).height()
        body_height = self.help_body.fontMetrics().boundingRect(QRect(0, 0, text_width, 10_000), flags, self.help_body.text()).height()
        self.help_title.setFixedHeight(title_height + 2)
        self.help_body.setFixedHeight(body_height + 4)
        bubble_height = 15 + title_height + 2 + 8 + body_height + 4 + 17
        self.help_bubble.setFixedHeight(bubble_height)
        self.help_bubble.layout().activate(); self.left_rail.layout().activate()
        rail_layout = self.left_rail.layout()
        rail_height = (
            rail_layout.contentsMargins().top() + rail_layout.contentsMargins().bottom()
            + rail_layout.spacing() * 2
            + rail_layout.itemAt(0).widget().sizeHint().height()
            + rail_layout.itemAt(1).widget().sizeHint().height()
            + bubble_height
        )
        self.left_rail.setFixedSize(260, rail_height)
        self._position_side_rails()

    def _register_help(self, widget: QWidget, title: str, body: str) -> None:
        topic = (title, body); self._help_topics[widget] = topic
        widget.setProperty("contextHelpTitle", title); widget.setProperty("contextHelpBody", body)
        widget.setMouseTracking(True); widget.installEventFilter(self)
        for child in widget.findChildren(QWidget):
            self._help_topics[child] = topic
            child.setProperty("contextHelpTitle", title); child.setProperty("contextHelpBody", body)
            child.setMouseTracking(True); child.installEventFilter(self)

    def _setup_context_help(self) -> None:
        topics = [
            (self.mode_buttons["Create"], "Create a new model", "Start a new model family without a parent model. LabelForge adds _v1 to the name and keeps the result separate from existing models."),
            (self.mode_buttons["Refine"], "Refine an existing model", "Continue the same model family. Select the parent .pt model and LabelForge proposes the next version, for example v3 → v4."),
            (self.mode_buttons["Specialize"], "Specialize a model", "Adapt a proven parent model to a cohort, setup or task. The new model keeps a link to its source but receives its own family name."),
            (self.backend, "Training backend", "Choose the software that understands the model. Facemap preserves genuine Facemap .pt compatibility; DeepLabCut expects a DLC project and config.yaml."),
            (self.local_environment, "Local Python environment", "Only relevant for training on this computer. It must contain the selected backend. It disappears automatically for remote or HPC runs."),
            (self.fm_install_button, "Install or repair Facemap", "Creates or repairs a local Conda environment containing Facemap. This is only needed when training on this computer."),
            (self.dlc_install_button, "Install or repair DeepLabCut", "Creates or repairs a local Conda environment containing DeepLabCut. Remote and HPC installations are handled on the target computer."),
            (self.parent_model, "Parent model", "The existing .pt model used as the starting point for Refine or Specialize. It is read-only and will never be overwritten."),
            (self.training_data, "Training images or project", "Facemap expects the folder containing labeled training images. DeepLabCut expects the project folder containing config.yaml."),
            (self.labels_config, "Labels and keypoints", "Facemap uses the LabelForge labels.csv; DLC uses config.yaml. Keypoint order and individually missing keypoints must remain intact."),
            (self.model_name, "New model name", "This is the identity of the output model. Refine proposes the next version automatically; Create and Specialize add _v1 when needed."),
            (self.output_dir, "Local package location", "LabelForge creates a reproducible staging folder here. For remote runs this package is later transferred to the training computer."),
            (self.init_video, "Facemap initialization video", "Facemap uses this video to initialize the model. It is required for Facemap whether training runs locally or on JUSUF."),
            (self.qc_enabled, "Automatic visual QC", "Creates a short result video after training: the full frame plus a focused zoom panel with the predicted keypoints. It is fetched together with the model and logs."),
            (self.qc_video, "QC source video", "Optional video used to test the newly trained model. Leave it empty to reuse the Facemap initialization video."),
            (self.qc_focus_label, "QC focus keypoint", "Pick a keypoint from the list to zoom in on it. The zoom is a static region — LabelForge computes the median position across all frames. 'Auto' prefers pupil, eye, tongue, nose or mouth labels."),
            (self.qc_duration, "QC preview length", "Choose 30–120 seconds. Sixty seconds is usually long enough to spot drift, swaps or poor confidence without producing a huge result file."),
            (self.qc_zoom, "QC zoom context", "Balanced is close to the established iris QC. More context shows surrounding anatomy; closer detail magnifies the selected keypoint."),
            (self.advanced_training_button, "Advanced training settings", "Opens optional expert controls. The normal workflow already provides safe defaults, so beginners can leave this section closed."),
            (self.training_script, "Facemap training adapter", "Optional versioned Python adapter that defines a custom Facemap training call. Leave it empty to use the standard LabelForge adapter."),
            (self.epochs, "Training epochs", "Maximum number of complete training passes. Higher values can improve convergence but take longer; the default is intended as a safe starting point."),
            (self.batch, "Batch size", "Number of samples processed together. Larger batches need more memory. Keep the default unless the backend or hardware requires a change."),
            (self.lr, "Learning rate", "Controls how strongly the model updates each step. This is an expert setting; an unsuitable value can prevent useful learning."),
            (self.seed, "Random seed", "Makes data sampling and initialization reproducible so the same package can be investigated or repeated later."),
            (self.execution_target, "Where training runs", "Local uses this computer. Remote workstation uses SSH directly. HPC submits a Slurm job so heavy training runs on a compute node, never the login node."),
            (self.sync_mode, "How files reach the target", "The recommended option copies one self-contained package. The advanced option references files that already exist at known paths on the target."),
            (self.remote_host, "SSH host", "The login address LabelForge connects to. For JUSUF this is jusuf.fz-juelich.de."),
            (self.remote_user, "JSC username", "Your account name on JUSUF. It is checked against whoami during the remote preflight."),
            (self.identity_file, "Private SSH key", "Only this local path is passed to Windows OpenSSH. The key itself and its passphrase are never copied, bundled or committed."),
            (self.ssh_agent_status, "Windows SSH Agent", "The agent holds the unlocked key in memory. Unlock it once per login session so LabelForge can connect without storing a passphrase."),
            (self.ssh_setup_button, "Enable and unlock SSH", "Opens a visible administrator PowerShell. It enables Windows SSH Agent and asks for the key passphrase once."),
            (self.ssh_refresh_button, "Refresh SSH status", "Checks the Windows SSH Agent again after a key was added or unlocked. It does not connect to JUSUF."),
            (self.totp_code, "JUSUF verification code", "Enter the current code from Google Authenticator immediately before the remote test. It is masked, used for this one SSH login only, and cleared as soon as the test starts."),
            (self.remote_root, "Remote workspace", "A predictable folder on JUSUF used for LabelForge packages, logs and results. The preflight creates it and confirms that it is writable."),
            (self.remote_environment, "Remote Python environment", "The environment that will eventually run Facemap or DLC on the target. Environment creation is the next implementation step after SSH preflight."),
            (self.account, "Slurm account", "The compute project charged for the job. The confirmed JUSUF account is training2636."),
            (self.partition, "Slurm partition", "The compute queue used by the job. Your confirmed association is batch; GPU partitions are not currently authorized."),
            (self.advanced_target_button, "Advanced target settings", "Shows optional Slurm timing and resource controls. Required account information remains visible outside this section."),
            (self.walltime, "Maximum run time", "The Slurm time limit in hours, minutes and seconds. JUSUF stops the job if it exceeds this value."),
            (self.gpus, "Requested GPUs", "Number of GPUs requested from Slurm. Your currently confirmed JUSUF batch setup uses zero GPUs."),
            (self.cpus, "Requested CPU cores", "CPU cores reserved for the training job. More cores only help when the training backend can use them."),
            (self.memory, "Requested memory", "RAM reserved for the Slurm job in gigabytes. Request enough for the dataset without unnecessarily blocking cluster resources."),
            (self.remote_test_button, "Test remote setup", "Connects safely without transferring data or starting training. It validates the key, user, workspace, Slurm and training2636 / batch association."),
            (self.validate_button, "Check inputs", "Verifies paths, required fields and safe defaults. Nothing is copied or changed during this step."),
            (self.generate_button, "Build training package", "Creates the portable recipe containing model references, labels, parameters, launch files and—when selected—the training data."),
            (self.sync_button, "Transfer package", "Copies the completed package to the tested remote workspace. This unlocks only after the remote preflight succeeds."),
            (self.start_button, "Start training", "Local runs start directly. HPC runs submit slurm_job.sh with sbatch and store the returned job ID."),
            (self.status_button, "Monitor the job", "Queries Slurm for the recorded job and shows whether it is queued, running, completed or failed."),
            (self.fetch_button, "Fetch results", "Copies logs and trained outputs back into the local bundle so the final .pt model can later be registered in the model library."),
        ]
        for widget, title, body in topics: self._register_help(widget, title, body)
        for button in self.findChildren(QPushButton):
            if button.objectName() == "BrowseButton":
                field = button.parentWidget().findChild(QLineEdit)
                if field and field.property("contextHelpTitle"):
                    self._register_help(button, field.property("contextHelpTitle"), field.property("contextHelpBody"))
                else:
                    self._register_help(button, "Choose a file or folder", "Opens a picker with the expected file type shown in its title and filter.")

        # Guarantee full coverage even when a new control is added later. The
        # specific topics above win; these fallbacks keep every interactive
        # surface explainable from its first build onward.
        interactive = (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton)
        for control_type in interactive:
            for widget in self.findChildren(control_type):
                if widget.property("contextHelpTitle"): continue
                if isinstance(widget, QLineEdit):
                    title = widget.placeholderText() or "Input field"
                    body = "Enter the requested value here. The surrounding step validates it before LabelForge unlocks the next action."
                elif isinstance(widget, QComboBox):
                    title, body = "Choose an option", "Click the dropdown and select one option deliberately. Scrolling the page cannot change this value."
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    title, body = "Numeric setting", "Enter a value or use the step buttons. This setting is validated before the training package is created."
                else:
                    title = widget.text().replace("…", "").strip() or "Action"
                    body = "Click to perform this action. LabelForge keeps source models and labels unchanged unless the description explicitly says otherwise."
                self._register_help(widget, title, body)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() in (QEvent.Enter, QEvent.HoverEnter, QEvent.MouseMove) and watched.property("contextHelpTitle") and hasattr(self, "help_title"):
            title, body = watched.property("contextHelpTitle"), watched.property("contextHelpBody")
            self.help_title.setText(title); self.help_body.setText(body)
            QTimer.singleShot(0, self._resize_help_bubble)
        return super().eventFilter(watched, event)

    def _card(self, title: str, text: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(); card.setObjectName("WizardPanel")
        box = QVBoxLayout(card); box.setContentsMargins(24, 20, 24, 22); box.setSpacing(12)
        heading = QLabel(title); heading.setObjectName("CardTitle")
        hint = QLabel(text); hint.setObjectName("FieldHint"); hint.setWordWrap(True)
        box.addWidget(heading); box.addWidget(hint)
        return card, box

    def _mode_card(self) -> QFrame:
        card, box = self._card(
            "1  What do you want to train?",
            "Choose the goal. LabelForge will suggest the correct new model name and keep the parent untouched.",
        )
        # Quick-load shortcut — always visible at the top of the card
        shortcut_row = QHBoxLayout(); shortcut_row.setSpacing(10)
        shortcut_label = QLabel("Already have a bundle?"); shortcut_label.setObjectName("FieldHint")
        self.early_load_button = QPushButton("Use existing package…")
        self.early_load_button.setObjectName("SecondaryActionButton")
        self.early_load_button.clicked.connect(self._load_existing_bundle_early)
        shortcut_row.addWidget(shortcut_label); shortcut_row.addWidget(self.early_load_button); shortcut_row.addStretch(1)
        box.addLayout(shortcut_row)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setObjectName("CardSeparator")
        box.addWidget(sep)
        row = QHBoxLayout(); row.setSpacing(12)
        self.mode_group = QButtonGroup(self); self.mode_group.setExclusive(True)
        choices = [
            ("Create", "Create\nStart a completely new model"),
            ("Refine", "Refine\nImprove an existing model"),
            ("Specialize", "Specialize\nAdapt a model to a new use case"),
        ]
        self.mode_buttons: dict[str, QPushButton] = {}
        for mode, text in choices:
            button = QPushButton(text); button.setObjectName("ModeCard"); button.setCheckable(True)
            button.setMinimumHeight(82); self.mode_group.addButton(button); row.addWidget(button, 1)
            button.clicked.connect(lambda checked=False, value=mode: self._mode_changed(value))
            self.mode_buttons[mode] = button
        box.addLayout(row)
        self.mode_prompt = QLabel("Choose one option to continue ↓"); self.mode_prompt.setObjectName("StepPrompt")
        box.addWidget(self.mode_prompt)
        return card

    def _software_card(self) -> QFrame:
        card, box = self._card(
            "2  Local software check",
            "Only needed when training on this computer. LabelForge detects Facemap and DeepLabCut in every local Conda environment.",
        )
        grid = QGridLayout(); grid.setHorizontalSpacing(16)
        self.conda_status = QLabel(); self.facemap_status = QLabel(); self.dlc_status = QLabel()
        for status in [self.conda_status, self.facemap_status, self.dlc_status]: status.setObjectName("SoftwareStatus")
        self.fm_install_button = QPushButton("Install / repair Facemap"); self.dlc_install_button = QPushButton("Install / repair DeepLabCut")
        self.fm_install_button.setObjectName("SecondaryActionButton"); self.dlc_install_button.setObjectName("SecondaryActionButton")
        self.fm_install_button.clicked.connect(lambda: self._install_backend("facemap"))
        self.dlc_install_button.clicked.connect(lambda: self._install_backend("deeplabcut"))
        grid.addWidget(QLabel("Conda"), 0, 0); grid.addWidget(self.conda_status, 0, 1)
        grid.addWidget(QLabel("Facemap"), 1, 0); grid.addWidget(self.facemap_status, 1, 1); grid.addWidget(self.fm_install_button, 1, 2)
        grid.addWidget(QLabel("DeepLabCut"), 2, 0); grid.addWidget(self.dlc_status, 2, 1); grid.addWidget(self.dlc_install_button, 2, 2)
        grid.setColumnStretch(1, 1); box.addLayout(grid)
        return card

    def _line(self, placeholder: str = "") -> QLineEdit:
        field = QLineEdit(); field.setObjectName("TextInput"); field.setPlaceholderText(placeholder)
        field.setMinimumHeight(44); field.textChanged.connect(field.setToolTip)
        return field

    def _path_row(self, field: QLineEdit, directory: bool = False, file_filter: str = "All files (*)", title: str = "Select file") -> QWidget:
        row = QWidget(); row.setMinimumHeight(44)
        layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0); layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(field, 1)
        button = QPushButton("Browse…"); button.setObjectName("BrowseButton")
        button.setMinimumHeight(44)
        button.clicked.connect(lambda: self._browse(field, directory, file_filter, title))
        layout.addWidget(button)
        return row

    def _with_hint(self, control: QWidget, text: str) -> QWidget:
        wrapper = QWidget(); layout = QVBoxLayout(wrapper); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(3)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        layout.addWidget(control)
        hint = QLabel(text); hint.setObjectName("InlineHint"); hint.setWordWrap(True); layout.addWidget(hint)
        return wrapper

    def _run_card(self) -> QFrame:
        card, box = self._card(
            "3  Choose the training material",
            "Add the model and labeled data. LabelForge keeps the keypoint order locked and never overwrites an existing model.",
        )
        form = QFormLayout(); self.training_form = form
        self.backend = ClearComboBox(); self.backend.addItems(["Facemap", "DeepLabCut"]); self.backend.setMaximumWidth(280)
        self.backend.currentTextChanged.connect(self._backend_changed)
        self.local_environment = ClearComboBox()
        self.parent_model = self._line("Parent .pt model")
        self.training_data = self._line("FM: image folder  •  DLC: project folder")
        self.labels_config = self._line("FM: labels.csv  •  DLC: config.yaml")
        self.init_video = self._line("Facemap initialization video")
        self.qc_enabled = QCheckBox("Create a short visual quality-check preview after training")
        self.qc_enabled.setChecked(True)
        self.qc_video = self._line("Optional — uses init video if empty")
        self.qc_focus_label = ClearComboBox()
        self.qc_focus_label.addItem("Auto — prefers pupil, eye, tongue, nose or mouth", "")
        self.qc_duration = QSpinBox(); self.qc_duration.setRange(30, 120); self.qc_duration.setValue(60); self.qc_duration.setSuffix(" seconds")
        self.qc_zoom = ClearComboBox(); self.qc_zoom.addItem("Balanced context  —  recommended", 1.0); self.qc_zoom.addItem("More surrounding context", 1.4); self.qc_zoom.addItem("Closer detail", 0.75)
        self.training_script = self._line("Versioned Facemap training adapter")
        self.output_dir = self._line("Folder in which LabelForge should save the training package")
        self.model_name = self._line("e.g. SideView_Face_2P_v2")
        self.parent_label = QLabel("Parent model")
        self.parent_row = self._path_row(self.parent_model, False, "PyTorch model (*.pt)", "Choose the existing .pt model")
        form.addRow("Backend", self.backend)
        form.addRow("Local environment", self.local_environment)
        form.addRow(self.parent_label, self.parent_row)
        training_control = self._path_row(self.training_data, True, title="Choose the image folder (Facemap) or DLC project folder")
        form.addRow("Training data", self._with_hint(training_control, "Facemap: folder containing the training images.  DeepLabCut: project folder containing config.yaml."))
        labels_control = self._path_row(self.labels_config, False, "Label/config (*.csv *.yaml *.yml)", "Choose labels.csv (Facemap) or config.yaml (DeepLabCut)")
        form.addRow("Labels / config", self._with_hint(labels_control, "Facemap: LabelForge labels file (*.csv).  DeepLabCut: project configuration (*.yaml or *.yml)."))
        form.addRow("New model name", self.model_name)
        package_control = self._path_row(self.output_dir, True, title="Choose where the new training package should be saved")
        package_wrapper = QWidget(); package_layout = QVBoxLayout(package_wrapper); package_layout.setContentsMargins(0, 0, 0, 0); package_layout.setSpacing(3); package_layout.setSizeConstraint(QLayout.SetMinimumSize)
        package_layout.addWidget(package_control)
        self.bundle_destination_hint = QLabel("Choose a folder and model name to preview the exact package location.")
        self.bundle_destination_hint.setObjectName("InlineHint"); self.bundle_destination_hint.setWordWrap(True)
        package_layout.addWidget(self.bundle_destination_hint)
        form.addRow("Save training package in", package_wrapper)
        self.init_video_control = self._path_row(self.init_video, False, "Videos (*.avi *.mp4 *.mkv *.mov)", "Choose the Facemap initialization video")
        self.init_video_row = self._with_hint(self.init_video_control, "Required by Facemap to initialize the model (*.avi, *.mp4, *.mkv or *.mov).")
        self.init_video_label = QLabel("Initialization video")
        form.addRow(self.init_video_label)
        form.addRow(self.init_video_row)

        form.addRow("Visual QC", self.qc_enabled)
        qc_video_control = self._path_row(self.qc_video, False, "Videos (*.avi *.mp4 *.mkv *.mov)", "Choose the video for the post-training QC preview")
        form.addRow("QC video", self._with_hint(qc_video_control, "Optional. If empty, LabelForge uses the initialization video. The selected 30–120 second segment is only used for prediction and visual checking."))
        qc_options = QWidget(); qc_row = QHBoxLayout(qc_options); qc_row.setContentsMargins(0, 0, 0, 0); qc_row.addWidget(QLabel("Focus")); qc_row.addWidget(self.qc_focus_label, 2); qc_row.addWidget(QLabel("Length")); qc_row.addWidget(self.qc_duration); qc_row.addWidget(QLabel("Zoom")); qc_row.addWidget(self.qc_zoom, 2)
        form.addRow("QC preview options", qc_options)

        self.advanced_training_button = QPushButton("Advanced training settings  ▸")
        self.advanced_training_button.setObjectName("AdvancedToggle"); self.advanced_training_button.setCheckable(True)
        self.advanced_training_button.clicked.connect(self._toggle_training_advanced)
        self.advanced_training_panel = QFrame(); self.advanced_training_panel.setObjectName("AdvancedPanel")
        advanced = QFormLayout(self.advanced_training_panel)
        advanced.addRow("Facemap adapter", self._path_row(self.training_script, False, "Python scripts (*.py);;All files (*)"))
        params = QWidget(); row = QHBoxLayout(params); row.setContentsMargins(0, 0, 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 10_000_000); self.epochs.setValue(100)
        self.batch = QSpinBox(); self.batch.setRange(1, 4096); self.batch.setValue(1)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(8); self.lr.setRange(0.00000001, 1.0); self.lr.setValue(0.00005)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(20260828)
        for label, widget in [("Epochs", self.epochs), ("Batch", self.batch), ("LR", self.lr), ("Seed", self.seed)]:
            row.addWidget(QLabel(label)); row.addWidget(widget)
        row.addStretch(1); advanced.addRow("Parameters", params)
        self.advanced_training_panel.setVisible(False)
        box.addLayout(form); box.addWidget(self.advanced_training_button); box.addWidget(self.advanced_training_panel)
        self.material_prompt = QLabel("○  Select valid files and a save folder to unlock step 4.")
        self.material_prompt.setObjectName("StepPrompt"); box.addWidget(self.material_prompt)
        self.parent_model.textChanged.connect(self._parent_changed)
        self.model_name.editingFinished.connect(self._normalize_model_name)
        self.model_name.textChanged.connect(self._update_bundle_destination)
        self.output_dir.textChanged.connect(self._update_bundle_destination)
        for field in [self.parent_model, self.training_data, self.labels_config, self.output_dir, self.model_name, self.init_video, self.qc_video]:
            field.textChanged.connect(self._configuration_changed)
        self.labels_config.textChanged.connect(self._update_focus_dropdown)
        self.qc_focus_label.currentIndexChanged.connect(self._configuration_changed)
        return card

    def _update_bundle_destination(self, *_args) -> None:
        if not hasattr(self, "bundle_destination_hint"): return
        root, model = self.output_dir.text().strip(), self.model_name.text().strip()
        if root and model:
            destination = Path(root).expanduser() / f"{safe_name(model)}_training_bundle"
            self.bundle_destination_hint.setText(f"Package will be created here:  {destination}")
            self.bundle_destination_hint.setToolTip(str(destination))
        else:
            self.bundle_destination_hint.setText("Choose a folder and model name to preview the exact package location.")

    def _update_focus_dropdown(self, labels_path: str = "") -> None:
        """Populate the QC focus keypoint dropdown from the selected labels.csv."""
        if not hasattr(self, "qc_focus_label"): return
        path = Path(labels_path or "")
        current = self.qc_focus_label.currentData()
        self.qc_focus_label.blockSignals(True)
        self.qc_focus_label.clear()
        self.qc_focus_label.addItem("Auto — prefers pupil, eye, tongue, nose or mouth", "")
        if path.suffix.lower() == ".csv" and path.is_file():
            try:
                for label in read_facemap_labels_csv(str(path)):
                    self.qc_focus_label.addItem(label, label)
            except Exception:
                pass
        idx = self.qc_focus_label.findData(current)
        self.qc_focus_label.setCurrentIndex(idx if idx >= 0 else 0)
        self.qc_focus_label.blockSignals(False)

    def _execution_card(self) -> QFrame:
        card, box = self._card(
            "4  Where should it run?",
            "Local runs directly. Remote targets use SSH/SCP; HPC additionally submits through Slurm. Passwords and browser tokens are never stored.",
        )
        form = QFormLayout(); self.execution_form = form
        self.execution_target = ClearComboBox(); self.execution_target.addItems(["Local", "Remote workstation", "HPC (Slurm)"])
        self.execution_target.setMaximumWidth(360)
        self.execution_target.currentTextChanged.connect(self._target_changed)
        self.sync_mode = ClearComboBox()
        self.sync_mode.addItem("Copy everything to the training computer  —  recommended", "Copy complete training package")
        self.sync_mode.addItem("Files already exist on the training computer  —  advanced", "Reuse paths already available on target")
        self.sync_mode.currentIndexChanged.connect(self._sync_mode_changed)
        self.remote_host = self._line("SSH hostname or IP")
        self.remote_user = self._line("SSH username")
        self.identity_file = self._line("Path to the private SSH key (never copied or stored in a bundle)")
        self.identity_file.setText(str(Path.home() / ".ssh" / "id_ed25519"))
        self.remote_root = self._line("Remote training workspace")
        self.remote_environment = self._line("Environment name or full venv path")
        self.account = self._line("Slurm budget/account"); self.partition = self._line("Slurm partition")
        self.walltime = self._line(); self.walltime.setText("04:00:00")
        resources = QWidget(); row = QHBoxLayout(resources); row.setContentsMargins(0, 0, 0, 0)
        self.gpus = QSpinBox(); self.gpus.setRange(0, 16); self.gpus.setValue(0)
        self.cpus = QSpinBox(); self.cpus.setRange(1, 256); self.cpus.setValue(8)
        self.memory = QSpinBox(); self.memory.setRange(1, 2048); self.memory.setValue(64)
        for label, widget in [("GPUs", self.gpus), ("CPUs", self.cpus), ("RAM GB", self.memory)]:
            row.addWidget(QLabel(label)); row.addWidget(widget)
        row.addStretch(1)
        self.target_explanation = QLabel(); self.target_explanation.setObjectName("TargetExplanation"); self.target_explanation.setWordWrap(True)
        form.addRow("Execution target", self.execution_target); form.addRow("", self.target_explanation)
        form.addRow("How should the training files get there?", self.sync_mode)
        self.sync_explanation = QLabel(); self.sync_explanation.setObjectName("TargetExplanation"); self.sync_explanation.setWordWrap(True)
        form.addRow("", self.sync_explanation)
        form.addRow("Host", self.remote_host); form.addRow("Username", self.remote_user)
        self.identity_control = self._path_row(self.identity_file, False, "SSH private key (id_*);;All files (*)", "Choose the private SSH key used for JUSUF")
        form.addRow("SSH key", self._with_hint(self.identity_control, "Only the local file path is used. The key and passphrase are never copied into LabelForge or a training package."))
        self.ssh_agent_status = QLabel(); self.ssh_agent_status.setObjectName("AgentStatus"); self.ssh_agent_status.setWordWrap(True)
        self.ssh_setup_button = QPushButton("Enable / unlock SSH key…"); self.ssh_setup_button.setObjectName("SecondaryActionButton")
        self.ssh_refresh_button = QPushButton("Refresh"); self.ssh_refresh_button.setObjectName("SecondaryActionButton")
        self.ssh_setup_button.clicked.connect(self._open_ssh_setup); self.ssh_refresh_button.clicked.connect(self._refresh_ssh_agent_status)
        agent_buttons = QHBoxLayout(); agent_buttons.addWidget(self.ssh_setup_button); agent_buttons.addWidget(self.ssh_refresh_button); agent_buttons.addStretch(1)
        agent_box = QWidget(); agent_layout = QVBoxLayout(agent_box); agent_layout.setContentsMargins(0, 0, 0, 0)
        agent_layout.addWidget(self.ssh_agent_status); agent_layout.addLayout(agent_buttons)
        form.addRow("SSH access", agent_box)
        self.totp_code = self._line("Current Google Authenticator code")
        self.totp_code.setEchoMode(QLineEdit.Password); self.totp_code.setMaxLength(12)
        self.totp_control = self._with_hint(
            self.totp_code, "Required only for this connection test. It is never saved, logged or added to the training package."
        )
        form.addRow("One-time JUSUF code", self.totp_control)
        form.addRow("Remote workspace", self.remote_root); form.addRow("Remote environment", self.remote_environment)
        self.account_control = self._with_hint(self.account, "Required for HPC: the project or compute-budget name used by Slurm.")
        form.addRow("Slurm account / budget", self.account_control)
        box.addLayout(form)
        self.advanced_target_button = QPushButton("Advanced target settings  ▸")
        self.advanced_target_button.setObjectName("AdvancedToggle"); self.advanced_target_button.setCheckable(True)
        self.advanced_target_button.clicked.connect(self._toggle_target_advanced)
        self.advanced_target_panel = QFrame(); self.advanced_target_panel.setObjectName("AdvancedPanel")
        target_advanced = QFormLayout(self.advanced_target_panel)
        target_advanced.addRow("Partition", self.partition)
        target_advanced.addRow("Walltime", self.walltime); target_advanced.addRow("Resources", resources)
        self.advanced_target_panel.setVisible(False)
        box.addWidget(self.advanced_target_button); box.addWidget(self.advanced_target_panel)
        self.remote_test_button = QPushButton("Test remote setup")
        self.remote_test_button.setObjectName("PrimaryNextButton"); self.remote_test_button.clicked.connect(self._test_remote)
        self.remote_test_status = QLabel("○  Not tested yet"); self.remote_test_status.setObjectName("RemoteTestStatus")
        self.remote_test_status.setProperty("state", "neutral")
        self.remote_test_status.setWordWrap(True); self.remote_test_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.remote_test_status.setMinimumHeight(48)
        test_row = QHBoxLayout(); test_row.setAlignment(Qt.AlignTop)
        test_row.addWidget(self.remote_test_button, 0, Qt.AlignTop); test_row.addWidget(self.remote_test_status, 1)
        box.addLayout(test_row)
        self._remote_fields = [self.sync_mode, self.sync_explanation, self.remote_host, self.remote_user, self.identity_control, agent_box, self.totp_control, self.remote_root, self.remote_environment]
        self._slurm_fields = [self.account, self.partition, self.walltime, self.gpus, self.cpus, self.memory]
        for field in [self.remote_host, self.remote_user, self.identity_file, self.remote_root, self.remote_environment, self.account, self.partition]:
            field.textChanged.connect(self._configuration_changed)
        self._target_changed("Local")
        self._sync_mode_changed()
        self._refresh_ssh_agent_status()
        return card

    def _actions_card(self) -> QFrame:
        card, box = self._card(
            "5  Ready to train",
            "Follow the buttons from left to right. Only the next available action is enabled.",
        )
        self.readiness = QLabel(); self.readiness.setObjectName("ReadinessChecklist"); self.readiness.setWordWrap(True)
        box.addWidget(self.readiness)
        self.action_road = QLabel("1  Check inputs   →   2  Build package   →   3  Transfer   →   4  Start   →   5  Monitor   →   6  Fetch results")
        self.action_road.setObjectName("ActionRoad"); self.action_road.setWordWrap(True); box.addWidget(self.action_road)
        first = QHBoxLayout()
        self.validate_button = QPushButton("1  Check inputs"); self.generate_button = QPushButton("2  Build training package")
        self.load_bundle_button = QPushButton("Use existing package…"); self.load_bundle_button.setObjectName("SecondaryActionButton")
        self.load_bundle_button.clicked.connect(self._load_existing_bundle)
        self.validate_button.setObjectName("SecondaryActionButton"); self.generate_button.setObjectName("PrimaryNextButton")
        self.validate_button.clicked.connect(self._validate); self.generate_button.clicked.connect(self._generate)
        self.generate_button.setEnabled(False)
        first.addWidget(self.validate_button); first.addWidget(self.generate_button); first.addWidget(self.load_bundle_button); first.addStretch(1)
        first_help = QLabel("Create a new package, or open an existing one and continue without rebuilding it.")
        first_help.setObjectName("InlineHint")
        second = QHBoxLayout()
        self.sync_button = QPushButton("4  Transfer package"); self.start_button = QPushButton("5  Start training")
        self.status_button = QPushButton("6  Check status"); self.fetch_button = QPushButton("7  Fetch results")
        for button in [self.sync_button, self.start_button, self.status_button, self.fetch_button]:
            button.setObjectName("SecondaryActionButton"); button.setEnabled(False); second.addWidget(button)
        second.addStretch(1)
        self.sync_button.clicked.connect(self._synchronize); self.start_button.clicked.connect(self._start_training)
        self.status_button.clicked.connect(self._check_status); self.fetch_button.clicked.connect(self._fetch_results)
        action_help = QLabel(
            "Synchronize copies the package to the selected computer.  Start training launches it.  "
            "Check status reads the current job state.  Fetch results copies logs and trained outputs back."
        )
        action_help.setObjectName("InlineHint"); action_help.setWordWrap(True)
        self._job_id_label = QLabel(); self._job_id_label.setObjectName("InlineHint")
        self._job_id_label.setWordWrap(True); self._job_id_label.setVisible(False)
        box.addLayout(first); box.addWidget(first_help); box.addSpacing(5); box.addLayout(second); box.addWidget(action_help); box.addWidget(self._job_id_label)
        self.log = QTextEdit(); self.log.setObjectName("TextInput"); self.log.setReadOnly(True)
        self.log.setMaximumHeight(190); self.log.setPlaceholderText("Validation, synchronization and training output appears here.")
        box.addWidget(self.log)
        self._update_readiness()
        self._set_action_stage(1)
        return card

    def _browse(self, field: QLineEdit, directory: bool, file_filter: str, title: str) -> None:
        if hasattr(self, "training_data") and field is self.training_data:
            title = "Choose the image folder (*.png, *.jpg, *.jpeg, *.tif)" if self.backend.currentText() == "Facemap" else "Choose the DeepLabCut project folder (contains config.yaml)"
        elif hasattr(self, "labels_config") and field is self.labels_config:
            title = "Choose the LabelForge labels file (*.csv)" if self.backend.currentText() == "Facemap" else "Choose the DeepLabCut project configuration (*.yaml, *.yml)"
        if directory: path = QFileDialog.getExistingDirectory(self, title, field.text())
        else: path, _ = QFileDialog.getOpenFileName(self, title, field.text(), file_filter)
        if path: field.setText(path)

    def _training_mode(self) -> str:
        for name, button in self.mode_buttons.items():
            if button.isChecked(): return name
        return ""

    def _mode_changed(self, mode: str) -> None:
        if hasattr(self, "mode_prompt"): self.mode_prompt.setText(f"✓  {mode} selected — continue with the training material below.")
        if hasattr(self, "run_card"): self.run_card.setVisible(True)
        needs_parent = mode != "Create"
        if hasattr(self, "parent_row"):
            self.parent_row.setVisible(needs_parent); self.parent_label.setVisible(needs_parent)
        if not hasattr(self, "model_name"): return
        if mode == "Refine":
            self.model_name.setReadOnly(True)
            self.model_name.setPlaceholderText("Suggested automatically from the parent model")
            self._parent_changed(self.parent_model.text())
        elif mode == "Specialize":
            self.model_name.setReadOnly(False)
            self.model_name.clear(); self.model_name.setPlaceholderText("New specialized model name (version added automatically)")
        else:
            self.model_name.setReadOnly(False)
            self.parent_model.clear(); self.model_name.clear()
            self.model_name.setPlaceholderText("New model name (version added automatically)")
        self._update_readiness()

    def _parent_changed(self, parent: str) -> None:
        if hasattr(self, "mode_buttons") and self._training_mode() == "Refine":
            self.model_name.setText(next_refinement_name(parent))
        self._update_readiness()

    def _normalize_model_name(self) -> None:
        if self._training_mode() != "Refine":
            self.model_name.setText(ensure_v1(self.model_name.text()))

    def _configuration_changed(self, *_args) -> None:
        self._validated = False
        self._last_bundle = None
        self._remote_test_passed = False
        if hasattr(self, "remote_test_status"): self._set_remote_test_status(False, "Not tested yet")
        if hasattr(self, "generate_button"):
            self.generate_button.setEnabled(False)
            for button in [self.sync_button, self.start_button, self.status_button, self.fetch_button]:
                button.setEnabled(False)
            self._set_action_stage(1)
        self._update_readiness()

    def _set_action_stage(self, active: int) -> None:
        if not hasattr(self, "action_road"): return
        remote = hasattr(self, "execution_target") and self.execution_target.currentText() != "Local"
        names = ["Check inputs", "Build package"]
        if remote: names.extend(["Test remote", "Transfer", "Start", "Monitor", "Fetch results"])
        else: names.append("Start")
        parts = []
        for index, name in enumerate(names, 1):
            if index < active: color, marker = "#75c995", "✓"
            elif index == active: color, marker = "#f0a354", "●"
            else: color, marker = "#7f8793", "○"
            parts.append(f'<span style="color:{color}; font-weight:700">{marker}&nbsp; {index} {name}</span>')
        self.action_road.setText("&nbsp;&nbsp; → &nbsp;&nbsp;".join(parts))

    def _toggle_training_advanced(self, checked: bool) -> None:
        self.advanced_training_panel.setVisible(checked)
        self.advanced_training_button.setText("Advanced training settings  ▾" if checked else "Advanced training settings  ▸")

    def _toggle_target_advanced(self, checked: bool) -> None:
        self.advanced_target_panel.setVisible(checked)
        self.advanced_target_button.setText("Advanced target settings  ▾" if checked else "Advanced target settings  ▸")

    def _update_readiness(self, *_args) -> None:
        mode = self._training_mode()
        if hasattr(self, "run_card"): self.run_card.setVisible(bool(mode))
        if hasattr(self, "training_data"):
            data_path = self.training_data.text().strip(); labels_path = self.labels_config.text().strip()
            output_path = self.output_dir.text().strip(); parent_path = self.parent_model.text().strip()
            material_ready = bool(
                mode and data_path and Path(data_path).exists()
                and labels_path and Path(labels_path).is_file()
                and output_path and Path(output_path).is_dir()
                and self.model_name.text().strip()
                and (mode == "Create" or (parent_path and Path(parent_path).is_file()))
            )
            if hasattr(self, "execution_card"): self.execution_card.setVisible(material_ready)
            if hasattr(self, "actions_card"): self.actions_card.setVisible(material_ready)
            if hasattr(self, "material_prompt"):
                self.material_prompt.setText(
                    "✓  Training material complete — choose where it should run below."
                    if material_ready else "○  Select valid files and a save folder to unlock step 4."
                )
        if not hasattr(self, "readiness"): return
        parent_ok = mode == "Create" or bool(self.parent_model.text().strip())
        data_ok = bool(self.training_data.text().strip() and self.labels_config.text().strip())
        name_ok = bool(self.model_name.text().strip())
        environment_ok = self.execution_target.currentText() != "Local" or self.local_environment.currentText() != "Not installed"
        items = [
            (parent_ok, "Starting model selected" if mode != "Create" else "New model selected"),
            (data_ok, "Training data and labels selected"),
            (name_ok, "New model name ready"),
            (environment_ok, "Training environment ready"),
        ]
        lines = []
        for complete, text in items:
            color, marker = ("#75c995", "✓") if complete else ("#aeb4bf", "○")
            lines.append(f'<span style="color:{color}; font-weight:600">{marker}&nbsp; {text}</span>')
        self.readiness.setText("&nbsp;&nbsp;&nbsp;&nbsp;".join(lines))
        self._refresh_route_rail()

    def _refresh_route_rail(self) -> None:
        if not hasattr(self, "route_labels"): return
        mode_ready = bool(self._training_mode())
        material_ready = hasattr(self, "execution_card") and not self.execution_card.isHidden()
        bundle_ready = getattr(self, "_last_bundle", None) is not None
        training_done = getattr(self, "_training_done", False)
        actions_visible = hasattr(self, "actions_card") and not self.actions_card.isHidden()
        states = [
            "complete" if mode_ready else "active",
            "complete" if self.conda_path else ("active" if mode_ready else "locked"),
            "complete" if material_ready else ("active" if mode_ready else "locked"),
            "complete" if bundle_ready else ("active" if material_ready else "locked"),
            "complete" if training_done else ("active" if actions_visible else "locked"),
        ]
        for label, state in zip(self.route_labels, states):
            label.setProperty("state", state); label.style().unpolish(label); label.style().polish(label)

    def _backend_changed(self, backend: str) -> None:
        facemap = backend.lower() == "facemap"
        if hasattr(self, "training_data"):
            self.training_data.setPlaceholderText(
                "Image folder (*.png, *.jpg, *.jpeg, *.tif)" if facemap else "DeepLabCut project folder (contains config.yaml)"
            )
            self.labels_config.setPlaceholderText("LabelForge labels (*.csv)" if facemap else "DLC configuration (*.yaml, *.yml)")
        self.init_video_row.setVisible(facemap); self.init_video_label.setVisible(facemap); self.training_script.setEnabled(facemap)
        self.local_environment.clear()
        environments = self.backend_environments.get(backend.lower(), [])
        self.local_environment.addItems(environments or ["Not installed"])
        if hasattr(self, "remote_environment"):
            default = "labelforge-facemap" if facemap else "labelforge-dlc"
            if not self.remote_environment.text() or self.remote_environment.text().startswith("labelforge-"):
                self.remote_environment.setText(default)
        self._update_readiness()

    def _target_changed(self, target: str) -> None:
        remote = target != "Local"; slurm = target == "HPC (Slurm)"
        explanations = {
            "Local": "Runs on this computer. Best for small tests and short training jobs; nothing is uploaded.",
            "Remote workstation": "Copies or reuses the package on another GPU computer via SSH, then starts training there.",
            "HPC (Slurm)": "Sends the package to a compute cluster and submits it to the Slurm queue (for example JUSUF).",
        }
        if hasattr(self, "target_explanation"): self.target_explanation.setText(explanations[target])
        for field in getattr(self, "_remote_fields", []):
            field.setVisible(remote)
            if hasattr(self, "execution_form") and self.execution_form.labelForField(field):
                self.execution_form.labelForField(field).setVisible(remote)
        for field in getattr(self, "_slurm_fields", []): field.setEnabled(slurm)
        if hasattr(self, "account_control"):
            self.account_control.setVisible(slurm)
            if self.execution_form.labelForField(self.account_control):
                self.execution_form.labelForField(self.account_control).setVisible(slurm)
        if hasattr(self, "advanced_target_button"):
            self.advanced_target_button.setVisible(slurm)
            if not slurm:
                self.advanced_target_button.setChecked(False); self._toggle_target_advanced(False)
        if hasattr(self, "remote_test_button"):
            self.remote_test_button.setVisible(remote); self.remote_test_status.setVisible(remote)
        if hasattr(self, "local_environment"):
            self.local_environment.setVisible(not remote)
            local_label = self.training_form.labelForField(self.local_environment)
            if local_label: local_label.setVisible(not remote)
        if slurm:
            if not self.remote_host.text(): self.remote_host.setText("jusuf.fz-juelich.de")
            if not self.remote_user.text(): self.remote_user.setText("daubenfeld1")
            if not self.remote_root.text() or self.remote_root.text() == "/p/home/jusers/daubenfeld1/jusuf/labelforge-training":
                self.remote_root.setText("/p/project1/training2636/daubenfeld1/labelforge/training-runs")
            if not self.account.text(): self.account.setText("training2636")
            if not self.remote_environment.text() or self.remote_environment.text().startswith("labelforge-"):
                self.remote_environment.setText(
                    "/p/project1/training2636/daubenfeld1/labelforge/environments/facemap"
                    if self.backend.currentText().lower() == "facemap" else "labelforge-dlc"
                )
            if not self.partition.text() or self.partition.text() == "gpus": self.partition.setText("batch")
            self.gpus.setValue(0)
        if hasattr(self, "sync_button"):
            self.sync_button.setVisible(remote); self.status_button.setVisible(remote); self.fetch_button.setVisible(remote)
            self.sync_button.setText("4  Transfer package")
            self.start_button.setText("5  Start training" if remote else "3  Start training")
            self.status_button.setText("6  Check status"); self.fetch_button.setText("7  Fetch results")
            self._configuration_changed()
        self._update_readiness()

    def _sync_mode_changed(self, *_args) -> None:
        if not hasattr(self, "sync_explanation"): return
        copy_all = self.sync_mode.currentData() == "Copy complete training package"
        self.sync_explanation.setText(
            "Recommended: LabelForge creates one self-contained package and transfers the model, labels and training data. "
            "This is easiest and safest, but large datasets take longer to copy."
            if copy_all else
            "Advanced: no training data is copied. Use this only when the same files already exist on the target computer "
            "and the remote paths are known and accessible."
        )
        self._configuration_changed()

    def _set_remote_test_status(self, passed: bool, text: str, state: str | None = None) -> None:
        state = state or ("neutral" if text == "Not tested yet" else ("ready" if passed else "failed"))
        self.remote_test_status.setProperty("state", state)
        marker = {"neutral": "○", "pending": "●", "ready": "✓", "failed": "✕"}[state]
        self.remote_test_status.setText(f"{marker}  {text}")
        self.remote_test_status.style().unpolish(self.remote_test_status); self.remote_test_status.style().polish(self.remote_test_status)

    def _refresh_ssh_agent_status(self) -> bool:
        if not hasattr(self, "ssh_agent_status"): return False
        try:
            result = subprocess.run(["ssh-add", "-l"], capture_output=True, text=True, timeout=6)
            output = (result.stdout + result.stderr).strip()
        except Exception as exc:
            result = None; output = str(exc)
        if result and result.returncode == 0:
            state, text = "ready", "✓  SSH agent ready — an unlocked key is available."
        elif "No such file" in output or "agent" in output.lower() and "connect" in output.lower():
            state, text = "failed", "✕  Windows SSH agent is not running. Enable and unlock it before testing JUSUF."
        else:
            state, text = "pending", "○  SSH agent is running, but no key is unlocked yet."
        self.ssh_agent_status.setProperty("state", state); self.ssh_agent_status.setText(text)
        self.ssh_agent_status.style().unpolish(self.ssh_agent_status); self.ssh_agent_status.style().polish(self.ssh_agent_status)
        return state == "ready"

    def _open_ssh_setup(self) -> None:
        if QMessageBox.question(
            self, "Enable and unlock the SSH key",
            "Windows will request administrator approval to enable the built-in SSH agent. "
            "A PowerShell window will then ask for the key passphrase once.\n\n"
            "The passphrase is kept by Windows SSH Agent for this login session and is never stored by LabelForge."
        ) != QMessageBox.Yes: return
        key = self.identity_file.text().strip().replace("'", "''")
        script = (
            "$ErrorActionPreference='Stop'; "
            "Set-Service -Name ssh-agent -StartupType Automatic; Start-Service ssh-agent; "
            f"Write-Host 'Unlocking SSH key for LabelForge...' -ForegroundColor Cyan; ssh-add '{key}'; "
            "if ($LASTEXITCODE -eq 0) { Write-Host 'SSH key is ready. Return to LabelForge and click Refresh.' -ForegroundColor Green } "
            "else { Write-Host 'The key could not be unlocked.' -ForegroundColor Red }; "
            "Read-Host 'Press Enter to close'"
        )
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        launcher = f"Start-Process powershell.exe -Verb RunAs -ArgumentList '-NoExit','-EncodedCommand','{encoded}'"
        try:
            subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", launcher])
        except Exception as exc:
            QMessageBox.critical(self, "Could not open SSH setup", str(exc))

    def _test_remote(self) -> None:
        profile = self._profile()
        if not profile.host or not profile.user or not profile.identity_file or not profile.root:
            self._set_remote_test_status(False, "Enter host, username, SSH key and remote workspace first")
            return
        if not self._refresh_ssh_agent_status():
            self._set_remote_test_status(False, "Unlock the SSH key in Windows SSH Agent first")
            return
        code = self.totp_code.text().strip()
        if not code and self._needs_totp():
            self._set_remote_test_status(False, "Enter the current JUSUF Google Authenticator code first")
            self.totp_code.setFocus(); return
        self._pending_totp = code
        self._remote_test_passed = False; self.remote_test_button.setEnabled(False)
        label = "Verifying SSH key and the one-time JUSUF code…" if self._needs_totp() else "Verifying SSH connection…"
        self._set_remote_test_status(False, label, "pending")
        try: command = preflight_command(profile, self.backend.currentText())
        except Exception as exc:
            self.remote_test_button.setEnabled(True); self._set_remote_test_status(False, str(exc)); return
        self._run_commands([command], "Testing remote setup")

    def _config(self) -> TrainingBundleConfig:
        return TrainingBundleConfig(
            training_mode=self._training_mode(),
            backend=self.backend.currentText(), local_environment=self.local_environment.currentText(),
            execution_target=self.execution_target.currentText(), sync_mode=self.sync_mode.currentData(),
            parent_model=self.parent_model.text().strip(), training_data=self.training_data.text().strip(),
            labels_or_config=self.labels_config.text().strip(), initialization_video=self.init_video.text().strip(),
            training_script=self.training_script.text().strip(), output_directory=self.output_dir.text().strip(),
            model_name=self.model_name.text().strip(), epochs=self.epochs.value(), batch_size=self.batch.value(),
            learning_rate=self.lr.value(), random_seed=self.seed.value(), remote_host=self.remote_host.text().strip(),
            remote_user=self.remote_user.text().strip(), remote_root=self.remote_root.text().strip(),
            remote_environment=self.remote_environment.text().strip(), slurm_account=self.account.text().strip(),
            slurm_partition=self.partition.text().strip(), walltime=self.walltime.text().strip(),
            gpus=self.gpus.value(), cpus=self.cpus.value(), memory_gb=self.memory.value(),
            qc_enabled=self.qc_enabled.isChecked(), qc_video=self.qc_video.text().strip(),
            qc_focus_label=self.qc_focus_label.currentData() or "", qc_duration_seconds=self.qc_duration.value(),
            qc_zoom_context=float(self.qc_zoom.currentData()),
        )

    def _profile(self) -> RemoteProfile:
        c = self._config()
        return RemoteProfile(
            c.execution_target, c.remote_host, c.remote_user, c.remote_root, c.remote_environment,
            self.identity_file.text().strip(), c.slurm_account, c.slurm_partition,
        )

    def _validate(self) -> bool:
        self._normalize_model_name()
        errors = validate_config(self._config())
        if not self._training_mode(): errors.insert(0, "Choose Create, Refine or Specialize first.")
        if not self.output_dir.text().strip(): errors.append("Choose where the training package should be saved.")
        if self.local_environment.currentText() == "Not installed" and self.execution_target.currentText() == "Local":
            errors.append("No suitable local environment is installed.")
        if errors:
            self._validated = False; self.generate_button.setEnabled(False); self._set_action_stage(1)
            self.log.setPlainText("Not ready yet:\n• " + "\n• ".join(errors)); return False
        self._validated = True; self.generate_button.setEnabled(True); self._set_action_stage(2)
        self.log.setPlainText("✓ Inputs checked. Nothing will be overwritten. Step 2 is now unlocked.")
        return True

    def _directory_size(self, path: Path) -> int:
        if path.is_file(): return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _generate(self) -> None:
        if not self._validate(): return
        config = self._config()
        if config.sync_mode == "Copy complete training package":
            destination = Path(config.output_directory).expanduser() / (safe_name(config.model_name) + "_training_bundle")
            if not confirm(
                self,
                "Build a self-contained training package",
                "LabelForge copies the model, labels and training data into a new portable folder. "
                f"The originals remain unchanged.\n\nPackage location:\n{destination}",
                confirm_text="Build package",
                cancel_text="Not yet",
            ):
                return
        self._start_bundle_worker(config)

    def _start_bundle_worker(self, config: TrainingBundleConfig) -> None:
        self.generate_button.setEnabled(False); self.validate_button.setEnabled(False)
        self._bundle_messages = [
            "Preparing a clean training workspace…",
            "Locking the model recipe and keypoint order…",
            "Gathering model, labels and training data…",
            "Packing the launch files…",
            "Almost ready for transfer…",
        ]
        self._bundle_message_index = 0
        self._bundle_progress = QProgressDialog(self._bundle_messages[0], "", 0, 0, self)
        self._bundle_progress.setWindowTitle("Building training package")
        self._bundle_progress.setCancelButton(None); self._bundle_progress.setWindowModality(Qt.WindowModal)
        self._bundle_progress.setMinimumDuration(0); self._bundle_progress.setMinimumSize(580, 180)
        self._bundle_progress.resize(620, 190); self._bundle_progress.show()
        self._bundle_timer = QTimer(self); self._bundle_timer.timeout.connect(self._advance_bundle_message); self._bundle_timer.start(1100)
        self._bundle_thread = QThread(self); self._bundle_worker = BundleWorker(config)
        self._bundle_worker.moveToThread(self._bundle_thread)
        self._bundle_thread.started.connect(self._bundle_worker.run)
        self._bundle_worker.finished.connect(self._bundle_ready)
        self._bundle_worker.failed.connect(self._bundle_failed)
        self._bundle_worker.finished.connect(self._bundle_thread.quit)
        self._bundle_worker.failed.connect(self._bundle_thread.quit)
        self._bundle_thread.finished.connect(self._bundle_worker.deleteLater)
        self._bundle_thread.start()

    def _advance_bundle_message(self) -> None:
        self._bundle_message_index = (self._bundle_message_index + 1) % len(self._bundle_messages)
        self._bundle_progress.setLabelText(self._bundle_messages[self._bundle_message_index])

    def _finish_bundle_worker(self) -> None:
        self._bundle_timer.stop(); self._bundle_progress.close(); self.validate_button.setEnabled(True)

    def _bundle_ready(self, bundle_path: str) -> None:
        self._finish_bundle_worker(); self._last_bundle = Path(bundle_path)
        remote = self.execution_target.currentText() != "Local"
        self.sync_button.setEnabled(remote and self._remote_test_passed); self.start_button.setEnabled(not remote)
        self.status_button.setEnabled(False); self.fetch_button.setEnabled(False)
        self._set_action_stage(4 if remote and self._remote_test_passed else 3)
        next_step = ("Test the remote setup, then transfer the package" if remote and not self._remote_test_passed else "Transfer the package") if remote else "Start training"
        self.log.append(f"\n✓ Training package created:\n{self._last_bundle}\n\nNext: {next_step}.")

    def _update_elapsed_label(self) -> None:
        if not hasattr(self, "_job_id_label") or not self._last_job_id: return
        import time as _time
        elapsed_sec = int(_time.monotonic() - self._job_submit_time) if self._job_submit_time else 0
        minutes = elapsed_sec // 60
        time_str = f"{minutes} min" if minutes < 60 else f"{minutes // 60}h {minutes % 60}min"
        self._job_id_label.setText(
            f"⏳  Job {self._last_job_id} · {time_str} seit Submit · "
            "Klick '6 Check status' um zu sehen ob er fertig ist, dann '7 Fetch results'."
        )
        self._job_id_label.setVisible(True)

    def _load_existing_bundle_early(self) -> None:
        """Load an existing bundle from the mode card — populates all form fields
        from the manifest and jumps straight to Transfer / Start."""
        # _set() calls below trigger _configuration_changed which resets
        # _remote_test_passed. Save and restore so an already-tested remote
        # stays unlocked after loading a bundle.
        _was_tested = self._remote_test_passed
        selected = QFileDialog.getExistingDirectory(self, "Choose an existing LabelForge training package", "D:\\")
        if not selected:
            return
        bundle = Path(selected)
        required = ["training_manifest.json", "training_entry.py"]
        missing = [n for n in required if not (bundle / n).is_file()]
        if missing:
            QMessageBox.warning(self, "Not a complete training package", "This folder is missing: " + ", ".join(missing))
            return
        try:
            manifest = json.loads((bundle / "training_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Unreadable training package", str(exc))
            return
        backend = str(manifest.get("backend", "Facemap"))
        if backend.lower() == "facemap":
            for name in ["facemap_training_adapter.py"]:
                if not (bundle / name).is_file():
                    QMessageBox.warning(self, "Older incomplete Facemap package",
                        f"Missing {name}. Rebuild with the current LabelForge version.")
                    return

        # ── Set mode ──────────────────────────────────────────────────────────
        mode = str(manifest.get("training_mode", "Create"))
        for btn in self.mode_buttons.values(): btn.setChecked(False)
        if mode in self.mode_buttons: self.mode_buttons[mode].setChecked(True)
        self._mode_changed(mode)

        # ── Populate material fields ───────────────────────────────────────────
        backend_idx = self.backend.findText(backend)
        if backend_idx >= 0: self.backend.setCurrentIndex(backend_idx)

        def _set(field, value):
            if value and hasattr(self, field):
                getattr(self, field).setText(str(value))

        _set("parent_model",  manifest.get("parent_model", ""))
        _set("training_data", manifest.get("training_data", ""))
        _set("labels_config", manifest.get("labels_or_config", ""))
        _set("init_video",    manifest.get("initialization_video", ""))
        _set("model_name",    manifest.get("model_name", ""))
        _set("output_dir",    str(bundle.parent))
        _set("qc_video",      manifest.get("qc_video", ""))
        self.qc_enabled.setChecked(bool(manifest.get("qc_enabled", True)))
        self.qc_duration.setValue(int(manifest.get("qc_duration_seconds", 60)))
        zoom_val = float(manifest.get("qc_zoom_context", 1.0))
        for i in range(self.qc_zoom.count()):
            if abs(float(self.qc_zoom.itemData(i)) - zoom_val) < 0.01:
                self.qc_zoom.setCurrentIndex(i); break
        focus = str(manifest.get("qc_focus_label", ""))
        self._update_focus_dropdown(manifest.get("labels_or_config", ""))
        if focus:
            idx = self.qc_focus_label.findData(focus)
            if idx >= 0: self.qc_focus_label.setCurrentIndex(idx)

        # ── Populate execution fields if present ───────────────────────────────
        target = str(manifest.get("execution_target", "HPC (Slurm)"))
        target_idx = self.execution_target.findText(target)
        if target_idx >= 0:
            self.execution_target.setCurrentIndex(target_idx)
            self._target_changed(target)
        _set("remote_host",        manifest.get("remote_host", ""))
        _set("remote_user",        manifest.get("remote_user", ""))
        _set("remote_root",        manifest.get("remote_root", ""))
        _set("remote_environment", manifest.get("remote_environment", ""))
        _set("account",            manifest.get("slurm_account", ""))
        _set("partition",          manifest.get("slurm_partition", ""))

        # ── Advanced training params ───────────────────────────────────────────
        self.epochs.setValue(int(manifest.get("epochs", 100)))
        self.batch.setValue(int(manifest.get("batch_size", 1)))
        self.lr.setValue(float(manifest.get("learning_rate", 0.00005)))
        self.seed.setValue(int(manifest.get("random_seed", 20260828)))

        # ── Activate bundle and unlock action buttons ──────────────────────────
        self._remote_test_passed = _was_tested  # restore before computing remote_ready
        self._last_bundle = bundle
        self._validated = True
        remote = target != "Local"
        remote_ready = remote and self._remote_test_passed
        self.run_card.setVisible(True); self.execution_card.setVisible(True); self.actions_card.setVisible(True)
        self.generate_button.setEnabled(False)
        self.sync_button.setEnabled(remote_ready)
        self.start_button.setEnabled(not remote or remote_ready)
        self.status_button.setEnabled(remote_ready)
        self.fetch_button.setEnabled(remote_ready)
        self._set_action_stage(4 if remote_ready else 3)
        self.log.append(
            f"\n✓ Bundle geladen:\n{bundle}\n\n"
            f"Weiter mit {'Transfer → Start' if remote else 'Start'}.\n"
            f"Fetched files: {bundle / 'remote_results'}"
        )
        # Scroll to actions card
        QTimer.singleShot(100, lambda: self._workspace_scroll.ensureWidgetVisible(self.actions_card))

    def _load_existing_bundle(self) -> None:
        _was_tested = self._remote_test_passed
        start = self.output_dir.text().strip() if hasattr(self, "output_dir") else ""
        selected = QFileDialog.getExistingDirectory(self, "Choose an existing LabelForge training package", start)
        if not selected:
            return
        bundle = Path(selected)
        required = ["training_manifest.json", "training_entry.py"]
        if self.execution_target.currentText() == "HPC (Slurm)":
            required.append("slurm_job.sh")
        missing = [name for name in required if not (bundle / name).is_file()]
        if missing:
            QMessageBox.warning(
                self,
                "Not a complete training package",
                "This folder is missing: " + ", ".join(missing),
            )
            return
        try:
            manifest = json.loads((bundle / "training_manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Unreadable training package", str(exc))
            return
        backend = str(manifest.get("backend", ""))
        if backend.lower() == "facemap":
            facemap_required = ["facemap_training_adapter.py"]
            if manifest.get("qc_enabled", False):
                facemap_required.append("facemap_qc.py")
            facemap_missing = [name for name in facemap_required if not (bundle / name).is_file()]
            if facemap_missing:
                QMessageBox.warning(self, "Older incomplete Facemap package", "This package cannot run headless because it is missing: " + ", ".join(facemap_missing) + ". Build it once with the current LabelForge version; future runs can then reuse it without rebuilding.")
                return
        backend_index = self.backend.findText(backend)
        if backend_index >= 0:
            self.backend.setCurrentIndex(backend_index)
        self._remote_test_passed = _was_tested  # restore before computing remote_ready
        self._last_bundle = bundle
        remote = self.execution_target.currentText() != "Local"
        remote_ready = remote and self._remote_test_passed
        self.sync_button.setEnabled(remote_ready)
        self.start_button.setEnabled(not remote or remote_ready)
        self.status_button.setEnabled(remote_ready)
        self.fetch_button.setEnabled(remote_ready)
        self._set_action_stage(4 if remote_ready else 3)
        self.log.append(
            f"\n✓ Existing training package opened:\n{bundle}\n\n"
            "Choose Transfer to copy it again, or Start to reuse the package already on the selected target.\n"
            f"Fetched files will appear in:\n{bundle / 'remote_results'}"
        )
    def _bundle_failed(self, message: str) -> None:
        self._finish_bundle_worker(); self.generate_button.setEnabled(True)
        self.log.append(f"\nPackage build stopped: {message}")
        QMessageBox.critical(self, "Could not build the training package", message)

    def _needs_totp(self) -> bool:
        """True when the target host is JUSUF (requires Google Authenticator MFA)."""
        host = self.remote_host.text().strip()
        return "juelich.de" in host

    def _ask_totp(self) -> str:
        """Return the current TOTP code.

        For JUSUF hosts: reads the stored field or opens a popup — returns "" if the user cancels.
        For all other SSH hosts: MFA is not required, returns "" immediately.
        """
        if not self._needs_totp():
            return ""
        code = self.totp_code.text().strip()
        if code:
            return code
        code, ok = QInputDialog.getText(
            self, "JUSUF one-time code",
            "Enter the current Google Authenticator code:",
            QLineEdit.Password,
        )
        entered = code.strip() if ok else ""
        if entered:
            self.totp_code.setText(entered)
        return entered

    def _synchronize(self) -> None:
        if not self._last_bundle: return
        code = self._ask_totp()
        if not code and self._needs_totp():
            self.log.append("\nTransfer needs a fresh JUSUF Google Authenticator code.")
            return
        self._pending_totp = code
        try: commands = sync_commands(self._profile(), self._last_bundle)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot synchronize", str(exc)); return
        self._run_commands(commands, "Synchronizing training bundle")

    def _start_training(self) -> None:
        if not self._last_bundle: return
        if self.execution_target.currentText() == "Local":
            command = [self.conda_path or "conda", "run", "--no-capture-output", "-n", self.local_environment.currentText(), "python", "-u", "training_entry.py"]
            self._run_commands([command], "Running training locally", str(self._last_bundle))
        else:
            code = self._ask_totp()
            if not code and self._needs_totp():
                self.log.append("\nStarting on JUSUF needs a fresh Google Authenticator code.")
                return
            self._pending_totp = code
            try: command = start_command(self._profile(), self._last_bundle)
            except Exception as exc:
                QMessageBox.critical(self, "Cannot start training", str(exc)); return
            self._run_commands([command], "Starting remote training")

    def _check_status(self) -> None:
        code = self._ask_totp()
        if not code and self._needs_totp():
            self.log.append("\nChecking JUSUF needs a fresh Google Authenticator code.")
            return
        self._pending_totp = code
        try: command = status_command(self._profile(), self._last_job_id)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot check status", str(exc)); return
        self._run_commands([command], "Checking remote status")

    def _fetch_results(self) -> None:
        if not self._last_bundle: return
        code = self._ask_totp()
        if not code and self._needs_totp():
            self.log.append("\nFetching from JUSUF needs a fresh Google Authenticator code.")
            return
        self._pending_totp = code
        try: commands = fetch_commands(self._profile(), self._last_bundle)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot fetch results", str(exc)); return
        self._run_commands(commands, "Fetching logs and results")

    def _run_commands(self, commands: list[list[str]], operation: str, working_directory: str | None = None, tolerate_failures: bool = False) -> None:
        if self._process and self._process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Operation running", "Wait for the current operation to finish."); return
        self._command_queue = [(command, working_directory) for command in commands]
        self._operation = operation; self._tolerate_failures = tolerate_failures; self._command_output = ""
        self.log.append(f"\n{operation}…"); self._run_next_command()

    def _run_next_command(self) -> None:
        if not self._command_queue:
            self.log.append(f"{self._operation} complete.")
            if self._operation.startswith("Testing remote"):
                self._remote_test_passed = True; self.remote_test_button.setEnabled(True)
                self._set_remote_test_status(True, "JUSUF SSH, workspace, account and partition are ready")
                self.sync_button.setEnabled(bool(self._last_bundle)); self._set_action_stage(4)
            if self._operation.startswith("Synchronizing"):
                self.start_button.setEnabled(True); self._set_action_stage(5)
            if self._operation.startswith("Starting"):
                match = re.search(r"Submitted batch job\s+(\d+)", self._command_output)
                if match: self._last_job_id = match.group(1)
                if self.execution_target.currentText() == "Local": self._set_action_stage(4)
                else:
                    self.status_button.setEnabled(True); self.fetch_button.setEnabled(True); self._set_action_stage(6)
                    if self._last_job_id and hasattr(self, "_job_id_label"):
                        import time as _time
                        self._job_submit_time = _time.monotonic()
                        self._job_id_label.setVisible(True)
                        self._update_elapsed_label()
                        if self._elapsed_timer is None:
                            self._elapsed_timer = QTimer(self)
                            self._elapsed_timer.timeout.connect(self._update_elapsed_label)
                        self._elapsed_timer.start(30000)
            if self._operation.startswith("Checking"):
                self._set_action_stage(7)
                if hasattr(self, "_job_id_label") and self._last_job_id:
                    out = self._command_output
                    running = bool(re.search(r"\b(RUNNING|PENDING|R\b|PD\b)", out))
                    completed = bool(re.search(r"\bCOMPLETED\b", out))
                    failed = bool(re.search(r"\b(FAILED|TIMEOUT|CANCELLED)\b", out))
                    no_rows = self._last_job_id not in out
                    if running:
                        self._job_id_label.setText(f"⏳  Job {self._last_job_id} is still running — wait before fetching.")
                    elif failed:
                        self._job_id_label.setText(f"✕  Job {self._last_job_id} failed. Fetch logs to see the error.")
                    elif completed or no_rows:
                        self._job_id_label.setText(f"✓  Job {self._last_job_id} finished — ready to fetch results.")
                    self._job_id_label.setVisible(True)
            if self._operation.startswith("Running training locally"):
                self._training_done = True; self._refresh_route_rail()
            if self._operation.startswith("Fetching"):
                self._set_action_stage(8)
                if self._elapsed_timer:
                    self._elapsed_timer.stop()
                if hasattr(self, "_job_id_label") and self._last_job_id:
                    self._job_id_label.setText(f"✓  Job {self._last_job_id} · Ergebnisse wurden abgeholt.")
                if self._last_bundle:
                    self.log.append(f"Results saved locally in: {self._last_bundle / 'remote_results'}")
            return
        command, cwd = self._command_queue.pop(0)
        self._process = QProcess(self); self._process.setProgram(command[0]); self._process.setArguments(command[1:])
        if self._operation.startswith(("Testing remote", "Synchronizing", "Starting remote", "Checking remote", "Fetching")):
            environment = QProcessEnvironment.systemEnvironment()
            askpass = Path(sys.executable).with_name("LabelForgeAskpass.exe")
            if not askpass.is_file():
                self.remote_test_button.setEnabled(True)
                self._set_remote_test_status(False, "LabelForge MFA helper is missing. Reinstall or update LabelForge.")
                self._pending_totp = ""; self.totp_code.clear(); self._command_queue = []
                return
            environment.insert("SSH_ASKPASS", str(askpass))
            environment.insert("SSH_ASKPASS_REQUIRE", "force")
            environment.insert("DISPLAY", "LabelForge")
            environment.insert("LABELFORGE_SSH_ASKPASS", "1")
            environment.insert("LABELFORGE_TOTP", self._pending_totp)
            self._process.setProcessEnvironment(environment)
        if cwd: self._process.setWorkingDirectory(cwd)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_process_output)
        self._process.finished.connect(self._command_finished); self._process.start()
        if self._operation.startswith(("Testing remote", "Synchronizing", "Starting remote", "Checking remote", "Fetching")):
            self._pending_totp = ""; self.totp_code.clear()

    def _read_process_output(self) -> None:
        if not self._process: return
        output = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        self._command_output += output
        if output.strip(): self.log.append(output.rstrip())

    def _command_finished(self, exit_code: int) -> None:
        self._read_process_output()
        if exit_code != 0 and not self._tolerate_failures:
            if self._operation.startswith("Testing remote") and not self._command_output.strip():
                friendly = (
                    "JUSUF verification was not completed. The SSH key may already be valid; "
                    "enter a fresh Google Authenticator code in LabelForge and retry."
                )
            else:
                friendly = self._friendly_remote_error(self._command_output)
            self.log.append(f"{self._operation} stopped:\n{friendly}")
            if self._operation.startswith("Testing remote"):
                self._remote_test_passed = False; self.remote_test_button.setEnabled(True)
                self._set_remote_test_status(False, friendly)
            self._command_queue = []; return
        self._run_next_command()

    def _friendly_remote_error(self, output: str) -> str:
        value = output.lower()
        if "message authentication code incorrect" in value or "corrupted mac" in value:
            return "The secure SSH connection was corrupted. Reconnect the VPN/network, then run 'Test remote setup' again."
        if "connection timed out" in value or "connect to host" in value:
            return "The remote computer could not be reached on SSH port 22. Check VPN, hostname and network access."
        if "permission denied" in value:
            return (
                "JUSUF authentication was not completed. The key can be valid even when this appears; "
                "enter a fresh Google Authenticator code in LabelForge and retry the remote test."
            )
        if "ssh key not found" in value:
            return "The selected SSH key was not found. Choose C:\\Users\\daubenfeld\\.ssh\\id_ed25519 or another valid private key."
        if "user_mismatch" in value:
            return "SSH connected, but the remote username does not match the configured JSC username."
        if "workspace_not_writable" in value:
            return "SSH works, but the LabelForge workspace could not be created or is not writable."
        if "association_missing" in value:
            return "JUSUF is reachable, but the configured Slurm account/partition association was not found. Expected training2636 / batch."
        if "conda_missing" in value:
            return "The remote profile names a Conda environment, but Conda is not available. For JUSUF, use the full venv path instead."
        if "environment_missing" in value:
            return f"JUSUF is ready, but the remote environment '{self.remote_environment.text()}' does not exist yet. It must be created before training."
        if "backend_version_mismatch" in value:
            return f"The remote environment exists, but it does not contain the required Facemap 1.0.8 version."
        if "backend_missing" in value:
            return f"{self.backend.currentText()} is not importable in remote environment '{self.remote_environment.text()}'."
        if "slurm_missing" in value:
            return "Slurm (sbatch) is not available on this remote computer."
        return "The command failed. Review the technical output above, correct the remote setup, and retry."

    def _refresh_software_status(self) -> None:
        self.conda_status.setText(("✓  Ready: " + self.conda_path) if self.conda_path else "✕  Not found")
        self.conda_status.setProperty("ready", bool(self.conda_path)); self.conda_status.style().polish(self.conda_status)
        if not self.conda_path:
            for status in [self.facemap_status, self.dlc_status]:
                status.setText("✕  Conda required"); status.setProperty("ready", False); status.style().polish(status)
            return
        self.backend_environments = discover_backend_environments(self.conda_path)
        fm = self.backend_environments["facemap"]; dlc = self.backend_environments["deeplabcut"]
        self.facemap_status.setText("✓  Installed: " + ", ".join(fm) if fm else "✕  Not installed")
        self.dlc_status.setText("✓  Installed: " + ", ".join(dlc) if dlc else "✕  Not installed")
        for status, ready in [(self.facemap_status, bool(fm)), (self.dlc_status, bool(dlc))]:
            status.setProperty("ready", ready); status.style().unpolish(status); status.style().polish(status)
        self._backend_changed(self.backend.currentText())

    def _install_backend(self, backend: str) -> None:
        if not self.conda_path:
            QMessageBox.warning(self, "Conda not found", "Install Miniconda/Anaconda first."); return
        env = "labelforge-facemap" if backend == "facemap" else "labelforge-dlc"
        package = "git+https://github.com/MouseLand/facemap.git" if backend == "facemap" else "deeplabcut"
        python = "3.10" if backend == "facemap" else "3.12"
        has_gpu = shutil.which("nvidia-smi") is not None
        gpu_note = " with NVIDIA GPU acceleration" if has_gpu else " (CPU only — no NVIDIA GPU detected)"
        if QMessageBox.question(self, "Install software", f"Create/update '{env}' from the official source{gpu_note}?") != QMessageBox.Yes: return
        commands = [
            [self.conda_path, "create", "-n", env, f"python={python}", "pip", "-y"],
        ]
        commands.append([self.conda_path, "run", "-n", env, "python", "-m", "pip", "install", "--upgrade", package])
        if has_gpu and backend == "facemap":
            # Install CUDA-enabled PyTorch AFTER facemap: pip install facemap pulls in CPU-only
            # torch as a dependency and would overwrite a pre-installed CUDA build.  Installing
            # last guarantees the CUDA wheel wins.
            commands.append([
                self.conda_path, "run", "-n", env, "python", "-m", "pip", "install",
                "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124",
            ])
        if backend == "facemap":
            # Facemap 1.0.8 is not compatible with numpy 2.x — pin to the last 1.x release.
            # This runs last so it wins over any numpy 2.x pulled in by torch/torchvision.
            commands.append([
                self.conda_path, "run", "-n", env, "python", "-m", "pip", "install", "numpy<2",
            ])
        self._run_commands(commands, f"Setting up {env}")
