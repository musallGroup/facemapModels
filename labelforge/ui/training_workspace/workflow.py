from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QProgressDialog, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from .bundle import (
    TrainingBundleConfig, create_bundle, discover_backend_environments,
    find_conda, validate_config,
)
from .remote import (
    RemoteProfile, fetch_commands, preflight_command, start_command, status_command, sync_commands,
)
from .naming import ensure_v1, next_refinement_name


class ClearComboBox(QComboBox):
    """Theme-independent combo box with an unmistakable dropdown marker."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QColor("#eaeaea") if self.isEnabled() else QColor("#737984"))
        painter.drawText(self.width() - 29, 0, 28, self.height(), Qt.AlignCenter, "▼")


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
        self._validated = False
        self._remote_test_passed = False
        self._bundle_thread: QThread | None = None
        self._bundle_worker: BundleWorker | None = None
        self._build_ui()
        self._refresh_software_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(); body_layout = QHBoxLayout(body); body_layout.setContentsMargins(28, 30, 28, 42)
        content = QWidget(); content.setMaximumWidth(1700)
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
        layout.addStretch(1); body_layout.addWidget(content, 1, Qt.AlignHCenter)
        scroll.setWidget(body); outer.addWidget(scroll)
        self.run_card.setVisible(False); self.execution_card.setVisible(False); self.actions_card.setVisible(False)

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
            "2  Software check",
            "LabelForge detects Facemap and DeepLabCut in every Conda environment. Install only when no suitable environment exists.",
        )
        grid = QGridLayout(); grid.setHorizontalSpacing(16)
        self.conda_status = QLabel(); self.facemap_status = QLabel(); self.dlc_status = QLabel()
        for status in [self.conda_status, self.facemap_status, self.dlc_status]: status.setObjectName("SoftwareStatus")
        fm = QPushButton("Install / repair Facemap"); dlc = QPushButton("Install / repair DeepLabCut")
        fm.setObjectName("SecondaryActionButton"); dlc.setObjectName("SecondaryActionButton")
        fm.clicked.connect(lambda: self._install_backend("facemap"))
        dlc.clicked.connect(lambda: self._install_backend("deeplabcut"))
        grid.addWidget(QLabel("Conda"), 0, 0); grid.addWidget(self.conda_status, 0, 1)
        grid.addWidget(QLabel("Facemap"), 1, 0); grid.addWidget(self.facemap_status, 1, 1); grid.addWidget(fm, 1, 2)
        grid.addWidget(QLabel("DeepLabCut"), 2, 0); grid.addWidget(self.dlc_status, 2, 1); grid.addWidget(dlc, 2, 2)
        grid.setColumnStretch(1, 1); box.addLayout(grid)
        return card

    def _line(self, placeholder: str = "") -> QLineEdit:
        field = QLineEdit(); field.setObjectName("TextInput"); field.setPlaceholderText(placeholder)
        return field

    def _path_row(self, field: QLineEdit, directory: bool = False, file_filter: str = "All files (*)", title: str = "Select file") -> QWidget:
        row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        button = QPushButton("Browse…"); button.setObjectName("BrowseButton")
        button.clicked.connect(lambda: self._browse(field, directory, file_filter, title))
        layout.addWidget(button)
        return row

    def _with_hint(self, control: QWidget, text: str) -> QWidget:
        wrapper = QWidget(); layout = QVBoxLayout(wrapper); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(3)
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
        form.addRow("Save training package in", self._with_hint(package_control, "LabelForge creates a new self-contained folder here. Your original model and labels are not changed."))
        self.init_video_control = self._path_row(self.init_video, False, "Videos (*.avi *.mp4 *.mkv *.mov)", "Choose the Facemap initialization video")
        self.init_video_row = self._with_hint(self.init_video_control, "Required by Facemap to initialize the model (*.avi, *.mp4, *.mkv or *.mov).")
        form.addRow("Initialization video", self.init_video_row)

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
        for field in [self.parent_model, self.training_data, self.labels_config, self.output_dir, self.model_name, self.init_video]:
            field.textChanged.connect(self._configuration_changed)
        return card

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
        self.remote_root = self._line("Remote training workspace")
        self.remote_environment = self._line("Remote Conda environment")
        self.account = self._line("Slurm budget/account"); self.partition = self._line("Slurm partition")
        self.walltime = self._line(); self.walltime.setText("04:00:00")
        resources = QWidget(); row = QHBoxLayout(resources); row.setContentsMargins(0, 0, 0, 0)
        self.gpus = QSpinBox(); self.gpus.setRange(0, 16); self.gpus.setValue(1)
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
        test_row = QHBoxLayout(); test_row.addWidget(self.remote_test_button); test_row.addWidget(self.remote_test_status, 1)
        box.addLayout(test_row)
        self._remote_fields = [self.sync_mode, self.sync_explanation, self.remote_host, self.remote_user, self.remote_root, self.remote_environment]
        self._slurm_fields = [self.account, self.partition, self.walltime, self.gpus, self.cpus, self.memory]
        for field in [self.remote_host, self.remote_user, self.remote_root, self.remote_environment, self.account]:
            field.textChanged.connect(self._configuration_changed)
        self._target_changed("Local")
        self._sync_mode_changed()
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
        self.validate_button.setObjectName("SecondaryActionButton"); self.generate_button.setObjectName("PrimaryNextButton")
        self.validate_button.clicked.connect(self._validate); self.generate_button.clicked.connect(self._generate)
        self.generate_button.setEnabled(False)
        first.addWidget(self.validate_button); first.addWidget(self.generate_button); first.addStretch(1)
        first_help = QLabel("Check inputs first. When everything is valid, package creation unlocks automatically.")
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
        box.addLayout(first); box.addWidget(first_help); box.addSpacing(5); box.addLayout(second); box.addWidget(action_help)
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

    def _backend_changed(self, backend: str) -> None:
        facemap = backend.lower() == "facemap"
        if hasattr(self, "training_data"):
            self.training_data.setPlaceholderText(
                "Image folder (*.png, *.jpg, *.jpeg, *.tif)" if facemap else "DeepLabCut project folder (contains config.yaml)"
            )
            self.labels_config.setPlaceholderText("LabelForge labels (*.csv)" if facemap else "DLC configuration (*.yaml, *.yml)")
        self.init_video_row.setVisible(facemap); self.training_script.setEnabled(facemap)
        if self.training_form.labelForField(self.init_video_row):
            self.training_form.labelForField(self.init_video_row).setVisible(facemap)
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
        if slurm:
            if not self.remote_host.text(): self.remote_host.setText("jusuf.fz-juelich.de")
            if not self.remote_user.text(): self.remote_user.setText("daubenfeld1")
            if not self.remote_root.text(): self.remote_root.setText("/p/home/jusers/daubenfeld1/jusuf/labelforge-training")
            if not self.partition.text(): self.partition.setText("gpus")
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

    def _set_remote_test_status(self, passed: bool, text: str) -> None:
        self.remote_test_status.setProperty("ready", passed)
        self.remote_test_status.setText(("✓  " if passed else "✕  ") + text if text != "Not tested yet" else "○  Not tested yet")
        self.remote_test_status.style().unpolish(self.remote_test_status); self.remote_test_status.style().polish(self.remote_test_status)

    def _test_remote(self) -> None:
        profile = self._profile()
        if not profile.host or not profile.user or not profile.environment:
            self._set_remote_test_status(False, "Enter host, username and remote environment first")
            return
        self._remote_test_passed = False; self.remote_test_button.setEnabled(False)
        self._set_remote_test_status(False, "Testing SSH, Conda and training software…")
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
        )

    def _profile(self) -> RemoteProfile:
        c = self._config()
        return RemoteProfile(c.execution_target, c.remote_host, c.remote_user, c.remote_root, c.remote_environment)

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
            size_gb = self._directory_size(Path(config.training_data)) / (1024 ** 3)
            dialog = QMessageBox(self)
            dialog.setIcon(QMessageBox.Question); dialog.setWindowTitle("Build a self-contained training package")
            dialog.setText("Ready to assemble the training package?")
            dialog.setInformativeText(
                f"Training data size: approximately {size_gb:.2f} GB.\n\n"
                "LabelForge will copy the model, labels and training data into a new portable folder. "
                "The originals remain unchanged."
            )
            dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No); dialog.setDefaultButton(QMessageBox.Yes)
            dialog.button(QMessageBox.Yes).setText("Build package"); dialog.button(QMessageBox.No).setText("Not yet")
            if dialog.exec() != QMessageBox.Yes: return
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
        self._bundle_progress.setMinimumDuration(0); self._bundle_progress.setMinimumWidth(470); self._bundle_progress.show()
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

    def _bundle_failed(self, message: str) -> None:
        self._finish_bundle_worker(); self.generate_button.setEnabled(True)
        self.log.append(f"\nPackage build stopped: {message}")
        QMessageBox.critical(self, "Could not build the training package", message)

    def _synchronize(self) -> None:
        if not self._last_bundle: return
        try: commands = sync_commands(self._profile(), self._last_bundle)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot synchronize", str(exc)); return
        self._run_commands(commands, "Synchronizing training bundle")

    def _start_training(self) -> None:
        if not self._last_bundle: return
        if self.execution_target.currentText() == "Local":
            command = [self.conda_path or "conda", "run", "-n", self.local_environment.currentText(), "python", "training_entry.py"]
            self._run_commands([command], "Running training locally", str(self._last_bundle))
        else:
            try: command = start_command(self._profile(), self._last_bundle)
            except Exception as exc:
                QMessageBox.critical(self, "Cannot start training", str(exc)); return
            self._run_commands([command], "Starting remote training")

    def _check_status(self) -> None:
        try: command = status_command(self._profile(), self._last_job_id)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot check status", str(exc)); return
        self._run_commands([command], "Checking remote status")

    def _fetch_results(self) -> None:
        if not self._last_bundle: return
        try: commands = fetch_commands(self._profile(), self._last_bundle)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot fetch results", str(exc)); return
        self._run_commands(commands, "Fetching logs and results", tolerate_failures=True)

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
                self._set_remote_test_status(True, "SSH, Conda, backend and scheduler are ready")
                self.sync_button.setEnabled(bool(self._last_bundle)); self._set_action_stage(4)
            if self._operation.startswith("Synchronizing"):
                self.start_button.setEnabled(True); self._set_action_stage(5)
            if self._operation.startswith("Starting"):
                match = re.search(r"Submitted batch job\s+(\d+)", self._command_output)
                if match: self._last_job_id = match.group(1)
                if self.execution_target.currentText() == "Local": self._set_action_stage(4)
                else:
                    self.status_button.setEnabled(True); self.fetch_button.setEnabled(True); self._set_action_stage(6)
            if self._operation.startswith("Checking"): self._set_action_stage(7)
            if self._operation.startswith("Fetching"): self._set_action_stage(8)
            return
        command, cwd = self._command_queue.pop(0)
        self._process = QProcess(self); self._process.setProgram(command[0]); self._process.setArguments(command[1:])
        if cwd: self._process.setWorkingDirectory(cwd)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_process_output)
        self._process.finished.connect(self._command_finished); self._process.start()

    def _read_process_output(self) -> None:
        if not self._process: return
        output = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        self._command_output += output
        if output.strip(): self.log.append(output.rstrip())

    def _command_finished(self, exit_code: int) -> None:
        self._read_process_output()
        if exit_code != 0 and not self._tolerate_failures:
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
            return "SSH authentication failed. LabelForge needs a working SSH key or Windows SSH-agent login for this host."
        if "conda_missing" in value:
            return "Conda is not available in the remote non-interactive shell. Add it to the remote shell PATH."
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
        if QMessageBox.question(self, "Install software", f"Create/update '{env}' from the official source?") != QMessageBox.Yes: return
        commands = [
            [self.conda_path, "create", "-n", env, f"python={python}", "pip", "-y"],
            [self.conda_path, "run", "-n", env, "python", "-m", "pip", "install", "--upgrade", package],
        ]
        self._run_commands(commands, f"Setting up {env}")
