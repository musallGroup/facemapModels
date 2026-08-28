from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from .bundle import (
    TrainingBundleConfig, create_bundle, discover_backend_environments,
    find_conda, validate_config,
)
from .remote import (
    RemoteProfile, fetch_commands, start_command, status_command, sync_commands,
)
from .naming import ensure_v1, next_refinement_name


class ClearComboBox(QComboBox):
    """Theme-independent combo box with an unmistakable dropdown marker."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QColor("#eaeaea") if self.isEnabled() else QColor("#737984"))
        painter.drawText(self.width() - 29, 0, 28, self.height(), Qt.AlignCenter, "▼")


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
        self._build_ui()
        self._refresh_software_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(); body_layout = QHBoxLayout(body); body_layout.setContentsMargins(28, 30, 28, 42)
        content = QWidget(); content.setMaximumWidth(1180)
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
        layout.addStretch(1); body_layout.addStretch(1); body_layout.addWidget(content, 1); body_layout.addStretch(1)
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
        form = QFormLayout()
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
        form.addRow(self.parent_label, self.parent_row)
        training_control = self._path_row(self.training_data, True, title="Choose the image folder (Facemap) or DLC project folder")
        form.addRow("Training data", self._with_hint(training_control, "Facemap: folder containing the training images.  DeepLabCut: project folder containing config.yaml."))
        labels_control = self._path_row(self.labels_config, False, "Label/config (*.csv *.yaml *.yml)", "Choose labels.csv (Facemap) or config.yaml (DeepLabCut)")
        form.addRow("Labels / config", self._with_hint(labels_control, "Facemap: LabelForge labels file (*.csv).  DeepLabCut: project configuration (*.yaml or *.yml)."))
        form.addRow("New model name", self.model_name)
        package_control = self._path_row(self.output_dir, True, title="Choose where the new training package should be saved")
        form.addRow("Save training package in", self._with_hint(package_control, "LabelForge creates a new self-contained folder here. Your original model and labels are not changed."))

        self.advanced_training_button = QPushButton("Advanced training settings  ▸")
        self.advanced_training_button.setObjectName("AdvancedToggle"); self.advanced_training_button.setCheckable(True)
        self.advanced_training_button.clicked.connect(self._toggle_training_advanced)
        self.advanced_training_panel = QFrame(); self.advanced_training_panel.setObjectName("AdvancedPanel")
        advanced = QFormLayout(self.advanced_training_panel)
        advanced.addRow("Local environment", self.local_environment)
        advanced.addRow("Initialization video", self._path_row(self.init_video, False, "Videos (*.avi *.mp4 *.mkv *.mov);;All files (*)"))
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
        for field in [self.parent_model, self.training_data, self.labels_config, self.output_dir, self.model_name]:
            field.textChanged.connect(self._update_readiness)
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
        self.sync_mode = ClearComboBox(); self.sync_mode.addItems([
            "Reuse paths already available on target", "Copy complete training package"
        ])
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
        form.addRow("Data strategy", self.sync_mode)
        form.addRow("Host", self.remote_host); form.addRow("Username", self.remote_user)
        form.addRow("Remote workspace", self.remote_root); form.addRow("Remote environment", self.remote_environment)
        box.addLayout(form)
        self.advanced_target_button = QPushButton("Advanced target settings  ▸")
        self.advanced_target_button.setObjectName("AdvancedToggle"); self.advanced_target_button.setCheckable(True)
        self.advanced_target_button.clicked.connect(self._toggle_target_advanced)
        self.advanced_target_panel = QFrame(); self.advanced_target_panel.setObjectName("AdvancedPanel")
        target_advanced = QFormLayout(self.advanced_target_panel)
        target_advanced.addRow("Slurm account", self.account); target_advanced.addRow("Partition", self.partition)
        target_advanced.addRow("Walltime", self.walltime); target_advanced.addRow("Resources", resources)
        self.advanced_target_panel.setVisible(False)
        box.addWidget(self.advanced_target_button); box.addWidget(self.advanced_target_panel)
        self._remote_fields = [self.sync_mode, self.remote_host, self.remote_user, self.remote_root, self.remote_environment]
        self._slurm_fields = [self.account, self.partition, self.walltime, self.gpus, self.cpus, self.memory]
        self._target_changed("Local")
        return card

    def _actions_card(self) -> QFrame:
        card, box = self._card(
            "5  Ready to train",
            "Follow the buttons from left to right. Only the next available action is enabled.",
        )
        self.readiness = QLabel(); self.readiness.setObjectName("ReadinessChecklist"); self.readiness.setWordWrap(True)
        box.addWidget(self.readiness)
        first = QHBoxLayout()
        validate = QPushButton("Validate"); generate = QPushButton("Generate bundle")
        validate.setObjectName("SecondaryActionButton"); generate.setObjectName("PrimaryNextButton")
        validate.clicked.connect(self._validate); generate.clicked.connect(self._generate)
        first.addWidget(validate); first.addWidget(generate); first.addStretch(1)
        first_help = QLabel("1. Check inputs     2. Create a portable training folder")
        first_help.setObjectName("InlineHint")
        second = QHBoxLayout()
        self.sync_button = QPushButton("Synchronize"); self.start_button = QPushButton("Start training")
        self.status_button = QPushButton("Check status"); self.fetch_button = QPushButton("Fetch results")
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
        self.init_video.setEnabled(facemap); self.training_script.setEnabled(facemap)
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
        if hasattr(self, "advanced_target_button"):
            self.advanced_target_button.setVisible(slurm)
            if not slurm:
                self.advanced_target_button.setChecked(False); self._toggle_target_advanced(False)
        if slurm:
            if not self.remote_host.text(): self.remote_host.setText("jusuf.fz-juelich.de")
            if not self.remote_user.text(): self.remote_user.setText("daubenfeld1")
            if not self.remote_root.text(): self.remote_root.setText("/p/home/jusers/daubenfeld1/jusuf/labelforge-training")
            if not self.partition.text(): self.partition.setText("gpus")
        if hasattr(self, "sync_button"):
            self.sync_button.setVisible(remote); self.status_button.setVisible(remote); self.fetch_button.setVisible(remote)
        self._update_readiness()

    def _config(self) -> TrainingBundleConfig:
        return TrainingBundleConfig(
            training_mode=self._training_mode(),
            backend=self.backend.currentText(), local_environment=self.local_environment.currentText(),
            execution_target=self.execution_target.currentText(), sync_mode=self.sync_mode.currentText(),
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
        if not self.output_dir.text().strip(): errors.append("Bundle location is required.")
        if self.local_environment.currentText() == "Not installed" and self.execution_target.currentText() == "Local":
            errors.append("No suitable local environment is installed.")
        if errors:
            self.log.setPlainText("Validation failed:\n• " + "\n• ".join(errors)); return False
        self.log.setPlainText("Validation passed. Inputs exist and no existing model will be changed.")
        return True

    def _directory_size(self, path: Path) -> int:
        if path.is_file(): return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def _generate(self) -> None:
        if not self._validate(): return
        config = self._config()
        if config.sync_mode == "Copy complete training package":
            size_gb = self._directory_size(Path(config.training_data)) / (1024 ** 3)
            if QMessageBox.question(
                self, "Copy training package", f"Training data: approximately {size_gb:.2f} GB.\n\nCreate a self-contained copy?"
            ) != QMessageBox.Yes: return
        try: self._last_bundle = create_bundle(config)
        except Exception as exc:
            QMessageBox.critical(self, "Could not create bundle", str(exc)); return
        remote = config.execution_target != "Local"
        self.sync_button.setEnabled(remote); self.start_button.setEnabled(not remote)
        self.status_button.setEnabled(False); self.fetch_button.setEnabled(False)
        self.log.append(f"\nBundle created:\n{self._last_bundle}")

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
            if self._operation.startswith("Synchronizing"): self.start_button.setEnabled(True)
            if self._operation.startswith("Starting"):
                match = re.search(r"Submitted batch job\s+(\d+)", self._command_output)
                if match: self._last_job_id = match.group(1)
                self.status_button.setEnabled(True); self.fetch_button.setEnabled(True)
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
            self.log.append(f"{self._operation} stopped with exit code {exit_code}."); self._command_queue = []; return
        self._run_next_command()

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
