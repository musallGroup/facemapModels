from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .bundle import TrainingBundleConfig, create_bundle, find_conda, validate_config


class TrainingWorkspace(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.conda_path = find_conda()
        self.install_process: QProcess | None = None
        self._build_ui()
        self._refresh_software_status()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(54, 38, 54, 50)
        layout.setSpacing(18)

        title = QLabel("Training Workspace")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Prepare one reproducible training run and execute it locally or on JUSUF. "
            "Facemap and DeepLabCut stay in separate environments."
        )
        subtitle.setObjectName("PageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        layout.addWidget(self._software_card())
        layout.addWidget(self._run_card())
        layout.addWidget(self._jusuf_card())
        layout.addWidget(self._actions_card())
        layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    def _card(self, title: str, text: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("WizardPanel")
        box = QVBoxLayout(card)
        box.setContentsMargins(24, 20, 24, 22)
        box.setSpacing(12)
        heading = QLabel(title)
        heading.setObjectName("CardTitle")
        hint = QLabel(text)
        hint.setObjectName("FieldHint")
        hint.setWordWrap(True)
        box.addWidget(heading)
        box.addWidget(hint)
        return card, box

    def _software_card(self) -> QFrame:
        card, box = self._card(
            "1  Software environments",
            "One click creates an isolated environment. Installation uses the official package sources and can take several minutes.",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        self.conda_status = QLabel()
        self.facemap_status = QLabel()
        self.dlc_status = QLabel()
        fm = QPushButton("Install / repair Facemap")
        dlc = QPushButton("Install / repair DeepLabCut")
        fm.setObjectName("SecondaryActionButton")
        dlc.setObjectName("SecondaryActionButton")
        fm.clicked.connect(lambda: self._install_backend("facemap"))
        dlc.clicked.connect(lambda: self._install_backend("deeplabcut"))
        grid.addWidget(QLabel("Conda"), 0, 0)
        grid.addWidget(self.conda_status, 0, 1)
        grid.addWidget(QLabel("Facemap"), 1, 0)
        grid.addWidget(self.facemap_status, 1, 1)
        grid.addWidget(fm, 1, 2)
        grid.addWidget(QLabel("DeepLabCut"), 2, 0)
        grid.addWidget(self.dlc_status, 2, 1)
        grid.addWidget(dlc, 2, 2)
        grid.setColumnStretch(1, 1)
        box.addLayout(grid)
        return card

    def _line(self, placeholder: str = "") -> QLineEdit:
        field = QLineEdit()
        field.setObjectName("TextInput")
        field.setPlaceholderText(placeholder)
        return field

    def _path_row(self, field: QLineEdit, directory: bool = False, file_filter: str = "All files (*)") -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        button = QPushButton("Browse…")
        button.setObjectName("BrowseButton")
        button.clicked.connect(lambda: self._browse(field, directory, file_filter))
        layout.addWidget(button)
        return row

    def _run_card(self) -> QFrame:
        card, box = self._card(
            "2  Training run",
            "Select the promoted parent model and the labeled training data. Nothing is overwritten.",
        )
        form = QFormLayout()
        self.backend = QComboBox()
        self.backend.addItems(["Facemap", "DeepLabCut"])
        self.backend.currentTextChanged.connect(self._backend_changed)
        self.parent_model = self._line("Parent .pt model")
        self.training_data = self._line("Image folder or DLC project")
        self.labels_config = self._line("LabelForge labels.csv or DLC config.yaml")
        self.init_video = self._line("Facemap initialization video")
        self.training_script = self._line("Optional versioned Facemap training script")
        self.output_dir = self._line("Where the bundle will be created")
        self.model_name = self._line("e.g. SideView_Face_2P_v2")
        form.addRow("Backend", self.backend)
        form.addRow("Parent model", self._path_row(self.parent_model, False, "PyTorch models (*.pt);;All files (*)"))
        form.addRow("Training data", self._path_row(self.training_data, True))
        form.addRow("Labels / config", self._path_row(self.labels_config, False, "CSV/YAML (*.csv *.yaml *.yml);;All files (*)"))
        form.addRow("Initialization video", self._path_row(self.init_video, False, "Videos (*.avi *.mp4 *.mkv *.mov);;All files (*)"))
        form.addRow("Facemap adapter", self._path_row(self.training_script, False, "Python scripts (*.py);;All files (*)"))
        form.addRow("New model name", self.model_name)
        form.addRow("Bundle location", self._path_row(self.output_dir, True))

        params = QWidget()
        params_layout = QHBoxLayout(params)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 10_000_000); self.epochs.setValue(100)
        self.batch = QSpinBox(); self.batch.setRange(1, 4096); self.batch.setValue(1)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(8); self.lr.setRange(0.00000001, 1.0); self.lr.setValue(0.00005)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(20260828)
        for label, widget in [("Epochs", self.epochs), ("Batch", self.batch), ("LR", self.lr), ("Seed", self.seed)]:
            params_layout.addWidget(QLabel(label)); params_layout.addWidget(widget)
        params_layout.addStretch(1)
        form.addRow("Parameters", params)
        box.addLayout(form)
        return card

    def _jusuf_card(self) -> QFrame:
        card, box = self._card(
            "3  JUSUF profile",
            "The bundle uses SSH/SFTP and Slurm—not browser automation. Passwords and browser tokens are never stored.",
        )
        form = QFormLayout()
        self.remote_host = self._line(); self.remote_host.setText("jusuf.fz-juelich.de")
        self.remote_user = self._line(); self.remote_user.setText("daubenfeld1")
        self.remote_root = self._line(); self.remote_root.setText("/p/home/jusers/daubenfeld1/jusuf/labelforge-training")
        self.account = self._line("Required Slurm budget/account")
        self.partition = self._line(); self.partition.setText("gpus")
        self.walltime = self._line(); self.walltime.setText("04:00:00")
        resources = QWidget(); row = QHBoxLayout(resources); row.setContentsMargins(0,0,0,0)
        self.gpus = QSpinBox(); self.gpus.setRange(0, 16); self.gpus.setValue(1)
        self.cpus = QSpinBox(); self.cpus.setRange(1, 256); self.cpus.setValue(8)
        self.memory = QSpinBox(); self.memory.setRange(1, 2048); self.memory.setValue(64)
        for label, widget in [("GPUs", self.gpus), ("CPUs", self.cpus), ("RAM GB", self.memory)]:
            row.addWidget(QLabel(label)); row.addWidget(widget)
        row.addStretch(1)
        form.addRow("Host", self.remote_host)
        form.addRow("Username", self.remote_user)
        form.addRow("Remote workspace", self.remote_root)
        form.addRow("Slurm account", self.account)
        form.addRow("Partition", self.partition)
        form.addRow("Walltime", self.walltime)
        form.addRow("Resources", resources)
        box.addLayout(form)
        return card

    def _actions_card(self) -> QFrame:
        card, box = self._card(
            "4  Validate and prepare",
            "Generate first, inspect the manifest, then run locally or transfer the same folder to JUSUF.",
        )
        buttons = QHBoxLayout()
        validate = QPushButton("Validate")
        generate = QPushButton("Generate training bundle")
        validate.setObjectName("SecondaryActionButton")
        generate.setObjectName("PrimaryNextButton")
        validate.clicked.connect(self._validate)
        generate.clicked.connect(self._generate)
        buttons.addWidget(validate)
        buttons.addWidget(generate)
        buttons.addStretch(1)
        box.addLayout(buttons)
        self.log = QTextEdit()
        self.log.setObjectName("TextInput")
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setPlaceholderText("Validation and setup output appears here.")
        box.addWidget(self.log)
        return card

    def _browse(self, field: QLineEdit, directory: bool, file_filter: str) -> None:
        if directory:
            path = QFileDialog.getExistingDirectory(self, "Select folder", field.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", field.text(), file_filter)
        if path:
            field.setText(path)

    def _backend_changed(self, backend: str) -> None:
        self.init_video.setEnabled(backend.lower() == "facemap")
        self.training_script.setEnabled(backend.lower() == "facemap")

    def _config(self) -> TrainingBundleConfig:
        return TrainingBundleConfig(
            backend=self.backend.currentText(), parent_model=self.parent_model.text().strip(),
            training_data=self.training_data.text().strip(), labels_or_config=self.labels_config.text().strip(),
            initialization_video=self.init_video.text().strip(), output_directory=self.output_dir.text().strip(),
            training_script=self.training_script.text().strip(),
            model_name=self.model_name.text().strip(), epochs=self.epochs.value(), batch_size=self.batch.value(),
            learning_rate=self.lr.value(), random_seed=self.seed.value(), remote_host=self.remote_host.text().strip(),
            remote_user=self.remote_user.text().strip(), remote_root=self.remote_root.text().strip(),
            slurm_account=self.account.text().strip(), slurm_partition=self.partition.text().strip(),
            walltime=self.walltime.text().strip(), gpus=self.gpus.value(), cpus=self.cpus.value(),
            memory_gb=self.memory.value(),
        )

    def _validate(self) -> bool:
        errors = validate_config(self._config())
        if not self.output_dir.text().strip(): errors.append("Bundle location is required.")
        if errors:
            self.log.setPlainText("Validation failed:\n• " + "\n• ".join(errors))
            return False
        self.log.setPlainText("Validation passed. Inputs exist and no output has been changed.")
        return True

    def _generate(self) -> None:
        if not self._validate(): return
        try:
            bundle = create_bundle(self._config())
        except Exception as exc:
            QMessageBox.critical(self, "Could not create bundle", str(exc))
            return
        self.log.append(f"\nBundle created:\n{bundle}")
        QMessageBox.information(self, "Training bundle ready", f"Created:\n{bundle}")

    def _refresh_software_status(self) -> None:
        self.conda_status.setText(self.conda_path or "Not found")
        if not self.conda_path:
            self.facemap_status.setText("Conda required")
            self.dlc_status.setText("Conda required")
            return
        for name, label in [("labelforge-facemap", self.facemap_status), ("labelforge-dlc", self.dlc_status)]:
            result = subprocess.run([self.conda_path, "env", "list"], capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            label.setText("Installed" if name in result.stdout else "Not installed")

    def _install_backend(self, backend: str) -> None:
        if not self.conda_path:
            QMessageBox.warning(self, "Conda not found", "Install Miniconda/Anaconda first or make conda available.")
            return
        env = "labelforge-facemap" if backend == "facemap" else "labelforge-dlc"
        package = "git+https://github.com/MouseLand/facemap.git" if backend == "facemap" else "deeplabcut"
        python = "3.10" if backend == "facemap" else "3.12"
        answer = QMessageBox.question(self, "Install software", f"Create/update the isolated environment '{env}'?\n\nThis downloads software from the official package source.")
        if answer != QMessageBox.Yes: return
        self.log.setPlainText(f"Setting up {env}…")
        commands = [
            [self.conda_path, "create", "-n", env, f"python={python}", "pip", "-y"],
            [self.conda_path, "run", "-n", env, "python", "-m", "pip", "install", "--upgrade", package],
        ]
        # Keep the UI responsive by chaining QProcess commands.
        self._install_commands = commands
        self._run_next_install_command()

    def _run_next_install_command(self) -> None:
        if not self._install_commands:
            self.log.append("\nEnvironment setup complete.")
            self._refresh_software_status()
            return
        command = self._install_commands.pop(0)
        self.install_process = QProcess(self)
        self.install_process.setProgram(command[0])
        self.install_process.setArguments(command[1:])
        self.install_process.setProcessChannelMode(QProcess.MergedChannels)
        self.install_process.readyReadStandardOutput.connect(
            lambda: self.log.append(bytes(self.install_process.readAllStandardOutput()).decode(errors="replace"))
        )
        self.install_process.finished.connect(self._install_finished)
        self.install_process.start()

    def _install_finished(self, exit_code: int) -> None:
        if exit_code != 0:
            self.log.append(f"\nSetup stopped with exit code {exit_code}.")
            self._install_commands = []
            return
        self._run_next_install_command()
