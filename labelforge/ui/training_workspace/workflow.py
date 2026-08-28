from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
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
        body = QWidget(); layout = QVBoxLayout(body)
        layout.setContentsMargins(54, 38, 54, 50); layout.setSpacing(18)
        title = QLabel("Training Workspace"); title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Create one reproducible training package, then run it locally, on a remote GPU workstation, "
            "or through Slurm on an HPC system."
        )
        subtitle.setObjectName("PageSubtitle"); subtitle.setWordWrap(True)
        layout.addWidget(title); layout.addWidget(subtitle)
        layout.addWidget(self._software_card()); layout.addWidget(self._run_card())
        layout.addWidget(self._execution_card()); layout.addWidget(self._actions_card())
        layout.addStretch(1); scroll.setWidget(body); outer.addWidget(scroll)

    def _card(self, title: str, text: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame(); card.setObjectName("WizardPanel")
        box = QVBoxLayout(card); box.setContentsMargins(24, 20, 24, 22); box.setSpacing(12)
        heading = QLabel(title); heading.setObjectName("CardTitle")
        hint = QLabel(text); hint.setObjectName("FieldHint"); hint.setWordWrap(True)
        box.addWidget(heading); box.addWidget(hint)
        return card, box

    def _software_card(self) -> QFrame:
        card, box = self._card(
            "1  Software environments",
            "LabelForge detects Facemap and DeepLabCut in every Conda environment. Install only when no suitable environment exists.",
        )
        grid = QGridLayout(); grid.setHorizontalSpacing(16)
        self.conda_status = QLabel(); self.facemap_status = QLabel(); self.dlc_status = QLabel()
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

    def _path_row(self, field: QLineEdit, directory: bool = False, file_filter: str = "All files (*)") -> QWidget:
        row = QWidget(); layout = QHBoxLayout(row); layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(field, 1)
        button = QPushButton("Browse…"); button.setObjectName("BrowseButton")
        button.clicked.connect(lambda: self._browse(field, directory, file_filter))
        layout.addWidget(button)
        return row

    def _run_card(self) -> QFrame:
        card, box = self._card(
            "2  Training package",
            "Choose the promoted parent model and labeled data. The keypoint order remains locked and nothing is overwritten.",
        )
        form = QFormLayout()
        self.backend = QComboBox(); self.backend.addItems(["Facemap", "DeepLabCut"])
        self.backend.currentTextChanged.connect(self._backend_changed)
        self.local_environment = QComboBox()
        self.parent_model = self._line("Parent .pt model")
        self.training_data = self._line("Image folder or DLC project")
        self.labels_config = self._line("LabelForge labels.csv or DLC config.yaml")
        self.init_video = self._line("Facemap initialization video")
        self.training_script = self._line("Versioned Facemap training adapter")
        self.output_dir = self._line("Where the bundle will be created")
        self.model_name = self._line("e.g. SideView_Face_2P_v2")
        form.addRow("Backend", self.backend); form.addRow("Local environment", self.local_environment)
        form.addRow("Parent model", self._path_row(self.parent_model, False, "PyTorch models (*.pt);;All files (*)"))
        form.addRow("Training data", self._path_row(self.training_data, True))
        form.addRow("Labels / config", self._path_row(self.labels_config, False, "CSV/YAML (*.csv *.yaml *.yml);;All files (*)"))
        form.addRow("Initialization video", self._path_row(self.init_video, False, "Videos (*.avi *.mp4 *.mkv *.mov);;All files (*)"))
        form.addRow("Facemap adapter", self._path_row(self.training_script, False, "Python scripts (*.py);;All files (*)"))
        form.addRow("New model name", self.model_name); form.addRow("Bundle location", self._path_row(self.output_dir, True))
        params = QWidget(); row = QHBoxLayout(params); row.setContentsMargins(0, 0, 0, 0)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 10_000_000); self.epochs.setValue(100)
        self.batch = QSpinBox(); self.batch.setRange(1, 4096); self.batch.setValue(1)
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(8); self.lr.setRange(0.00000001, 1.0); self.lr.setValue(0.00005)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(20260828)
        for label, widget in [("Epochs", self.epochs), ("Batch", self.batch), ("LR", self.lr), ("Seed", self.seed)]:
            row.addWidget(QLabel(label)); row.addWidget(widget)
        row.addStretch(1); form.addRow("Parameters", params); box.addLayout(form)
        return card

    def _execution_card(self) -> QFrame:
        card, box = self._card(
            "3  Execution target and synchronization",
            "Local runs directly. Remote targets use SSH/SCP; HPC additionally submits through Slurm. Passwords and browser tokens are never stored.",
        )
        form = QFormLayout()
        self.execution_target = QComboBox(); self.execution_target.addItems(["Local", "Remote workstation", "HPC (Slurm)"])
        self.execution_target.currentTextChanged.connect(self._target_changed)
        self.sync_mode = QComboBox(); self.sync_mode.addItems([
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
        form.addRow("Execution target", self.execution_target); form.addRow("Data strategy", self.sync_mode)
        form.addRow("Host", self.remote_host); form.addRow("Username", self.remote_user)
        form.addRow("Remote workspace", self.remote_root); form.addRow("Remote environment", self.remote_environment)
        form.addRow("Slurm account", self.account); form.addRow("Partition", self.partition)
        form.addRow("Walltime", self.walltime); form.addRow("Resources", resources); box.addLayout(form)
        self._remote_fields = [self.sync_mode, self.remote_host, self.remote_user, self.remote_root, self.remote_environment]
        self._slurm_fields = [self.account, self.partition, self.walltime, self.gpus, self.cpus, self.memory]
        self._target_changed("Local")
        return card

    def _actions_card(self) -> QFrame:
        card, box = self._card(
            "4  Prepare, run and retrieve",
            "Validate and generate an immutable bundle. Then run locally or synchronize, start, monitor and retrieve it remotely.",
        )
        first = QHBoxLayout()
        validate = QPushButton("Validate"); generate = QPushButton("Generate bundle")
        validate.setObjectName("SecondaryActionButton"); generate.setObjectName("PrimaryNextButton")
        validate.clicked.connect(self._validate); generate.clicked.connect(self._generate)
        first.addWidget(validate); first.addWidget(generate); first.addStretch(1)
        second = QHBoxLayout()
        self.sync_button = QPushButton("Synchronize"); self.start_button = QPushButton("Start training")
        self.status_button = QPushButton("Check status"); self.fetch_button = QPushButton("Fetch results")
        for button in [self.sync_button, self.start_button, self.status_button, self.fetch_button]:
            button.setObjectName("SecondaryActionButton"); button.setEnabled(False); second.addWidget(button)
        second.addStretch(1)
        self.sync_button.clicked.connect(self._synchronize); self.start_button.clicked.connect(self._start_training)
        self.status_button.clicked.connect(self._check_status); self.fetch_button.clicked.connect(self._fetch_results)
        box.addLayout(first); box.addLayout(second)
        self.log = QTextEdit(); self.log.setObjectName("TextInput"); self.log.setReadOnly(True)
        self.log.setMaximumHeight(190); self.log.setPlaceholderText("Validation, synchronization and training output appears here.")
        box.addWidget(self.log)
        return card

    def _browse(self, field: QLineEdit, directory: bool, file_filter: str) -> None:
        if directory: path = QFileDialog.getExistingDirectory(self, "Select folder", field.text())
        else: path, _ = QFileDialog.getOpenFileName(self, "Select file", field.text(), file_filter)
        if path: field.setText(path)

    def _backend_changed(self, backend: str) -> None:
        facemap = backend.lower() == "facemap"
        self.init_video.setEnabled(facemap); self.training_script.setEnabled(facemap)
        self.local_environment.clear()
        environments = self.backend_environments.get(backend.lower(), [])
        self.local_environment.addItems(environments or ["Not installed"])
        if hasattr(self, "remote_environment"):
            default = "labelforge-facemap" if facemap else "labelforge-dlc"
            if not self.remote_environment.text() or self.remote_environment.text().startswith("labelforge-"):
                self.remote_environment.setText(default)

    def _target_changed(self, target: str) -> None:
        remote = target != "Local"; slurm = target == "HPC (Slurm)"
        for field in getattr(self, "_remote_fields", []): field.setEnabled(remote)
        for field in getattr(self, "_slurm_fields", []): field.setEnabled(slurm)
        if slurm:
            if not self.remote_host.text(): self.remote_host.setText("jusuf.fz-juelich.de")
            if not self.remote_user.text(): self.remote_user.setText("daubenfeld1")
            if not self.remote_root.text(): self.remote_root.setText("/p/home/jusers/daubenfeld1/jusuf/labelforge-training")
            if not self.partition.text(): self.partition.setText("gpus")
        if hasattr(self, "sync_button"):
            self.sync_button.setVisible(remote); self.status_button.setVisible(remote); self.fetch_button.setVisible(remote)

    def _config(self) -> TrainingBundleConfig:
        return TrainingBundleConfig(
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
        errors = validate_config(self._config())
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
        self.conda_status.setText(self.conda_path or "Not found")
        if not self.conda_path:
            self.facemap_status.setText("Conda required"); self.dlc_status.setText("Conda required"); return
        self.backend_environments = discover_backend_environments(self.conda_path)
        fm = self.backend_environments["facemap"]; dlc = self.backend_environments["deeplabcut"]
        self.facemap_status.setText("Installed: " + ", ".join(fm) if fm else "Not installed")
        self.dlc_status.setText("Installed: " + ", ".join(dlc) if dlc else "Not installed")
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
