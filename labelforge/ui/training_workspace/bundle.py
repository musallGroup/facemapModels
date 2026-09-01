from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class TrainingBundleConfig:
    training_mode: str
    backend: str
    local_environment: str
    execution_target: str
    sync_mode: str
    parent_model: str
    training_data: str
    labels_or_config: str
    initialization_video: str
    training_script: str
    output_directory: str
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int
    remote_host: str
    remote_user: str
    remote_root: str
    remote_environment: str
    slurm_account: str
    slurm_partition: str
    walltime: str
    gpus: int
    cpus: int
    memory_gb: int
    qc_enabled: bool = True
    qc_video: str = ""
    qc_focus_label: str = ""
    qc_duration_seconds: int = 60
    qc_zoom_context: float = 1.0

def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return cleaned or "training_run"


def validate_config(config: TrainingBundleConfig) -> list[str]:
    errors: list[str] = []
    backend = config.backend.lower()
    if backend not in {"facemap", "deeplabcut"}:
        errors.append("Backend must be Facemap or DeepLabCut.")
    if config.training_mode != "Create" and not Path(config.parent_model).is_file():
        errors.append("Parent model does not exist.")
    if not Path(config.training_data).exists():
        errors.append("Training data path does not exist.")
    if not Path(config.labels_or_config).is_file():
        errors.append("Labels/config file does not exist.")
    if backend == "facemap" and not Path(config.initialization_video).is_file():
        errors.append("Facemap requires an initialization video.")
    if backend == "facemap" and config.training_script and not Path(config.training_script).is_file():
        errors.append("The selected Facemap training adapter does not exist.")
    if backend == "facemap" and config.qc_enabled and config.qc_video and not Path(config.qc_video).is_file():
        errors.append("The selected QC video does not exist.")
    if not config.model_name.strip():
        errors.append("A new model name is required.")
    if config.epochs < 1 or config.batch_size < 1:
        errors.append("Epochs and batch size must be positive.")
    if config.learning_rate <= 0:
        errors.append("Learning rate must be positive.")
    if config.execution_target != "Local" and not config.remote_host.strip():
        errors.append("A remote host is required for remote execution.")
    if config.execution_target != "Local" and not config.remote_user.strip():
        errors.append("A remote username is required for remote execution.")
    if config.execution_target != "Local" and not config.remote_root.strip():
        errors.append("A remote workspace is required for remote execution.")
    if config.execution_target != "Local" and not config.remote_environment.strip():
        errors.append("A remote Python environment name or path is required.")
    if config.execution_target == "HPC (Slurm)" and not config.slurm_account.strip():
        errors.append("A Slurm account/budget is required for HPC execution.")
    if config.execution_target == "HPC (Slurm)" and not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", config.walltime):
        errors.append("Walltime must use HH:MM:SS.")
    return errors


def _environment_text(backend: str) -> str:
    if backend.lower() == "facemap":
        return """name: labelforge-facemap
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - git+https://github.com/MouseLand/facemap.git
      - opencv-python-headless
"""
    return """name: labelforge-dlc
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - deeplabcut
"""


def _runner_text() -> str:
    return r'''from __future__ import annotations

import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "training_manifest.json").read_text(encoding="utf-8"))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_dlc() -> None:
    import deeplabcut
    config_path = CONFIG.get("runtime_labels_or_config", CONFIG["labels_or_config"])
    deeplabcut.train_network(
        config_path,
        shuffle=1,
        trainingsetindex=0,
        maxiters=CONFIG["epochs"],
        allow_growth=True,
    )


def run_facemap() -> None:
    # Facemap projects differ in how their CSV and image tensors are assembled.
    # The bundle deliberately validates all inputs and delegates to the versioned
    # project training script until LabelForge's common Facemap adapter is ready.
    script = ROOT / "facemap_training_adapter.py"
    if not script.exists():
        raise RuntimeError(
            "Facemap bundle is valid, but facemap_training_adapter.py is missing. "
            "Generate/copy the project adapter before submitting this run."
        )
    namespace = {"TRAINING_MANIFEST": CONFIG, "__name__": "__main__"}
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
    if CONFIG.get("qc_enabled", True):
        try:
            qc = ROOT / "facemap_qc.py"
            if not qc.is_file():
                raise RuntimeError("The Facemap QC runner is missing from this package.")
            exec(compile(qc.read_text(encoding="utf-8"), str(qc), "exec"), {"__name__": "__main__"})
        except Exception as exc:
            failure = ROOT / "results" / "qc" / "QC_FAILED.txt"
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(str(exc), encoding="utf-8")
            print(f"Training completed, but QC preview failed: {exc}")

if __name__ == "__main__":
    set_seeds(int(CONFIG["random_seed"]))
    if CONFIG["backend"].lower() == "deeplabcut":
        run_dlc()
    else:
        run_facemap()
'''


def _slurm_text(config: TrainingBundleConfig) -> str:
    environment = config.remote_environment or (
        "labelforge-facemap" if config.backend.lower() == "facemap" else "labelforge-dlc"
    )
    account = f"#SBATCH --account={config.slurm_account}\n" if config.slurm_account else ""
    partition = f"#SBATCH --partition={config.slurm_partition}\n" if config.slurm_partition else ""
    gpu = f"#SBATCH --gres=gpu:{config.gpus}\n" if config.gpus else ""
    if environment.startswith("/"):
        module_setup = (
            "module purge\nmodule load Stages/2025 GCCcore/.13.3.0 PyTorch/2.5.1\n"
            if "juelich.de" in config.remote_host and config.backend.lower() == "facemap" else ""
        )
        launch = f"{shlex.quote(environment.rstrip('/'))}/bin/python training_entry.py"
    else:
        module_setup = "source \"$HOME/.bashrc\" || true\n"
        launch = f"conda run -n {shlex.quote(environment)} python training_entry.py"
    return f"""#!/bin/bash -l
# Generated by LabelForge. Submit with: sbatch slurm_job.sh
#SBATCH --job-name={safe_name(config.model_name)[:80]}
{account}{partition}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={config.cpus}
{gpu}#SBATCH --mem={config.memory_gb}G
#SBATCH --time={config.walltime}
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
mkdir -p logs
{module_setup}{launch}
"""


def create_bundle(config: TrainingBundleConfig) -> Path:
    errors = validate_config(config)
    if errors:
        raise ValueError("\n".join(errors))

    output_root = Path(config.output_directory).expanduser().resolve()
    bundle = output_root / f"{safe_name(config.model_name)}_training_bundle"
    if bundle.exists():
        raise FileExistsError(f"Bundle already exists: {bundle}")
    bundle.mkdir(parents=True)
    (bundle / "logs").mkdir()

    manifest = asdict(config)
    manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["bundle_format"] = 1
    manifest["source_computer"] = os.environ.get("COMPUTERNAME", "unknown")
    if config.sync_mode == "Copy complete training package":
        payload = bundle / "payload"
        payload.mkdir()
        if config.parent_model:
            parent_target = payload / "parent_model" / Path(config.parent_model).name
            parent_target.parent.mkdir()
            shutil.copy2(config.parent_model, parent_target)
            manifest["runtime_parent_model"] = str(parent_target.relative_to(bundle)).replace("\\", "/")
        labels_target = payload / "labels_or_config" / Path(config.labels_or_config).name
        labels_target.parent.mkdir()
        shutil.copy2(config.labels_or_config, labels_target)
        manifest["runtime_labels_or_config"] = str(labels_target.relative_to(bundle)).replace("\\", "/")
        if config.initialization_video:
            video_target = payload / "initialization_video" / Path(config.initialization_video).name
            video_target.parent.mkdir()
            shutil.copy2(config.initialization_video, video_target)
            manifest["runtime_initialization_video"] = str(video_target.relative_to(bundle)).replace("\\", "/")
        if config.qc_video:
            qc_video_target = payload / "qc_video" / Path(config.qc_video).name
            qc_video_target.parent.mkdir()
            shutil.copy2(config.qc_video, qc_video_target)
            manifest["runtime_qc_video"] = str(qc_video_target.relative_to(bundle)).replace("\\", "/")
        data_source = Path(config.training_data)
        data_target = payload / "training_data"
        if data_source.is_dir():
            shutil.copytree(data_source, data_target)
        else:
            data_target.mkdir()
            shutil.copy2(data_source, data_target / data_source.name)
        manifest["runtime_training_data"] = str(data_target.relative_to(bundle)).replace("\\", "/")

    (bundle / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (bundle / "environment.yml").write_text(
        _environment_text(config.backend), encoding="utf-8"
    )
    (bundle / "training_entry.py").write_text(_runner_text(), encoding="utf-8")
    if config.backend.lower() == "facemap":
        adapter = Path(config.training_script) if config.training_script else Path(__file__).with_name("facemap_training_adapter.py")
        if not adapter.is_file():
            raise FileNotFoundError("The built-in Facemap training adapter is missing.")
        shutil.copy2(adapter, bundle / "facemap_training_adapter.py")
        qc_runner = Path(__file__).with_name("facemap_qc.py")
        if not qc_runner.is_file():
            raise FileNotFoundError("The built-in Facemap QC runner is missing.")
        shutil.copy2(qc_runner, bundle / "facemap_qc.py")
    (bundle / "slurm_job.sh").write_text(_slurm_text(config), encoding="utf-8", newline="\n")
    env_name = config.local_environment or (
        "labelforge-facemap" if config.backend.lower() == "facemap" else "labelforge-dlc"
    )
    (bundle / "run_local.bat").write_text(
        f"@echo off\r\nconda run -n {env_name} python training_entry.py\r\n",
        encoding="utf-8",
    )
    (bundle / "README.txt").write_text(
        "LabelForge training bundle\n\n"
        "1. Review training_manifest.json.\n"
        "2. Create/update the environment from environment.yml.\n"
        "3. Local: run run_local.bat.\n"
        "4. Remote workstation: sync the folder and run training_entry.py.\n"
        "5. HPC: sync the folder and submit slurm_job.sh.\n"
        "6. Fetch results and import the completed model as the next version.\n",
        encoding="utf-8",
    )
    return bundle


def find_conda() -> str | None:
    candidates = [
        shutil.which("conda"),
        str(Path.home() / "miniconda3" / "Scripts" / "conda.exe"),
        r"D:\Miniconda\Scripts\conda.exe",
        r"C:\ProgramData\anaconda3\Scripts\conda.exe",
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


def discover_backend_environments(conda_path: str | None) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"facemap": [], "deeplabcut": []}
    if not conda_path:
        return found
    try:
        result = subprocess.run(
            [conda_path, "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        environments = json.loads(result.stdout).get("envs", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return found

    for environment_path in environments:
        path = Path(environment_path)
        python = path / ("python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            continue
        try:
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import importlib.util;"
                    "print(int(importlib.util.find_spec('facemap') is not None),"
                    "int(importlib.util.find_spec('deeplabcut') is not None))",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            facemap, deeplabcut = probe.stdout.strip().split()[-2:]
        except (OSError, subprocess.SubprocessError, ValueError):
            continue
        name = path.name if path != Path(conda_path).parent.parent else "base"
        if facemap == "1":
            found["facemap"].append(name)
        if deeplabcut == "1":
            found["deeplabcut"].append(name)
    return found
