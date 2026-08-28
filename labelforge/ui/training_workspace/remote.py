from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RemoteProfile:
    target: str
    host: str
    user: str
    root: str
    environment: str

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"

    def bundle_path(self, bundle: Path) -> str:
        return f"{self.root.rstrip('/')}/{bundle.name}"


def ssh_executable() -> str | None:
    return shutil.which("ssh")


def scp_executable() -> str | None:
    return shutil.which("scp")


def quote_remote(value: str) -> str:
    return shlex.quote(value)


def sync_commands(profile: RemoteProfile, bundle: Path) -> list[list[str]]:
    ssh = ssh_executable()
    scp = scp_executable()
    if not ssh or not scp:
        raise RuntimeError("Windows OpenSSH (ssh.exe and scp.exe) is required.")
    return [
        [ssh, profile.destination, f"mkdir -p {quote_remote(profile.root)}"],
        [scp, "-r", str(bundle), f"{profile.destination}:{profile.root.rstrip('/')}/"],
    ]


def preflight_command(profile: RemoteProfile, backend: str) -> list[str]:
    ssh = ssh_executable()
    if not ssh:
        raise RuntimeError("Windows OpenSSH (ssh.exe) is required.")
    environment = quote_remote(profile.environment)
    module = "facemap" if backend.lower() == "facemap" else "deeplabcut"
    checks = [
        "printf 'SSH_OK\\n'",
        "command -v conda >/dev/null && printf 'CONDA_OK\\n' || { printf 'CONDA_MISSING\\n'; exit 21; }",
        f"conda run -n {environment} python -c \"import {module}\" && printf 'BACKEND_OK\\n' || {{ printf 'BACKEND_MISSING\\n'; exit 22; }}",
    ]
    if profile.target == "HPC (Slurm)":
        checks.append("command -v sbatch >/dev/null && printf 'SLURM_OK\\n' || { printf 'SLURM_MISSING\\n'; exit 23; }")
    else:
        checks.append("command -v nvidia-smi >/dev/null && printf 'GPU_TOOLS_OK\\n' || printf 'GPU_TOOLS_WARNING\\n'")
    return [ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", profile.destination, " && ".join(checks)]


def start_command(profile: RemoteProfile, bundle: Path) -> list[str]:
    ssh = ssh_executable()
    if not ssh:
        raise RuntimeError("Windows OpenSSH (ssh.exe) is required.")
    remote_bundle = quote_remote(profile.bundle_path(bundle))
    if profile.target == "HPC (Slurm)":
        command = f"cd {remote_bundle} && sbatch slurm_job.sh"
    else:
        environment = quote_remote(profile.environment)
        command = (
            f"cd {remote_bundle} && mkdir -p logs && "
            f"nohup conda run -n {environment} python training_entry.py "
            f"> logs/remote.out 2> logs/remote.err < /dev/null & echo $!"
        )
    return [ssh, profile.destination, command]


def status_command(profile: RemoteProfile, job_id: str = "") -> list[str]:
    ssh = ssh_executable()
    if not ssh:
        raise RuntimeError("Windows OpenSSH (ssh.exe) is required.")
    if profile.target == "HPC (Slurm)":
        selector = f"-j {quote_remote(job_id)}" if job_id else f"-u {quote_remote(profile.user)}"
        command = f"squeue {selector} -o '%.18i %.12T %.24j %.10M %.10l %R'"
    else:
        command = "ps -u \"$USER\" -o pid,stat,etime,cmd | grep training_entry.py | grep -v grep || true"
    return [ssh, profile.destination, command]


def fetch_commands(profile: RemoteProfile, bundle: Path) -> list[list[str]]:
    scp = scp_executable()
    if not scp:
        raise RuntimeError("Windows OpenSSH (scp.exe) is required.")
    destination = bundle / "remote_results"
    destination.mkdir(exist_ok=True)
    remote = profile.bundle_path(bundle)
    return [
        [scp, "-r", f"{profile.destination}:{remote}/logs", str(destination)],
        [scp, "-r", f"{profile.destination}:{remote}/results", str(destination)],
    ]
