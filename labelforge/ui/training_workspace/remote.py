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
    identity_file: str = ""
    slurm_account: str = ""
    slurm_partition: str = ""

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


JUSUF_MAC = "hmac-sha2-256-etm@openssh.com"
JUSUF_FACEMAP_MODULES = "module purge && module load Stages/2025 GCCcore/.13.3.0 PyTorch/2.5.1"


def remote_python(profile: RemoteProfile, backend: str = "") -> tuple[str, str]:
    """Return setup commands and the Python launcher for a remote environment.

    Absolute paths name a venv. Environment names retain the portable Conda
    behavior used by remote workstations and older LabelForge profiles.
    """
    environment = profile.environment.strip()
    if environment.startswith("/"):
        python = f"{quote_remote(environment.rstrip('/'))}/bin/python"
        setup = JUSUF_FACEMAP_MODULES if "juelich.de" in profile.host and backend.lower() == "facemap" else ":"
        return setup, python
    return ":", f"conda run -n {quote_remote(environment)} python"


def ssh_transport_options(profile: RemoteProfile) -> list[str]:
    identity = Path(profile.identity_file)
    if not profile.identity_file or not identity.is_file():
        raise RuntimeError(f"SSH key not found: {profile.identity_file or 'no key selected'}")
    return ["-m", JUSUF_MAC, "-i", str(identity), "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def interactive_ssh_transport_options(profile: RemoteProfile) -> list[str]:
    """SSH options for the user-attended JUSUF MFA preflight only."""
    identity = Path(profile.identity_file)
    if not profile.identity_file or not identity.is_file():
        raise RuntimeError(f"SSH key not found: {profile.identity_file or 'no key selected'}")
    return [
        "-m", JUSUF_MAC, "-i", str(identity),
        "-o", "BatchMode=no",
        "-o", "PreferredAuthentications=publickey,keyboard-interactive",
        "-o", "ConnectTimeout=12",
    ]


def scp_transport_options(profile: RemoteProfile) -> list[str]:
    identity = Path(profile.identity_file)
    if not profile.identity_file or not identity.is_file():
        raise RuntimeError(f"SSH key not found: {profile.identity_file or 'no key selected'}")
    return ["-o", f"MACs={JUSUF_MAC}", "-i", str(identity), "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def interactive_scp_transport_options(profile: RemoteProfile) -> list[str]:
    """SCP options for one user-authorized transfer with a fresh JUSUF TOTP."""
    identity = Path(profile.identity_file)
    if not profile.identity_file or not identity.is_file():
        raise RuntimeError(f"SSH key not found: {profile.identity_file or 'no key selected'}")
    return [
        "-o", f"MACs={JUSUF_MAC}", "-i", str(identity),
        "-o", "BatchMode=no",
        "-o", "PreferredAuthentications=publickey,keyboard-interactive",
        "-o", "ConnectTimeout=12",
    ]


def sync_commands(profile: RemoteProfile, bundle: Path) -> list[list[str]]:
    ssh = ssh_executable()
    scp = scp_executable()
    if not ssh or not scp:
        raise RuntimeError("Windows OpenSSH (ssh.exe and scp.exe) is required.")
    # The preflight already created and validated profile.root. Keeping the
    # transfer to one SCP connection means JUSUF asks for MFA exactly once.
    return [[scp, *interactive_scp_transport_options(profile), "-r", str(bundle), f"{profile.destination}:{profile.root.rstrip('/')}/"]]


def preflight_command(profile: RemoteProfile, backend: str = "") -> list[str]:
    ssh = ssh_executable()
    if not ssh:
        raise RuntimeError("Windows OpenSSH (ssh.exe) is required.")
    workspace = quote_remote(profile.root)
    user = quote_remote(profile.user)
    account = quote_remote(profile.slurm_account)
    partition = quote_remote(profile.slurm_partition)
    checks = [
        "printf 'SSH_OK\\n'",
        f"test \"$(whoami)\" = {user} && printf 'USER_OK\\n' || {{ printf 'USER_MISMATCH\\n'; exit 20; }}",
        f"mkdir -p {workspace} && test -w {workspace} && printf 'WORKSPACE_OK\\n' || {{ printf 'WORKSPACE_NOT_WRITABLE\\n'; exit 21; }}",
    ]
    if profile.target == "HPC (Slurm)":
        association = (
            "sacctmgr show associations user=$USER format=Account,Partition -n -P "
            f"| grep -E '^'\"{account}\"'\\|'\"{partition}\"'($|\\|)' >/dev/null"
        )
        checks.append("command -v sbatch >/dev/null && command -v sacctmgr >/dev/null && printf 'SLURM_OK\\n' || { printf 'SLURM_MISSING\\n'; exit 22; }")
        checks.append(f"{association} && printf 'ASSOCIATION_OK\\n' || {{ printf 'ASSOCIATION_MISSING\\n'; exit 23; }}")
    if profile.environment:
        package = "facemap" if backend.lower() == "facemap" else "deeplabcut"
        setup, python = remote_python(profile, backend)
        if profile.environment.startswith("/"):
            checks.append(f"test -x {python} && printf 'ENVIRONMENT_OK\\n' || {{ printf 'ENVIRONMENT_MISSING\\n'; exit 25; }}")
        else:
            checks.append("command -v conda >/dev/null && printf 'CONDA_OK\\n' || { printf 'CONDA_MISSING\\n'; exit 24; }")
            environment = quote_remote(profile.environment)
            checks.append(
                f"conda env list | awk '{{print $1}}' | grep -Fx {environment} >/dev/null "
                "&& printf 'ENVIRONMENT_OK\\n' || { printf 'ENVIRONMENT_MISSING\\n'; exit 25; }"
            )
        version_probe = f"{setup} && {python} -c \"import importlib.metadata as m; print(m.version('{package}'))\" 2>/dev/null | tail -n 1"
        if package == "facemap":
            checks.append(
                f"test \"$({version_probe})\" = '1.0.8' && printf 'BACKEND_OK facemap=1.0.8\\n' "
                f"|| {{ printf 'BACKEND_VERSION_MISMATCH %s\\n' \"$({version_probe})\"; exit 26; }}"
            )
        else:
            checks.append(
                f"test -n \"$({version_probe})\" && printf 'BACKEND_OK deeplabcut\\n' "
                "|| { printf 'BACKEND_MISSING\\n'; exit 26; }"
            )
    return [ssh, *interactive_ssh_transport_options(profile), profile.destination, " && ".join(checks)]


def start_command(profile: RemoteProfile, bundle: Path) -> list[str]:
    ssh = ssh_executable()
    if not ssh:
        raise RuntimeError("Windows OpenSSH (ssh.exe) is required.")
    remote_bundle = quote_remote(profile.bundle_path(bundle))
    if profile.target == "HPC (Slurm)":
        command = f"cd {remote_bundle} && sbatch slurm_job.sh"
    else:
        setup, python = remote_python(profile, "facemap")
        command = (
            f"cd {remote_bundle} && mkdir -p logs && "
            f"{setup} && nohup {python} training_entry.py "
            f"> logs/remote.out 2> logs/remote.err < /dev/null & echo $!"
        )
    return [ssh, *interactive_ssh_transport_options(profile), profile.destination, command]


def status_command(profile: RemoteProfile, job_id: str = "") -> list[str]:
    ssh = ssh_executable()
    if not ssh:
        raise RuntimeError("Windows OpenSSH (ssh.exe) is required.")
    if profile.target == "HPC (Slurm)":
        selector = f"-j {quote_remote(job_id)}" if job_id else f"-u {quote_remote(profile.user)}"
        command = f"squeue {selector} -o '%.18i %.12T %.24j %.10M %.10l %R'"
    else:
        command = "ps -u \"$USER\" -o pid,stat,etime,cmd | grep training_entry.py | grep -v grep || true"
    return [ssh, *interactive_ssh_transport_options(profile), profile.destination, command]


def fetch_commands(profile: RemoteProfile, bundle: Path) -> list[list[str]]:
    scp = scp_executable()
    if not scp:
        raise RuntimeError("Windows OpenSSH (scp.exe) is required.")
    destination = bundle / "remote_results"
    destination.mkdir(exist_ok=True)
    remote = profile.bundle_path(bundle)
    # Fetch logs and results over one connection, so JUSUF asks for MFA once.
    # run_local.bat may also match [lr]*; it is tiny and harmless here.
    source = f"{profile.destination}:{remote}/[lr]*"
    return [[scp, *interactive_scp_transport_options(profile), "-r", source, str(destination)]]
