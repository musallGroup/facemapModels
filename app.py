import os
import sys
from pathlib import Path


# Windows OpenSSH starts this same frozen executable as its SSH_ASKPASS helper
# during the attended JUSUF preflight. Return the one-time code and exit before
# importing Qt or LabelForge. The value exists only in the child environment.
if os.environ.get("LABELFORGE_SSH_ASKPASS") == "1":
    sys.stdout.write(os.environ.get("LABELFORGE_TOTP", ""))
    sys.stdout.flush()
    raise SystemExit(0)


def _set_label_forge_working_directory() -> None:
    """
    Keep LabelForge independent of how Windows launches it.

    In a PyInstaller build, relative paths are resolved from the folder that
    contains LabelForge.exe. During normal Python development they are resolved
    from the project root containing this app.py.
    """
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    os.chdir(base_dir)


_set_label_forge_working_directory()

from labelforge.app import run_app


if __name__ == "__main__":
    run_app()
