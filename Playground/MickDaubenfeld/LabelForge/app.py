import os
import sys
from pathlib import Path


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
