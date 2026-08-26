from __future__ import annotations

import json
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from labelforge.ui.main_window import MainWindow


def load_label_context(app_root: Path) -> dict:
    context_path = app_root / ".dev" / "label_context.json"

    if not context_path.exists():
        raise RuntimeError(
            "No Label development context exists yet.\n\n"
            "Open LabelForge once, complete Project / Keypoints / Frames, "
            "and enter the Label tab. After that you can launch "
            "python dev_labeling.py directly."
        )

    return json.loads(
        context_path.read_text(encoding="utf-8")
    )


def apply_context_to_workflow(window: MainWindow, payload: dict) -> None:
    """
    Open the REAL LabelForge Create Base Model workflow directly on Step 4
    while restoring the last development context.

    This intentionally uses MainWindow + CreateBaseModelWorkflow instead of
    mounting LabelingPage in a separate QMainWindow. Dev mode therefore looks
    exactly like the real app.
    """
    workflow = window.create_page

    project_draft = payload.get("project_draft", {})
    keypoint_draft = payload.get("keypoint_draft", {})
    frame_draft = payload.get("frame_draft", {})

    # Populate the actual Project widgets as well as the draft.
    # go_to_step(3) saves Step 1 before leaving it, so these fields must contain
    # the restored values or the draft would otherwise be overwritten by blanks.
    workflow.name_input.setText(
        project_draft.get("name", "")
    )
    workflow.description_input.setPlainText(
        project_draft.get("description", "")
    )
    workflow.location_input.setText(
        project_draft.get("location", "")
    )

    workflow.project_draft = dict(project_draft)
    workflow.keypoint_draft = dict(keypoint_draft)
    workflow.frame_draft = dict(frame_draft)

    # Put the real MainWindow into Label Workspace / Create Base Model.
    if hasattr(window, "label_tab"):
        window.label_tab.setChecked(True)

    window.stack.setCurrentWidget(window.create_page)

    # Enter the REAL Label step. This calls the same set_context(...) path that
    # normal LabelForge uses.
    workflow.go_to_step(3)


def main() -> None:
    app_root = Path(__file__).resolve().parent
    payload = load_label_context(app_root)

    app = QApplication(sys.argv)
    app.setApplicationName("LabelForge")

    # Match the normal application icon when available.
    assets = app_root / "labelforge" / "assets"
    icon_path = assets / "labelforge_icon.ico"

    if not icon_path.exists():
        icon_path = assets / "labelforge_icon.png"

    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()

    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))

    apply_context_to_workflow(window, payload)

    window.showMaximized()

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
