from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


DIALOG_STYLE = """
QDialog {
    background: #1E2127;
}

QLabel#DialogTitle {
    color: #EAEAEA;
    font-family: "Segoe UI";
    font-size: 16px;
    font-weight: 700;
}

QLabel#DialogBody {
    color: #EAEAEA;
    font-family: "Segoe UI";
    font-size: 13px;
}

QLabel#DialogKind {
    color: #D18B47;
    font-family: "Segoe UI";
    font-size: 11px;
    font-weight: 700;
}

QPushButton#DialogPrimary {
    background: #D18B47;
    color: #111111;
    border: none;
    border-radius: 8px;
    min-width: 96px;
    padding: 8px 16px;
    font-family: "Segoe UI";
    font-weight: 700;
}

QPushButton#DialogPrimary:hover {
    background: #DFA15F;
}

QPushButton#DialogSecondary {
    background: #2A2E35;
    color: #EAEAEA;
    border: 1px solid #3A4049;
    border-radius: 8px;
    min-width: 96px;
    padding: 8px 16px;
    font-family: "Segoe UI";
    font-weight: 600;
}

QPushButton#DialogSecondary:hover {
    background: #343942;
    border: 1px solid #D18B47;
}
"""


class LabelForgeDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        text: str,
        *,
        kind: str = "INFO",
        confirm_text: str = "OK",
        cancel_text: str | None = None,
    ) -> None:
        super().__init__(parent)

        self.result_confirmed = False

        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(460)
        self.setMaximumWidth(620)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        kind_label = QLabel(kind.upper())
        kind_label.setObjectName("DialogKind")

        title_label = QLabel(title)
        title_label.setObjectName("DialogTitle")
        title_label.setWordWrap(True)

        body_label = QLabel(text)
        body_label.setObjectName("DialogBody")
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        layout.addWidget(kind_label)
        layout.addWidget(title_label)
        layout.addWidget(body_label)
        layout.addSpacing(8)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        if cancel_text is not None:
            cancel_button = QPushButton(cancel_text)
            cancel_button.setObjectName("DialogSecondary")
            cancel_button.clicked.connect(self.reject)
            buttons.addWidget(cancel_button)

        confirm_button = QPushButton(confirm_text)
        confirm_button.setObjectName("DialogPrimary")
        confirm_button.clicked.connect(self._confirm)
        confirm_button.setDefault(True)
        buttons.addWidget(confirm_button)

        layout.addLayout(buttons)

    def _confirm(self) -> None:
        self.result_confirmed = True
        self.accept()


def information(
    parent: QWidget | None,
    title: str,
    text: str,
) -> None:
    LabelForgeDialog(
        parent,
        title,
        text,
        kind="Info",
        confirm_text="OK",
    ).exec()


def warning(
    parent: QWidget | None,
    title: str,
    text: str,
) -> None:
    LabelForgeDialog(
        parent,
        title,
        text,
        kind="Warning",
        confirm_text="OK",
    ).exec()


def error(
    parent: QWidget | None,
    title: str,
    text: str,
) -> None:
    LabelForgeDialog(
        parent,
        title,
        text,
        kind="Error",
        confirm_text="OK",
    ).exec()


def confirm(
    parent: QWidget | None,
    title: str,
    text: str,
    *,
    confirm_text: str = "Continue",
    cancel_text: str = "Cancel",
) -> bool:
    dialog = LabelForgeDialog(
        parent,
        title,
        text,
        kind="Confirm",
        confirm_text=confirm_text,
        cancel_text=cancel_text,
    )
    dialog.exec()
    return dialog.result_confirmed
