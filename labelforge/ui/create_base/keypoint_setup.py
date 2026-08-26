from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..common.dialogs import information, warning, confirm


PALETTES: dict[str, list[str]] = {
    "Blue": [
        "#5DA9E9", "#4A98DD", "#3787D1", "#2476C5",
        "#1165B9", "#0E56A0", "#0B4787", "#08386E",
    ],
    "Green": [
        "#74D99B", "#60CE89", "#4CC377", "#38B865",
        "#24AD53", "#1E9146", "#187539", "#12592C",
    ],
    "Yellow": [
        "#E5CE68", "#DDC455", "#D5BA42", "#CDB02F",
        "#C5A61C", "#A88D18", "#8B7414", "#6E5B10",
    ],
    "Orange": [
        "#E7A15E", "#DF9550", "#D78A42", "#D18B47",
        "#C47735", "#A9632C", "#8E4F23", "#733B1A",
    ],
    "Red": [
        "#ED7A78", "#E86866", "#E35654", "#DE4442",
        "#D93230", "#B92A29", "#992222", "#791A1B",
    ],
    "Pink": [
        "#E78BB5", "#DE78A8", "#D5659B", "#CC528E",
        "#C33F81", "#A6356D", "#892B59", "#6C2145",
    ],
    "Purple": [
        "#B38AE5", "#A477DA", "#9564CF", "#8651C4",
        "#773EB9", "#65349E", "#532A83", "#412068",
    ],
    "Brown": [
        "#D2AE84", "#C49A6C", "#B7885B", "#A8764D",
        "#956440", "#805335", "#6A432C", "#553424",
    ],
}

SHORTCUT_OPTIONS = [
    "",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "-", "=", "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
]


LOCAL_STYLE = """
QComboBox#CompactCombo {
    background: #14161A;
    color: #EAEAEA;
    border: 1px solid #3A4049;
    border-radius: 8px;
    padding: 7px 30px 7px 9px;
    min-height: 20px;
}

QComboBox#CompactCombo:hover,
QComboBox#CompactCombo:focus {
    border: 1px solid #D18B47;
}

QComboBox#CompactCombo::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    background: transparent;
    border: none;
}

QComboBox#CompactCombo::down-arrow {
    image: none;
    width: 0px;
    height: 0px;
}

QComboBox#CompactCombo QAbstractItemView {
    background: #1E2127;
    color: #EAEAEA;
    border: 1px solid #3A4049;
    selection-background-color: #D18B47;
    selection-color: #111111;
    outline: 0;
    padding: 4px;
}

QComboBox#CompactCombo QAbstractItemView::item {
    min-height: 26px;
    padding: 4px 8px;
}

QComboBox#CompactCombo QAbstractItemView::item:hover {
    background: #343942;
    color: #EAEAEA;
}

QPushButton#RemoveGroupButton {
    background: transparent;
    color: #AEB4BF;
    border: 1px solid #3A4049;
    border-radius: 8px;
    min-width: 30px;
    max-width: 30px;
    min-height: 30px;
    max-height: 30px;
    font-size: 17px;
    font-weight: 700;
    padding: 0;
}

QPushButton#RemoveGroupButton:hover {
    color: #EAEAEA;
    background: #2A2E35;
    border: 1px solid #D18B47;
}

QPushButton#RemoveKeypointButton {
    background: transparent;
    color: #AEB4BF;
    border: 1px solid #3A4049;
    border-radius: 7px;
    padding: 6px 10px;
}

QPushButton#RemoveKeypointButton:hover {
    color: #EAEAEA;
    background: #2A2E35;
    border: 1px solid #D18B47;
}
"""


@dataclass
class KeypointDraft:
    name: str = ""
    shortcut: str = ""


@dataclass
class GroupDraft:
    name: str = ""
    palette: str = "Blue"
    keypoints: list[KeypointDraft] = field(default_factory=list)


class ColorDot(QLabel):
    def __init__(self, color: str, size: int = 16, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self.set_color(color)

    def set_color(self, color: str) -> None:
        self.setStyleSheet(
            f"background: {color}; border: 1px solid #111111; "
            f"border-radius: {self._size // 2}px;"
        )


class ChevronComboBox(QComboBox):
    """Dark combo box with a small, clean chevron inside the field."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.chevron = QLabel("▼", self)
        self.chevron.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.chevron.setAlignment(Qt.AlignCenter)
        self.chevron.setStyleSheet(
            "color: #AEB4BF; background: transparent; border: none; "
            "font-size: 10px; font-weight: 700;"
        )
        self.chevron.setFixedSize(22, 22)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        x = self.width() - self.chevron.width() - 4
        y = (self.height() - self.chevron.height()) // 2
        self.chevron.move(x, y)
        self.chevron.raise_()



class KeypointRow(QFrame):
    remove_requested = Signal(object)

    def __init__(
        self,
        index: int,
        color: str,
        name: str = "",
        shortcut: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("KeypointRow")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(10)

        self.index_label = QLabel()
        self.index_label.setFixedWidth(24)
        self.index_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.index_label.setObjectName("FieldHint")

        self.color_dot = ColorDot(color, 16)

        self.name_input = QLineEdit()
        self.name_input.setObjectName("CompactInput")
        self.name_input.setPlaceholderText("e.g. pupil_top")
        self.name_input.setText(name)

        shortcut_label = QLabel("Shortcut")
        shortcut_label.setObjectName("FieldHint")

        self.shortcut_combo = ChevronComboBox()
        self.shortcut_combo.setObjectName("CompactCombo")
        self.shortcut_combo.setMinimumWidth(76)
        self.shortcut_combo.addItems(SHORTCUT_OPTIONS)
        if shortcut in SHORTCUT_OPTIONS:
            self.shortcut_combo.setCurrentText(shortcut)

        self.remove_button = QPushButton("Remove")
        self.remove_button.setObjectName("RemoveKeypointButton")
        self.remove_button.setToolTip("Remove this keypoint")
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))

        layout.addWidget(self.index_label)
        layout.addWidget(self.color_dot)
        layout.addWidget(self.name_input, 1)
        layout.addWidget(shortcut_label)
        layout.addWidget(self.shortcut_combo)
        layout.addWidget(self.remove_button)

        self.set_index(index)

    def set_index(self, index: int) -> None:
        self.index_label.setText(f"{index + 1}.")

    def set_color(self, color: str) -> None:
        self.color_dot.set_color(color)

    def data(self) -> dict:
        return {
            "name": self.name_input.text().strip(),
            "shortcut": self.shortcut_combo.currentText().strip(),
        }


class GroupCard(QFrame):
    remove_requested = Signal(object)
    changed = Signal()

    def __init__(
        self,
        index: int,
        draft: GroupDraft | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("GroupCard")
        self.rows: list[KeypointRow] = []

        draft = draft or GroupDraft()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(10)

        self.group_number = QLabel()
        self.group_number.setObjectName("GroupNumber")
        self.group_number.setFixedWidth(68)

        self.name_input = QLineEdit()
        self.name_input.setObjectName("CompactInput")
        self.name_input.setPlaceholderText("e.g. Eye")
        self.name_input.setText(draft.name)

        palette_label = QLabel("Palette")
        palette_label.setObjectName("FieldHint")

        self.palette_combo = ChevronComboBox()
        self.palette_combo.setObjectName("CompactCombo")
        self.palette_combo.setMinimumWidth(112)
        self.palette_combo.addItems(PALETTES.keys())
        self.palette_combo.setCurrentText(draft.palette)
        self.palette_combo.currentTextChanged.connect(self.refresh_colors)

        self.remove_group_button = QPushButton("×")
        self.remove_group_button.setObjectName("RemoveGroupButton")
        self.remove_group_button.setToolTip("Remove this group and its keypoints")
        self.remove_group_button.clicked.connect(
            lambda: self.remove_requested.emit(self)
        )

        header.addWidget(self.group_number)
        header.addWidget(self.name_input, 1)
        header.addWidget(palette_label)
        header.addWidget(self.palette_combo)
        header.addWidget(self.remove_group_button)

        outer.addLayout(header)

        self.palette_preview = QHBoxLayout()
        self.palette_preview.setSpacing(5)
        self.palette_preview.addSpacing(80)

        self.palette_dots: list[ColorDot] = []
        for color in PALETTES[self.palette_combo.currentText()]:
            dot = ColorDot(color, 14)
            self.palette_dots.append(dot)
            self.palette_preview.addWidget(dot)

        self.palette_preview.addStretch(1)
        outer.addLayout(self.palette_preview)

        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)
        outer.addWidget(self.rows_container)

        add_keypoint = QPushButton("+ Add Keypoint")
        add_keypoint.setObjectName("SecondaryActionButton")
        add_keypoint.setFixedWidth(145)
        add_keypoint.clicked.connect(lambda checked=False: self.add_keypoint())

        add_row = QHBoxLayout()
        add_row.addSpacing(80)
        add_row.addWidget(add_keypoint)
        add_row.addStretch(1)
        outer.addLayout(add_row)

        self.set_group_index(index)

        if draft.keypoints:
            for kp in draft.keypoints:
                self.add_keypoint(kp.name, kp.shortcut)
        else:
            self.add_keypoint()

    def set_group_index(self, index: int) -> None:
        self.group_number.setText(f"Group {index + 1}")

    def add_keypoint(self, name: str = "", shortcut: str = "") -> None:
        palette = PALETTES[self.palette_combo.currentText()]
        color = palette[len(self.rows) % len(palette)]

        row = KeypointRow(
            index=len(self.rows),
            color=color,
            name=name,
            shortcut=shortcut,
        )
        row.remove_requested.connect(self.remove_keypoint)

        self.rows.append(row)
        self.rows_layout.addWidget(row)
        self.changed.emit()

    def remove_keypoint(self, row: KeypointRow) -> None:
        if row not in self.rows:
            return

        self.rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.reindex_rows()
        self.changed.emit()

    def reindex_rows(self) -> None:
        palette = PALETTES[self.palette_combo.currentText()]

        for index, row in enumerate(self.rows):
            row.set_index(index)
            row.set_color(palette[index % len(palette)])

    def refresh_colors(self) -> None:
        palette = PALETTES[self.palette_combo.currentText()]

        for dot, color in zip(self.palette_dots, palette):
            dot.set_color(color)

        self.reindex_rows()
        self.changed.emit()

    def has_user_content(self) -> bool:
        if self.name_input.text().strip():
            return True

        return any(row.name_input.text().strip() for row in self.rows)

    def data(self) -> dict:
        palette_name = self.palette_combo.currentText()
        palette = PALETTES[palette_name]

        keypoints = []
        for index, row in enumerate(self.rows):
            values = row.data()
            keypoints.append(
                {
                    "name": values["name"],
                    "shortcut": values["shortcut"],
                    "color": palette[index % len(palette)],
                }
            )

        return {
            "name": self.name_input.text().strip(),
            "palette": palette_name,
            "keypoints": keypoints,
        }


class KeypointSetupPage(QWidget):
    previous_requested = Signal()
    next_requested = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Local styling fixes the Qt dropdown popup without touching main_window.py.
        self.setStyleSheet(LOCAL_STYLE)

        self.group_cards: list[GroupCard] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        intro = QHBoxLayout()

        intro_text = QLabel(
            "Create logical keypoint groups. Each group gets one color family; "
            "individual keypoints automatically receive different shades."
        )
        intro_text.setObjectName("PageSubtitle")
        intro_text.setWordWrap(True)

        add_group = QPushButton("+ Add Group")
        add_group.setObjectName("PrimaryNextButton")
        add_group.setToolTip(
            "Create another logical group, e.g. Pupil, Nose or Mouth"
        )
        add_group.clicked.connect(lambda: self.add_group())

        intro.addWidget(intro_text, 1)
        intro.addWidget(add_group)
        layout.addLayout(intro)

        header = QHBoxLayout()
        header.setContentsMargins(18, 0, 18, 0)

        group_h = QLabel("GROUP / KEYPOINT")
        group_h.setObjectName("FieldHint")

        shortcut_h = QLabel(
            "Names should be backend-safe (letters, numbers, underscore)."
        )
        shortcut_h.setObjectName("FieldHint")
        shortcut_h.setAlignment(Qt.AlignRight)

        header.addWidget(group_h)
        header.addStretch(1)
        header.addWidget(shortcut_h)
        layout.addLayout(header)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("KeypointScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.scroll_content = QWidget()
        self.groups_layout = QVBoxLayout(self.scroll_content)
        self.groups_layout.setContentsMargins(0, 0, 6, 0)
        self.groups_layout.setSpacing(10)
        self.groups_layout.addStretch(1)

        self.scroll.setWidget(self.scroll_content)
        layout.addWidget(self.scroll, 1)

        bottom = QHBoxLayout()

        previous = QPushButton("← Previous")
        previous.setObjectName("BackButton")
        previous.clicked.connect(self.previous_requested.emit)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("FieldHint")
        self.summary_label.setAlignment(Qt.AlignCenter)

        next_button = QPushButton("Continue →")
        next_button.setObjectName("PrimaryNextButton")
        next_button.clicked.connect(self.validate_and_continue)

        bottom.addWidget(previous)
        bottom.addStretch(1)
        bottom.addWidget(self.summary_label)
        bottom.addStretch(1)
        bottom.addWidget(next_button)

        layout.addLayout(bottom)

        # One starter group helps explain the interaction without creating data.
        self.add_group(name="Eye", palette="Blue")

    def add_group(self, name: str = "", palette: str = "Blue") -> None:
        draft = GroupDraft(name=name, palette=palette)

        card = GroupCard(
            index=len(self.group_cards),
            draft=draft,
        )
        card.remove_requested.connect(self.remove_group)
        card.changed.connect(self.update_summary)

        self.group_cards.append(card)
        self.groups_layout.insertWidget(
            self.groups_layout.count() - 1,
            card,
        )

        self.reindex_groups()
        self.update_summary()

        # Ensure a newly added group is visible.
        self.scroll.ensureWidgetVisible(card)

    def remove_group(self, card: GroupCard) -> None:
        if card not in self.group_cards:
            return

        if len(self.group_cards) == 1:
            information(
                self,
                "At least one group",
                "A new base model needs at least one keypoint group.",
            )
            return

        if card.has_user_content():
            if not confirm(
                self,
                "Remove group?",
                "Remove this group and all keypoints inside it?",
                confirm_text="Remove",
                cancel_text="Cancel",
            ):
                return

        self.group_cards.remove(card)
        card.setParent(None)
        card.deleteLater()

        self.reindex_groups()
        self.update_summary()

    def reindex_groups(self) -> None:
        for index, card in enumerate(self.group_cards):
            card.set_group_index(index)

    def update_summary(self) -> None:
        keypoint_count = sum(len(card.rows) for card in self.group_cards)
        self.summary_label.setText(
            f"{len(self.group_cards)} group(s) · {keypoint_count} keypoint(s)"
        )

    def load_data(self, data: dict) -> None:
        """
        Replace the current editor contents with a previously saved keypoint draft.
        This is used when navigating away from Keypoints and returning later.
        """
        groups = data.get("groups", []) if data else []

        if not groups:
            return

        for card in list(self.group_cards):
            self.group_cards.remove(card)
            card.setParent(None)
            card.deleteLater()

        for group in groups:
            draft = GroupDraft(
                name=group.get("name", ""),
                palette=group.get("palette", "Blue"),
                keypoints=[
                    KeypointDraft(
                        name=kp.get("name", ""),
                        shortcut=kp.get("shortcut", ""),
                    )
                    for kp in group.get("keypoints", [])
                ],
            )

            card = GroupCard(
                index=len(self.group_cards),
                draft=draft,
            )
            card.remove_requested.connect(self.remove_group)
            card.changed.connect(self.update_summary)

            self.group_cards.append(card)
            self.groups_layout.insertWidget(
                self.groups_layout.count() - 1,
                card,
            )

        self.reindex_groups()
        self.update_summary()

    def collect_data(self) -> dict:
        return {
            "groups": [card.data() for card in self.group_cards]
        }

    def validate_and_continue(self) -> None:
        data = self.collect_data()

        if not data["groups"]:
            warning(
                self,
                "No groups",
                "Please create at least one keypoint group.",
            )
            return

        seen_names: set[str] = set()
        seen_shortcuts: set[str] = set()

        for group in data["groups"]:
            if not group["name"]:
                warning(
                    self,
                    "Group name required",
                    "Every keypoint group needs a name.",
                )
                return

            if not group["keypoints"]:
                warning(
                    self,
                    "Keypoints required",
                    f'Group "{group["name"]}" does not contain any keypoints.',
                )
                return

            for keypoint in group["keypoints"]:
                name = keypoint["name"]

                if not name:
                    warning(
                        self,
                        "Keypoint name required",
                        f'Every keypoint in group "{group["name"]}" needs a name.',
                    )
                    return

                safe = name.replace("_", "")
                if not safe.isalnum():
                    warning(
                        self,
                        "Invalid keypoint name",
                        f'"{name}" contains unsupported characters.\n\n'
                        "Use letters, numbers and underscores only.",
                    )
                    return

                lowered = name.lower()
                if lowered in seen_names:
                    warning(
                        self,
                        "Duplicate keypoint",
                        f'The keypoint name "{name}" is used more than once.',
                    )
                    return
                seen_names.add(lowered)

                shortcut = keypoint["shortcut"]
                if shortcut:
                    normalized_shortcut = shortcut.upper()
                    if normalized_shortcut in seen_shortcuts:
                        warning(
                            self,
                            "Duplicate shortcut",
                            f'The shortcut "{shortcut}" is assigned more than once.',
                        )
                        return
                    seen_shortcuts.add(normalized_shortcut)

        self.next_requested.emit(data)
