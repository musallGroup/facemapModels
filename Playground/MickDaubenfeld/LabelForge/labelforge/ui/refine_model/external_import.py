from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,QDialog,QDialogButtonBox,QFileDialog,QFormLayout,QHBoxLayout,QHeaderView,
    QLabel,QLineEdit,QPushButton,QSpinBox,QTableWidget,QTableWidgetItem,
    QTextEdit,QVBoxLayout,
)
from ...model_metadata import (
    ModelMetadataError,discover_facemap_keypoints,read_dlc_bodyparts,
    load_parent_metadata,read_facemap_labels_csv,read_facemap_refined_bodyparts,
    write_external_metadata,
)
from ..common.dialogs import warning
from ..create_base.keypoint_setup import PALETTES

COLORS=["#D18B47","#E0A15F","#C86B4A","#B65E82","#8F6FB5","#5E8FA8","#67A37A","#B4A24C"]
GROUP_PALETTES={
    "Eye":"Blue",
    "Mouth":"Red",
    "Nose":"Orange",
    "Whiskers":"Purple",
    "Paw":"Green",
    "Keypoints":"Yellow",
}

class ExternalModelImportDialog(QDialog):
    def __init__(self,selection,parent=None):
        super().__init__(parent); self.selection=Path(selection); self.saved_path=None; self.keypoint_source=None
        self.setWindowTitle("Import External Model"); self.resize(860,680)
        self.setStyleSheet("""
            QDialog {
                background: #17191E;
                color: #EAEAEA;
                font-family: "Segoe UI";
                font-size: 13px;
            }
            QLabel {
                color: #EAEAEA;
                background: transparent;
            }
            QLabel#PageTitle {
                color: #F4F4F4;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#PageSubtitle {
                color: #C8CDD5;
                font-size: 13px;
            }
            QLabel#FieldHint {
                color: #AEB5C0;
                font-size: 12px;
            }
            QLineEdit, QTextEdit, QSpinBox {
                background: #111318;
                color: #F2F2F2;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 8px 10px;
                selection-background-color: #D18B47;
                selection-color: #111111;
            }
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {
                border: 1px solid #D18B47;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: #2A2E35;
                border-left: 1px solid #3A4049;
                width: 20px;
            }
            QTableWidget {
                background: #111318;
                alternate-background-color: #191C22;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 6px;
                gridline-color: #30343C;
                selection-background-color: #6E4B2D;
                selection-color: #FFFFFF;
            }
            QTableWidget::item {
                padding: 7px;
            }
            QHeaderView::section {
                background: #252930;
                color: #EAEAEA;
                border: none;
                border-right: 1px solid #3A4049;
                border-bottom: 1px solid #3A4049;
                padding: 8px;
                font-weight: 600;
            }
            QTableCornerButton::section {
                background: #252930;
                border: none;
            }
            QPushButton {
                background: #2A2E35;
                color: #F0F0F0;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 9px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #D18B47;
                background: #343942;
            }
            QPushButton:disabled {
                background: #202329;
                color: #727985;
                border-color: #30343A;
            }
            QDialogButtonBox QPushButton {
                min-width: 90px;
            }
            QPushButton#DialogPrimary {
                background: #D18B47;
                color: #111111;
                border: none;
                font-weight: 700;
            }
            QPushButton#DialogPrimary:hover {
                background: #DFA15F;
            }
            QScrollBar:vertical {
                background: #17191E;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #444A54;
                border-radius: 5px;
                min-height: 28px;
            }
        """)
        outer=QVBoxLayout(self); outer.setContentsMargins(24,20,24,20); outer.setSpacing(12)
        title=QLabel("Import External Facemap Model" if self.selection.is_file() else "Import External DeepLabCut Model")
        title.setObjectName("PageTitle")
        hint=QLabel("This creates LabelForge metadata only. The trained model and DLC project are not modified. Keep the exact keypoint order used during training.")
        hint.setObjectName("PageSubtitle"); hint.setWordWrap(True)
        form=QFormLayout(); self.family=QLineEdit(self.selection.stem if self.selection.is_file() else self.selection.name); self.family.setObjectName("TextInput")
        self.version=QSpinBox(); self.version.setRange(1,999); self.version.setValue(1)
        form.addRow("Model family",self.family); form.addRow("Current version",self.version)
        source_row=QHBoxLayout(); self.manual=QTextEdit(); self.manual.setPlaceholderText("One keypoint name per line, in the exact model-output order."); self.manual.setFixedHeight(100)
        csv_button=QPushButton("Import keypoints…"); csv_button.setObjectName("BrowseButton"); csv_button.clicked.connect(self.import_keypoints)
        apply_button=QPushButton("Use these keypoints"); apply_button.setObjectName("BrowseButton"); apply_button.clicked.connect(self.apply_manual)
        auto_group_button=QPushButton("Auto-group"); auto_group_button.setObjectName("BrowseButton"); auto_group_button.clicked.connect(self.auto_group)
        source_buttons=QVBoxLayout(); source_buttons.addWidget(csv_button); source_buttons.addWidget(apply_button); source_buttons.addWidget(auto_group_button); source_buttons.addStretch(1)
        source_row.addWidget(self.manual,1); source_row.addLayout(source_buttons)
        self.table=QTableWidget(0,4); self.table.setHorizontalHeaderLabels(["Keypoint (locked order)","Group","Color","Shortcut"])
        self.table.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch); self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self.pick_color)
        note=QLabel("Groups and colors are suggested automatically. Edit group names directly, double-click a color to choose it, or use Auto-group to restore the suggestions. Keypoint order stays locked.")
        note.setObjectName("FieldHint"); note.setWordWrap(True)
        self.source_note=QLabel(""); self.source_note.setObjectName("FieldHint"); self.source_note.setWordWrap(True)
        buttons=QDialogButtonBox(QDialogButtonBox.Cancel|QDialogButtonBox.Save)
        buttons.button(QDialogButtonBox.Save).setObjectName("DialogPrimary")
        buttons.button(QDialogButtonBox.Cancel).setObjectName("DialogSecondary")
        buttons.rejected.connect(self.reject); buttons.accepted.connect(self.save)
        outer.addWidget(title); outer.addWidget(hint); outer.addLayout(form); outer.addLayout(source_row); outer.addWidget(self.source_note); outer.addWidget(note); outer.addWidget(self.table,1); outer.addWidget(buttons)
        try:
            existing=load_parent_metadata(self.selection)
        except ModelMetadataError:
            existing=None
        if existing is not None:
            self.family.setText(existing.model_family); self.version.setValue(existing.version)
            self.keypoint_source=existing.source_path; self._show_existing(existing)
        elif self.selection.is_dir():
            try:
                self.keypoint_source=self.selection/"config.yaml"
                names=read_dlc_bodyparts(self.keypoint_source); self._show_detected(names,self.keypoint_source)
            except ModelMetadataError as exc: warning(self,"DLC keypoints could not be read",str(exc))
        else:
            try:
                names,self.keypoint_source=discover_facemap_keypoints(self.selection)
                self._show_detected(names,self.keypoint_source)
            except ModelMetadataError as exc:
                self.source_note.setText(f"No keypoints detected automatically: {exc}")
    def _show_existing(self,metadata):
        groups=metadata.keypoint_schema["groups"]
        names=[keypoint["name"] for group in groups for keypoint in group["keypoints"]]
        self.manual.setPlainText("\n".join(names)); self.table.setRowCount(len(names))
        row=0
        for group in groups:
            group_name=group.get("name","Keypoints")
            saved_palette_name=group.get("palette")
            default_palette_name=GROUP_PALETTES.get(group_name,"Yellow")

            # Older metadata may have palette="Custom" because every keypoint
            # in the group was accidentally saved with the same color.
            # For known semantic groups, always restore the same palette family
            # as Create Base so the keypoints get distinct shades.
            if saved_palette_name in PALETTES:
                palette_name=saved_palette_name
            else:
                palette_name=default_palette_name

            palette=PALETTES.get(palette_name)

            for group_index,keypoint in enumerate(group["keypoints"]):
                name_item=QTableWidgetItem(keypoint["name"]); name_item.setFlags(name_item.flags()&~Qt.ItemIsEditable)
                self.table.setItem(row,0,name_item)
                self.table.setItem(row,1,QTableWidgetItem(group_name))

                # Match Create Base exactly: one palette per group,
                # one shade per keypoint within that group.
                #
                # Palette-based groups intentionally regenerate their shades
                # from the saved palette name. This also fixes older metadata
                # where all keypoints in a group were accidentally stored with
                # the exact same color.
                #
                # Truly custom groups keep their saved individual colors.
                if palette is not None:
                    color=palette[group_index % len(palette)]
                else:
                    color=keypoint.get("color",COLORS[row % len(COLORS)])

                self._set_color_item(row,color)
                self.table.setItem(row,3,QTableWidgetItem(keypoint.get("shortcut","")))
                row+=1
        self.source_note.setText(f"Loaded {len(names)} keypoints and saved styling from: {metadata.source_path}")
    def _show_detected(self,names,source):
        self.manual.setPlainText("\n".join(names)); self.set_keypoints(names)
        self.source_note.setText(f"Detected {len(names)} keypoints from: {source}")
    def import_keypoints(self):
        path,_=QFileDialog.getOpenFileName(
            self,"Import Facemap keypoints","",
            "Keypoint metadata (*.csv *.npy);;Facemap refined data (*.npy);;Label CSV (*.csv)",
        )
        if not path:return
        try:
            self.keypoint_source=Path(path)
            if self.keypoint_source.suffix.lower()==".npy":
                names=read_facemap_refined_bodyparts(path)
            else:
                names=read_facemap_labels_csv(path)
            self._show_detected(names,self.keypoint_source)
        except ModelMetadataError as exc: warning(self,"Could not import keypoints",str(exc))
    def apply_manual(self):
        names=[line.strip() for line in self.manual.toPlainText().splitlines() if line.strip()]
        if len(names)!=len(set(names)): warning(self,"Duplicate keypoints","Every keypoint name must be unique."); return
        self.keypoint_source=None; self.source_note.setText("Using manually entered keypoints.")
        self.set_keypoints(names)
    def set_keypoints(self,names):
        self.table.setRowCount(len(names))
        for row,name in enumerate(names):
            name_item=QTableWidgetItem(name); name_item.setFlags(name_item.flags()&~Qt.ItemIsEditable)
            self.table.setItem(row,0,name_item)
            if not self.table.item(row,3): self.table.setItem(row,3,QTableWidgetItem(""))
        self.auto_group()
    @staticmethod
    def suggested_group(name):
        lowered=name.lower()
        if lowered.startswith("eye"): return "Eye"
        if "mouth" in lowered or "lip" in lowered: return "Mouth"
        if lowered.startswith("nose"): return "Nose"
        if lowered.startswith("whisker"): return "Whiskers"
        if lowered.startswith("paw"): return "Paw"
        return "Keypoints"
    def auto_group(self):
        group_indices={}
        for row in range(self.table.rowCount()):
            name_item=self.table.item(row,0)
            if name_item is None: continue
            group=self.suggested_group(name_item.text())
            self.table.setItem(row,1,QTableWidgetItem(group))
            index=group_indices.get(group,0); group_indices[group]=index+1
            palette=PALETTES[GROUP_PALETTES.get(group,"Yellow")]
            self._set_color_item(row,palette[index%len(palette)])
    def _set_color_item(self,row,color_name):
        color=QColor(color_name)
        item=QTableWidgetItem(color.name().upper())
        item.setBackground(color)
        item.setForeground(QColor("#111111") if color.lightness()>150 else QColor("#FFFFFF"))
        self.table.setItem(row,2,item)
    def pick_color(self,row,column):
        if column!=2:return
        current=self.table.item(row,column)
        initial=QColor(current.text()) if current else QColor(COLORS[row%len(COLORS)])
        chosen=QColorDialog.getColor(initial,self,"Choose keypoint color")
        if chosen.isValid(): self._set_color_item(row,chosen.name())
    def collect_groups(self):
        groups={}
        for row in range(self.table.rowCount()):
            name=self.table.item(row,0).text().strip(); group=(self.table.item(row,1).text().strip() if self.table.item(row,1) else "") or "Keypoints"
            color=(self.table.item(row,2).text().strip() if self.table.item(row,2) else "") or COLORS[row%len(COLORS)]
            shortcut=self.table.item(row,3).text().strip() if self.table.item(row,3) else ""
            groups.setdefault(group,[]).append({"name":name,"color":color,"shortcut":shortcut})
        result=[]
        for name,keypoints in groups.items():
            colors=[keypoint["color"].upper() for keypoint in keypoints]
            palette_name="Custom"
            for candidate,palette in PALETTES.items():
                expected=[palette[index%len(palette)].upper() for index in range(len(colors))]
                if colors==expected: palette_name=candidate; break
            result.append({"name":name,"palette":palette_name,"keypoints":keypoints})
        return result
    def save(self):
        try: self.saved_path=write_external_metadata(self.selection,self.family.text(),self.version.value(),self.collect_groups(),self.keypoint_source)
        except ModelMetadataError as exc: warning(self,"Could not create model metadata",str(exc)); return
        self.accept()
