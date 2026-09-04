from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFileDialog,QFrame,QHBoxLayout,QLabel,QLineEdit,QPushButton,QStackedWidget,QVBoxLayout,QWidget
from ...model_metadata import ModelMetadataError,load_parent_metadata
from ..common.dialogs import confirm,warning
from ..create_base.workflow import StepNavigation
from ..create_base.frame_picker import FramePickerPage
from ..create_base.labeling import LabelingPage
from ..create_base.review import ReviewPage
from ..create_base.export import ExportPage
from .external_import import ExternalModelImportDialog

class ParentModelPage(QWidget):
    next_requested=Signal(object,str,str)
    def __init__(self):
        super().__init__(); self.metadata=None
        outer=QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        panel=QFrame(); panel.setObjectName("WizardPanel"); box=QVBoxLayout(panel); box.setContentsMargins(30,28,30,28)
        title=QLabel("Select a trained parent model"); title.setObjectName("CardTitle")
        hint=QLabel("Choose a Facemap .pt model or a DeepLabCut project folder. A LabelForge metadata sidecar is required."); hint.setObjectName("PageSubtitle"); hint.setWordWrap(True)
        row=QHBoxLayout(); self.path=QLineEdit(); self.path.setObjectName("TextInput"); self.path.setReadOnly(True)
        fm=QPushButton("Facemap .pt…"); fm.setObjectName("BrowseButton"); fm.clicked.connect(self.choose_facemap)
        dlc=QPushButton("DLC project…"); dlc.setObjectName("BrowseButton"); dlc.clicked.connect(self.choose_dlc)
        row.addWidget(self.path,1); row.addWidget(fm); row.addWidget(dlc)
        self.summary=QLabel("No parent model selected."); self.summary.setObjectName("FieldHint"); self.summary.setWordWrap(True)
        self.edit_metadata_button=QPushButton("Edit groups & colors…"); self.edit_metadata_button.setObjectName("BrowseButton"); self.edit_metadata_button.setEnabled(False); self.edit_metadata_button.clicked.connect(self.edit_metadata)
        loc_label=QLabel("New LabelForge project location"); loc_label.setObjectName("FieldLabel")
        loc_row=QHBoxLayout(); self.location=QLineEdit(); self.location.setObjectName("TextInput")
        browse=QPushButton("Browse…"); browse.setObjectName("BrowseButton"); browse.clicked.connect(self.choose_location)
        loc_row.addWidget(self.location,1); loc_row.addWidget(browse)
        self.continue_button=QPushButton("Continue to Frames →"); self.continue_button.setObjectName("PrimaryNextButton"); self.continue_button.setEnabled(False); self.continue_button.clicked.connect(self.continue_workflow)
        box.addWidget(title); box.addWidget(hint); box.addLayout(row); box.addWidget(self.summary); box.addWidget(self.edit_metadata_button,0,Qt.AlignLeft); box.addWidget(loc_label); box.addLayout(loc_row)
        nav=QHBoxLayout(); nav.addStretch(1); nav.addWidget(self.continue_button)
        outer.addWidget(panel); outer.addLayout(nav); outer.addStretch(1)
    def choose_facemap(self):
        path,_=QFileDialog.getOpenFileName(self,"Select trained Facemap model","","PyTorch model (*.pt)")
        if path: self.load_selection(path)
    def choose_dlc(self):
        path=QFileDialog.getExistingDirectory(self,"Select DeepLabCut project")
        if path: self.load_selection(path)
    def choose_location(self):
        path=QFileDialog.getExistingDirectory(self,"Choose refined LabelForge project location")
        if path: self.location.setText(path)
    def load_selection(self,path):
        self.edit_metadata_button.setEnabled(False)
        try: self.metadata=load_parent_metadata(path)
        except ModelMetadataError as exc:
            self.metadata=None; self.continue_button.setEnabled(False)
            if "No LabelForge metadata sidecar" not in str(exc):
                warning(self,"Parent model metadata required",str(exc)); return
            should_import=confirm(
                self,
                "External model detected",
                f"{exc}\n\nImport this external model into LabelForge now? "
                "You can define its identity, groups, colors and shortcuts. "
                "The trained model itself will not be changed.",
                confirm_text="Import External Model",
                cancel_text="Cancel",
            )
            if not should_import:return
            dialog=ExternalModelImportDialog(path,self)
            if not dialog.exec():return
            try:self.metadata=load_parent_metadata(path)
            except ModelMetadataError as retry_exc:
                warning(self,"Metadata import failed",str(retry_exc)); return
        self.path.setText(path)
        self.refresh_summary()
        self.edit_metadata_button.setEnabled(True)
        self.continue_button.setEnabled(True)
    def refresh_summary(self):
        if not self.metadata:return
        names=[k["name"] for g in self.metadata.keypoint_schema["groups"] for k in g["keypoints"]]
        self.summary.setText(f"BACKEND\n{self.metadata.backend.upper()}\n\nPARENT MODEL\n{self.metadata.model_name}\n\nKEYPOINT SCHEMA — LOCKED\n{', '.join(names)}\n\nSUGGESTED NEXT VERSION\n{self.metadata.suggested_next_version}")
    def edit_metadata(self):
        path=self.path.text().strip()
        if not path:return
        dialog=ExternalModelImportDialog(path,self)
        if not dialog.exec():return
        try:self.metadata=load_parent_metadata(path)
        except ModelMetadataError as exc:
            warning(self,"Metadata update failed",str(exc)); return
        self.refresh_summary()
    def continue_workflow(self):
        if not self.metadata: return
        if not self.location.text().strip(): warning(self,"Project location required","Choose where the refined project should be stored."); return
        self.next_requested.emit(self.metadata,self.path.text(),self.location.text().strip())

class RefineModelWorkflow(QWidget):
    back_to_home_requested=Signal()
    STEPS=["Parent Model","Frames","Label","Review","Export"]
    def __init__(self,parent=None):
        super().__init__(parent); self.project_draft={}; self.keypoint_draft={"locked":True,"groups":[]}; self.frame_draft={}; self.label_draft={}; self.current_step=0
        outer=QVBoxLayout(self); outer.setContentsMargins(34,16,34,20)
        header=QHBoxLayout(); back=QPushButton("← Back"); back.setObjectName("BackButton"); back.clicked.connect(self.back_to_home_requested.emit)
        block=QVBoxLayout(); title=QLabel("Refine Existing Model"); title.setObjectName("PageTitle"); self.subtitle=QLabel(); self.subtitle.setObjectName("PageSubtitle"); block.addWidget(title); block.addWidget(self.subtitle)
        self.counter=QLabel(); self.counter.setObjectName("PageSubtitle"); self.counter.setAlignment(Qt.AlignRight|Qt.AlignTop)
        header.addWidget(back); header.addLayout(block,1); header.addWidget(self.counter)
        self.navigation=StepNavigation(self.STEPS); self.navigation.step_requested.connect(self.go_to_step); self.stack=QStackedWidget()
        for index,button in enumerate(self.navigation.buttons):
            button.setEnabled(index==0)
            if index:
                button.setToolTip("Select and confirm a parent model first.")
        self.model_page=ParentModelPage(); self.frames_page=FramePickerPage(); self.label_page=LabelingPage(); self.review_page=ReviewPage(); self.export_page=ExportPage()
        self.model_page.next_requested.connect(self.parent_selected); self.frames_page.previous_requested.connect(lambda:self.go_to_step(0)); self.frames_page.next_requested.connect(self.frames_selected)
        self.label_page.previous_requested.connect(lambda:self.go_to_step(1)); self.label_page.next_requested.connect(self.labels_selected); self.review_page.previous_requested.connect(lambda:self.go_to_step(2)); self.review_page.fix_requested.connect(self.open_issue); self.review_page.export_requested.connect(lambda:self.go_to_step(4)); self.export_page.previous_requested.connect(lambda:self.go_to_step(3))
        for page in (self.model_page,self.frames_page,self.label_page,self.review_page,self.export_page): self.stack.addWidget(page)
        outer.addLayout(header); outer.addWidget(self.navigation); outer.addWidget(self.stack,1); self.go_to_step(0)
    def parent_selected(self,metadata,parent_path,location):
        self.keypoint_draft=metadata.to_keypoint_draft(); self.project_draft={"name":metadata.suggested_next_version,"description":f"Refinement of {metadata.model_name}","location":location,"workflow_type":"refine","backend":metadata.backend,"parent_model":parent_path,"parent_model_name":metadata.model_name,"parent_metadata":str(metadata.source_path),"suggested_version":metadata.suggested_next_version}
        for button in self.navigation.buttons:
            button.setEnabled(True)
            button.setToolTip("")
        self.go_to_step(1)
    def frames_selected(self,data): self.frame_draft=data; self.go_to_step(2)
    def labels_selected(self,data): self.label_draft=data; self.go_to_step(3)
    def open_issue(self,index,name): self.go_to_step(2); self.label_page.jump_to_annotation(index,name)
    def go_to_step(self,index):
        index=max(0,min(index,4))
        if index and not self.project_draft: index=0
        if self.current_step==1: self.frame_draft=self.frames_page.collect_data()
        elif self.current_step==2: self.label_page.save_all_labels(); self.label_draft=self.label_page.collect_data()
        self.current_step=index; self.stack.setCurrentIndex(index); self.navigation.set_active_step(index); self.counter.setText(f"Step {index+1} of 5")
        if index==1: self.frames_page.set_project_context(self.project_draft)
        elif index==2: self.label_page.set_context(self.project_draft,self.keypoint_draft,self.frame_draft)
        elif index==3: self.review_page.set_context(self.project_draft,self.keypoint_draft,self.frame_draft,self.label_draft)
        elif index==4: self.export_page.set_context(self.project_draft,self.keypoint_draft,self.frame_draft,self.label_draft)
        self.subtitle.setText(["Select the trained parent model.","Add representative frames.","Label with the locked parent schema.","Check the dataset.","Create a backend-ready export."][index])
