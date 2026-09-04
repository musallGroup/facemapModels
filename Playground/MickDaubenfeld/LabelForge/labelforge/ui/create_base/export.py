from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..common.dialogs import information, warning


STATE_VISIBLE = "visible"
STATE_NOT_VISIBLE = "not_visible"
STATE_UNSET = "unset"


class ExportPage(QWidget):
    previous_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.project_draft: dict = {}
        self.keypoint_draft: dict = {}
        self.frame_draft: dict = {}
        self.label_draft: dict = {}

        self.frame_paths: list[Path] = []
        self.keypoint_names: list[str] = []
        self.labels_csv: Path | None = None

        self.setStyleSheet("""
            QFrame#ExportCard {
                background: #1D2026;
                border: 1px solid #2D323A;
                border-radius: 14px;
            }

            QLabel#ExportTitle {
                color: #EAEAEA;
                font-size: 18px;
                font-weight: 700;
            }

            QLabel#ExportMeta {
                color: #AEB4BF;
                font-size: 12px;
            }

            QPushButton#ExportPrimary {
                background: #D18B47;
                color: #111111;
                border: none;
                border-radius: 9px;
                padding: 10px 18px;
                font-weight: 700;
            }

            QPushButton#ExportPrimary:hover {
                background: #DFA15F;
            }

            QPushButton#ExportSecondary {
                background: #2A2E35;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 9px;
                padding: 10px 18px;
                font-weight: 600;
            }

            QPushButton#ExportSecondary:hover {
                border: 1px solid #D18B47;
                background: #343942;
            }

            QPushButton#ExportSecondary:disabled {
                color: #6F7682;
                border: 1px solid #2D323A;
                background: #202329;
            }
        """)

        self.build_ui()

    def build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        intro = QLabel(
            "Export the finished LabelForge dataset into a backend-ready format. "
            "The LabelForge project itself is never modified."
        )
        intro.setObjectName("PageSubtitle")
        intro.setWordWrap(True)
        root.addWidget(intro)

        cards = QHBoxLayout()
        cards.setSpacing(12)

        # ----------------------------------------------------------
        # Facemap
        # ----------------------------------------------------------
        facemap_card = QFrame()
        facemap_card.setObjectName("ExportCard")

        facemap_layout = QVBoxLayout(facemap_card)
        facemap_layout.setContentsMargins(18, 18, 18, 18)
        facemap_layout.setSpacing(10)

        facemap_title = QLabel("Facemap")
        facemap_title.setObjectName("ExportTitle")

        facemap_text = QLabel(
            "Create a training-ready frame folder and label CSV compatible "
            "with the current LabelForge / Facemap training workflow."
        )
        facemap_text.setObjectName("ExportMeta")
        facemap_text.setWordWrap(True)

        self.facemap_summary = QLabel("")
        self.facemap_summary.setObjectName("ExportMeta")
        self.facemap_summary.setWordWrap(True)

        facemap_button = QPushButton("Export Facemap Dataset")
        facemap_button.setObjectName("ExportPrimary")
        facemap_button.clicked.connect(self.export_facemap)

        facemap_layout.addWidget(facemap_title)
        facemap_layout.addWidget(facemap_text)
        facemap_layout.addSpacing(6)
        facemap_layout.addWidget(self.facemap_summary)
        facemap_layout.addStretch(1)
        facemap_layout.addWidget(facemap_button)

        # ----------------------------------------------------------
        # DLC
        # ----------------------------------------------------------
        dlc_card = QFrame()
        dlc_card.setObjectName("ExportCard")

        dlc_layout = QVBoxLayout(dlc_card)
        dlc_layout.setContentsMargins(18, 18, 18, 18)
        dlc_layout.setSpacing(10)

        dlc_title = QLabel("DeepLabCut")
        dlc_title.setObjectName("ExportTitle")

        dlc_text = QLabel(
            "Create a DLC labeled-data package with CollectedData CSV/H5 files. "
            "Not-visible keypoints are exported as missing coordinates."
        )
        dlc_text.setObjectName("ExportMeta")
        dlc_text.setWordWrap(True)

        scorer_label = QLabel("Scorer / annotator name")
        scorer_label.setObjectName("ExportMeta")

        self.dlc_scorer_input = QLineEdit()
        self.dlc_scorer_input.setObjectName("TextInput")
        self.dlc_scorer_input.setText("LabelForge")
        self.dlc_scorer_input.setPlaceholderText("e.g. Mick")

        self.dlc_summary = QLabel("")
        self.dlc_summary.setObjectName("ExportMeta")
        self.dlc_summary.setWordWrap(True)

        dlc_button = QPushButton("Export DLC Dataset")
        dlc_button.setObjectName("ExportPrimary")
        dlc_button.clicked.connect(self.export_dlc)

        dlc_layout.addWidget(dlc_title)
        dlc_layout.addWidget(dlc_text)
        dlc_layout.addSpacing(6)
        dlc_layout.addWidget(scorer_label)
        dlc_layout.addWidget(self.dlc_scorer_input)
        dlc_layout.addWidget(self.dlc_summary)
        dlc_layout.addStretch(1)
        dlc_layout.addWidget(dlc_button)

        cards.addWidget(facemap_card, 1)
        cards.addWidget(dlc_card, 1)

        root.addLayout(cards, 1)

        bottom = QHBoxLayout()

        previous = QPushButton("← Previous")
        previous.setObjectName("BackButton")
        previous.clicked.connect(self.previous_requested.emit)

        self.status_label = QLabel("")
        self.status_label.setObjectName("ExportMeta")
        self.status_label.setAlignment(Qt.AlignCenter)

        bottom.addWidget(previous)
        bottom.addStretch(1)
        bottom.addWidget(self.status_label)
        bottom.addStretch(1)

        root.addLayout(bottom)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def set_context(
        self,
        project_draft: dict,
        keypoint_draft: dict,
        frame_draft: dict,
        label_draft: dict,
    ) -> None:
        self.project_draft = dict(project_draft)
        self.keypoint_draft = dict(keypoint_draft)
        self.frame_draft = dict(frame_draft)
        self.label_draft = dict(label_draft)

        self.keypoint_names = [
            keypoint.get("name", "")
            for group in keypoint_draft.get("groups", [])
            for keypoint in group.get("keypoints", [])
            if keypoint.get("name", "")
        ]

        self.frame_paths = self._load_frame_paths(frame_draft)

        labels_path = label_draft.get("labels_csv", "")
        self.labels_csv = Path(labels_path) if labels_path else None

        if self.labels_csv is None or not self.labels_csv.exists():
            location = project_draft.get("location", "").strip()
            name = project_draft.get("name", "").strip()

            if location and name:
                candidate = (
                    Path(location)
                    / self._safe_name(name)
                    / "labels"
                    / "labels.csv"
                )
                if candidate.exists():
                    self.labels_csv = candidate

        self.facemap_summary.setText(
            f"{len(self.frame_paths)} frame(s)\n"
            f"{len(self.keypoint_names)} keypoint(s)\n"
            "Partial visibility is preserved per keypoint."
        )

        self.dlc_summary.setText(
            f"{len(self.frame_paths)} frame(s)\n"
            f"{len(self.keypoint_names)} bodypart(s)\n"
            "Exports labeled-data + CollectedData CSV/H5."
        )

    # ------------------------------------------------------------------
    # Facemap export
    # ------------------------------------------------------------------

    def export_facemap(self) -> None:
        if not self.frame_paths:
            warning(
                self,
                "No frames",
                "There are no frames available for export.",
            )
            return

        if not self.keypoint_names:
            warning(
                self,
                "No keypoints",
                "There are no keypoints configured for export.",
            )
            return

        if self.labels_csv is None or not self.labels_csv.exists():
            warning(
                self,
                "Labels missing",
                "The LabelForge label CSV could not be found.",
            )
            return

        rows = self._load_label_rows()

        missing = self._validate_complete(rows)

        if missing:
            preview = "\n".join(
                f"Frame {frame_index + 1}: {keypoint}"
                for frame_index, keypoint in missing[:20]
            )

            if len(missing) > 20:
                preview += f"\n... and {len(missing) - 20} more"

            warning(
                self,
                "Export blocked",
                "The dataset still contains unfinished labels:\n\n"
                f"{preview}",
            )
            return

        start_dir = self.project_draft.get("location", "").strip() or str(Path.home())

        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose Facemap export location",
            start_dir,
        )

        if not selected:
            return

        project_name = self._safe_name(
            self.project_draft.get("name", "").strip()
            or "LabelForge_Project"
        )

        export_root = (
            Path(selected)
            / f"{project_name}_Facemap"
        )

        if export_root.exists():
            warning(
                self,
                "Export already exists",
                "This Facemap export folder already exists:\n\n"
                f"{export_root}\n\n"
                "Choose another location or rename/remove the existing export.",
            )
            return

        frames_dir = export_root / "Frames"
        labels_dir = export_root / "Labels"

        frames_dir.mkdir(parents=True, exist_ok=False)
        labels_dir.mkdir(parents=True, exist_ok=False)

        source_rows = {
            self._normalize_path(row.get("png_path", "")): row
            for row in rows
            if row.get("png_path", "").strip()
        }

        export_csv = labels_dir / "labels.csv"

        fieldnames = [
            "png_path",
            "image",
            "image_folder",
        ]

        for keypoint in self.keypoint_names:
            fieldnames.extend(
                [
                    f"{keypoint}_x",
                    f"{keypoint}_y",
                    f"{keypoint}_visible",
                    f"{keypoint}_state",
                ]
            )

        exported_rows: list[dict] = []

        for frame_path in self.frame_paths:
            source_row = source_rows.get(
                self._normalize_path(frame_path),
                {},
            )

            destination_frame = frames_dir / frame_path.name
            shutil.copy2(frame_path, destination_frame)

            row = {
                "png_path": str(destination_frame),
                "image": destination_frame.name,
                "image_folder": str(frames_dir),
            }

            for keypoint in self.keypoint_names:
                state = source_row.get(
                    f"{keypoint}_state",
                    STATE_UNSET,
                ).strip()

                x_text = source_row.get(
                    f"{keypoint}_x",
                    "",
                ).strip()

                y_text = source_row.get(
                    f"{keypoint}_y",
                    "",
                ).strip()

                visible_text = source_row.get(
                    f"{keypoint}_visible",
                    "0",
                ).strip()

                if (
                    state == STATE_VISIBLE
                    and x_text
                    and y_text
                ):
                    row[f"{keypoint}_x"] = x_text
                    row[f"{keypoint}_y"] = y_text
                    row[f"{keypoint}_visible"] = 1
                    row[f"{keypoint}_state"] = STATE_VISIBLE
                else:
                    row[f"{keypoint}_x"] = ""
                    row[f"{keypoint}_y"] = ""
                    row[f"{keypoint}_visible"] = 0
                    row[f"{keypoint}_state"] = STATE_NOT_VISIBLE

            exported_rows.append(row)

        with export_csv.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(exported_rows)

        keypoint_colors = {
            keypoint.get("name", ""): keypoint.get("color", "#D18B47")
            for group in self.keypoint_draft.get("groups", [])
            for keypoint in group.get("keypoints", [])
            if keypoint.get("name", "")
        }

        manifest = {
            "format": "LabelForge Facemap Export",
            "project_name": self.project_draft.get("name", ""),
            "frame_count": len(self.frame_paths),
            "keypoint_count": len(self.keypoint_names),
            "keypoints": self.keypoint_names,
            "keypoint_colors": keypoint_colors,
            "frames_dir": str(frames_dir),
            "labels_csv": str(export_csv),
            "visibility_rule": (
                "visible keypoints retain x/y; not_visible keypoints export "
                "with blank x/y, visible=0, state=not_visible"
            ),
            "note": (
                "Keypoint names and order come directly from the LabelForge project. "
                "A compatible custom Facemap model/trainer must use the same semantic "
                "keypoint order when continuing/fine-tuning an existing model."
            ),
        }

        manifest_path = export_root / "facemap_export.json"

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.status_label.setText(
            f"Facemap export created: {export_root}"
        )

        information(
            self,
            "Facemap export complete",
            "Facemap dataset exported successfully.\n\n"
            f"Frames: {len(self.frame_paths)}\n"
            f"Keypoints: {len(self.keypoint_names)}",
        )

    # ------------------------------------------------------------------
    # DeepLabCut export
    # ------------------------------------------------------------------

    def export_dlc(self) -> None:
        """
        Export a complete single-animal DeepLabCut project package.

        LabelForge remains the source of truth. The export contains:
        - config.yaml
        - labeled-data/<source-video-stem>/frames
        - CollectedData_<scorer>.csv per source video
        - CollectedData_<scorer>.h5 when PyTables is available
        - videos/ folder
        - training-datasets/
        - dlc-models/
        - dlc-models-pytorch/
        - a ready-to-run DLC training script

        Frames are grouped by source video using extraction_manifest.csv.
        """
        if not self.frame_paths:
            warning(
                self,
                "No frames",
                "There are no frames available for export.",
            )
            return

        if not self.keypoint_names:
            warning(
                self,
                "No keypoints",
                "There are no keypoints configured for export.",
            )
            return

        if self.labels_csv is None or not self.labels_csv.exists():
            warning(
                self,
                "Labels missing",
                "The LabelForge label CSV could not be found.",
            )
            return

        scorer = self.dlc_scorer_input.text().strip() or "LabelForge"
        safe_scorer = self._safe_name(scorer)

        rows = self._load_label_rows()
        missing = self._validate_complete(rows)

        if missing:
            preview = "\n".join(
                f"Frame {frame_index + 1}: {keypoint}"
                for frame_index, keypoint in missing[:20]
            )
            if len(missing) > 20:
                preview += f"\n... and {len(missing) - 20} more"

            warning(
                self,
                "Export blocked",
                "The dataset still contains unfinished labels:\n\n"
                f"{preview}",
            )
            return

        manifest_path = Path(
            self.frame_draft.get("manifest_path", "")
        )

        if not manifest_path.is_file():
            warning(
                self,
                "Frame manifest missing",
                "A full DLC project export needs extraction_manifest.csv "
                "so LabelForge can group frames by their source videos.",
            )
            return

        manifest_rows = self._load_manifest_rows(manifest_path)

        if not manifest_rows:
            warning(
                self,
                "Frame manifest empty",
                "The extraction manifest contains no usable frame entries.",
            )
            return

        start_dir = (
            self.project_draft.get("location", "").strip()
            or str(Path.home())
        )

        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose DeepLabCut project export location",
            start_dir,
        )

        if not selected:
            return

        task_name = self._safe_name(
            self.project_draft.get("name", "").strip()
            or "LabelForge_Project"
        )

        date_tag = datetime.now().strftime("%Y-%m-%d")
        export_root = (
            Path(selected)
            / f"{task_name}-{safe_scorer}-{date_tag}"
        )

        if export_root.exists():
            warning(
                self,
                "Export already exists",
                "This DeepLabCut project folder already exists:\n\n"
                f"{export_root}\n\n"
                "Choose another location or rename/remove the existing export.",
            )
            return

        labeled_data_root = export_root / "labeled-data"
        videos_dir = export_root / "videos"
        training_dir = export_root / "training-datasets"
        models_tf_dir = export_root / "dlc-models"
        models_pt_dir = export_root / "dlc-models-pytorch"
        eval_pt_dir = export_root / "evaluation-results-pytorch"

        for directory in [
            labeled_data_root,
            videos_dir,
            training_dir,
            models_tf_dir,
            models_pt_dir,
            eval_pt_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        label_rows_by_path = {
            self._normalize_path(row.get("png_path", "")): row
            for row in rows
            if row.get("png_path", "").strip()
        }

        manifest_by_saved_image = {
            self._normalize_path(row.get("saved_image", "")): row
            for row in manifest_rows
            if row.get("saved_image", "").strip()
        }

        # Keep source-video order exactly as it first appears in the manifest.
        grouped: dict[str, dict] = {}

        for frame_path in self.frame_paths:
            manifest_row = manifest_by_saved_image.get(
                self._normalize_path(frame_path)
            )

            if manifest_row is None:
                warning(
                    self,
                    "Manifest mismatch",
                    "Could not find this labeled frame in extraction_manifest.csv:\n\n"
                    f"{frame_path}",
                )
                shutil.rmtree(export_root, ignore_errors=True)
                return

            video_path_text = manifest_row.get("video_path", "").strip()
            video_name = manifest_row.get("video_name", "").strip()

            if not video_path_text:
                warning(
                    self,
                    "Source video missing",
                    "The extraction manifest does not contain a source video path "
                    f"for:\n\n{frame_path.name}",
                )
                shutil.rmtree(export_root, ignore_errors=True)
                return

            video_path = Path(video_path_text)
            video_stem = self._safe_name(
                Path(video_name or video_path.name).stem
            )

            # Two videos with identical stems must not silently collide.
            group_key = self._normalize_path(video_path)
            if group_key not in grouped:
                grouped[group_key] = {
                    "video_path": video_path,
                    "video_name": video_name or video_path.name,
                    "folder_name": video_stem,
                    "frames": [],
                }

            grouped[group_key]["frames"].append(
                (frame_path, manifest_row)
            )

        # Make duplicate folder stems unique, if necessary.
        used_folder_names: set[str] = set()
        for group in grouped.values():
            base = group["folder_name"]
            candidate = base
            counter = 2
            while candidate.lower() in used_folder_names:
                candidate = f"{base}_{counter}"
                counter += 1
            group["folder_name"] = candidate
            used_folder_names.add(candidate.lower())

        columns = pd.MultiIndex.from_product(
            [
                [scorer],
                self.keypoint_names,
                ["x", "y"],
            ],
            names=["scorer", "bodyparts", "coords"],
        )

        h5_success_count = 0
        h5_failures: list[str] = []
        config_video_sets: list[tuple[str, int, int]] = []
        exported_frame_count = 0

        for group in grouped.values():
            video_path: Path = group["video_path"]
            folder_name: str = group["folder_name"]
            folder = labeled_data_root / folder_name
            folder.mkdir(parents=True, exist_ok=False)

            data_rows: list[list[float]] = []
            index_rows: list[tuple[str, str, str]] = []
            first_size: tuple[int, int] | None = None

            for frame_path, _manifest_row in group["frames"]:
                source_row = label_rows_by_path.get(
                    self._normalize_path(frame_path),
                    {},
                )

                destination_frame = folder / frame_path.name
                shutil.copy2(frame_path, destination_frame)

                if first_size is None:
                    with Image.open(destination_frame) as image:
                        first_size = image.size

                index_rows.append(
                    (
                        "labeled-data",
                        folder_name,
                        destination_frame.name,
                    )
                )

                values: list[float] = []

                for keypoint in self.keypoint_names:
                    state = source_row.get(
                        f"{keypoint}_state",
                        STATE_UNSET,
                    ).strip()

                    x_text = source_row.get(
                        f"{keypoint}_x",
                        "",
                    ).strip()
                    y_text = source_row.get(
                        f"{keypoint}_y",
                        "",
                    ).strip()

                    if (
                        state == STATE_VISIBLE
                        and x_text
                        and y_text
                    ):
                        values.extend(
                            [float(x_text), float(y_text)]
                        )
                    else:
                        values.extend(
                            [np.nan, np.nan]
                        )

                data_rows.append(values)
                exported_frame_count += 1

            index = pd.MultiIndex.from_tuples(
                index_rows,
                names=["labeled-data", "folder", "image"],
            )

            dataframe = pd.DataFrame(
                data_rows,
                index=index,
                columns=columns,
                dtype=float,
            )

            collected_base = (
                folder
                / f"CollectedData_{safe_scorer}"
            )
            csv_path = collected_base.with_suffix(".csv")
            h5_path = collected_base.with_suffix(".h5")

            dataframe.to_csv(csv_path)

            try:
                dataframe.to_hdf(
                    h5_path,
                    key="df_with_missing",
                    format="table",
                    mode="w",
                )
                h5_success_count += 1
            except Exception as error:
                h5_failures.append(
                    f"{folder_name}: {error}"
                )

            width, height = first_size or (0, 0)
            config_video_sets.append(
                (
                    str(video_path),
                    width,
                    height,
                )
            )

        config_path = export_root / "config.yaml"
        config_path.write_text(
            self._build_dlc_config_yaml(
                task_name=task_name,
                scorer=scorer,
                project_path=export_root,
                video_sets=config_video_sets,
                bodyparts=self.keypoint_names,
                numframes=len(self.frame_paths),
                date_tag=date_tag,
            ),
            encoding="utf-8",
        )

        bodyparts_path = export_root / "bodyparts.txt"
        bodyparts_path.write_text(
            "\n".join(self.keypoint_names) + "\n",
            encoding="utf-8",
        )

        # Small runner for the future Training Workspace and for manual testing.
        runner_path = export_root / "run_dlc_training.py"
        runner_path.write_text(
            self._build_dlc_training_runner(scorer),
            encoding="utf-8",
        )

        smoke_path = export_root / "check_dlc_project.py"
        smoke_path.write_text(
            self._build_dlc_project_smoke_runner(scorer),
            encoding="utf-8",
        )

        export_info = {
            "format": "LabelForge Full DeepLabCut Project Export",
            "task": task_name,
            "scorer": scorer,
            "project_path": str(export_root),
            "config_yaml": str(config_path),
            "frame_count": exported_frame_count,
            "bodyparts": self.keypoint_names,
            "source_video_count": len(grouped),
            "source_videos": [
                {
                    "video_path": str(group["video_path"]),
                    "video_name": group["video_name"],
                    "labeled_data_folder": group["folder_name"],
                    "frame_count": len(group["frames"]),
                }
                for group in grouped.values()
            ],
            "h5_created_for_all_folders": (
                h5_success_count == len(grouped)
            ),
            "h5_failures": h5_failures,
            "training_entrypoint": str(runner_path),
            "smoke_test_entrypoint": str(smoke_path),
            "note": (
                "This export is designed as a backend-ready single-animal DLC "
                "project. Source videos are referenced from config.yaml rather "
                "than duplicated. The generated run_dlc_training.py can convert "
                "CSV labels to H5 inside a DLC environment before creating the "
                "training dataset."
            ),
        }

        info_path = export_root / "labelforge_dlc_export.json"
        info_path.write_text(
            json.dumps(export_info, indent=2),
            encoding="utf-8",
        )

        self.status_label.setText(
            f"DLC project exported: {export_root}"
        )

        information(
            self,
            "DLC project export complete",
            "DeepLabCut project exported successfully.\n\n"
            f"Frames: {exported_frame_count}\n"
            f"Source videos: {len(grouped)}\n"
            f"Bodyparts: {len(self.keypoint_names)}\n"
            f"Scorer: {scorer}\n\n"
            "Project is ready for DeepLabCut.",
        )

    def _load_manifest_rows(
        self,
        manifest_path: Path,
    ) -> list[dict[str, str]]:
        with manifest_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _yaml_quote(value: str | Path) -> str:
        text = str(value)
        return "'" + text.replace("'", "''") + "'"

    def _build_dlc_config_yaml(
        self,
        *,
        task_name: str,
        scorer: str,
        project_path: Path,
        video_sets: list[tuple[str, int, int]],
        bodyparts: list[str],
        numframes: int,
        date_tag: str,
    ) -> str:
        lines = [
            "# LabelForge-generated DeepLabCut project",
            "",
            f"Task: {self._yaml_quote(task_name)}",
            f"scorer: {self._yaml_quote(scorer)}",
            f"date: {self._yaml_quote(date_tag)}",
            "multianimalproject: false",
            "identity:",
            "",
            f"project_path: {self._yaml_quote(project_path)}",
            "",
            "engine: pytorch",
            "",
            "video_sets:",
        ]

        for video_path, width, height in video_sets:
            lines.append(
                f"  {self._yaml_quote(video_path)}:"
            )
            lines.append(
                f"    crop: 0, {width}, 0, {height}"
            )

        lines.extend(
            [
                "",
                "bodyparts:",
            ]
        )

        for bodypart in bodyparts:
            lines.append(
                f"  - {self._yaml_quote(bodypart)}"
            )

        lines.extend(
            [
                "",
                "start: 0",
                "stop: 1",
                f"numframes2pick: {max(1, int(numframes))}",
                "",
                "skeleton: []",
                "skeleton_color: black",
                "pcutoff: 0.6",
                "dotsize: 6",
                "alphavalue: 0.7",
                "colormap: rainbow",
                "",
                "TrainingFraction:",
                "  - 0.95",
                "iteration: 0",
                "default_net_type: resnet_50",
                "default_augmenter: default",
                "snapshotindex: -1",
                "detector_snapshotindex: -1",
                "batch_size: 8",
                "detector_batch_size: 1",
                "",
                "cropping: false",
                "x1: 0",
                "x2: 640",
                "y1: 277",
                "y2: 624",
                "",
                "corner2move2:",
                "  - 50",
                "  - 50",
                "move2corner: true",
                "",
                "SuperAnimalConversionTables:",
                "",
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def _build_dlc_training_runner(scorer: str) -> str:
        return f"""from pathlib import Path
import deeplabcut

PROJECT = Path(__file__).resolve().parent
CONFIG = PROJECT / "config.yaml"
SCORER = {scorer!r}

print("LabelForge DLC training package")
print("Config:", CONFIG)

# Ensure externally generated CSV labels have DLC-native H5 companions.
deeplabcut.convertcsv2h5(
    str(CONFIG),
    scorer=SCORER,
)

# Build the actual DLC train/test dataset on THIS machine.
deeplabcut.create_training_dataset(
    str(CONFIG),
)

print()
print("Training dataset created successfully.")
print("Start training with:")
print(f"deeplabcut.train_network(r'{{CONFIG}}')")
"""

    @staticmethod
    def _build_dlc_project_smoke_runner(scorer: str) -> str:
        return f"""from pathlib import Path
import pandas as pd
import deeplabcut

PROJECT = Path(__file__).resolve().parent
CONFIG = PROJECT / "config.yaml"
SCORER = {scorer!r}


def convert_label_csvs_to_h5():
    labeled_data = PROJECT / "labeled-data"
    csv_files = sorted(
        labeled_data.glob(f"*/CollectedData_{{SCORER}}.csv")
    )

    if not csv_files:
        raise RuntimeError(
            f"No CollectedData_{{SCORER}}.csv files found."
        )

    print(f"Found {{len(csv_files)}} labeled-data folder(s).")

    for csv_path in csv_files:
        print(f"\\nReading: {{csv_path}}")
        df = pd.read_csv(
            csv_path,
            header=[0, 1, 2],
            index_col=[0, 1, 2],
        )
        h5_path = csv_path.with_suffix(".h5")
        df.to_hdf(
            h5_path,
            key="df_with_missing",
            mode="w",
        )

        check = pd.read_hdf(
            h5_path,
            key="df_with_missing",
        )

        if list(df.index) != list(check.index):
            raise RuntimeError(
                f"H5 row index mismatch: {{h5_path}}"
            )
        if list(df.columns) != list(check.columns):
            raise RuntimeError(
                f"H5 column mismatch: {{h5_path}}"
            )

        print(
            f"✓ H5 created and re-opened: {{h5_path.name}}"
        )


def main():
    print("=== LabelForge -> DeepLabCut real smoke test ===")
    print(f"Project: {{PROJECT}}")
    print(f"Config:  {{CONFIG}}\\n")

    if not CONFIG.is_file():
        raise RuntimeError(
            f"config.yaml not found: {{CONFIG}}"
        )

    print(f"DeepLabCut version: {{deeplabcut.__version__}}")

    print("\\nSTEP 1/2 - Create DLC H5 label files")
    convert_label_csvs_to_h5()

    print(
        "\\nSTEP 2/2 - Ask DeepLabCut to create the training dataset"
    )
    print("No network training will be started.\\n")

    deeplabcut.create_training_dataset(str(CONFIG))

    print("\\n==============================================")
    print("LABELFORGE -> DEEPLABCUT READY TO TRAIN")
    print("==============================================")
    print("DeepLabCut accepted the LabelForge project and")
    print("successfully created its training dataset.")
    print("No model training was started.")


if __name__ == "__main__":
    main()
"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_label_rows(self) -> list[dict[str, str]]:
        assert self.labels_csv is not None

        with self.labels_csv.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            return list(csv.DictReader(handle))

    def _validate_complete(
        self,
        rows: list[dict[str, str]],
    ) -> list[tuple[int, str]]:
        rows_by_path = {
            self._normalize_path(row.get("png_path", "")): row
            for row in rows
            if row.get("png_path", "").strip()
        }

        missing: list[tuple[int, str]] = []

        for frame_index, frame_path in enumerate(self.frame_paths):
            row = rows_by_path.get(
                self._normalize_path(frame_path),
                {},
            )

            for keypoint in self.keypoint_names:
                state = row.get(
                    f"{keypoint}_state",
                    "",
                ).strip()

                if state not in {
                    STATE_VISIBLE,
                    STATE_NOT_VISIBLE,
                }:
                    missing.append(
                        (frame_index, keypoint)
                    )

        return missing

    def _load_frame_paths(
        self,
        frame_draft: dict,
    ) -> list[Path]:
        manifest = frame_draft.get("manifest_path", "")

        if manifest and Path(manifest).exists():
            paths: list[Path] = []

            with Path(manifest).open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                for row in csv.DictReader(handle):
                    saved_image = row.get(
                        "saved_image",
                        "",
                    ).strip()

                    if saved_image and Path(saved_image).exists():
                        paths.append(Path(saved_image))

            return paths

        output_dir = frame_draft.get("output_dir", "")

        if output_dir and Path(output_dir).exists():
            return sorted(
                Path(output_dir).glob("*.png")
            )

        return []

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        return str(Path(path)).replace("/", "\\").lower()

    @staticmethod
    def _safe_name(text: str) -> str:
        safe = "".join(
            character
            if character.isalnum() or character in "._- "
            else "_"
            for character in text
        )
        return safe.strip().replace(" ", "_") or "LabelForge_Project"
