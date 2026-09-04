from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


STATE_UNSET = "unset"
STATE_VISIBLE = "visible"
STATE_NOT_VISIBLE = "not_visible"


class ReviewPage(QWidget):
    previous_requested = Signal()
    export_requested = Signal()
    fix_requested = Signal(int, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.project_draft: dict = {}
        self.keypoint_draft: dict = {}
        self.frame_draft: dict = {}
        self.label_draft: dict = {}

        self.frame_paths: list[Path] = []
        self.keypoint_names: list[str] = []
        self.labels_csv: Path | None = None
        self.missing: list[tuple[int, Path, str]] = []

        self.setStyleSheet("""
            QFrame#ReviewCard {
                background: #1D2026;
                border: 1px solid #2D323A;
                border-radius: 14px;
            }

            QFrame#ReviewIssue {
                background: #14161A;
                border: 1px solid #3A4049;
                border-radius: 10px;
            }

            QLabel#ReviewBigNumber {
                color: #EAEAEA;
                font-size: 24px;
                font-weight: 700;
            }

            QLabel#ReviewGood {
                color: #7FD69A;
                font-weight: 700;
            }

            QLabel#ReviewWarning {
                color: #D18B47;
                font-weight: 700;
            }

            QLabel#ReviewMeta {
                color: #AEB4BF;
                font-size: 12px;
            }

            QPushButton#ReviewFixButton {
                background: #2A2E35;
                color: #EAEAEA;
                border: 1px solid #3A4049;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
            }

            QPushButton#ReviewFixButton:hover {
                border: 1px solid #D18B47;
                background: #343942;
            }

            QFrame#ReviewProgressPanel {
                background: #1D2026;
                border: 1px solid #2D323A;
                border-radius: 14px;
            }

            QLabel#ReviewScoreTitle {
                color: #AEB4BF;
                font-size: 11px;
                font-weight: 700;
            }

            QLabel#ReviewScore {
                color: #F2F2F2;
                font-size: 27px;
                font-weight: 700;
            }

            QLabel#ReviewScoreGood {
                color: #F2F2F2;
                font-size: 27px;
                font-weight: 700;
            }

            QProgressBar#ReviewProgress {
                min-height: 7px;
                max-height: 7px;
                border: none;
                border-radius: 3px;
                background: #2D323A;
                text-align: center;
            }

            QProgressBar#ReviewProgress::chunk {
                background: #D18B47;
                border-radius: 3px;
            }

            QLabel#ReviewIssueTitle {
                color: #EAEAEA;
                font-size: 13px;
                font-weight: 700;
            }

            QLabel#ReviewMissingTag {
                color: #D18B47;
                font-size: 12px;
                font-weight: 700;
            }
        """)

        self.build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        intro = QLabel(
            "Final dataset check before export. LabelForge verifies that every "
            "Frame × Keypoint assignment is accounted for."
        )
        intro.setObjectName("PageSubtitle")
        intro.setWordWrap(True)
        root.addWidget(intro)

        # --------------------------------------------------------------
        # Dataset progress cards
        # --------------------------------------------------------------
        progress_title = QLabel("DATASET PROGRESS")
        progress_title.setObjectName("ReviewScoreTitle")
        root.addWidget(progress_title)

        score_row = QHBoxLayout()
        score_row.setSpacing(10)

        # FRAMES card
        frames_card = QFrame()
        frames_card.setObjectName("ReviewProgressPanel")
        frames_layout = QVBoxLayout(frames_card)
        frames_layout.setContentsMargins(16, 13, 16, 13)
        frames_layout.setSpacing(5)

        frames_title = QLabel("FRAMES")
        frames_title.setObjectName("ReviewScoreTitle")
        self.frames_value = QLabel("—")
        self.frames_value.setObjectName("ReviewScore")
        self.frames_progress = QProgressBar()
        self.frames_progress.setObjectName("ReviewProgress")
        self.frames_progress.setRange(0, 100)
        self.frames_progress.setTextVisible(False)
        self.frames_detail = QLabel("")
        self.frames_detail.setObjectName("ReviewMeta")

        frames_layout.addWidget(frames_title)
        frames_layout.addWidget(self.frames_value)
        frames_layout.addWidget(self.frames_progress)
        frames_layout.addWidget(self.frames_detail)

        # LABELS card
        labels_card = QFrame()
        labels_card.setObjectName("ReviewProgressPanel")
        labels_layout = QVBoxLayout(labels_card)
        labels_layout.setContentsMargins(16, 13, 16, 13)
        labels_layout.setSpacing(5)

        labels_title = QLabel("LABELS")
        labels_title.setObjectName("ReviewScoreTitle")
        self.annotations_value = QLabel("—")
        self.annotations_value.setObjectName("ReviewScore")
        self.annotations_progress = QProgressBar()
        self.annotations_progress.setObjectName("ReviewProgress")
        self.annotations_progress.setRange(0, 100)
        self.annotations_progress.setTextVisible(False)
        self.annotations_detail = QLabel("")
        self.annotations_detail.setObjectName("ReviewMeta")

        labels_layout.addWidget(labels_title)
        labels_layout.addWidget(self.annotations_value)
        labels_layout.addWidget(self.annotations_progress)
        labels_layout.addWidget(self.annotations_detail)

        # STATUS card
        status_card = QFrame()
        status_card.setObjectName("ReviewProgressPanel")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(16, 13, 16, 13)
        status_layout.setSpacing(5)

        status_title = QLabel("STATUS")
        status_title.setObjectName("ReviewScoreTitle")
        self.status_value = QLabel("—")
        self.status_value.setObjectName("ReviewWarning")
        self.status_detail = QLabel("")
        self.status_detail.setObjectName("ReviewMeta")
        self.visibility_value = QLabel("")
        self.visibility_value.setObjectName("ReviewMeta")
        self.visibility_detail = QLabel("")
        self.visibility_detail.setObjectName("ReviewMeta")

        status_layout.addWidget(status_title)
        status_layout.addWidget(self.status_value)
        status_layout.addWidget(self.status_detail)
        status_layout.addStretch(1)
        status_layout.addWidget(self.visibility_value)

        score_row.addWidget(frames_card, 2)
        score_row.addWidget(labels_card, 2)
        score_row.addWidget(status_card, 1)

        root.addLayout(score_row)

        # --------------------------------------------------------------
        # Issues grouped by frame
        # --------------------------------------------------------------
        issues_panel = QFrame()
        issues_panel.setObjectName("ReviewCard")

        issues_layout = QVBoxLayout(issues_panel)
        issues_layout.setContentsMargins(14, 14, 14, 14)
        issues_layout.setSpacing(8)

        issues_header = QHBoxLayout()

        issues_title = QLabel("Issues")
        issues_title.setObjectName("FieldLabel")

        self.issue_count_label = QLabel("")
        self.issue_count_label.setObjectName("ReviewMeta")
        self.issue_count_label.setAlignment(Qt.AlignRight)

        issues_header.addWidget(issues_title)
        issues_header.addStretch(1)
        issues_header.addWidget(self.issue_count_label)
        issues_layout.addLayout(issues_header)

        self.issue_scroll = QScrollArea()
        self.issue_scroll.setWidgetResizable(True)
        self.issue_scroll.setFrameShape(QFrame.NoFrame)
        self.issue_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollArea > QWidget > QWidget { background: transparent; }"
        )
        self.issue_scroll.viewport().setStyleSheet("background: transparent;")

        self.issue_container = QWidget()
        self.issue_container.setStyleSheet("background: transparent;")

        self.issue_list = QVBoxLayout(self.issue_container)
        self.issue_list.setContentsMargins(0, 4, 0, 0)
        self.issue_list.setSpacing(7)
        self.issue_list.addStretch(1)

        self.issue_scroll.setWidget(self.issue_container)
        issues_layout.addWidget(self.issue_scroll, 1)

        self.no_issues_label = QLabel(
            "Everything is accounted for. This dataset is ready for export. 🔥"
        )
        self.no_issues_label.setObjectName("ReviewGood")
        self.no_issues_label.setAlignment(Qt.AlignCenter)
        self.no_issues_label.setWordWrap(True)
        self.no_issues_label.setVisible(False)
        issues_layout.addWidget(self.no_issues_label, 1)

        root.addWidget(issues_panel, 1)

        bottom = QHBoxLayout()

        previous = QPushButton("← Previous")
        previous.setObjectName("BackButton")
        previous.clicked.connect(self.previous_requested.emit)

        self.ready_label = QLabel("")
        self.ready_label.setAlignment(Qt.AlignCenter)
        self.ready_label.setObjectName("ReviewMeta")

        self.export_button = QPushButton("Continue to Export →")
        self.export_button.setObjectName("PrimaryNextButton")
        self.export_button.clicked.connect(self.export_requested.emit)

        bottom.addWidget(previous)
        bottom.addStretch(1)
        bottom.addWidget(self.ready_label)
        bottom.addStretch(1)
        bottom.addWidget(self.export_button)

        root.addLayout(bottom)

    # Kept for compatibility with older code that may still call it.
    def _summary_card(self, title: str):
        card = QFrame()
        card.setObjectName("ReviewCard")
        layout = QVBoxLayout(card)
        value = QLabel("—")
        detail = QLabel("")
        layout.addWidget(QLabel(title))
        layout.addWidget(value)
        layout.addWidget(detail)
        return card, value, detail

    # ------------------------------------------------------------------
    # Context / analysis
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

        if (
            self.labels_csv is None
            or not self.labels_csv.exists()
        ):
            # Fallback to the standard LabelForge project path.
            location = project_draft.get("location", "").strip()
            name = project_draft.get("name", "").strip()

            if location and name:
                candidate = Path(location) / name / "labels" / "labels.csv"
                if candidate.exists():
                    self.labels_csv = candidate

        stats = self._calculate_stats()
        self._render_stats(stats)

    def _load_frame_paths(self, frame_draft: dict) -> list[Path]:
        manifest = frame_draft.get("manifest_path", "")

        if manifest and Path(manifest).exists():
            paths: list[Path] = []

            with Path(manifest).open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                for row in csv.DictReader(handle):
                    saved_image = row.get("saved_image", "").strip()
                    if saved_image and Path(saved_image).exists():
                        paths.append(Path(saved_image))

            return paths

        output_dir = frame_draft.get("output_dir", "")

        if output_dir and Path(output_dir).exists():
            return sorted(Path(output_dir).glob("*.png"))

        return []

    @staticmethod
    def _normalize_path(path: str | Path) -> str:
        return str(Path(path)).replace("/", "\\").lower()

    def _calculate_stats(self) -> dict:
        rows_by_path: dict[str, dict] = {}

        if self.labels_csv is not None and self.labels_csv.exists():
            with self.labels_csv.open(
                "r",
                newline="",
                encoding="utf-8-sig",
            ) as handle:
                for row in csv.DictReader(handle):
                    png_path = row.get("png_path", "").strip()

                    if png_path:
                        rows_by_path[self._normalize_path(png_path)] = row

        visible = 0
        not_visible = 0
        missing: list[tuple[int, Path, str]] = []

        for frame_index, frame_path in enumerate(self.frame_paths):
            row = rows_by_path.get(
                self._normalize_path(frame_path),
                {},
            )

            for name in self.keypoint_names:
                state = row.get(f"{name}_state", "").strip()

                if state == STATE_VISIBLE:
                    visible += 1
                elif state == STATE_NOT_VISIBLE:
                    not_visible += 1
                else:
                    missing.append(
                        (frame_index, frame_path, name)
                    )

        total = len(self.frame_paths) * len(self.keypoint_names)
        completed = visible + not_visible

        completed_frames = 0

        for frame_index, frame_path in enumerate(self.frame_paths):
            if not any(
                issue_frame == frame_index
                for issue_frame, _path, _name in missing
            ):
                completed_frames += 1

        self.missing = missing

        return {
            "frame_count": len(self.frame_paths),
            "completed_frames": completed_frames,
            "total_annotations": total,
            "completed_annotations": completed,
            "visible": visible,
            "not_visible": not_visible,
            "missing": missing,
        }

    def _render_stats(self, stats: dict) -> None:
        frames = stats["frame_count"]
        completed_frames = stats["completed_frames"]
        total = stats["total_annotations"]
        completed = stats["completed_annotations"]
        not_visible = stats["not_visible"]
        missing = stats["missing"]

        frame_percent = round((completed_frames / frames) * 100) if frames else 0
        annotation_percent = round((completed / total) * 100) if total else 0

        frame_fire = " 🔥" if frames and completed_frames == frames else ""
        annotation_fire = " 🔥" if total and completed == total else ""

        self.frames_value.setText(f"{completed_frames} / {frames}{frame_fire}")
        self.frames_progress.setValue(frame_percent)
        remaining_frames = max(frames - completed_frames, 0)
        self.frames_detail.setText(
            "All frames complete"
            if frames and remaining_frames == 0
            else f"{remaining_frames} frame(s) incomplete · {frame_percent}%"
        )

        self.annotations_value.setText(
            f"{completed} / {total}{annotation_fire}"
        )
        self.annotations_progress.setValue(annotation_percent)
        remaining_annotations = max(total - completed, 0)
        self.annotations_detail.setText(
            "All labels complete"
            if total and remaining_annotations == 0
            else f"{remaining_annotations} label(s) remaining · "
                 f"{annotation_percent}%"
        )

        self.visibility_value.setText(
            f"{not_visible} marked not visible"
        )
        self.visibility_detail.setText("")

        if missing:
            affected_frames = len({frame_index for frame_index, _, _ in missing})
            self.status_value.setText("Needs attention")
            self.status_value.setObjectName("ReviewWarning")
            self.status_detail.setText(
                f"{len(missing)} missing label(s) across "
                f"{affected_frames} frame(s)"
            )
            self.ready_label.setText(
                "Resolve the remaining items before export."
            )
            self.export_button.setEnabled(False)
        else:
            self.status_value.setText("Ready for export")
            self.status_value.setObjectName("ReviewGood")
            self.status_detail.setText("Dataset check complete")
            self.ready_label.setText("Ready for export ✓")
            self.export_button.setEnabled(True)

        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)

        self._clear_issue_rows()

        if missing:
            grouped: dict[int, dict] = {}
            for frame_index, frame_path, keypoint_name in missing:
                entry = grouped.setdefault(
                    frame_index,
                    {"path": frame_path, "keypoints": []},
                )
                entry["keypoints"].append(keypoint_name)

            self.issue_count_label.setText(
                f"{len(missing)} issue(s) · {len(grouped)} frame(s)"
            )
            self.issue_scroll.setVisible(True)
            self.no_issues_label.setVisible(False)

            for frame_index, data in grouped.items():
                self._add_issue_group(
                    frame_index,
                    data["path"],
                    data["keypoints"],
                )
        else:
            self.issue_count_label.setText("No issues")
            self.issue_scroll.setVisible(False)
            self.no_issues_label.setVisible(True)

    def _clear_issue_rows(self) -> None:
        while self.issue_list.count() > 1:
            item = self.issue_list.takeAt(0)
            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def _add_issue_group(
        self,
        frame_index: int,
        frame_path: Path,
        keypoint_names: list[str],
    ) -> None:
        row = QFrame()
        row.setObjectName("ReviewIssue")

        outer = QHBoxLayout(row)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(14)

        text_box = QVBoxLayout()
        text_box.setSpacing(4)

        title_row = QHBoxLayout()
        frame_title = QLabel(
            f"Frame {frame_index + 1} · {frame_path.name}"
        )
        frame_title.setObjectName("ReviewIssueTitle")

        missing_tag = QLabel(f"{len(keypoint_names)} missing")
        missing_tag.setObjectName("ReviewMissingTag")

        title_row.addWidget(frame_title, 1)
        title_row.addWidget(missing_tag)

        missing_text = QLabel(
            "Missing: " + "  ·  ".join(keypoint_names)
        )
        missing_text.setObjectName("ReviewMeta")
        missing_text.setWordWrap(True)

        text_box.addLayout(title_row)
        text_box.addWidget(missing_text)

        fix = QPushButton("Go to frame →")
        fix.setObjectName("ReviewFixButton")
        first_keypoint = keypoint_names[0] if keypoint_names else ""
        fix.clicked.connect(
            lambda checked=False,
            idx=frame_index,
            name=first_keypoint: self.fix_requested.emit(idx, name)
        )

        outer.addLayout(text_box, 1)
        outer.addWidget(fix)

        self.issue_list.insertWidget(
            self.issue_list.count() - 1,
            row,
        )
