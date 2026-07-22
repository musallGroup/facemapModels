"""
Tongue Frame Tinder GUI
=======================

Purpose
-------
1. Add up to three Cam0 videos.
2. Extract frames from a chosen interval (default: 60 seconds).
3. Review each frame Tinder-style:
   - KEEP: move to "Kept"
   - REJECT: move to "Rejected"
4. Undo the last decision at any time.

Nothing is permanently deleted during review. Rejected frames are moved into a
separate folder so accidental clicks can be recovered safely.

Keyboard shortcuts
------------------
Right arrow / D / K : KEEP
Left arrow  / A / R : REJECT
Backspace / U       : Undo
Escape              : Exit

Dependencies
------------
- Python 3
- opencv-python
- Pillow

Typical installation inside the facemap environment:
    pip install opencv-python pillow
"""

from __future__ import annotations

import csv
import shutil
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "Tongue Frame Tinder"
MAX_VIDEOS = 3
DEFAULT_DURATION_SECONDS = 60.0
DEFAULT_FRAME_STEP = 1
IMAGE_EXTENSION = ".png"
PNG_COMPRESSION = 3


@dataclass
class VideoJob:
    video_path: Path
    start_minutes: float
    duration_seconds: float
    frame_step: int


@dataclass
class Decision:
    source: Path
    destination: Path
    action: str


def safe_name(text: str) -> str:
    """Create a Windows-safe folder/file component."""
    invalid = '<>:"/\\|?*'
    cleaned = "".join("_" if char in invalid else char for char in text)
    cleaned = cleaned.strip().strip(".")
    return cleaned or "video"


def unique_destination(folder: Path, filename: str) -> Path:
    """Avoid overwriting an existing frame."""
    candidate = folder / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class TongueFrameTinderApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1280x860")
        self.minsize(950, 650)

        self.jobs: list[VideoJob] = []
        self.review_root: Optional[Path] = None
        self.pending_dir: Optional[Path] = None
        self.kept_dir: Optional[Path] = None
        self.rejected_dir: Optional[Path] = None
        self.log_path: Optional[Path] = None

        self.pending_frames: list[Path] = []
        self.current_frame: Optional[Path] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.history: list[Decision] = []
        self.review_active = False
        self.extraction_running = False

        self.status_var = tk.StringVar(value="Videos hinzufügen oder vorhandenen Review-Ordner öffnen.")
        self.progress_var = tk.StringVar(value="Noch keine Frames geladen.")
        self.video_count_var = tk.StringVar(value=f"0/{MAX_VIDEOS} Videos")
        self.current_name_var = tk.StringVar(value="")
        self.start_minutes_var = tk.StringVar(value="0")
        self.duration_seconds_var = tk.StringVar(value=str(DEFAULT_DURATION_SECONDS))
        self.frame_step_var = tk.StringVar(value=str(DEFAULT_FRAME_STEP))

        self._build_ui()
        self._bind_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        setup = ttk.LabelFrame(outer, text="1. Videos und Extraktion", padding=10)
        setup.pack(fill=tk.X)

        controls = ttk.Frame(setup)
        controls.pack(fill=tk.X)

        ttk.Button(
            controls,
            text="Video hinzufügen",
            command=self.add_video,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            controls,
            text="Ausgewähltes Video entfernen",
            command=self.remove_selected_video,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            controls,
            text="Vorhandenen Review-Ordner öffnen",
            command=self.open_existing_review,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(controls, textvariable=self.video_count_var).pack(side=tk.RIGHT)

        params = ttk.Frame(setup)
        params.pack(fill=tk.X, pady=(10, 8))

        ttk.Label(params, text="Start (Minuten):").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.start_minutes_var, width=10).grid(
            row=0, column=1, padx=(5, 18), sticky="w"
        )

        ttk.Label(params, text="Dauer (Sekunden):").grid(row=0, column=2, sticky="w")
        ttk.Entry(params, textvariable=self.duration_seconds_var, width=10).grid(
            row=0, column=3, padx=(5, 18), sticky="w"
        )

        ttk.Label(params, text="Jedes n-te Frame:").grid(row=0, column=4, sticky="w")
        ttk.Entry(params, textvariable=self.frame_step_var, width=10).grid(
            row=0, column=5, padx=(5, 18), sticky="w"
        )

        ttk.Label(
            params,
            text="1 = wirklich jedes Frame; 2 = jedes zweite; 3 = jedes dritte",
        ).grid(row=0, column=6, sticky="w")

        table_frame = ttk.Frame(setup)
        table_frame.pack(fill=tk.X)

        self.video_table = ttk.Treeview(
            table_frame,
            columns=("video", "start", "duration", "step"),
            show="headings",
            height=4,
        )
        self.video_table.heading("video", text="Video")
        self.video_table.heading("start", text="Start (min)")
        self.video_table.heading("duration", text="Dauer (s)")
        self.video_table.heading("step", text="Frame-Schritt")
        self.video_table.column("video", width=720)
        self.video_table.column("start", width=90, anchor=tk.CENTER)
        self.video_table.column("duration", width=90, anchor=tk.CENTER)
        self.video_table.column("step", width=100, anchor=tk.CENTER)
        self.video_table.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scrollbar = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self.video_table.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.video_table.configure(yscrollcommand=scrollbar.set)

        extraction_row = ttk.Frame(setup)
        extraction_row.pack(fill=tk.X, pady=(10, 0))

        self.extract_button = ttk.Button(
            extraction_row,
            text="Frames extrahieren und Review starten",
            command=self.start_extraction,
        )
        self.extract_button.pack(side=tk.LEFT)

        self.extraction_progress = ttk.Progressbar(
            extraction_row,
            mode="determinate",
            length=420,
        )
        self.extraction_progress.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        # Review area
        review = ttk.LabelFrame(outer, text="2. Frame Review", padding=10)
        review.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        header = ttk.Frame(review)
        header.pack(fill=tk.X)

        ttk.Label(
            header,
            textvariable=self.progress_var,
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.LEFT)

        ttk.Label(
            header,
            textvariable=self.current_name_var,
        ).pack(side=tk.RIGHT)

        image_container = ttk.Frame(review)
        image_container.pack(fill=tk.BOTH, expand=True, pady=10)

        self.image_label = ttk.Label(
            image_container,
            text="Noch kein Frame geladen.",
            anchor=tk.CENTER,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(review)
        buttons.pack(fill=tk.X)

        self.reject_button = tk.Button(
            buttons,
            text="← NICHT HALTEN",
            command=self.reject_current,
            font=("Segoe UI", 14, "bold"),
            height=2,
        )
        self.reject_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.undo_button = ttk.Button(
            buttons,
            text="Letzte Entscheidung rückgängig",
            command=self.undo_last,
        )
        self.undo_button.pack(side=tk.LEFT, padx=6)

        self.keep_button = tk.Button(
            buttons,
            text="HALTEN →",
            command=self.keep_current,
            font=("Segoe UI", 14, "bold"),
            height=2,
        )
        self.keep_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        shortcuts = ttk.Label(
            review,
            text=(
                "Tastatur: ← / A / R = nicht halten | "
                "→ / D / K = halten | Rücktaste / U = rückgängig"
            ),
            anchor=tk.CENTER,
        )
        shortcuts.pack(fill=tk.X, pady=(8, 0))

        status_bar = ttk.Label(
            outer,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(6, 4),
        )
        status_bar.pack(fill=tk.X, pady=(8, 0))

        self._set_review_buttons(False)

    def _bind_shortcuts(self) -> None:
        self.bind("<Right>", lambda _event: self.keep_current())
        self.bind("<Left>", lambda _event: self.reject_current())
        self.bind("<KeyPress-d>", lambda _event: self.keep_current())
        self.bind("<KeyPress-D>", lambda _event: self.keep_current())
        self.bind("<KeyPress-k>", lambda _event: self.keep_current())
        self.bind("<KeyPress-K>", lambda _event: self.keep_current())
        self.bind("<KeyPress-a>", lambda _event: self.reject_current())
        self.bind("<KeyPress-A>", lambda _event: self.reject_current())
        self.bind("<KeyPress-r>", lambda _event: self.reject_current())
        self.bind("<KeyPress-R>", lambda _event: self.reject_current())
        self.bind("<BackSpace>", lambda _event: self.undo_last())
        self.bind("<KeyPress-u>", lambda _event: self.undo_last())
        self.bind("<KeyPress-U>", lambda _event: self.undo_last())
        self.bind("<Configure>", self._on_resize)

    def _set_review_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.keep_button.configure(state=state)
        self.reject_button.configure(state=state)
        self.undo_button.configure(state=tk.NORMAL if self.history else tk.DISABLED)

    # ------------------------------------------------------------------
    # Video setup
    # ------------------------------------------------------------------

    def _parse_parameters(self) -> tuple[float, float, int]:
        try:
            start_minutes = float(self.start_minutes_var.get().replace(",", "."))
            duration_seconds = float(self.duration_seconds_var.get().replace(",", "."))
            frame_step = int(self.frame_step_var.get())
        except ValueError as exc:
            raise ValueError(
                "Start, Dauer und Frame-Schritt müssen gültige Zahlen sein."
            ) from exc

        if start_minutes < 0:
            raise ValueError("Die Startzeit darf nicht negativ sein.")
        if duration_seconds <= 0:
            raise ValueError("Die Dauer muss größer als 0 sein.")
        if frame_step < 1:
            raise ValueError("Der Frame-Schritt muss mindestens 1 sein.")

        return start_minutes, duration_seconds, frame_step

    def add_video(self) -> None:
        if len(self.jobs) >= MAX_VIDEOS:
            messagebox.showinfo(
                APP_TITLE,
                f"Es sind bereits {MAX_VIDEOS} Videos eingetragen.",
            )
            return

        try:
            start_minutes, duration_seconds, frame_step = self._parse_parameters()
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return

        selected = filedialog.askopenfilename(
            title="Cam0-Video auswählen",
            filetypes=[
                ("Videodateien", "*.avi *.mp4 *.mov *.mkv *.m4v"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not selected:
            return

        video_path = Path(selected)
        if any(job.video_path == video_path for job in self.jobs):
            messagebox.showinfo(APP_TITLE, "Dieses Video ist bereits eingetragen.")
            return

        job = VideoJob(
            video_path=video_path,
            start_minutes=start_minutes,
            duration_seconds=duration_seconds,
            frame_step=frame_step,
        )
        self.jobs.append(job)

        self.video_table.insert(
            "",
            tk.END,
            iid=str(len(self.jobs) - 1),
            values=(
                str(video_path),
                f"{start_minutes:g}",
                f"{duration_seconds:g}",
                frame_step,
            ),
        )
        self.video_count_var.set(f"{len(self.jobs)}/{MAX_VIDEOS} Videos")
        self.status_var.set(f"Video hinzugefügt: {video_path.name}")

    def remove_selected_video(self) -> None:
        selection = self.video_table.selection()
        if not selection:
            return

        selected_indices = sorted((int(item) for item in selection), reverse=True)
        for index in selected_indices:
            if 0 <= index < len(self.jobs):
                self.jobs.pop(index)

        for item in self.video_table.get_children():
            self.video_table.delete(item)

        for index, job in enumerate(self.jobs):
            self.video_table.insert(
                "",
                tk.END,
                iid=str(index),
                values=(
                    str(job.video_path),
                    f"{job.start_minutes:g}",
                    f"{job.duration_seconds:g}",
                    job.frame_step,
                ),
            )

        self.video_count_var.set(f"{len(self.jobs)}/{MAX_VIDEOS} Videos")
        self.status_var.set("Ausgewähltes Video entfernt.")

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def start_extraction(self) -> None:
        if self.extraction_running:
            return
        if not self.jobs:
            messagebox.showerror(
                APP_TITLE,
                "Bitte mindestens ein Video hinzufügen.",
            )
            return

        output_parent = filedialog.askdirectory(
            title="Übergeordneten Ausgabeordner auswählen"
        )
        if not output_parent:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.review_root = (
            Path(output_parent) / f"Tongue_Frame_Review_{timestamp}"
        )
        self._configure_review_folders(self.review_root)

        self.extraction_running = True
        self.extract_button.configure(state=tk.DISABLED)
        self.extraction_progress.configure(value=0, maximum=100)
        self.status_var.set("Extraktion läuft …")

        worker = threading.Thread(
            target=self._extract_worker,
            args=(list(self.jobs),),
            daemon=True,
        )
        worker.start()

    def _configure_review_folders(self, root: Path) -> None:
        self.review_root = root
        self.pending_dir = root / "Pending"
        self.kept_dir = root / "Kept"
        self.rejected_dir = root / "Rejected"
        self.log_path = root / "review_log.csv"

        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.kept_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)

        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(
                    [
                        "timestamp",
                        "action",
                        "filename",
                        "source_path",
                        "destination_path",
                    ]
                )

    def _extract_worker(self, jobs: list[VideoJob]) -> None:
        try:
            estimates: list[tuple[VideoJob, int]] = []
            total_expected = 0

            for job in jobs:
                capture = cv2.VideoCapture(str(job.video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"Video konnte nicht geöffnet werden:\n{job.video_path}")

                fps = float(capture.get(cv2.CAP_PROP_FPS))
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                capture.release()

                if fps <= 0:
                    raise RuntimeError(f"Ungültige FPS für:\n{job.video_path}")

                start_frame = int(round(job.start_minutes * 60.0 * fps))
                requested_end = start_frame + int(round(job.duration_seconds * fps))
                end_frame = min(total_frames, requested_end)

                if start_frame >= total_frames:
                    raise RuntimeError(
                        f"Startzeit liegt hinter dem Videoende:\n{job.video_path}"
                    )

                expected = max(0, (end_frame - start_frame + job.frame_step - 1) // job.frame_step)
                estimates.append((job, expected))
                total_expected += expected

            if total_expected <= 0:
                raise RuntimeError("Für die gewählten Intervalle wurden keine Frames gefunden.")

            extracted_total = 0

            for video_index, (job, _expected) in enumerate(estimates, start=1):
                capture = cv2.VideoCapture(str(job.video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"Video konnte nicht geöffnet werden:\n{job.video_path}")

                fps = float(capture.get(cv2.CAP_PROP_FPS))
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                start_frame = int(round(job.start_minutes * 60.0 * fps))
                requested_end = start_frame + int(round(job.duration_seconds * fps))
                end_frame = min(total_frames, requested_end)

                capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                frame_index = start_frame
                video_prefix = f"V{video_index:02d}_{safe_name(job.video_path.stem)}"

                while frame_index < end_frame:
                    success, frame = capture.read()
                    if not success:
                        break

                    offset = frame_index - start_frame
                    if offset % job.frame_step == 0:
                        time_seconds = frame_index / fps
                        filename = (
                            f"{video_prefix}_"
                            f"sourceframe_{frame_index:09d}_"
                            f"time_{time_seconds:011.3f}s"
                            f"{IMAGE_EXTENSION}"
                        )
                        destination = unique_destination(self.pending_dir, filename)

                        write_ok = cv2.imwrite(
                            str(destination),
                            frame,
                            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
                        )
                        if not write_ok:
                            capture.release()
                            raise RuntimeError(
                                f"Frame konnte nicht gespeichert werden:\n{destination}"
                            )

                        extracted_total += 1
                        percent = (extracted_total / total_expected) * 100.0
                        self.after(
                            0,
                            self._update_extraction_progress,
                            percent,
                            extracted_total,
                            total_expected,
                            job.video_path.name,
                        )

                    frame_index += 1

                capture.release()

            self.after(0, self._extraction_finished, extracted_total)

        except Exception as exc:
            details = traceback.format_exc()
            self.after(0, self._extraction_failed, str(exc), details)

    def _update_extraction_progress(
        self,
        percent: float,
        extracted: int,
        total: int,
        video_name: str,
    ) -> None:
        self.extraction_progress.configure(value=min(percent, 100.0))
        self.status_var.set(
            f"Extraktion: {extracted}/{total} Frames – {video_name}"
        )

    def _extraction_finished(self, extracted_total: int) -> None:
        self.extraction_running = False
        self.extract_button.configure(state=tk.NORMAL)
        self.extraction_progress.configure(value=100)
        self.status_var.set(
            f"Extraktion abgeschlossen: {extracted_total} Frames."
        )
        self.load_review()

    def _extraction_failed(self, error: str, details: str) -> None:
        self.extraction_running = False
        self.extract_button.configure(state=tk.NORMAL)
        self.status_var.set("Extraktion fehlgeschlagen.")
        messagebox.showerror(
            APP_TITLE,
            f"{error}\n\nTechnische Details:\n{details}",
        )

    # ------------------------------------------------------------------
    # Existing review
    # ------------------------------------------------------------------

    def open_existing_review(self) -> None:
        selected = filedialog.askdirectory(
            title="Tongue_Frame_Review-Ordner auswählen"
        )
        if not selected:
            return

        root = Path(selected)
        if not (root / "Pending").exists():
            messagebox.showerror(
                APP_TITLE,
                "Der ausgewählte Ordner enthält keinen 'Pending'-Unterordner.",
            )
            return

        self._configure_review_folders(root)
        self.load_review()

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def load_review(self) -> None:
        if self.pending_dir is None:
            return

        self.pending_frames = sorted(
            path
            for path in self.pending_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        self.history.clear()
        self.review_active = bool(self.pending_frames)
        self._show_next_frame()

    def _show_next_frame(self) -> None:
        if not self.pending_frames:
            self.current_frame = None
            self.current_photo = None
            self.image_label.configure(
                image="",
                text=(
                    "Review abgeschlossen.\n\n"
                    f"Gehalten: {self._count_files(self.kept_dir)}\n"
                    f"Nicht gehalten: {self._count_files(self.rejected_dir)}"
                ),
            )
            self.progress_var.set("Keine ausstehenden Frames mehr.")
            self.current_name_var.set("")
            self.review_active = False
            self._set_review_buttons(False)
            self.status_var.set("Review abgeschlossen.")
            return

        self.current_frame = self.pending_frames[0]
        self.current_name_var.set(self.current_frame.name)
        self._render_current_frame()
        self._update_review_progress()
        self.review_active = True
        self._set_review_buttons(True)

    def _render_current_frame(self) -> None:
        if self.current_frame is None:
            return

        try:
            image = Image.open(self.current_frame).convert("RGB")
        except Exception as exc:
            messagebox.showerror(
                APP_TITLE,
                f"Frame konnte nicht geöffnet werden:\n{self.current_frame}\n\n{exc}",
            )
            return

        available_width = max(self.image_label.winfo_width() - 20, 400)
        available_height = max(self.image_label.winfo_height() - 20, 300)

        image.thumbnail(
            (available_width, available_height),
            Image.Resampling.LANCZOS,
        )

        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.current_photo, text="")

    def _on_resize(self, _event: tk.Event) -> None:
        if self.current_frame is not None:
            if hasattr(self, "_resize_after_id"):
                self.after_cancel(self._resize_after_id)
            self._resize_after_id = self.after(120, self._render_current_frame)

    def _count_files(self, folder: Optional[Path]) -> int:
        if folder is None or not folder.exists():
            return 0
        return sum(1 for path in folder.iterdir() if path.is_file())

    def _update_review_progress(self) -> None:
        kept = self._count_files(self.kept_dir)
        rejected = self._count_files(self.rejected_dir)
        pending = len(self.pending_frames)
        reviewed = kept + rejected
        total = reviewed + pending

        self.progress_var.set(
            f"Bearbeitet: {reviewed}/{total} | "
            f"Halten: {kept} | Nicht halten: {rejected} | Offen: {pending}"
        )
        self.undo_button.configure(
            state=tk.NORMAL if self.history else tk.DISABLED
        )

    def keep_current(self) -> None:
        self._decide_current("KEEP")

    def reject_current(self) -> None:
        self._decide_current("REJECT")

    def _decide_current(self, action: str) -> None:
        if not self.review_active or self.current_frame is None:
            return

        if action == "KEEP":
            destination_folder = self.kept_dir
        elif action == "REJECT":
            destination_folder = self.rejected_dir
        else:
            raise ValueError(f"Unbekannte Aktion: {action}")

        if destination_folder is None:
            return

        source = self.current_frame
        destination = unique_destination(destination_folder, source.name)

        try:
            shutil.move(str(source), str(destination))
        except Exception as exc:
            messagebox.showerror(
                APP_TITLE,
                f"Frame konnte nicht verschoben werden:\n{source}\n\n{exc}",
            )
            return

        decision = Decision(
            source=source,
            destination=destination,
            action=action,
        )
        self.history.append(decision)
        self._append_log(action, source, destination)

        if self.pending_frames and self.pending_frames[0] == source:
            self.pending_frames.pop(0)
        else:
            self.pending_frames = [path for path in self.pending_frames if path != source]

        self.status_var.set(
            "Gehalten." if action == "KEEP" else "Nicht gehalten."
        )
        self._show_next_frame()

    def undo_last(self) -> None:
        if not self.history:
            return

        decision = self.history.pop()
        if not decision.destination.exists():
            messagebox.showerror(
                APP_TITLE,
                f"Die Datei für Undo wurde nicht gefunden:\n{decision.destination}",
            )
            return

        restored = unique_destination(self.pending_dir, decision.source.name)

        try:
            shutil.move(str(decision.destination), str(restored))
        except Exception as exc:
            messagebox.showerror(
                APP_TITLE,
                f"Entscheidung konnte nicht rückgängig gemacht werden:\n{exc}",
            )
            return

        self._append_log(
            f"UNDO_{decision.action}",
            decision.destination,
            restored,
        )

        self.pending_frames.insert(0, restored)
        self.status_var.set("Letzte Entscheidung rückgängig gemacht.")
        self._show_next_frame()

    def _append_log(
        self,
        action: str,
        source: Path,
        destination: Path,
    ) -> None:
        if self.log_path is None:
            return

        with self.log_path.open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    action,
                    destination.name,
                    str(source),
                    str(destination),
                ]
            )


def main() -> None:
    app = TongueFrameTinderApp()
    app.mainloop()


if __name__ == "__main__":
    main()