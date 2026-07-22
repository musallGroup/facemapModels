"""
Video Frame Picker GUI

Features:
- Open one video
- Seek with a timeline slider
- Play/pause and playback speed
- Step by 1, 10, or 100 frames
- Save only selected frames as PNG
- Keep a mapping.csv with source frame and timestamp

Keyboard:
Space = play/pause
Left/Right = 1 frame
Shift+Left/Right = 10 frames
Ctrl+Left/Right = 100 frames
S = save frame
Backspace = remove last saved frame

Install:
pip install opencv-python pillow
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "Video Frame Picker"
PNG_COMPRESSION = 3


@dataclass
class SavedFrame:
    path: Path
    frame_index: int
    time_seconds: float


def safe_name(text: str) -> str:
    invalid = '<>:"/\\|?*'
    return "".join("_" if c in invalid else c for c in text).strip(" .") or "video"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x880")
        self.minsize(980, 680)

        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        self.mapping_path: Optional[Path] = None

        self.fps = 0.0
        self.total_frames = 0
        self.frame_index = 0
        self.frame_bgr = None
        self.photo: Optional[ImageTk.PhotoImage] = None

        self.playing = False
        self.speed = 1.0
        self.after_id: Optional[str] = None
        self.internal_slider_update = False

        self.saved: list[SavedFrame] = []
        self.saved_indices: set[int] = set()

        self.status_var = tk.StringVar(value="Video öffnen.")
        self.info_var = tk.StringVar(value="Kein Video geladen.")
        self.saved_var = tk.StringVar(value="Gespeichert: 0")

        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Video öffnen", command=self.open_video).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(top, text="Ausgabeordner wählen", command=self.choose_output).pack(
            side=tk.LEFT, padx=(0, 8)
        )

        self.video_name = ttk.Label(top, text="Kein Video")
        self.video_name.pack(side=tk.LEFT, padx=10)

        ttk.Label(top, textvariable=self.saved_var).pack(side=tk.RIGHT)

        viewer = ttk.LabelFrame(root, text="Video", padding=8)
        viewer.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.image_label = ttk.Label(
            viewer,
            text="Öffne ein Video, um zu beginnen.",
            anchor=tk.CENTER,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.slider = ttk.Scale(
            root,
            from_=0,
            to=1,
            orient=tk.HORIZONTAL,
            command=self.slider_moved,
        )
        self.slider.pack(fill=tk.X, pady=(10, 0))

        info = ttk.Frame(root)
        info.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(info, textvariable=self.info_var, font=("Segoe UI", 10, "bold")).pack(
            side=tk.LEFT
        )

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, pady=(10, 0))

        for text, delta in [("|<", None), ("-100", -100), ("-10", -10), ("-1", -1)]:
            if delta is None:
                command = lambda: self.seek(0)
            else:
                command = lambda d=delta: self.step(d)
            ttk.Button(controls, text=text, width=7, command=command).pack(
                side=tk.LEFT, padx=2
            )

        self.play_button = ttk.Button(
            controls, text="Play", width=10, command=self.toggle_play
        )
        self.play_button.pack(side=tk.LEFT, padx=8)

        for text, delta in [("+1", 1), ("+10", 10), ("+100", 100), (">|", None)]:
            if delta is None:
                command = lambda: self.seek(max(0, self.total_frames - 1))
            else:
                command = lambda d=delta: self.step(d)
            ttk.Button(controls, text=text, width=7, command=command).pack(
                side=tk.LEFT, padx=2
            )

        speed_frame = ttk.Frame(controls)
        speed_frame.pack(side=tk.RIGHT)
        ttk.Label(speed_frame, text="Tempo:").pack(side=tk.LEFT, padx=(0, 4))

        self.speed_combo = ttk.Combobox(
            speed_frame,
            state="readonly",
            width=7,
            values=("0.25×", "0.5×", "1.0×", "2.0×", "4.0×"),
        )
        self.speed_combo.set("1.0×")
        self.speed_combo.pack(side=tk.LEFT)
        self.speed_combo.bind("<<ComboboxSelected>>", self.change_speed)

        save_row = ttk.Frame(root)
        save_row.pack(fill=tk.X, pady=(12, 0))

        tk.Button(
            save_row,
            text="AKTUELLES FRAME SPEICHERN  [S]",
            command=self.save_frame,
            font=("Segoe UI", 13, "bold"),
            height=2,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        ttk.Button(
            save_row,
            text="Letztes gespeichertes Frame entfernen",
            command=self.remove_last,
        ).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(
            root,
            text=(
                "Leertaste: Play/Pause | ←/→: 1 Frame | "
                "Shift+←/→: 10 | Ctrl+←/→: 100 | "
                "S: speichern | Rücktaste: letztes entfernen"
            ),
            anchor=tk.CENTER,
        ).pack(fill=tk.X, pady=(8, 0))

        ttk.Label(
            root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(6, 4),
        ).pack(fill=tk.X, pady=(8, 0))

    def _bind_keys(self) -> None:
        self.bind("<space>", lambda _e: self.toggle_play())
        self.bind("<Left>", lambda _e: self.step(-1))
        self.bind("<Right>", lambda _e: self.step(1))
        self.bind("<Shift-Left>", lambda _e: self.step(-10))
        self.bind("<Shift-Right>", lambda _e: self.step(10))
        self.bind("<Control-Left>", lambda _e: self.step(-100))
        self.bind("<Control-Right>", lambda _e: self.step(100))
        self.bind("<KeyPress-s>", lambda _e: self.save_frame())
        self.bind("<KeyPress-S>", lambda _e: self.save_frame())
        self.bind("<BackSpace>", lambda _e: self.remove_last())
        self.bind("<Home>", lambda _e: self.seek(0))
        self.bind("<End>", lambda _e: self.seek(max(0, self.total_frames - 1)))
        self.bind("<Configure>", self.on_resize)

    def open_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Video auswählen",
            filetypes=[
                ("Videos", "*.avi *.mp4 *.mov *.mkv *.m4v"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if not selected:
            return

        self.stop()
        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            messagebox.showerror(APP_TITLE, f"Video konnte nicht geöffnet werden:\n{selected}")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total <= 0:
            cap.release()
            messagebox.showerror(APP_TITLE, "FPS oder Frame-Anzahl konnten nicht gelesen werden.")
            return

        self.cap = cap
        self.video_path = Path(selected)
        self.fps = fps
        self.total_frames = total
        self.frame_index = 0
        self.saved.clear()
        self.saved_indices.clear()

        self.slider.configure(from_=0, to=max(0, total - 1))
        self.video_name.configure(text=self.video_path.name)
        self.saved_var.set("Gespeichert: 0")

        if self.output_dir is None:
            self.output_dir = self.video_path.parent / (
                safe_name(self.video_path.stem) + "_Selected_Frames"
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.mapping_path = self.output_dir / "mapping.csv"
            self.ensure_mapping_header()

        self.seek(0)
        self.status_var.set(f"Video geladen: {self.video_path.name}")

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Ausgabeordner wählen")
        if not selected:
            return

        self.output_dir = Path(selected)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mapping_path = self.output_dir / "mapping.csv"
        self.ensure_mapping_header()
        self.status_var.set(f"Ausgabeordner: {self.output_dir}")

    def seek(self, index: int) -> None:
        if self.cap is None or self.total_frames <= 0:
            return

        index = max(0, min(int(index), self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set(f"Frame {index} konnte nicht gelesen werden.")
            return

        self.frame_index = index
        self.frame_bgr = frame
        self.render(frame)
        self.update_info()

    def step(self, delta: int) -> None:
        if self.cap is None:
            return
        self.stop()
        self.seek(self.frame_index + delta)

    def slider_moved(self, value: str) -> None:
        if self.internal_slider_update or self.cap is None:
            return
        self.stop()
        self.seek(int(float(value)))

    def toggle_play(self) -> None:
        if self.cap is None:
            return

        if self.playing:
            self.stop()
            return

        self.playing = True
        self.play_button.configure(text="Pause")
        self.schedule_next()

    def stop(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

    def schedule_next(self) -> None:
        if not self.playing:
            return
        delay = max(1, int(1000 / (self.fps * self.speed)))
        self.after_id = self.after(delay, self.play_tick)

    def play_tick(self) -> None:
        self.after_id = None
        if not self.playing or self.cap is None:
            return

        if self.frame_index + 1 >= self.total_frames:
            self.stop()
            return

        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return

        self.frame_index += 1
        self.frame_bgr = frame
        self.render(frame)
        self.update_info()
        self.schedule_next()

    def change_speed(self, _event: tk.Event) -> None:
        self.speed = float(self.speed_combo.get().replace("×", ""))

    def render(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        width = max(self.image_label.winfo_width() - 20, 400)
        height = max(self.image_label.winfo_height() - 20, 300)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo, text="")

    def on_resize(self, _event: tk.Event) -> None:
        if self.frame_bgr is None:
            return
        if hasattr(self, "_resize_job"):
            try:
                self.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self._resize_job = self.after(120, lambda: self.render(self.frame_bgr))

    def update_info(self) -> None:
        self.internal_slider_update = True
        self.slider.set(self.frame_index)
        self.internal_slider_update = False

        current = self.frame_index / self.fps
        total = (self.total_frames - 1) / self.fps
        self.info_var.set(
            f"Frame {self.frame_index:,} / {self.total_frames - 1:,} | "
            f"Zeit {self.format_time(current)} / {self.format_time(total)}"
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        rest = seconds - minutes * 60
        return f"{minutes:02d}:{rest:06.3f}"

    def save_frame(self) -> None:
        if self.frame_bgr is None or self.video_path is None:
            return

        if self.output_dir is None:
            self.choose_output()
            if self.output_dir is None:
                return

        if self.frame_index in self.saved_indices:
            self.status_var.set("Dieses Frame wurde bereits gespeichert.")
            return

        time_seconds = self.frame_index / self.fps
        filename = (
            f"{safe_name(self.video_path.stem)}_"
            f"sourceframe_{self.frame_index:09d}_"
            f"time_{time_seconds:011.3f}s.png"
        )
        target = self.output_dir / filename

        ok = cv2.imwrite(
            str(target),
            self.frame_bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )
        if not ok:
            messagebox.showerror(APP_TITLE, f"Frame konnte nicht gespeichert werden:\n{target}")
            return

        saved = SavedFrame(target, self.frame_index, time_seconds)
        self.saved.append(saved)
        self.saved_indices.add(self.frame_index)
        self.append_mapping(saved)

        self.saved_var.set(f"Gespeichert: {len(self.saved)}")
        self.status_var.set(f"Gespeichert: {target.name}")

    def remove_last(self) -> None:
        if not self.saved:
            self.status_var.set("Noch kein Frame gespeichert.")
            return

        saved = self.saved.pop()
        try:
            if saved.path.exists():
                saved.path.unlink()
        except OSError as exc:
            self.saved.append(saved)
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self.saved_indices.discard(saved.frame_index)
        self.rewrite_mapping()
        self.saved_var.set(f"Gespeichert: {len(self.saved)}")
        self.status_var.set(f"Entfernt: {saved.path.name}")

    def ensure_mapping_header(self) -> None:
        if self.mapping_path is None or self.mapping_path.exists():
            return
        with self.mapping_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                ["saved_png", "source_video", "source_frame", "time_seconds", "saved_at"]
            )

    def append_mapping(self, saved: SavedFrame) -> None:
        if self.mapping_path is None or self.video_path is None:
            return
        self.ensure_mapping_header()
        with self.mapping_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                [
                    saved.path.name,
                    str(self.video_path),
                    saved.frame_index,
                    f"{saved.time_seconds:.6f}",
                    datetime.now().isoformat(timespec="seconds"),
                ]
            )

    def rewrite_mapping(self) -> None:
        if self.mapping_path is None or self.video_path is None:
            return
        with self.mapping_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                ["saved_png", "source_video", "source_frame", "time_seconds", "saved_at"]
            )
            for saved in self.saved:
                writer.writerow(
                    [
                        saved.path.name,
                        str(self.video_path),
                        saved.frame_index,
                        f"{saved.time_seconds:.6f}",
                        "",
                    ]
                )

    def close(self) -> None:
        self.stop()
        if self.cap is not None:
            self.cap.release()
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()