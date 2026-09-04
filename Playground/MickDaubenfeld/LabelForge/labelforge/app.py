import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from .ui.main_window import MainWindow


def asset_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / filename


def run_app() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LabelForge")

    icon_path = asset_path("labelforge_icon.ico")
    if not icon_path.exists():
        icon_path = asset_path("labelforge_icon.png")

    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    splash = None
    splash_path = asset_path("labelforge_extended.png")

    if splash_path.exists():
        pixmap = QPixmap(str(splash_path))

        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                760,
                260,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )

            splash = QSplashScreen(pixmap)
            splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            splash.show()
            app.processEvents()

    window = MainWindow()

    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))

    def show_main_window() -> None:
        window.showMaximized()
        if splash is not None:
            splash.finish(window)

    if splash is not None:
        QTimer.singleShot(1200, show_main_window)
    else:
        show_main_window()

    sys.exit(app.exec())
