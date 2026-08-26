from __future__ import annotations

import numpy as np
import cv2

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush, QImage, QMouseEvent, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)


class ImageViewer(QGraphicsView):
    """
    Reusable LabelForge image viewer.

    Controls
    --------
    Mouse wheel:
        Zoom at cursor.
    Shift + mouse wheel:
        Brightness.
    Ctrl + mouse wheel:
        Contrast.
    Alt + mouse wheel:
        Gamma.
    Middle mouse + drag:
        Pan image.
    Double middle click:
        Reset view.

    Important
    ---------
    Brightness / contrast / gamma are DISPLAY ONLY.
    The original image array is never modified.
    """

    display_changed = Signal(float, float, float)
    view_reset = Signal()
    image_left_clicked = Signal(float, float)
    image_right_clicked = Signal(float, float)

    MIN_ZOOM = 1.0
    MAX_ZOOM = 12.0
    ZOOM_STEP = 1.20

    MIN_DISPLAY_FACTOR = 0.20
    MAX_DISPLAY_FACTOR = 3.00
    DISPLAY_STEP = 0.10

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setObjectName("ImageViewer")
        self.setScene(QGraphicsScene(self))

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setAcceptedMouseButtons(Qt.NoButton)
        self.scene().addItem(self._pixmap_item)
        self._overlay_items: list = []

        self._original_bgr: np.ndarray | None = None
        self._zoom_factor = 1.0

        self._brightness = 1.0
        self._contrast = 1.0
        self._gamma = 1.0

        self._panning = False
        self._last_pan_pos = None

        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setFrameShape(QGraphicsView.NoFrame)
        self.setAlignment(Qt.AlignCenter)
        self.setBackgroundBrush(Qt.black)

        self.setStyleSheet("""
            QGraphicsView#ImageViewer {
                background: #0F1114;
                border: 1px solid #2D323A;
                border-radius: 12px;
            }
        """)

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    def set_bgr_image(
        self,
        frame_bgr: np.ndarray,
        *,
        preserve_view: bool = False,
    ) -> None:
        """
        Show a new image.

        preserve_view=False:
            Fit/reset the image as before.

        preserve_view=True:
            Keep the current zoom and pan. This is useful for Keypoint mode,
            where the same anatomical location is inspected across frames.
        """
        self._original_bgr = frame_bgr.copy()
        self._render_display()

        if not preserve_view:
            self.reset_view()

    def clear_image(self) -> None:
        self.clear_overlays()
        self._original_bgr = None
        self._pixmap_item.setPixmap(QPixmap())
        self.scene().setSceneRect(0, 0, 1, 1)

    def original_bgr(self) -> np.ndarray | None:
        if self._original_bgr is None:
            return None
        return self._original_bgr.copy()

    # ------------------------------------------------------------------
    # Display adjustments
    # ------------------------------------------------------------------

    @property
    def brightness(self) -> float:
        return self._brightness

    @property
    def contrast(self) -> float:
        return self._contrast

    @property
    def gamma(self) -> float:
        return self._gamma

    def set_brightness(self, value: float) -> None:
        self._brightness = self._clamp_display(value)
        self._render_display()
        self.display_changed.emit(
            self._brightness,
            self._contrast,
            self._gamma,
        )

    def set_contrast(self, value: float) -> None:
        self._contrast = self._clamp_display(value)
        self._render_display()
        self.display_changed.emit(
            self._brightness,
            self._contrast,
            self._gamma,
        )

    def set_gamma(self, value: float) -> None:
        self._gamma = self._clamp_display(value)
        self._render_display()
        self.display_changed.emit(
            self._brightness,
            self._contrast,
            self._gamma,
        )

    def reset_display(self) -> None:
        self._brightness = 1.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._render_display()
        self.display_changed.emit(1.0, 1.0, 1.0)

    def _clamp_display(self, value: float) -> float:
        return max(
            self.MIN_DISPLAY_FACTOR,
            min(float(value), self.MAX_DISPLAY_FACTOR),
        )

    def _apply_display_adjustments(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Return an adjusted DISPLAY COPY.
        The stored original frame remains untouched.
        """
        image = frame_bgr.astype(np.float32) / 255.0

        # Brightness
        image *= self._brightness

        # Contrast around the mid-point
        image = (image - 0.5) * self._contrast + 0.5

        image = np.clip(image, 0.0, 1.0)

        # Gamma
        if abs(self._gamma - 1.0) > 1e-9:
            image = np.power(image, 1.0 / self._gamma)

        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image

    def _render_display(self) -> None:
        if self._original_bgr is None:
            return

        display_bgr = self._apply_display_adjustments(
            self._original_bgr
        )
        rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)

        height, width, channels = rgb.shape
        bytes_per_line = channels * width

        qimage = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item.setPixmap(pixmap)

        rect = self._pixmap_item.boundingRect()
        margin_x = max(rect.width() * 2.0, 2000.0)
        margin_y = max(rect.height() * 2.0, 2000.0)

        self.scene().setSceneRect(
            rect.adjusted(
                -margin_x,
                -margin_y,
                margin_x,
                margin_y,
            )
        )

    # ------------------------------------------------------------------
    # Label overlays
    # ------------------------------------------------------------------

    def clear_overlays(self) -> None:
        for item in self._overlay_items:
            self.scene().removeItem(item)
        self._overlay_items = []

    def set_overlays(self, overlays: list[dict]) -> None:
        """
        Draw keypoints in original-image coordinates while keeping marker and
        text size stable on screen during zoom.

        Each overlay dict may contain:
        x, y, color, name, selected
        """
        self.clear_overlays()

        for overlay in overlays:
            x = float(overlay["x"])
            y = float(overlay["y"])
            color = QColor(overlay.get("color", "#D18B47"))
            selected = bool(overlay.get("selected", False))
            name = str(overlay.get("name", ""))

            # These dimensions are SCREEN-SIZE-like because the graphics items
            # ignore the viewer transform. This prevents giant points when zooming.
            # Constant screen-space size in every labeling mode.
            # 12 px diameter: slightly larger than the previous markers,
            # while remaining compact even at high zoom.
            radius = 6.0

            if selected:
                halo_radius = radius + 3.0
                halo = QGraphicsEllipseItem(
                    -halo_radius,
                    -halo_radius,
                    halo_radius * 2,
                    halo_radius * 2,
                )
                halo.setPos(x, y)
                halo.setFlag(
                    QGraphicsEllipseItem.ItemIgnoresTransformations,
                    True,
                )
                halo.setPen(QPen(QColor("#EAEAEA"), 1.6))
                halo.setBrush(QBrush(Qt.NoBrush))
                halo.setZValue(9)
                self.scene().addItem(halo)
                self._overlay_items.append(halo)

            dot = QGraphicsEllipseItem(
                -radius,
                -radius,
                radius * 2,
                radius * 2,
            )
            dot.setPos(x, y)
            dot.setFlag(
                QGraphicsEllipseItem.ItemIgnoresTransformations,
                True,
            )
            dot.setPen(QPen(QColor("#111111"), 1.2))
            dot.setBrush(QBrush(color))
            dot.setZValue(10)
            self.scene().addItem(dot)
            self._overlay_items.append(dot)

            if name:
                label = QGraphicsSimpleTextItem(name)
                label.setBrush(QBrush(color))
                label.setFlag(
                    QGraphicsSimpleTextItem.ItemIgnoresTransformations,
                    True,
                )
                label.setScale(0.90)
                label.setPos(x + 9, y - 12)
                label.setZValue(11)
                self.scene().addItem(label)
                self._overlay_items.append(label)

    def _view_position_to_image(
        self,
        event: QMouseEvent,
    ) -> tuple[float, float] | None:
        if self._original_bgr is None:
            return None

        scene_pos = self.mapToScene(event.position().toPoint())
        item_pos = self._pixmap_item.mapFromScene(scene_pos)
        image_rect = self._pixmap_item.boundingRect()

        if not image_rect.contains(item_pos):
            return None

        return float(item_pos.x()), float(item_pos.y())

    # ------------------------------------------------------------------
    # View control
    # ------------------------------------------------------------------

    def reset_view(self) -> None:
        self.resetTransform()
        self._zoom_factor = 1.0

        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(
                self._pixmap_item,
                Qt.KeepAspectRatio,
            )

        self.centerOn(self._pixmap_item.boundingRect().center())
        self.view_reset.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

        # Keep auto-fit only while at the baseline view.
        if (
            abs(self._zoom_factor - 1.0) < 1e-9
            and not self._pixmap_item.pixmap().isNull()
        ):
            self.fitInView(
                self._pixmap_item,
                Qt.KeepAspectRatio,
            )

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._original_bgr is None:
            return

        direction = 1 if event.angleDelta().y() > 0 else -1
        modifiers = event.modifiers()

        if modifiers & Qt.ShiftModifier:
            self.set_brightness(
                self._brightness + direction * self.DISPLAY_STEP
            )
            event.accept()
            return

        if modifiers & Qt.ControlModifier:
            self.set_contrast(
                self._contrast + direction * self.DISPLAY_STEP
            )
            event.accept()
            return

        if modifiers & Qt.AltModifier:
            self.set_gamma(
                self._gamma + direction * self.DISPLAY_STEP
            )
            event.accept()
            return

        old_zoom = self._zoom_factor

        if direction > 0:
            new_zoom = min(
                self.MAX_ZOOM,
                self._zoom_factor * self.ZOOM_STEP,
            )
        else:
            new_zoom = max(
                self.MIN_ZOOM,
                self._zoom_factor / self.ZOOM_STEP,
            )

        if abs(new_zoom - old_zoom) < 1e-9:
            return

        factor = new_zoom / old_zoom
        self._zoom_factor = new_zoom

        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._last_pan_pos = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        coordinates = self._view_position_to_image(event)

        if coordinates is not None and event.button() == Qt.LeftButton:
            self.image_left_clicked.emit(*coordinates)
            event.accept()
            return

        if coordinates is not None and event.button() == Qt.RightButton:
            self.image_right_clicked.emit(*coordinates)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning and self._last_pan_pos is not None:
            delta = event.position() - self._last_pan_pos
            self._last_pan_pos = event.position()

            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )

            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self._last_pan_pos = None
            self.unsetCursor()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MiddleButton:
            self.reset_view()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)
