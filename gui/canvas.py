from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QRubberBand
from PyQt6.QtGui import QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent, QColor
from PyQt6.QtCore import Qt, QRectF, QRect, QPoint, QSize, pyqtSignal
import numpy as np

class SmartCanvas(QGraphicsView):
    """
    A custom QGraphicsView that handles:
    1. Zooming (Mouse Wheel)
    2. Panning (Middle Mouse Button)
    3. ROI Selection (Left Click + Drag)
    4. Image & Overlay Display
    """
    
    # ROI seçimi bittiğinde koordinatları gönderen sinyal
    roi_selected = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # 1. Katman: Orijinal Görüntü
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # 2. Katman: Analiz Sonucu (Overlay)
        # Z-Value 1 yaparak resmin üstünde durmasını sağlıyoruz
        self.overlay_item = QGraphicsPixmapItem()
        self.overlay_item.setZValue(1) 
        self.scene.addItem(self.overlay_item)
        
        # Canvas Ayarları
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        
        # ROI Seçim Araçları
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self._is_selecting = False

    def set_image(self, image_data: np.ndarray) -> None:
        """
        Converts a numpy array to QPixmap and displays it.
        """
        if image_data is None:
            return

        # Görüntü formatını belirle
        height, width = image_data.shape[:2]
        
        if len(image_data.shape) == 2:
            bytes_per_line = width
            image_format = QImage.Format.Format_Grayscale8
        elif len(image_data.shape) == 3:
            bytes_per_line = image_data.shape[2] * width
            image_format = QImage.Format.Format_RGB888 if image_data.shape[2] == 3 else QImage.Format.Format_RGBA8888
        else:
            return

        if not image_data.flags['C_CONTIGUOUS']:
            image_data = np.ascontiguousarray(image_data)

        q_img = QImage(image_data.data, width, height, bytes_per_line, image_format)
        self.pixmap_item.setPixmap(QPixmap.fromImage(q_img))
        
        # Sahneyi görüntü boyutuna ayarla
        self.setSceneRect(QRectF(0, 0, width, height))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlay(self, mask: np.ndarray, color=(0, 255, 0), alpha=100) -> None:
        """
        Displays a boolean mask as a semi-transparent overlay on top of the image.
        """
        if mask is None:
            self.overlay_item.setPixmap(QPixmap())
            return

        h, w = mask.shape
        
        # RGBA (Red, Green, Blue, Alpha) formatında boş bir resim oluştur
        overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Maskenin True olduğu yerleri boya
        overlay_rgba[mask, 0] = color[0] # R
        overlay_rgba[mask, 1] = color[1] # G
        overlay_rgba[mask, 2] = color[2] # B
        overlay_rgba[mask, 3] = alpha    # A (Şeffaflık)

        # NumPy dizisini QImage'a çevir
        q_img = QImage(overlay_rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        
        # Canvas üzerine yerleştir
        self.overlay_item.setPixmap(QPixmap.fromImage(q_img))
        
        # Overlay'in konumu ana resimle aynı olmalı
        self.overlay_item.setPos(self.pixmap_item.pos())

    # --- Mouse Etkileşimleri ---

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # ROI Seçimini Başlat
            self._is_selecting = True
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            
        elif event.button() == Qt.MouseButton.MiddleButton:
            # Pan Moduna Geç
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            
            # Koordinat düzeltmesi (PyQt6 QPointF ister)
            fake_event = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(fake_event)
            
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._is_selecting:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._is_selecting:
            self._is_selecting = False
            # Seçilen alanı sahne (görüntü) koordinatlarına çevir
            viewport_rect = self.rubber_band.geometry()
            scene_rect = self.mapToScene(viewport_rect).boundingRect().toRect()
            
            # Sinyali gönder
            self.roi_selected.emit(scene_rect)
            # rubber_band'i gizlemiyoruz ki kullanıcı ne seçtiğini görsün
        
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)