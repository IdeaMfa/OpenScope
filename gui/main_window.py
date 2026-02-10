import sys
import traceback
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QStatusBar, QFileDialog, QLabel, QFrame, 
                             QComboBox, QPushButton, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, QRect
from pathlib import Path
import numpy as np
import cv2
import tifffile  # Kayıpsız TIFF kaydı için

# Core & Plugins
try:
    from core.loader import load_tiff_image, normalize_for_display
    from gui.canvas import SmartCanvas
    from plugins.basic_blob import BasicBlobDetector
    from plugins.advanced_ge import AdvancedBacteriaDetector
    from plugins.advanced_bf import BrightfieldBacteriaDetectorPlugin
except ImportError as e:
    print(f"Kritik Import Hatası: {e}")

class OpenScopeMain(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # --- Veri Modelleri ---
        self.raw_image = None       # Ham NumPy dizisi (Orijinal Veri)
        self.current_roi = None     # Seçili alan (x, y, w, h) veya None
        self.current_mask = None    # Son analizden elde edilen maske (Kayıt için)
        
        # Plugin Kaydı
        self.plugins = {}
        
        # 1. Basic Plugin
        try:
            self.plugins["Basic Blob Detector"] = BasicBlobDetector()
        except Exception as e:
            print(f"Hata: Basic Plugin yüklenemedi: {e}")

        # 2. Advanced Plugin
        try:
            self.plugins["Advanced Bacteria Detector (v4.5)"] = AdvancedBacteriaDetector()
        except Exception as e:
            print(f"Hata: Advanced Plugin yüklenemedi: {e}")
        
        # 3. Advanced Plugin 2
        try:
            self.plugins["Brightfield Bacteria Detector"] = BrightfieldBacteriaDetectorPlugin()
        except Exception as e:
            print(f"Hata: Advanced Plugin yüklenemedi: {e}")

        self.setWindowTitle("OpenScope - Scientific Image Analysis Platform")
        self.resize(1200, 800)

        # --- Arayüz Kurulumu ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self._init_sidebar()
        self._init_canvas()
        self._init_menu()
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _init_sidebar(self):
        """Sol paneli oluşturur."""
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(320)
        self.sidebar.setStyleSheet("background-color: #f8f9fa; border-right: 1px solid #ddd;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        
        # Header
        lbl_sidebar = QLabel("Analysis Controls")
        lbl_sidebar.setStyleSheet("font-weight: bold; font-size: 16px; color: #333333; margin-bottom: 10px; border: none;")
        sidebar_layout.addWidget(lbl_sidebar)

        # Algoritma Seçici
        self.algo_group = QGroupBox("Select Algorithm")
        self.algo_group.setStyleSheet("QGroupBox { color: #333333; font-weight: bold; border: 1px solid #ccc; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        algo_layout = QVBoxLayout()
        
        self.algo_selector = QComboBox()
        self.algo_selector.setStyleSheet("""
            QComboBox { color: #000000; background-color: #ffffff; border: 1px solid #cccccc; border-radius: 4px; padding: 5px; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView { color: #000000; background-color: #ffffff; selection-background-color: #007bff; selection-color: #ffffff; }
        """)
        self.algo_selector.addItems(list(self.plugins.keys()))
        algo_layout.addWidget(self.algo_selector)
        self.algo_group.setLayout(algo_layout)
        sidebar_layout.addWidget(self.algo_group)

        # Parametre Alanı
        self.params_container = QGroupBox("Parameters")
        self.params_container.setStyleSheet("QGroupBox { color: #333333; font-weight: bold; border: 1px solid #ccc; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        self.params_layout = QVBoxLayout()
        lbl_placeholder = QLabel("Default parameters applied.")
        lbl_placeholder.setStyleSheet("color: #555555; font-style: italic; border: none;")
        self.params_layout.addWidget(lbl_placeholder)
        self.params_container.setLayout(self.params_layout)
        sidebar_layout.addWidget(self.params_container)

        # Aksiyon Butonları
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setStyleSheet("""
            QPushButton { background-color: #007bff; color: white; padding: 10px; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #0056b3; }
        """)
        self.btn_run.clicked.connect(self.run_analysis)
        sidebar_layout.addWidget(self.btn_run)

        # --- YENİ: Kaydet Butonu ---
        self.btn_save = QPushButton("Save Result as TIFF")
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; padding: 10px; font-weight: bold; border-radius: 4px; margin-top: 5px;}
            QPushButton:hover { background-color: #218838; }
        """)
        self.btn_save.clicked.connect(self.save_analysis_result)
        sidebar_layout.addWidget(self.btn_save)
        # ---------------------------

        sidebar_layout.addStretch()
        self.main_layout.addWidget(self.sidebar)

    def _init_canvas(self):
        self.canvas = SmartCanvas()
        self.canvas.roi_selected.connect(self.handle_roi_selection)
        self.main_layout.addWidget(self.canvas, stretch=1)

    def _init_menu(self) -> None:
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False) 

        file_menu = menubar.addMenu("&File")
        
        open_action = file_menu.addAction("&Open TIFF...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)
        
        # --- YENİ: Menüye Kaydet Eklendi ---
        save_action = file_menu.addAction("&Save Result...")
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_analysis_result)
        # -----------------------------------
        
        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

    def open_file_dialog(self) -> None:
        options = QFileDialog.Option.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()), "TIFF Files (*.tif *.tiff);;All Files (*)", options=options
        )
        
        if file_name:
            self.status_bar.showMessage(f"Loading: {file_name}...")
            image = load_tiff_image(file_name)
            
            if image is not None:
                self.raw_image = image
                self.current_roi = None
                self.current_mask = None # Yeni resim yüklenince eski maskeyi unut
                
                display_image = normalize_for_display(self.raw_image)
                self.canvas.set_image(display_image)
                self.canvas.set_overlay(None)
                
                self.status_bar.showMessage(f"Loaded: {Path(file_name).name}")
            else:
                self.status_bar.showMessage("Error: Could not load image.")

    def handle_roi_selection(self, scene_rect: QRect):
        if self.raw_image is None: return

        x, y, w, h = int(scene_rect.x()), int(scene_rect.y()), int(scene_rect.width()), int(scene_rect.height())
        img_h, img_w = self.raw_image.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w > 0 and h > 0:
            self.current_roi = (x, y, w, h)
            self.status_bar.showMessage(f"ROI Selected: x={x}, y={y}, {w}x{h} px")
        else:
            self.current_roi = None

    def run_analysis(self):
        if self.raw_image is None:
            self.status_bar.showMessage("Error: No image loaded.")
            return

        plugin_name = self.algo_selector.currentText()
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return

        if self.current_roi:
            x, y, w, h = self.current_roi
            input_image = self.raw_image[y:y+h, x:x+w]
        else:
            input_image = self.raw_image

        params = plugin.parameters 

        try:
            self.status_bar.showMessage(f"Running {plugin_name}...")
            result_mask = plugin.run(input_image, params)

            if result_mask is None:
                self.status_bar.showMessage("Analysis returned no result.")
                return

            # Reconstruction
            full_h, full_w = self.raw_image.shape[:2]
            final_mask = np.zeros((full_h, full_w), dtype=bool)

            if self.current_roi:
                x, y, w, h = self.current_roi
                if result_mask.shape == (h, w):
                    final_mask[y:y+h, x:x+w] = result_mask
            else:
                final_mask = result_mask

            # --- DEĞİŞİKLİK: Maskeyi hafızaya at ---
            self.current_mask = final_mask
            # --------------------------------------

            self.display_overlay(final_mask)
            
            count = np.count_nonzero(final_mask)
            self.status_bar.showMessage(f"Analysis Complete. Detected: {count} px. Ready to save.")

        except Exception as e:
            print(f"Analysis Error: {e}")
            traceback.print_exc()
            self.status_bar.showMessage(f"Error during analysis: {e}")

    def display_overlay(self, mask: np.ndarray):
        self.canvas.set_overlay(mask, color=(0, 255, 0), alpha=100)

    # --- YENİ FONKSİYON: SONUCU KAYDETME ---
    def save_analysis_result(self):
        """
        Orijinal görüntü ile analiz maskesini birleştirip kaydeder.
        """
        if self.raw_image is None or self.current_mask is None:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek bir analiz sonucu yok. Lütfen önce analiz çalıştırın.")
            return

        # 1. Kayıt Yeri Seçimi
        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Analysis Result", 
            str(Path.home() / "analysis_result.tif"), 
            "TIFF Files (*.tif)", 
            options=options
        )

        if not file_path:
            return

        if not file_path.endswith('.tif'):
            file_path += '.tif'

        try:
            print(f"DEBUG: Saving to {file_path}...")
            
            # 2. Görüntüyü Hazırla (Original + Green Overlay)
            # Orijinal görüntünün kopyasını al
            save_image = self.raw_image.copy()

            # Eğer görüntü Grayscale ise (2D), RGB'ye (3D) çevir
            # Çünkü renkli (yeşil) maske ekleyeceğiz.
            if len(save_image.shape) == 2:
                save_image = cv2.cvtColor(save_image, cv2.COLOR_GRAY2RGB)
            elif save_image.shape[2] == 4: # RGBA ise RGB yap
                save_image = cv2.cvtColor(save_image, cv2.COLOR_RGBA2RGB)

            # 3. Maskeyi Yak (Yeşil: [0, 255, 0])
            # Sadece maskenin True olduğu pikselleri boyuyoruz
            save_image[self.current_mask] = [0, 255, 0]

            # 4. Kayıpsız Kayıt (TIFF + LZW Compression)
            # imagecodecs kütüphanesini kullanarak LZW sıkıştırması yapar
            tifffile.imwrite(file_path, save_image, compression='lzw')
            
            self.status_bar.showMessage(f"Saved successfully: {Path(file_path).name}")
            QMessageBox.information(self, "Başarılı", f"Dosya kaydedildi:\n{file_path}")

        except Exception as e:
            print(f"Save Error: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Hata", f"Dosya kaydedilirken hata oluştu:\n{e}")