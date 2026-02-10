import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QStatusBar, QFileDialog, QLabel, QFrame, 
                             QComboBox, QPushButton, QGroupBox)
from PyQt6.QtCore import Qt
from pathlib import Path

class OpenScopeMain(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("OpenScope - Scientific Image Analysis Platform")
        self.resize(1200, 800)

        # 1. Main Layout Setup
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # 2. Sidebar Setup
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(320)
        self.sidebar.setStyleSheet("background-color: #f8f9fa; border-right: 1px solid #ddd;")
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        
        # --- Sidebar Header ---
        lbl_sidebar = QLabel("Analysis Controls")
        lbl_sidebar.setStyleSheet("font-weight: bold; font-size: 16px; color: #333; margin-bottom: 10px;")
        self.sidebar_layout.addWidget(lbl_sidebar)

        # --- Algorithm Selector Section ---
        self.algo_group = QGroupBox("Select Algorithm")
        self.algo_layout = QVBoxLayout()
        
        self.algo_selector = QComboBox()
        self.algo_selector.addItems(["Select an algorithm...", "Basic Blob Detection", "Advanced Segmentation"])
        # Not: Phase 3'te burası plugins klasöründen otomatik dolacak.
        
        self.algo_layout.addWidget(self.algo_selector)
        self.algo_group.setLayout(self.algo_layout)
        self.sidebar_layout.addWidget(self.algo_group)

        # --- Dynamic Parameters Placeholder ---
        # Seçilen algoritmaya göre burası değişecek (Phase 3)
        self.params_container = QGroupBox("Parameters")
        self.params_layout = QVBoxLayout()
        lbl_placeholder = QLabel("No algorithm selected.")
        lbl_placeholder.setStyleSheet("color: #777; font-style: italic;")
        self.params_layout.addWidget(lbl_placeholder)
        self.params_container.setLayout(self.params_layout)
        
        self.sidebar_layout.addWidget(self.params_container)

        # --- Action Buttons ---
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #007bff; color: white; padding: 10px; 
                font-weight: bold; border-radius: 4px;
            }
            QPushButton:hover { background-color: #0056b3; }
        """)
        self.sidebar_layout.addWidget(self.btn_run)
        
        self.sidebar_layout.addStretch() # Push everything up
        self.main_layout.addWidget(self.sidebar)

        # 3. Canvas Area (Placeholder)
        self.canvas_placeholder = QFrame()
        self.canvas_placeholder.setStyleSheet("background-color: #2b2b2b;")
        lbl_info = QLabel("Image Canvas\n(Drop File Here)")
        lbl_info.setStyleSheet("color: #888; font-size: 20px;")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        cv_layout = QVBoxLayout(self.canvas_placeholder)
        cv_layout.addWidget(lbl_info)
        
        self.main_layout.addWidget(self.canvas_placeholder, stretch=1)

        # 4. Status Bar & Menu
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._init_menu()

    def _init_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_action = file_menu.addAction("&Open TIFF...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file_dialog)

    def open_file_dialog(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", str(Path.home()), "TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_name:
            self.status_bar.showMessage(f"Loaded: {file_name}")
            print(f"File selected: {file_name}")