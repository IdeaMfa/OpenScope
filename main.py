import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import OpenScopeMain

def main() -> None:
    """
    Application Entry Point.
    Initializes the QApplication and the Main Window.
    """
    app = QApplication(sys.argv)
    
    # Optional: Set a global application style
    app.setStyle("Fusion") 

    window = OpenScopeMain()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()