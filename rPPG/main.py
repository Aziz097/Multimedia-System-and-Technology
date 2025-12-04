"""Launcher for rPPG Heart Rate Monitor Application.

This is the main entry point for the rPPG system GUI application.
Run this file to start the heart rate monitoring interface.
"""

import sys
from PyQt6.QtWidgets import QApplication
from rppg_system.gui.gui_app import MainWindow


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
