"""Start the CopyVocal desktop interface.

Run through run_gui.bat, or directly:

    .\\python\\pythonw.exe -X utf8 app_gui.py
"""
from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from gui.window import MainWindow
    application = QApplication(sys.argv)
    application.setApplicationName("CopyVocal")
    application.setOrganizationName("CopyVocal")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
