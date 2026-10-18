from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_app_icon():
    from PySide6.QtGui import QIcon

    icon = ROOT / "icon.png"
    return QIcon(str(icon)) if icon.is_file() else QIcon()


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from gui import __version__
    from gui.window import MainWindow
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    application = QApplication(sys.argv)
    application.setApplicationName("CopyVocal")
    application.setOrganizationName("CopyVocal")
    application.setWindowIcon(_load_app_icon())
    window = MainWindow(version=__version__)
    window.show()
    return application.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
