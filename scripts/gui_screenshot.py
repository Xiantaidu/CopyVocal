"""Render the window to PNG so the layout can be inspected without a display.

Run with:  .\\python\\python.exe -X utf8 scripts\\gui_screenshot.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from gui.window import MainWindow


def main() -> int:
    application = QApplication([])
    window = MainWindow()
    window.teacher_edit.setText(str(ROOT / "Teacher.wav"))
    window.student_edit.setText(str(ROOT / "Student.wav"))
    window.output_edit.setText(str(ROOT / "outputs" / "short"))
    window.advanced.setChecked(True)
    window.refresh_results()

    target = ROOT / "outputs" / "gui_screenshot.png"
    window.resize(1180, 860)
    window.show()
    application.processEvents()
    # Exercise the progress display so the shot shows a mid-run state too.
    for stage, state in (("analysis", "done"), ("alignment", "done"), ("render", "running")):
        window._update_stage(stage, state)
    window._update_progress(72)
    window.detail_label.setText("正在渲染：full")
    application.processEvents()
    window.grab().save(str(target))
    print("saved", target, target.stat().st_size, "bytes")

    window.tabs.setCurrentIndex(1)
    window.tables.setCurrentIndex(0)
    application.processEvents()
    tables = ROOT / "outputs" / "gui_screenshot_results.png"
    window.grab().save(str(tables))
    print("saved", tables, tables.stat().st_size, "bytes")

    window.tables.setCurrentIndex(1)
    application.processEvents()
    regions = ROOT / "outputs" / "gui_screenshot_regions.png"
    window.grab().save(str(regions))
    print("saved", regions, regions.stat().st_size, "bytes")

    window.tables.setCurrentIndex(2)
    application.processEvents()
    warnings = ROOT / "outputs" / "gui_screenshot_warnings.png"
    window.grab().save(str(warnings))
    print("saved", warnings, warnings.stat().st_size, "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
