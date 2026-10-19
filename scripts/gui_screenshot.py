from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from gui.i18n import tr
from gui.window import MainWindow


def save(window, application, name: str) -> None:
    application.processEvents()
    target = ROOT / "outputs" / name
    window.grab().save(str(target))
    print("saved", target, target.stat().st_size, "bytes")


def main() -> int:
    application = QApplication([])
    window = MainWindow()
    window.teacher_edit.setText(str(ROOT / "Teacher.wav"))
    window.student_edit.setText(str(ROOT / "Student.wav"))
    window.output_edit.setText(str(ROOT / "outputs" / "short"))
    window.refresh_results()
    window.resize(1260, 1000)
    window.show()
    application.processEvents()

    # Exercise the progress display so the shot shows a mid-run state too.
    for stage, state in (("analysis", "done"), ("alignment", "done"), ("render", "running")):
        window._update_stage(stage, state)
    window._update_progress(72)
    window.progress_bar.setVisible(True)
    window.detail_label.setText(tr("detail_running"))
    window.stage_list.setVisible(True)
    save(window, application, "gui_screenshot.png")

    for expander in (window.options.run_expander, window.options.advanced_expander):
        expander.expand()
    window.navigationInterface.setCurrentItem(window.settings.objectName())
    save(window, application, "gui_screenshot_expanded.png")
    for expander in (window.options.run_expander, window.options.advanced_expander):
        expander.collapse()

    window.navigationInterface.setCurrentItem(window.results.objectName())
    for index, name in ((0, "gui_screenshot_results.png"),
                        (1, "gui_screenshot_regions.png"),
                        (2, "gui_screenshot_warnings.png")):
        window.tables.setCurrentIndex(index)
        save(window, application, name)

    window.navigationInterface.setCurrentItem(window.settings.objectName())
    save(window, application, "gui_screenshot_settings.png")

    window.settings.apply_theme("dark")
    window.navigationInterface.setCurrentItem(window.correction.objectName())
    save(window, application, "gui_screenshot_dark.png")
    window.settings.apply_theme("light")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
