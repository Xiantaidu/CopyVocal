from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from gui.window import MainWindow


def main() -> int:
    application = QApplication([])
    window = MainWindow()
    output = ROOT / "outputs" / "gui_probe_out"
    window.teacher_edit.setText(str(ROOT / "Teacher.wav"))
    window.student_edit.setText(str(ROOT / "Student.wav"))
    window.output_edit.setText(str(output))
    # Keep the probe cheap while still exercising every stage the window shows.
    window.game_check.setChecked(False)
    window.experiments_check.setChecked(True)
    window.verify_check.setChecked(True)
    print("preview:", window.command_label.text()[:160], "...")

    loop = QEventLoop()
    outcome: dict = {}
    window.runner.finished.connect(lambda result: (outcome.update(result), loop.quit()))
    window.start_job()
    if not window.runner.running:
        print("FAIL: the job did not start")
        return 1
    QTimer.singleShot(600_000, loop.quit)  # safety net
    loop.exec()

    print("outcome:", outcome.get("status"), "exit", outcome.get("exit_code"),
          f"in {outcome.get('elapsed_seconds', 0):.1f}s")
    print("stages:", window.runner.stage_states())
    print("progress bar:", window.progress_bar.value())
    print("last progress line:", window.detail_label.text())
    window.refresh_results()
    rows = window.metric_table.rowCount()
    print("result metric rows:", rows)
    print("regions:", window.region_table.rowCount(), "| clips:", window.audio_combo.count())
    print("summary:", window.summary_label.text().replace("<br>", " ")[:200])
    window.close()
    status = outcome.get("status")
    if status != "succeeded":
        print("FAIL: pipeline did not succeed")
        return 1
    if window.progress_bar.value() != 100:
        print("FAIL: progress did not reach 100")
        return 1
    if rows < 5 or window.audio_combo.count() < 4:
        print("FAIL: result panel is not populated")
        return 1
    print("OK: end-to-end run through the interface completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
