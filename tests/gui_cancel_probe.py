"""Check that cancelling stops a running job and its workers.

Cancellation is the one control the interface implements itself, so it needs a
real child process to be meaningful. Run with:

    .\\python\\python.exe -X utf8 tests\\gui_cancel_probe.py
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from gui.window import MainWindow


def python_processes() -> int:
    """Count interpreter processes, to confirm workers did not survive."""
    completed = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/NH"],
                               capture_output=True, text=True)
    return completed.stdout.lower().count("python.exe")


def main() -> int:
    application = QApplication([])
    window = MainWindow()
    # A long pair with the slow CPU vocoder keeps the job alive long enough to cancel.
    window.teacher_edit.setText(str(ROOT / "teacher_long_hard.wav"))
    window.student_edit.setText(str(ROOT / "student_long_hard.wav"))
    window.output_edit.setText(str(ROOT / "outputs" / "gui_cancel_probe"))
    window.vocoder_combo.setCurrentText("cpu")
    window.onnx_combo.setCurrentText("cpu")
    window.verify_check.setChecked(False)

    before = python_processes()
    loop = QEventLoop()
    outcome: dict = {}
    window.runner.finished.connect(lambda result: (outcome.update(result), loop.quit()))
    window.start_job()
    if not window.runner.running:
        print("FAIL: the job did not start")
        return 1
    print("job started; python processes before:", before)
    QTimer.singleShot(6000, window.runner.cancel)   # let analysis begin, then stop it
    QTimer.singleShot(120_000, loop.quit)           # safety net
    loop.exec()

    time.sleep(3)  # let taskkill finish reaping the tree
    after = python_processes()
    print("outcome:", outcome.get("status"), "exit", outcome.get("exit_code"),
          f"in {outcome.get('elapsed_seconds', 0):.1f}s")
    print("python processes after:", after)
    print("stages:", window.runner.stage_states())
    print("run button enabled again:", window.run_button.isEnabled())
    print("cancel button disabled:", not window.cancel_button.isEnabled())
    window.close()
    failures = []
    if outcome.get("status") != "cancelled":
        failures.append("the job did not report cancellation")
    if not window.run_button.isEnabled():
        failures.append("the Run button stayed disabled")
    if after > before + 1:
        failures.append(f"worker processes may have survived ({before} -> {after})")
    for item in failures:
        print("FAIL:", item)
    if failures:
        return 1
    print("OK: cancellation stopped the job and its workers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
