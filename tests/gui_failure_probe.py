"""Check how the window behaves when a job fails.

A failing job must surface the error, re-enable Run, and leave unfinished stages
unfinished rather than showing them as complete. Run with:

    .\\python\\python.exe -X utf8 tests\\gui_failure_probe.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox

from gui.window import MainWindow


def main() -> int:
    application = QApplication([])
    window = MainWindow()
    # The interface validates its inputs before starting, so drive the failure
    # through a path it accepts but the pipeline cannot use: a real, empty output
    # directory named on the command line as a source file. The interfaces cannot
    # know this is wrong -- the analysis stage discovers it -- so the job really is
    # started and really does fail.
    output = ROOT / "outputs" / "failure_probe"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("teacher.analysis.json", "student.analysis.json"):
        (output / name).unlink(missing_ok=True)
    blocked = ROOT / "outputs" / "blocked_file.tmp"
    blocked.write_text("not audio", encoding="utf-8")
    window.teacher_edit.setText(str(blocked))       # exists, but is not decodable audio
    window.student_edit.setText(str(ROOT / "Student.wav"))
    window.output_edit.setText(str(output))
    window.game_check.setChecked(False)
    window.verify_check.setChecked(False)

    preview = window._collect().validate()
    print("interface-level validation:", preview or "none (the pipeline must fail instead)")

    outcome: dict = {}
    window.runner.finished.connect(outcome.update)
    dialogs = []
    def record(*args, **kwargs):
        dialogs.append(args[2] if len(args) > 2 else "")
        return QMessageBox.StandardButton.Ok
    with patch("gui.window.QMessageBox.critical", side_effect=record) as critical, \
         patch("gui.window.QMessageBox.warning", side_effect=record) as warning:
        window.start_job()
        if window.runner.running:
            window.runner.wait_for_finish(180_000)
            application.processEvents()

    print("outcome:", outcome.get("status"), "exit", outcome.get("exit_code"))
    print("stages:", window.runner.stage_states())
    print("run re-enabled:", window.run_button.isEnabled(),
          "| cancel disabled:", not window.cancel_button.isEnabled())
    print("error dialog shown:", bool(dialogs))
    if dialogs:
        print("dialog text:", str(dialogs[0]).replace("\n", " ")[:160])

    # Closing must not block on a modal dialog once the job has ended.
    window.close()
    blocked.unlink(missing_ok=True)

    failures = []
    if outcome.get("status") != "failed":
        failures.append("a failing job did not report failure")
    if not window.run_button.isEnabled():
        failures.append("the Run button stayed disabled")
    if window.cancel_button.isEnabled():
        failures.append("the Cancel button stayed enabled")
    if not dialogs:
        failures.append("no error was shown to the user")
    for item in failures:
        print("FAIL:", item)
    if failures:
        return 1
    print("OK: a failing job is reported, stages stay unfinished, and the window recovers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
