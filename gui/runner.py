"""Run the command-line pipeline in a child process and report its progress.

The interface adds no inference of its own: the pipeline already isolates GPU
backends behind worker processes, so this class starts one child, streams its
merged output, and translates the messages the workers actually write into
pipeline stages.

Two properties are deliberate. Cancelling kills the whole process tree,
because the child owns its own workers. And the stage table contains only
markers observed in real runs: the correspondence stage prints nothing at all,
so it is derived from the surrounding stages rather than guessed from prose.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from application.artifacts import ROOT
from gui import text as T

# Ordered stages as the pipeline runs them, with what each one accomplishes. The
# keys are the pipeline's own vocabulary; the labels are what the window shows.
STAGES: tuple[tuple[str, str], ...] = T.STAGE_LABELS

# Percent shown when each stage has just finished. The correspondence stage runs
# in-process between two workers and is much shorter than analysis or synthesis.
_STAGE_PERCENT = {"analysis": 58, "alignment": 63, "render": 80,
                  "verification": 90, "timing_verification": 95, "review": 100}

# (pattern, stage, state), matched against the message without its log prefix.
# Each of these strings appears in verbatim worker output; see tests/test_gui.py.
_MARKERS: tuple[tuple[re.Pattern, str, str], ...] = (
    (re.compile(r"^Starting analysis_worker\.py\."), "analysis", "running"),
    (re.compile(r"^Analysis artifacts saved\.$"), "analysis", "done"),
    # No worker is launched for correspondence: run_job computes it in-process.
    (re.compile(r"^Starting render_worker\.py\."), "render", "running"),
    (re.compile(r"^Render artifacts saved\.$"), "render", "done"),
    (re.compile(r"^Starting verify_worker\.py\."), "verification", "running"),
    (re.compile(r"^Verification saved: "), "verification", "done"),
    (re.compile(r"^Starting timing_verify_worker\.py\."), "timing_verification", "running"),
    (re.compile(r"^Timing verification saved: "), "timing_verification", "done"),
    # write_report runs immediately before this line, so review is complete.
    (re.compile(r"^Saved corrected vocal: "), "review", "done"),
)

# Activity text shown to the user, taken verbatim from the pipeline's progress hook.
_DETAIL: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^Extracting continuous pitch from (\w+)\.$"), r"Continuous pitch: \1"),
    (re.compile(r"^Rendering (\w+)\.$"), r"Rendering \1"),
    (re.compile(r"^Loading PC-NSF-HiFiGAN on (\w+)\.$"), r"Loading vocoder on \1"),
)

# Chunk progress lines, with the shorter label the window shows.
_CHUNK = re.compile(r"^(Transcribing|Aligning|Detecting notes in) (\w+) chunk (\d+)/(\d+)\.$")
_CHUNK_LABELS = {"Transcribing": "Lyrics", "Aligning": "Phonemes", "Detecting notes in": "Notes"}
# Log text that carries values is rebuilt in Chinese from the parts the window
# already parses, so no worker message format has to change.
def _track(value: str) -> str:
    return T.STAGE_TRACK_LABELS.get(value, value)
_PITCH = re.compile(r"^Extracting continuous pitch from (\w+)\.$")

# Analysis phases in the order analyze_pair performs them, each with the share of
# the analysis bar it covers. Tracks are processed one after the other.
_ANALYSIS_SLOTS: tuple[tuple[str, str, float, float], ...] = (
    ("Transcribing", "teacher", 0.00, 0.17),
    ("Transcribing", "student", 0.17, 0.34),
    ("Aligning", "teacher", 0.34, 0.50),
    ("Aligning", "student", 0.50, 0.62),
    ("pitch", "teacher", 0.62, 0.70),
    ("pitch", "student", 0.70, 0.78),
    ("Detecting notes in", "teacher", 0.78, 0.89),
    ("Detecting notes in", "student", 0.89, 1.00),
)


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


class PipelineRunner(QObject):
    """Own the child process and expose its progress as signals."""

    line = Signal(str)                     # one line of merged output
    stage_changed = Signal(str, str)       # stage key, state (pending/running/done/skipped)
    detail = Signal(str)                   # latest human-readable activity
    progress = Signal(int)                 # 0-100, monotonic within a job
    finished = Signal(dict)                # {status, exit_code, elapsed_seconds, error}

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        self._stages: dict[str, str] = {key: "pending" for key, _ in STAGES}
        self._started = 0.0
        self._cancelled = False
        self._percent = 0
        self._active = False   # True between start() and the finish signal
        self._tail = ""        # partial line held between reads

    # ── Queries ───────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._active and self._process is not None

    def stage_states(self) -> dict[str, str]:
        return dict(self._stages)

    @staticmethod
    def is_stage_enabled(key: str, request) -> bool:
        """Whether this request is expected to reach a stage at all.

        Correspondence runs in the parent process from whatever analysis the
        output directory holds, so it happens even when analysis is reused.
        """
        if key == "analysis":
            return not request.skip_analysis
        if key in {"alignment", "render", "review"}:
            return not request.analyze_only
        if key == "verification":
            return not request.analyze_only and request.verify
        if key == "timing_verification":
            return not request.analyze_only and request.verify_timing
        return True

    # ── Control ───────────────────────────────────────────────────
    def start(self, request) -> None:
        """Write the settings file and launch scripts/copyvocal_cli.py."""
        if self.running:
            raise RuntimeError("A job is already running.")
        request.write_config(request.config_path())
        self._stages = {key: ("pending" if self.is_stage_enabled(key, request) else "skipped")
                        for key, _ in STAGES}
        self._started, self._cancelled, self._percent = time.perf_counter(), False, 0
        for key, _ in STAGES:
            self.stage_changed.emit(key, self._stages[key])
        # Without analysis, correspondence is the first thing the parent does.
        if request.skip_analysis and not request.analyze_only:
            self._set_stage("alignment", "running")
        self.progress.emit(0)
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-X", "utf8", "-u", str(ROOT / "scripts" / "copyvocal_cli.py"),
                              *request.argv()])
        process.setWorkingDirectory(str(ROOT))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("V2M_FFMPEG", str(ROOT / "ffmpeg.exe"))
        environment.insert("PATH", str(ROOT) + os.pathsep + environment.value("PATH"))
        process.setProcessEnvironment(environment)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_error)
        self._process = process
        self._active = True
        self.detail.emit(T.DETAIL_STARTING)
        process.start()

    def cancel(self) -> None:
        """Stop the job and the workers it spawned."""
        if not self.running or self._process is None:
            return
        self._cancelled = True
        pid = int(self._process.processId())
        self.detail.emit(T.DETAIL_CANCELLING)
        threading.Thread(target=self._kill_tree, args=(pid,), daemon=True).start()

    @staticmethod
    def _kill_tree(pid: int) -> None:
        """Kill a process and its children; the child owns model subprocesses."""
        if pid <= 0:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.killpg(os.getpgid(pid), 15)
            except (OSError, ProcessLookupError):
                pass

    def wait_for_finish(self, timeout_ms: int = 15000) -> bool:
        """Block briefly, for shutdown while a job is still running."""
        if self._process is None:
            return True
        return self._process.waitForFinished(timeout_ms)

    # ── Output handling ───────────────────────────────────────────
    def _read(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput().data())
        if not data:
            return
        # Complete lines only; a partial tail waits for the next read. The child is
        # started unbuffered, so this stays prompt.
        pending = self._tail + _decode(data)
        lines = pending.split("\n")
        self._tail = lines.pop()
        for line in lines:
            self._consume(line.rstrip("\r"))

    def _consume(self, raw: str) -> None:
        # logging writes "<timestamp> LEVEL <message>"; match the message alone.
        message = re.sub(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} \w+ ", "", raw)
        self.line.emit(raw)
        for pattern, stage, state in _MARKERS:
            if pattern.search(message):
                self._set_stage(stage, state)
                if stage == "analysis" and state == "done":
                    self._set_stage("alignment", "running")
                elif stage == "render" and state == "running":
                    # Render starts only after correspondence finished.
                    self._set_stage("alignment", "done")
                break
        self._track_detail(message)

    def _set_stage(self, stage: str, state: str) -> None:
        """Move a stage forward; a finished or skipped stage never regresses."""
        if self._stages.get(stage) in {"done", "skipped"} or self._stages.get(stage) == state:
            return
        self._stages[stage] = state
        self.stage_changed.emit(stage, state)
        if state == "done":
            self._set_percent(_STAGE_PERCENT.get(stage, self._percent))

    def _set_percent(self, value: int) -> None:
        """Progress only ever moves forward within a job."""
        value = max(0, min(100, int(value)))
        if value > self._percent and self._percent < 100:
            self._percent = value
            self.progress.emit(value)

    def _track_detail(self, message: str) -> None:
        chunk = _CHUNK.match(message)
        if chunk:
            phase, track = chunk.group(1), chunk.group(2)
            index, total = int(chunk.group(3)), int(chunk.group(4))
            self.detail.emit(T.DETAIL_CHUNK.format(
                phase=T.PHASE_LABELS[_CHUNK_LABELS[phase]], track=_track(track),
                index=index, total=total))
            self._analysis_percent(phase, track, index / max(1, total))
            return
        pitch = _PITCH.match(message)
        if pitch:
            self.detail.emit(T.DETAIL_PITCH.format(track=_track(pitch.group(1))))
            self._analysis_percent("pitch", pitch.group(1), 1.0)
            return
        for pattern, replacement in _DETAIL:
            match = pattern.search(message)
            if not match:
                continue
            value = match.group(1)
            if replacement == r"Continuous pitch: \1":
                self.detail.emit(T.DETAIL_PITCH.format(track=_track(value)))
            elif replacement == r"Rendering \1":
                self.detail.emit(T.DETAIL_RENDER.format(mode=value))
            else:
                self.detail.emit(T.DETAIL_LOADING_VOCODER.format(device=value))
            return

    def _analysis_percent(self, phase: str, track: str, fraction: float) -> None:
        """Map an analysis sub-step onto the analysis share of the bar."""
        for name, label, start, stop in _ANALYSIS_SLOTS:
            if name == phase and label == track:
                span = _STAGE_PERCENT["analysis"]
                self._set_percent(round((start + (stop - start) * fraction) * span))
                return

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._emit_finish(-1, T.STATUS_FAILED[0])

    def _on_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._read()  # drain anything written just before exit
        trailing = self._tail.strip()
        self._tail = ""
        if trailing:
            self.line.emit(trailing)
        error = T.DIALOG_PIPELINE_FAILED if status == QProcess.ExitStatus.CrashExit else ""
        self._emit_finish(exit_code, error)

    def _emit_finish(self, exit_code: int, error: str) -> None:
        """Settle the job once; a start failure and a normal exit both land here."""
        if not self._active:
            return
        self._active, self._process = False, None
        if self._cancelled:
            status = "cancelled"
        elif exit_code == 0:
            status = "succeeded"
        else:
            status = "failed"
        # A stage still running when the job ends did not complete.
        for key, state in list(self._stages.items()):
            if state == "running":
                self._set_stage(key, "done" if status == "succeeded" else "pending")
        if status == "succeeded":
            self._set_percent(100)
        self.finished.emit({"status": status, "exit_code": exit_code, "error": error,
                            "elapsed_seconds": time.perf_counter() - self._started})
