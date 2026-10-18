from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSplitter, QTabWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from application.artifacts import ROOT
from gui import text as T
from gui.job import AUDIO_SUFFIXES, LANGUAGES, PREFS_PATH, JobRequest, suggested_output
from gui.results import JobArtifacts
from gui.runner import STAGES, PipelineRunner

STAGE_ICONS = {"pending": "○", "running": "▶", "done": "✔", "skipped": "–"}
MAX_LOG_LINES = 4000
FILTER_AUDIO = "音频文件 (" + " ".join(f"*{item}" for item in AUDIO_SUFFIXES) + ");;所有文件 (*)"
PENDING_BRUSH, DONE_BRUSH = QBrush(QColor("#666666")), QBrush(QColor("#1a7f37"))


def _monospace() -> QFont:
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


class MainWindow(QMainWindow):
    """Collect options, run the pipeline, and present its artifacts."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(T.WINDOW_TITLE)
        self.resize(1180, 820)
        self.setAcceptDrops(True)
        self.request = JobRequest.from_defaults()
        self.runner = PipelineRunner(self)
        self.player, self.audio_output = QMediaPlayer(self), QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self._player_target = ""
        self._load_prefs()
        self._build()
        self._connect()
        self._sync_enabled()
        self.refresh_results()

    def _build(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._inputs_box())
        layout.addWidget(self._options_box())
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._progress_box())
        splitter.addWidget(self._results_tabs())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        # Push the restored request into the widgets only once the whole window
        # exists, so no signal handler can run against a missing control.
        self._apply(self.request)
        self.statusBar().showMessage(T.STATUSBAR_READY)

    def _inputs_box(self) -> QGroupBox:
        box = QGroupBox(T.GROUP_RECORDINGS)
        grid = QGridLayout(box)
        self.teacher_edit, self.student_edit, self.output_edit = QLineEdit(), QLineEdit(), QLineEdit()
        for row, (label, edit, handler) in enumerate((
                (T.LABEL_TEACHER, self.teacher_edit, self._browse_teacher),
                (T.LABEL_STUDENT, self.student_edit, self._browse_student),
                (T.LABEL_OUTPUT, self.output_edit, self._browse_output))):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            button = QPushButton(T.BUTTON_BROWSE)
            button.clicked.connect(handler)
            grid.addWidget(button, row, 2)
        grid.setColumnStretch(1, 1)
        self.teacher_edit.setText(self.request.teacher)
        self.student_edit.setText(self.request.student)
        self.output_edit.setText(self.request.output)
        hint = QLabel(T.HINT_INPUTS)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        grid.addWidget(hint, 3, 0, 1, 3)
        return box

    def _options_box(self) -> QGroupBox:
        box = QGroupBox(T.GROUP_OPTIONS)
        outer = QVBoxLayout(box)
        grid = QGridLayout()
        self.onnx_combo, self.vocoder_combo, self.language_combo = QComboBox(), QComboBox(), QComboBox()
        self.onnx_combo.addItems(("dml", "cpu"))
        self.onnx_combo.setToolTip(T.TOOLTIP_ONNX)
        self.vocoder_combo.addItems(("auto", "cuda", "cpu"))
        self.vocoder_combo.setToolTip(T.TOOLTIP_VOCODER)
        self.language_combo.addItems(LANGUAGES)
        for column, (label, widget) in enumerate(((T.LABEL_LYRIC_BACKEND, self.onnx_combo),
                                                 (T.LABEL_VOCODER_DEVICE, self.vocoder_combo),
                                                 (T.LABEL_LANGUAGE, self.language_combo))):
            grid.addWidget(QLabel(label), 0, column * 2)
            grid.addWidget(widget, 0, column * 2 + 1)
        outer.addLayout(grid)

        self.game_check = QCheckBox(T.CHECK_GAME)
        self.experiments_check = QCheckBox(T.CHECK_EXPERIMENTS)
        self.verify_check = QCheckBox(T.CHECK_VERIFY)
        self.verify_timing_check = QCheckBox(T.CHECK_VERIFY_TIMING)
        self.analyze_check = QCheckBox(T.CHECK_ANALYZE_ONLY)
        self.skip_check = QCheckBox(T.CHECK_SKIP_ANALYSIS)
        self.fallback_check = QCheckBox(T.CHECK_FALLBACK)
        for widget, tip in ((self.game_check, T.TOOLTIP_CHECK_GAME),
                            (self.experiments_check, T.TOOLTIP_CHECK_EXPERIMENTS),
                            (self.verify_check, T.TOOLTIP_CHECK_VERIFY),
                            (self.verify_timing_check, T.TOOLTIP_CHECK_VERIFY_TIMING),
                            (self.analyze_check, T.TOOLTIP_CHECK_ANALYZE_ONLY),
                            (self.skip_check, T.TOOLTIP_CHECK_SKIP_ANALYSIS),
                            (self.fallback_check, T.TOOLTIP_CHECK_FALLBACK)):
            widget.setToolTip(tip)
        toggles = QGridLayout()
        for index, widget in enumerate((self.game_check, self.experiments_check, self.verify_check,
                                        self.verify_timing_check, self.analyze_check, self.skip_check,
                                        self.fallback_check)):
            toggles.addWidget(widget, index // 3, index % 3)
        outer.addLayout(toggles)

        # The pitch switch is a three-way choice rather than a checkbox, because
        # "leave it alone", "move it to the teacher" and "place each note inside
        # the student's own register" are three different answers to one question.
        self.timing_check = QCheckBox(T.CHECK_TIMING)
        self.pitch_combo = QComboBox()
        self.pitch_combo.addItems(T.PITCH_MODE_LABELS)
        self.pitch_map_check = QCheckBox(T.CHECK_PITCH_MAP)
        self.timing_check.setToolTip(T.TOOLTIP_CHECK_TIMING)
        self.pitch_combo.setToolTip(T.TOOLTIP_PITCH_MODE)
        self.pitch_map_check.setToolTip(T.TOOLTIP_CHECK_PITCH_MAP)
        axis_row = QHBoxLayout()
        axis_row.addWidget(self.timing_check)
        axis_row.addWidget(QLabel(T.LABEL_PITCH_MODE))
        axis_row.addWidget(self.pitch_combo)
        axis_row.addWidget(self.pitch_map_check)
        axis_row.addStretch(1)
        outer.addLayout(axis_row)
        hint = QLabel(T.AXIS_HINT)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        outer.addWidget(hint)

        self.advanced = QGroupBox(T.GROUP_ADVANCED)
        self.advanced.setCheckable(True)
        form = QFormLayout(self.advanced)
        self.chunk_spin = self._spin(1.0, 600.0, self.request.max_chunk_seconds, 1.0, " s")
        self.context_spin = self._spin(0.01, 10.0, self.request.context_seconds, 0.05, " s")
        self.quality_spin = self._spin(0.01, 1.0, self.request.minimum_alignment_quality, 0.05, "")
        self.stretch_spin = self._spin(1.0, 100.0, self.request.maximum_stretch, 0.5, " x")
        self.floor_spin = self._spin(1e-9, 1.0, self.request.log_floor, 1e-5, "", decimals=9)
        for label, widget, tip in (
                (T.ADVANCED_MAX_CHUNK, self.chunk_spin, T.TOOLTIP_MAX_CHUNK),
                (T.ADVANCED_CONTEXT, self.context_spin, T.TOOLTIP_CONTEXT),
                (T.ADVANCED_QUALITY, self.quality_spin, T.TOOLTIP_QUALITY),
                (T.ADVANCED_STRETCH, self.stretch_spin, T.TOOLTIP_STRETCH),
                (T.ADVANCED_FLOOR, self.floor_spin, T.TOOLTIP_FLOOR)):
            widget.setToolTip(tip)
            form.addRow(label, widget)
        outer.addWidget(self.advanced)
        return box

    def _spin(self, low: float, high: float, value: float, step: float, suffix: str,
              decimals: int = 3) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(self._sync_enabled)
        return spin

    def _progress_box(self) -> QGroupBox:
        box = QGroupBox(T.GROUP_PROGRESS)
        layout = QVBoxLayout(box)
        toolbar = QHBoxLayout()
        self.run_button = QPushButton(T.BUTTON_RUN)
        self.run_button.setDefault(True)
        self.cancel_button = QPushButton(T.BUTTON_CANCEL)
        self.cancel_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        toolbar.addWidget(self.run_button)
        toolbar.addWidget(self.cancel_button)
        toolbar.addWidget(self.progress_bar, 1)
        layout.addLayout(toolbar)
        self.command_label = QLabel()
        self.command_label.setFont(_monospace())
        self.command_label.setWordWrap(True)
        self.command_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.command_label.setStyleSheet("color: #666;")
        layout.addWidget(self.command_label)
        self.stage_list = QListWidget()
        self.stage_list.setMaximumHeight(150)
        for key, label in STAGES:
            item = QListWidgetItem(f"{STAGE_ICONS['pending']}  {label}")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.stage_list.addItem(item)
        layout.addWidget(self.stage_list)
        self.detail_label = QLabel(T.DETAIL_NO_JOB)
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        return box

    def _results_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(_monospace())
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        self.log_view.setPlaceholderText(T.LOG_PLACEHOLDER)
        self.tabs.addTab(self.log_view, T.TAB_LOG)

        panel = QWidget()
        results_layout = QVBoxLayout(panel)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        results_layout.addWidget(self.summary_label)
        self.tables = QTabWidget()
        self.metric_table = self._table((T.HEADER_MEASUREMENT, T.HEADER_VALUE))
        self.region_table = self._table(T.HEADERS_REGIONS)
        self.warning_view = QPlainTextEdit()
        self.warning_view.setReadOnly(True)
        self.warning_view.setPlaceholderText(T.WARNING_PLACEHOLDER)
        self.tables.addTab(self.metric_table, T.TABLE_METRICS)
        self.tables.addTab(self.region_table, T.TABLE_REGIONS)
        self.tables.addTab(self.warning_view, T.TABLE_WARNINGS)
        results_layout.addWidget(self.tables, 1)
        audio_row = QHBoxLayout()
        self.audio_combo = QComboBox()
        self.audio_combo.setMinimumWidth(220)
        self.play_button, self.stop_button = QPushButton(T.BUTTON_PLAY), QPushButton(T.BUTTON_STOP)
        self.open_report_button = QPushButton(T.BUTTON_OPEN_REPORT)
        self.open_output_button = QPushButton(T.BUTTON_OPEN_OUTPUT)
        for widget in (QLabel(T.LABEL_LISTENING), self.audio_combo, self.play_button, self.stop_button):
            audio_row.addWidget(widget)
        audio_row.addStretch(1)
        audio_row.addWidget(self.open_report_button)
        audio_row.addWidget(self.open_output_button)
        results_layout.addLayout(audio_row)
        self.tabs.addTab(panel, T.TAB_RESULTS)
        return self.tabs

    def _table(self, headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def _connect(self) -> None:
        self.run_button.clicked.connect(self.start_job)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.runner.line.connect(self._append_log)
        self.runner.stage_changed.connect(self._update_stage)
        self.runner.detail.connect(self.detail_label.setText)
        self.runner.progress.connect(self._update_progress)
        self.runner.finished.connect(self._on_finished)
        self.play_button.clicked.connect(self._play)
        self.stop_button.clicked.connect(self.player.stop)
        self.audio_combo.currentIndexChanged.connect(self._pick_audio)
        self.open_report_button.clicked.connect(self._open_report)
        self.open_output_button.clicked.connect(self._open_output)
        for edit in (self.teacher_edit, self.student_edit, self.output_edit):
            edit.textChanged.connect(self._sync_enabled)
        for combo in (self.onnx_combo, self.vocoder_combo, self.language_combo):
            combo.currentIndexChanged.connect(self._sync_enabled)
        for check in (self.game_check, self.experiments_check, self.verify_check,
                      self.verify_timing_check, self.analyze_check, self.skip_check, self.fallback_check,
                      self.timing_check, self.pitch_map_check):
            check.stateChanged.connect(self._sync_enabled)
        self.pitch_combo.currentIndexChanged.connect(self._sync_enabled)
        self.player.errorOccurred.connect(self._on_player_error)

    @Slot()
    def _sync_enabled(self) -> None:
        """Keep the option combination legal and the command preview current."""
        analyze, skip = self.analyze_check.isChecked(), self.skip_check.isChecked()
        self.skip_check.setEnabled(not analyze)
        self.analyze_check.setEnabled(not skip)
        self.verify_timing_check.setEnabled(not analyze)
        self.experiments_check.setEnabled(not analyze)
        # The pitch switch decides where the pitch lands and the contour checkbox
        # decides whose pitch movement is sung inside it; 'off' has no contour to
        # choose, so the checkbox is disabled there rather than silently ignored.
        self.pitch_map_check.setEnabled(self._pitch_mode() != "off")
        if analyze:
            self.verify_timing_check.setChecked(False)
        self.command_label.setText(self._collect().command_line())

    def _pitch_mode(self) -> str:
        index = self.pitch_combo.currentIndex()
        return T.PITCH_MODE_VALUES[index] if 0 <= index < len(T.PITCH_MODE_VALUES) else "off"

    def _collect(self) -> JobRequest:
        request = self.request
        request.teacher, request.student = self.teacher_edit.text().strip(), self.student_edit.text().strip()
        request.output = self.output_edit.text().strip()
        request.onnx_device = self.onnx_combo.currentText()
        request.vocoder_device = self.vocoder_combo.currentText()
        request.language = self.language_combo.currentText()
        request.max_chunk_seconds = self.chunk_spin.value()
        request.context_seconds = self.context_spin.value()
        request.minimum_alignment_quality = self.quality_spin.value()
        request.maximum_stretch = self.stretch_spin.value()
        request.log_floor = self.floor_spin.value()
        request.include_game = self.game_check.isChecked()
        request.experiments = self.experiments_check.isChecked()
        request.verify = self.verify_check.isChecked()
        request.verify_timing = self.verify_timing_check.isChecked()
        request.analyze_only = self.analyze_check.isChecked()
        request.skip_analysis = self.skip_check.isChecked()
        request.allow_acoustic_fallback = self.fallback_check.isChecked()
        request.change_timing = self.timing_check.isChecked()
        request.pitch_mode = self._pitch_mode()
        request.map_teacher_pitch = self.pitch_map_check.isChecked()
        return request

    def _apply(self, request: JobRequest) -> None:
        self.teacher_edit.setText(request.teacher)
        self.student_edit.setText(request.student)
        self.output_edit.setText(request.output)
        self.onnx_combo.setCurrentText(request.onnx_device)
        self.vocoder_combo.setCurrentText(request.vocoder_device)
        self.language_combo.setCurrentText(request.language)
        self.chunk_spin.setValue(request.max_chunk_seconds)
        self.context_spin.setValue(request.context_seconds)
        self.quality_spin.setValue(request.minimum_alignment_quality)
        self.stretch_spin.setValue(request.maximum_stretch)
        self.floor_spin.setValue(request.log_floor)
        self.game_check.setChecked(request.include_game)
        self.experiments_check.setChecked(request.experiments)
        self.verify_check.setChecked(request.verify)
        self.verify_timing_check.setChecked(request.verify_timing)
        self.analyze_check.setChecked(request.analyze_only)
        self.skip_check.setChecked(request.skip_analysis)
        self.fallback_check.setChecked(request.allow_acoustic_fallback)
        self.timing_check.setChecked(request.change_timing)
        index = T.PITCH_MODE_VALUES.index(request.pitch_mode) if request.pitch_mode in T.PITCH_MODE_VALUES else 0
        self.pitch_combo.setCurrentIndex(index)
        self.pitch_map_check.setChecked(request.map_teacher_pitch)

    def _browse_teacher(self) -> None:
        self._choose_file(self.teacher_edit, T.LABEL_TEACHER)

    def _browse_student(self) -> None:
        self._choose_file(self.student_edit, T.LABEL_STUDENT)

    def _choose_file(self, target: QLineEdit, title: str) -> None:
        start = target.text().strip() or str(ROOT)
        chosen, _ = QFileDialog.getOpenFileName(self, title, start, FILTER_AUDIO)
        if chosen:
            self._set_audio_path(target, chosen)

    def _set_audio_path(self, target: QLineEdit, path: str) -> None:
        target.setText(path)
        if target is self.student_edit:
            current = self.output_edit.text().strip()
            if not current or Path(current).name in {"gui", "outputs"}:
                self.output_edit.setText(str(suggested_output(path)))

    def _browse_output(self) -> None:
        start = self.output_edit.text().strip() or str(ROOT / "outputs")
        chosen = QFileDialog.getExistingDirectory(self, T.LABEL_OUTPUT, start)
        if chosen:
            self.output_edit.setText(chosen)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() and any(
                item.toLocalFile().lower().endswith(AUDIO_SUFFIXES) for item in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(item.toLocalFile()) for item in event.mimeData().urls()
                 if item.toLocalFile().lower().endswith(AUDIO_SUFFIXES)]
        if not paths:
            return
        # Fill empty fields first, so one drop sets the teacher and a second the student.
        for path in paths[:2]:
            current = {self.teacher_edit.text().strip(), self.student_edit.text().strip()}
            if str(path) in current:
                continue
            if not self.teacher_edit.text().strip():
                self._set_audio_path(self.teacher_edit, str(path))
            elif not self.student_edit.text().strip():
                self._set_audio_path(self.student_edit, str(path))
            else:
                self._set_audio_path(self.teacher_edit, str(path))
        event.acceptProposedAction()

    @Slot()
    def start_job(self) -> None:
        request = self._collect()
        problems = request.validate()
        if problems:
            QMessageBox.warning(self, T.DIALOG_CANNOT_START,
                                "\n".join(T.BULLET_ITEM.format(item=item) for item in problems))
            return
        if request.skip_analysis and QMessageBox.question(
                self, T.DIALOG_REUSE_TITLE, T.DIALOG_REUSE_BODY) \
                != QMessageBox.StandardButton.Yes:
            return
        self.log_view.clear()
        self._save_prefs()
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.summary_label.setText(f"<b>{T.DETAIL_RUNNING}</b>")
        self.statusBar().showMessage(T.STATUSBAR_RUNNING)
        try:
            self.runner.start(request)
        except Exception as error:  # a start failure must not close the window
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            QMessageBox.critical(self, T.DIALOG_CANNOT_START, str(error))

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _update_stage(self, stage: str, state: str) -> None:
        for row in range(self.stage_list.count()):
            item = self.stage_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) != stage:
                continue
            item.setText(f"{STAGE_ICONS.get(state, '○')}  {dict(STAGES)[stage]}")
            item.setForeground(DONE_BRUSH if state == "done" else PENDING_BRUSH)
            if state == "running":
                self.stage_list.setCurrentItem(item)
            return

    def _update_progress(self, value: int) -> None:
        self.progress_bar.setValue(max(0, value))

    def _on_finished(self, outcome: dict) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        status = outcome.get("status")
        headline = T.RUN_STATUS_HEADLINES.get(status, status)
        self.detail_label.setText(T.DETAIL_FINISHED.format(
            status=headline, seconds=outcome.get("elapsed_seconds", 0.0)))
        self.statusBar().showMessage(T.STATUSBAR_JOB.format(status=headline))
        self.refresh_results()
        if status == "failed":
            self.tabs.setCurrentIndex(0)
            # The child process usually exits non-zero without printing the reason,
            # so the recorded job error is what the user actually needs to read.
            recorded = JobArtifacts(Path(self.output_edit.text().strip())).job().get("error")
            QMessageBox.critical(self, T.DIALOG_PIPELINE_FAILED, T.DIALOG_FAILED_BODY.format(
                error=T.job_error_text(recorded) if recorded
                else (outcome.get("error") or T.DIALOG_FAILED_FALLBACK)))
        elif status == "cancelled":
            self.detail_label.setText(T.DETAIL_CANCELLED)
        else:
            self.tabs.setCurrentIndex(1)

    def closeEvent(self, event) -> None:
        if self.runner.running:
            if QMessageBox.question(self, T.DIALOG_JOB_RUNNING, T.DIALOG_JOB_RUNNING_BODY) \
                    != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.cancel()
            self.runner.wait_for_finish(8000)
        self.player.stop()
        self._save_prefs()
        event.accept()

    def refresh_results(self) -> None:
        artifacts = JobArtifacts(Path(self.output_edit.text().strip() or ROOT / "outputs"))
        job = artifacts.job()
        headline, note = artifacts.summary()
        self.summary_label.setText(self._summary_html(artifacts, job, headline, note))
        rows = artifacts.metrics() + artifacts.pitch_rows() + artifacts.timing_rows() + artifacts.diagnostics()
        self._fill_table(self.metric_table, [(key, value) for key, value in rows])
        self._fill_table(self.region_table, [
            (str(row["teacher"]), str(row["student"]), str(row["status"]), str(row["mode"]),
             "—" if row["quality"] is None else f"{float(row['quality']):.2f}",
             self._time(row["teacher_start"]), self._time(row["teacher_end"]), str(row["reason"]))
            for row in artifacts.regions()])
        warnings = artifacts.warnings()
        self.warning_view.setPlainText("\n\n".join(T.BULLET_ITEM.format(item=item) for item in warnings)
                                       if warnings else T.NO_WARNINGS)
        self._reload_audio(artifacts)
        self.open_report_button.setEnabled(artifacts.report_path() is not None)
        self.open_output_button.setEnabled(Path(self.output_edit.text().strip()).is_dir())

    @staticmethod
    def _summary_html(artifacts: JobArtifacts, job: dict, headline: str, note: str) -> str:
        corrected = artifacts.corrected_path()
        line = (f"<br>{T.SUMMARY_CORRECTED_PREFIX}<code>{corrected.name}</code>"
                if corrected is not None else "")
        return (f"<b>{headline}</b><br>{note}{line}<br>"
                f"<span style='color:#666'>{T.SUMMARY_DISCLAIMER}</span>")

    def _reload_audio(self, artifacts: JobArtifacts) -> None:
        previous = self.audio_combo.currentData()
        available = artifacts.available_audio()
        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        for label, path in available:
            self.audio_combo.addItem(label, str(path))
        self.audio_combo.blockSignals(False)
        keep = next((index for index, (_, path) in enumerate(available) if str(path) == previous), None)
        self.audio_combo.setCurrentIndex(keep if keep is not None else min(2, len(available) - 1))
        enabled = bool(available)
        self.audio_combo.setEnabled(enabled)
        self.play_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self._pick_audio(self.audio_combo.currentIndex())

    @staticmethod
    def _time(value) -> str:
        try:
            return T.VALUE_SECONDS.format(value=float(value))
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _fill_table(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    def _pick_audio(self, index: int) -> None:
        path = self.audio_combo.itemData(index) if index >= 0 else None
        if path and path != self._player_target:
            self._player_target = path
            self.player.setSource(QUrl.fromLocalFile(path))

    def _play(self) -> None:
        if self._player_target:
            self.player.play()

    def _on_player_error(self, error, message: str) -> None:
        if error != QMediaPlayer.Error.NoError:
            self.statusBar().showMessage(T.STATUSBAR_PLAYBACK_UNAVAILABLE.format(message=message))

    def _open_report(self) -> None:
        report = JobArtifacts(Path(self.output_edit.text().strip())).report_path()
        if report is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(report)))

    def _open_output(self) -> None:
        path = Path(self.output_edit.text().strip())
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _load_prefs(self) -> None:
        try:
            values = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(values, dict):
            self.request = JobRequest.from_prefs(values, self.request)

    def _save_prefs(self) -> None:
        try:
            PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
            PREFS_PATH.write_text(json.dumps(self._collect().to_prefs(), indent=2) + "\n",
                                  encoding="utf-8")
        except OSError:
            self.statusBar().showMessage(T.STATUSBAR_PREFS_FAILED)
