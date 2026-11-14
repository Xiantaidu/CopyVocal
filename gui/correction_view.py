from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QListWidgetItem,
                               QMessageBox, QVBoxLayout, QWidget)

from application.artifacts import ROOT
from gui import text as T
from gui.fluent_compat import (BodyLabel, CardWidget, ComboBox, FluentIcon, HintLabel,
                               IndeterminateProgressRing, LineEdit, ListWidget, PrimaryPushButton,
                               PushButton, ScrollArea, SettingField, SubtitleLabel, SwitchButton,
                               cell_grid)
from gui.i18n import tr
from gui.job import AUDIO_SUFFIXES, LANGUAGES, PREFS_PATH, JobRequest, suggested_output
from gui.log_terminal import LogTerminal
from gui.results import JobArtifacts
from gui.run_options import RunOptionsCard
from gui.runner import STAGES, PipelineRunner

STAGE_ICONS = {"pending": "○", "running": "▶", "done": "✔", "skipped": "–"}
PENDING_BRUSH, DONE_BRUSH = QBrush(QColor("#666666")), QBrush(QColor("#1a7f37"))
FILTER_AUDIO = "音频文件 (" + " ".join(f"*{item}" for item in AUDIO_SUFFIXES) + ");;所有文件 (*)"


class CorrectionInterface(ScrollArea):
    """Pick the recordings and the correction settings, then run the pipeline."""

    finished_signal = Signal(str, str)

    def __init__(self, options: RunOptionsCard, parent=None) -> None:
        super().__init__(parent=parent)
        self.setObjectName("correctionInterface")
        self.setAcceptDrops(True)
        self.request = JobRequest.from_defaults()
        self.options = options
        self.runner = PipelineRunner(self)
        self._tr_bindings: list = []
        self._load_prefs()
        self._build()
        self._connect()
        self._apply(self.request)
        self._sync_enabled()

    def _build(self) -> None:
        self.view = QWidget(self)
        self.view.setObjectName("view")
        self.vBoxLayout = QVBoxLayout(self.view)
        self.vBoxLayout.setContentsMargins(36, 16, 36, 24)
        self.vBoxLayout.setSpacing(14)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        self.title_label = SubtitleLabel(tr("nav_correction"))
        self.vBoxLayout.addWidget(self.title_label)
        self._bind_tr(lambda: self.title_label.setText(tr("nav_correction")))

        self._build_recordings()
        self._build_settings()
        self._build_progress()
        self.vBoxLayout.addStretch(1)

    def _card(self, title_key: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)
        label = BodyLabel(tr(title_key), card)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(label)
        self._bind_tr(lambda label=label: label.setText(tr(title_key)))
        return card, layout

    def _build_recordings(self) -> None:
        card, layout = self._card("card_recordings")
        self.teacher_edit, self.student_edit, self.output_edit = LineEdit(card), LineEdit(card), LineEdit(card)
        rows = (("label_teacher", self.teacher_edit, self._browse_teacher),
                ("label_student", self.student_edit, self._browse_student),
                ("label_output", self.output_edit, self._browse_output))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        for row, (label_key, edit, handler) in enumerate(rows):
            label = BodyLabel(tr(label_key), card)
            self._bind_tr(lambda label=label, key=label_key: label.setText(tr(key)))
            button = PushButton(tr("button_browse"), card, FluentIcon.FOLDER)
            self._bind_tr(lambda button=button: button.setText(tr("button_browse")))
            button.clicked.connect(handler)
            grid.addWidget(label, row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(button, row, 2)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.hint_label = HintLabel(tr("hint_inputs"), card)
        self.hint_label.setWordWrap(True)
        self._bind_tr(lambda: self.hint_label.setText(tr("hint_inputs")))
        layout.addWidget(self.hint_label)
        self.vBoxLayout.addWidget(card)

    def _build_settings(self) -> None:
        card, layout = self._card("card_axes")
        self.timing_switch = self._switch(card, "check_timing")
        self.pitch_combo = ComboBox(card)
        self.pitch_combo.setFixedWidth(185)
        self.pitch_map_switch = self._switch(card, "check_pitch_map")
        self.language_combo = ComboBox(card)
        self.language_combo.setFixedWidth(115)
        for item in LANGUAGES:
            self.language_combo.addItem(item, userData=item)
        cells = [self._field(card, "check_timing", self.timing_switch, "tooltip_check_timing"),
                 self._field(card, "label_pitch_mode", self.pitch_combo, "tooltip_pitch_mode"),
                 self._field(card, "check_pitch_map", self.pitch_map_switch, "tooltip_check_pitch_map"),
                 self._field(card, "label_language", self.language_combo, "tooltip_language")]
        layout.addLayout(cell_grid(cells, columns=4))
        self._fill_pitch_modes()
        self._bind_tr(self._fill_pitch_modes)
        self.axis_hint = HintLabel(tr("axis_hint"), card)
        self.axis_hint.setWordWrap(True)
        self._bind_tr(lambda: self.axis_hint.setText(tr("axis_hint")))
        layout.addWidget(self.axis_hint)
        self.vBoxLayout.addWidget(card)

    def _switch(self, parent, key: str) -> SwitchButton:
        switch = SwitchButton(tr("switch_on"), parent)
        switch.setOffText(tr("switch_off"))
        switch.setChecked(False)
        self._bind_tr(lambda switch=switch: self._retranslate_switch(switch))
        return switch

    @staticmethod
    def _retranslate_switch(switch: SwitchButton) -> None:
        switch.setOnText(tr("switch_on"))
        switch.setOffText(tr("switch_off"))

    def _field(self, parent, key: str, control: QWidget, tip_key: str | None = None) -> QWidget:
        field = SettingField(tr(key), control, parent)
        self._bind_tr(lambda field=field, key=key: field.setText(tr(key)))
        if tip_key:
            field.label().setToolTip(tr(tip_key))
            control.setToolTip(tr(tip_key))
            self._bind_tr(lambda: (field.label().setToolTip(tr(tip_key)),
                                   control.setToolTip(tr(tip_key))))
        return field

    def _build_progress(self) -> None:
        card, layout = self._card("card_progress")
        row = QHBoxLayout()
        row.setSpacing(12)
        self.run_button = PrimaryPushButton(tr("button_run"), card, FluentIcon.PLAY)
        self._bind_tr(lambda: self.run_button.setText(tr("button_run")))
        self.run_button.setMinimumWidth(120)
        self.cancel_button = PushButton(tr("button_cancel"), card, FluentIcon.PAUSE)
        self.cancel_button.setEnabled(False)
        self._bind_tr(lambda: self.cancel_button.setText(tr("button_cancel")))
        self.progress_ring = IndeterminateProgressRing(card)
        self.progress_ring.setFixedSize(28, 28)
        self.progress_ring.hide()
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        row.addWidget(self.progress_ring)
        row.addStretch(1)
        layout.addLayout(row)

        self.progress_bar = _ProgressBar(card)
        # An idle bar reads as an empty divider; it belongs to a running job.
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.detail_label = BodyLabel(tr("detail_no_job"), card)
        self.detail_label.setWordWrap(True)
        self._bind_tr(lambda: self.detail_label.setText(tr("detail_no_job")))
        layout.addWidget(self.detail_label)

        self.stage_list = ListWidget(card)
        self.stage_list.setMaximumHeight(150)
        for key, _label in STAGES:
            item = QListWidgetItem("")
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.stage_list.addItem(item)
        self._refresh_stage_labels()
        self._bind_tr(self._refresh_stage_labels)
        # Nothing has run yet: the stage list appears with the first job, so an
        # idle window shows the decisions rather than six rows of pending text.
        self.stage_list.setVisible(False)
        layout.addWidget(self.stage_list)

        # The log stays open: it is the only place a running job explains itself.
        self.log_label = BodyLabel(tr("expander_log"), card)
        self._bind_tr(lambda: self.log_label.setText(tr("expander_log")))
        layout.addWidget(self.log_label)
        self.log_view = LogTerminal(card)
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_view)
        layout.addLayout(log_layout)
        self.vBoxLayout.addWidget(card)

    def _refresh_stage_labels(self) -> None:
        labels = dict(T.stage_labels())
        for row in range(self.stage_list.count()):
            item = self.stage_list.item(row)
            key = item.data(Qt.ItemDataRole.UserRole)
            state = self.runner.stage_states().get(key, "pending") if hasattr(self, "runner") else "pending"
            item.setText(f"{STAGE_ICONS.get(state, '○')}  {labels.get(key, key)}")

    def _fill_pitch_modes(self) -> None:
        current = self.pitch_combo.currentData() or "off"
        self.pitch_combo.blockSignals(True)
        self.pitch_combo.clear()
        for value, label in zip(T.PITCH_MODE_VALUES, T.pitch_mode_labels()):
            self.pitch_combo.addItem(label, userData=value)
        index = self.pitch_combo.findData(current)
        self.pitch_combo.setCurrentIndex(index if index >= 0 else 0)
        self.pitch_combo.blockSignals(False)

    def _connect(self) -> None:
        self.run_button.clicked.connect(self.start_job)
        self.cancel_button.clicked.connect(self.runner.cancel)
        self.runner.line.connect(self._append_log)
        self.runner.stage_changed.connect(self._update_stage)
        self.runner.detail.connect(self.detail_label.setText)
        self.runner.progress.connect(self._update_progress)
        self.runner.finished.connect(self._on_finished)
        for edit in (self.teacher_edit, self.student_edit, self.output_edit):
            edit.textChanged.connect(self._sync_enabled)
        for switch in (self.timing_switch, self.pitch_map_switch):
            switch.checkedChanged.connect(self._sync_enabled)
        for combo in (self.pitch_combo, self.language_combo):
            combo.currentIndexChanged.connect(self._sync_enabled)
        self.options.changed.connect(self._sync_enabled)

    @Slot()
    def _sync_enabled(self) -> None:
        """Keep the correction axes legal; the run options keep themselves legal."""
        # 'off' has no contour to choose, so the checkbox is disabled rather than
        # silently ignored.
        self.pitch_map_switch.setEnabled(self._pitch_mode() != "off")

    def _pitch_mode(self) -> str:
        value = self.pitch_combo.currentData()
        return value if value in T.PITCH_MODE_VALUES else "off"

    @property
    def timing_check(self) -> SwitchButton:
        return self.timing_switch

    @property
    def pitch_map_check(self) -> SwitchButton:
        return self.pitch_map_switch

    def _collect(self) -> JobRequest:
        request = self.request
        request.teacher, request.student = self.teacher_edit.text().strip(), self.student_edit.text().strip()
        request.output = self.output_edit.text().strip()
        request.language = self.language_combo.currentData() or self.language_combo.currentText()
        request.change_timing = self.timing_check.isChecked()
        request.pitch_mode = self._pitch_mode()
        request.map_teacher_pitch = self.pitch_map_check.isChecked()
        return self.options.collect(request)

    def _apply(self, request: JobRequest) -> None:
        self.teacher_edit.setText(request.teacher)
        self.student_edit.setText(request.student)
        self.output_edit.setText(request.output)
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(request.language)))
        self.timing_check.setChecked(request.change_timing)
        index = self.pitch_combo.findData(request.pitch_mode)
        self.pitch_combo.setCurrentIndex(index if index >= 0 else 0)
        self.pitch_map_check.setChecked(request.map_teacher_pitch)
        self.options.apply(request)

    def _browse_teacher(self) -> None:
        self._choose_file(self.teacher_edit, tr("label_teacher"))

    def _browse_student(self) -> None:
        self._choose_file(self.student_edit, tr("label_student"))

    def _choose_file(self, target: LineEdit, title: str) -> None:
        start = target.text().strip() or str(ROOT)
        chosen, _ = QFileDialog.getOpenFileName(self, title, start, FILTER_AUDIO)
        if chosen:
            self._set_audio_path(target, chosen)

    def _set_audio_path(self, target: LineEdit, path: str) -> None:
        target.setText(path)
        if target is self.student_edit:
            current = self.output_edit.text().strip()
            if not current or Path(current).name in {"gui", "outputs"}:
                self.output_edit.setText(str(suggested_output(path)))

    def _browse_output(self) -> None:
        start = self.output_edit.text().strip() or str(ROOT / "outputs")
        chosen = QFileDialog.getExistingDirectory(self, tr("label_output"), start)
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
            QMessageBox.warning(self, tr("dialog_cannot_start"),
                                "\n".join(tr("bullet_item", item=item) for item in problems))
            return
        if request.skip_analysis and QMessageBox.question(
                self, tr("dialog_reuse_title"), tr("dialog_reuse_body")) \
                != QMessageBox.StandardButton.Yes:
            return
        self.log_view.clear()
        self.stage_list.setVisible(True)
        self.progress_bar.setVisible(True)
        self._save_prefs()
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_ring.show()
        self.detail_label.setText(tr("detail_running"))
        try:
            self.runner.start(request)
        except Exception as error:  # a start failure must not close the window
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.progress_ring.hide()
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, tr("dialog_cannot_start"), str(error))

    def _append_log(self, text: str) -> None:
        self.log_view.log(text)

    def _update_stage(self, stage: str, state: str) -> None:
        labels = dict(T.stage_labels())
        for row in range(self.stage_list.count()):
            item = self.stage_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) != stage:
                continue
            item.setText(f"{STAGE_ICONS.get(state, '○')}  {labels.get(stage, stage)}")
            item.setForeground(DONE_BRUSH if state == "done" else PENDING_BRUSH)
            if state == "running":
                self.stage_list.setCurrentItem(item)
            return

    def _update_progress(self, value: int) -> None:
        self.progress_bar.setValue(max(0, value))

    def _on_finished(self, outcome: dict) -> None:
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress_ring.hide()
        self.progress_bar.setVisible(False)
        status = outcome.get("status")
        headline = T.run_status_headline(status)
        self.detail_label.setText(tr("detail_finished", status=headline,
                                     seconds=outcome.get("elapsed_seconds", 0.0)))
        self.finished_signal.emit(status, headline)
        if status == "failed":
            # The child process usually exits non-zero without printing the reason,
            # so the recorded job error is what the user actually needs to read.
            recorded = JobArtifacts(Path(self.output_edit.text().strip())).job().get("error")
            QMessageBox.critical(self, tr("dialog_pipeline_failed"), tr("dialog_failed_body").format(
                error=T.job_error_text(recorded) if recorded
                else (outcome.get("error") or tr("dialog_failed_fallback"))))
        elif status == "cancelled":
            self.detail_label.setText(tr("detail_cancelled"))

    def _load_prefs(self) -> None:
        import json

        try:
            values = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(values, dict):
            self.request = JobRequest.from_prefs(values, self.request)

    def _save_prefs(self) -> None:
        import json

        try:
            PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Merge instead of replacing: this also runs when a job starts, and
            # the window owns theme/ui_language, which must survive that write.
            try:
                values = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                values = {}
            if not isinstance(values, dict):
                values = {}
            values.update(self._collect().to_prefs())
            PREFS_PATH.write_text(json.dumps(values, indent=2) + "\n",
                                  encoding="utf-8")
        except OSError:
            pass

    def retranslate_ui(self) -> None:
        for fn in self._tr_bindings:
            fn()

    def _bind_tr(self, fn) -> None:
        self._tr_bindings.append(fn)
        fn()

    def is_running(self) -> bool:
        return self.runner.running

    def request_stop(self) -> None:
        self.runner.cancel()

    def wait_for_worker(self, timeout_ms: int) -> bool:
        return self.runner.wait_for_finish(timeout_ms)

    def on_worker_settled(self, callback) -> None:
        self.runner.finished.connect(lambda _outcome: callback())


class _ProgressBar(QWidget):
    """Thin progress bar drawn in the theme accent colour."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QProgressBar

        self._bar = QProgressBar(self)
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._bar)

    def setValue(self, value: int) -> None:
        self._bar.setValue(value)

    def value(self) -> int:
        return self._bar.value()
