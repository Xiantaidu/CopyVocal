from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QListWidgetItem,
                               QMessageBox, QVBoxLayout, QWidget)

from application.artifacts import ROOT
from gui import text as T
from gui.fluent_compat import (BodyLabel, CardWidget, ComboBox, DoubleSpinBox, Expander, FluentIcon,
                               HintLabel, IndeterminateProgressRing, LineEdit, ListWidget,
                               PrimaryPushButton, PushButton, ScrollArea, SettingField, SettingRow,
                               SubtitleLabel, SwitchButton)
from gui.i18n import tr
from gui.job import AUDIO_SUFFIXES, LANGUAGES, PREFS_PATH, JobRequest, suggested_output
from gui.log_terminal import LogTerminal
from gui.results import JobArtifacts
from gui.runner import STAGES, PipelineRunner

STAGE_ICONS = {"pending": "○", "running": "▶", "done": "✔", "skipped": "–"}
PENDING_BRUSH, DONE_BRUSH = QBrush(QColor("#666666")), QBrush(QColor("#1a7f37"))
FILTER_AUDIO = "音频文件 (" + " ".join(f"*{item}" for item in AUDIO_SUFFIXES) + ");;所有文件 (*)"


def _monospace() -> QFont:
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


class CorrectionInterface(ScrollArea):
    """Collect options, run the pipeline, and show its progress and log."""

    finished_signal = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setObjectName("correctionInterface")
        self.setAcceptDrops(True)
        self.request = JobRequest.from_defaults()
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
        self._build_correction()
        self._build_options()
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

    def _build_correction(self) -> None:
        """The three correction axes side by side, so the whole choice costs one row."""
        card, layout = self._card("card_axes")
        self._switches: dict[str, SwitchButton] = {}
        self.timing_switch = self._switch(card, "check_timing")
        self.pitch_combo = ComboBox(card)
        self.pitch_combo.setFixedWidth(180)
        self.pitch_map_switch = self._switch(card, "check_pitch_map")
        cells = [self._field(card, "check_timing", self.timing_switch, "tooltip_check_timing"),
                 self._field(card, "label_pitch_mode", self.pitch_combo, "tooltip_pitch_mode"),
                 self._field(card, "check_pitch_map", self.pitch_map_switch, "tooltip_check_pitch_map")]
        layout.addLayout(self._cell_grid(cells, columns=3))
        self._fill_pitch_modes()
        self._bind_tr(self._fill_pitch_modes)
        self.axis_hint = HintLabel(tr("axis_hint"), card)
        self.axis_hint.setWordWrap(True)
        self._bind_tr(lambda: self.axis_hint.setText(tr("axis_hint")))
        layout.addWidget(self.axis_hint)
        self.vBoxLayout.addWidget(card)

    def _build_options(self) -> None:
        card, layout = self._card("card_options")
        self.onnx_combo, self.vocoder_combo, self.language_combo = ComboBox(card), ComboBox(card), ComboBox(card)
        for combo in (self.onnx_combo, self.vocoder_combo, self.language_combo):
            combo.setFixedWidth(110)
        for combo, items in ((self.onnx_combo, ("dml", "cpu")),
                             (self.vocoder_combo, ("auto", "cuda", "cpu")),
                             (self.language_combo, LANGUAGES)):
            for item in items:
                combo.addItem(item, userData=item)
        fields = [self._field(card, label_key, widget, tip_key)
                  for label_key, widget, tip_key in (
                      ("label_lyric_backend", self.onnx_combo, "tooltip_onnx"),
                      ("label_vocoder_device", self.vocoder_combo, "tooltip_vocoder"),
                      ("label_language", self.language_combo, None))]
        layout.addLayout(self._cell_grid(fields, columns=3))

        toggles = [self._switch_row(card, key) for key in ("check_game", "check_verify")]
        layout.addLayout(self._cell_grid(toggles, columns=2))

        self.run_expander = Expander(tr("expander_run_modes"), card)
        self._bind_tr(lambda: self.run_expander.setTitle(tr("expander_run_modes")))
        more = [self._switch_row(self.run_expander, key)
                for key in ("check_experiments", "check_verify_timing", "check_analyze_only",
                            "check_skip_analysis", "check_fallback")]
        self.run_expander.setContentWidget(self._grid_holder(more, 2, self.run_expander))
        layout.addWidget(self.run_expander)

        self._build_advanced(layout)
        self.vBoxLayout.addWidget(card)

    def _switch(self, parent, key: str) -> SwitchButton:
        """One switch showing only its on/off state, registered for retranslation."""
        switch = SwitchButton(tr("switch_on"), parent)
        switch.setOffText(tr("switch_off"))
        switch.setChecked(False)
        self._bind_tr(lambda switch=switch: self._retranslate_switch(switch))
        self._switches[key] = switch
        return switch

    @staticmethod
    def _retranslate_switch(switch: SwitchButton) -> None:
        switch.setOnText(tr("switch_on"))
        switch.setOffText(tr("switch_off"))

    def _setting_row(self, parent, key: str, control: QWidget, tip_key: str | None = None) -> QWidget:
        """Caption on the left, its control on the right, as in system settings."""
        row = SettingRow(tr(key), control, parent)
        self._bind_tr(lambda row=row, key=key: row.setText(tr(key)))
        if tip_key:
            label = row.label()
            label.setToolTip(tr(tip_key))
            control.setToolTip(tr(tip_key))
            self._bind_tr(lambda: (label.setToolTip(tr(tip_key)), control.setToolTip(tr(tip_key))))
        return row

    def _field(self, parent, key: str, control: QWidget, tip_key: str | None = None) -> QWidget:
        """Caption above its control, so several fit side by side on one row."""
        field = SettingField(tr(key), control, parent)
        self._bind_tr(lambda field=field, key=key: field.setText(tr(key)))
        if tip_key:
            field.label().setToolTip(tr(tip_key))
            control.setToolTip(tr(tip_key))
            self._bind_tr(lambda: (field.label().setToolTip(tr(tip_key)),
                                   control.setToolTip(tr(tip_key))))
        return field

    def _switch_row(self, parent, key: str, switch: SwitchButton | None = None) -> QWidget:
        """A switch on the right of its sentence, so the state never reads as a prefix."""
        switch = switch if switch is not None else self._switch(parent, key)
        return self._setting_row(parent, key, switch, f"tooltip_{key}")

    def _cell_grid(self, cells, columns: int) -> QGridLayout:
        """Pack cells sideways into columns, filling left to right row by row."""
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)
        for index, cell in enumerate(cells):
            row, column = divmod(index, columns)
            grid.addWidget(cell, row, column)
        # Columns take their natural width and the slack collects at the end:
        # squeezing a caption below its width is what forces it onto two lines.
        grid.setColumnStretch(columns, 1)
        return grid

    def _grid_holder(self, cells, columns: int, parent) -> QWidget:
        """A grid inside a widget, for containers that adopt widgets, not layouts."""
        holder = QWidget(parent)
        holder.setLayout(self._cell_grid(cells, columns))
        return holder

    def _build_advanced(self, layout) -> None:
        """The numeric knobs, folded into the options card behind an expander."""
        self.advanced_expander = Expander(tr("expander_advanced"))
        self._bind_tr(lambda: self.advanced_expander.setTitle(tr("expander_advanced")))
        self.chunk_spin = self._spin(1.0, 600.0, self.request.max_chunk_seconds, 1.0, " s")
        self.context_spin = self._spin(0.01, 10.0, self.request.context_seconds, 0.05, " s")
        self.quality_spin = self._spin(0.01, 1.0, self.request.minimum_alignment_quality, 0.05, "")
        self.stretch_spin = self._spin(1.0, 100.0, self.request.maximum_stretch, 0.5, " x")
        self.floor_spin = self._spin(1e-9, 1.0, self.request.log_floor, 1e-5, "", decimals=9)
        cells = [self._spin_row(self.advanced_expander, label_key, spin, tip_key)
                 for label_key, spin, tip_key in (
                     ("advanced_max_chunk", self.chunk_spin, "tooltip_max_chunk"),
                     ("advanced_context", self.context_spin, "tooltip_context"),
                     ("advanced_quality", self.quality_spin, "tooltip_quality"),
                     ("advanced_stretch", self.stretch_spin, "tooltip_stretch"),
                     ("advanced_floor", self.floor_spin, "tooltip_floor"))]
        self.advanced_expander.setContentWidget(
            self._grid_holder(cells, 2, self.advanced_expander))
        layout.addWidget(self.advanced_expander)

    def _spin_row(self, parent, label_key: str, spin: DoubleSpinBox, tip_key: str) -> QWidget:
        spin.setToolTip(tr(tip_key))
        self._bind_tr(lambda: spin.setToolTip(tr(tip_key)))
        return self._setting_row(parent, label_key, spin, tip_key)

    def _spin(self, low: float, high: float, value: float, step: float, suffix: str,
              decimals: int = 3) -> DoubleSpinBox:
        spin = DoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        # A spin box left to stretch absorbs the whole column, so the value ends
        # up marooned at the left of a very wide field.
        spin.setFixedWidth(160)
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(self._sync_enabled)
        return spin

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

        self.command_label = BodyLabel("", card)
        self.command_label.setFont(_monospace())
        self.command_label.setWordWrap(True)
        self.command_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.command_expander = Expander(tr("expander_command"), card, expanded=False)
        self._bind_tr(lambda: self.command_expander.setTitle(tr("expander_command")))
        command_layout = QVBoxLayout()
        command_layout.addWidget(self.command_label)
        self.command_expander.setContentLayout(command_layout)
        layout.addWidget(self.command_expander)
        self._build_log(layout)
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

    def _build_log(self, layout) -> None:
        """The log, folded into the run card so it costs nothing until a job runs."""
        self.log_expander = Expander(tr("expander_log"))
        self._bind_tr(lambda: self.log_expander.setTitle(tr("expander_log")))
        self.log_view = LogTerminal(self.log_expander)
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_view)
        self.log_expander.setContentLayout(log_layout)
        layout.addWidget(self.log_expander)

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
        for combo in (self.onnx_combo, self.vocoder_combo, self.language_combo):
            combo.currentIndexChanged.connect(self._sync_enabled)
        for switch in (*self._switches.values(), self.timing_switch, self.pitch_map_switch):
            switch.checkedChanged.connect(self._sync_enabled)
        self.pitch_combo.currentIndexChanged.connect(self._sync_enabled)

    @Slot()
    def _sync_enabled(self) -> None:
        """Keep the option combination legal and the command preview current."""
        analyze = self.analyze_check.isChecked()
        skip = self.skip_check.isChecked()
        self.skip_check.setEnabled(not analyze)
        self.analyze_check.setEnabled(not skip)
        self.verify_timing_check.setEnabled(not analyze)
        self.experiments_check.setEnabled(not analyze)
        # 'off' has no contour to choose, so the checkbox is disabled rather than
        # silently ignored.
        self.pitch_map_switch.setEnabled(self._pitch_mode() != "off")
        if analyze:
            self.verify_timing_check.setChecked(False)
        self.command_label.setText(self._collect().command_line())

    @property
    def game_check(self) -> SwitchButton:
        return self._switches["check_game"]

    @property
    def experiments_check(self) -> SwitchButton:
        return self._switches["check_experiments"]

    @property
    def verify_check(self) -> SwitchButton:
        return self._switches["check_verify"]

    @property
    def verify_timing_check(self) -> SwitchButton:
        return self._switches["check_verify_timing"]

    @property
    def analyze_check(self) -> SwitchButton:
        return self._switches["check_analyze_only"]

    @property
    def skip_check(self) -> SwitchButton:
        return self._switches["check_skip_analysis"]

    @property
    def fallback_check(self) -> SwitchButton:
        return self._switches["check_fallback"]

    @property
    def timing_check(self) -> SwitchButton:
        return self.timing_switch

    @property
    def pitch_map_check(self) -> SwitchButton:
        return self.pitch_map_switch

    def _pitch_mode(self) -> str:
        value = self.pitch_combo.currentData()
        return value if value in T.PITCH_MODE_VALUES else "off"

    def _collect(self) -> JobRequest:
        request = self.request
        request.teacher, request.student = self.teacher_edit.text().strip(), self.student_edit.text().strip()
        request.output = self.output_edit.text().strip()
        request.onnx_device = self.onnx_combo.currentData() or self.onnx_combo.currentText()
        request.vocoder_device = self.vocoder_combo.currentData() or self.vocoder_combo.currentText()
        request.language = self.language_combo.currentData() or self.language_combo.currentText()
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
        self.onnx_combo.setCurrentIndex(max(0, self.onnx_combo.findData(request.onnx_device)))
        self.vocoder_combo.setCurrentIndex(max(0, self.vocoder_combo.findData(request.vocoder_device)))
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(request.language)))
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
        index = self.pitch_combo.findData(request.pitch_mode)
        self.pitch_combo.setCurrentIndex(index if index >= 0 else 0)
        self.pitch_map_check.setChecked(request.map_teacher_pitch)

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
        self.log_expander.expand()
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
            PREFS_PATH.write_text(json.dumps(self._collect().to_prefs(), indent=2) + "\n",
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
