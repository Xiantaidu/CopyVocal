from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget

from gui.fluent_compat import (BodyLabel, CardWidget, ComboBox, DoubleSpinBox, Expander, SettingField,
                               SettingRow, SwitchButton, cell_grid, grid_holder)
from gui.i18n import tr
from gui.job import JobRequest

ONNX_DEVICES = ("dml", "cpu")
VOCODER_DEVICES = ("auto", "cuda", "cpu")
RUN_MODE_KEYS = ("check_experiments", "check_verify_timing", "check_analyze_only",
                 "check_skip_analysis", "check_fallback")
# Options the pipeline accepts only one of; the rest are disabled instead.
EXCLUSIVE_KEYS = ("check_analyze_only", "check_skip_analysis")


class RunOptionsCard(CardWidget):
    """Run options: shown on the settings page, read by the page that runs a job."""

    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tr_bindings: list = []
        self._switches: dict[str, SwitchButton] = {}
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)
        title = BodyLabel(tr("card_options"), self)
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: title.setText(tr("card_options")))
        layout.addWidget(title)

        self.onnx_combo, self.vocoder_combo = ComboBox(self), ComboBox(self)
        for combo, items in ((self.onnx_combo, ONNX_DEVICES), (self.vocoder_combo, VOCODER_DEVICES)):
            combo.setFixedWidth(115)
            for item in items:
                combo.addItem(item, userData=item)
        layout.addLayout(cell_grid([self._field(self, "label_lyric_backend", self.onnx_combo, "tooltip_onnx"),
                                    self._field(self, "label_vocoder_device", self.vocoder_combo,
                                                "tooltip_vocoder")], columns=2))

        toggles = [self._switch_row(self, key)
                   for key in ("check_game", "check_verify", "check_export_opentune")]
        layout.addLayout(cell_grid(toggles, columns=2))

        self.run_expander = Expander(tr("expander_run_modes"), self)
        self._bind_tr(lambda: self.run_expander.setTitle(tr("expander_run_modes")))
        modes = [self._switch_row(self.run_expander, key) for key in RUN_MODE_KEYS]
        self.run_expander.setContentWidget(grid_holder(modes, 2, self.run_expander))
        layout.addWidget(self.run_expander)

        self._build_advanced(layout)
        for switch in self._switches.values():
            switch.checkedChanged.connect(self._on_option_changed)
        for combo in (self.onnx_combo, self.vocoder_combo):
            combo.currentIndexChanged.connect(self._on_option_changed)
        for spin in (self.chunk_spin, self.context_spin, self.quality_spin,
                     self.stretch_spin, self.floor_spin):
            spin.valueChanged.connect(self._on_option_changed)

    @Slot()
    def _on_option_changed(self) -> None:
        self._sync_modes()
        self.changed.emit()

    def _build_advanced(self, layout) -> None:
        defaults = JobRequest.from_defaults()
        self.advanced_expander = Expander(tr("expander_advanced"), self)
        self._bind_tr(lambda: self.advanced_expander.setTitle(tr("expander_advanced")))
        self.chunk_spin = self._spin(1.0, 600.0, defaults.max_chunk_seconds, 1.0, " s")
        self.context_spin = self._spin(0.01, 10.0, defaults.context_seconds, 0.05, " s")
        self.quality_spin = self._spin(0.01, 1.0, defaults.minimum_alignment_quality, 0.05, "")
        self.stretch_spin = self._spin(1.0, 100.0, defaults.maximum_stretch, 0.5, " x")
        self.floor_spin = self._spin(1e-9, 1.0, defaults.log_floor, 1e-5, "", decimals=9)
        cells = [self._setting_row(self.advanced_expander, label_key, spin, tip_key)
                 for label_key, spin, tip_key in (
                     ("advanced_max_chunk", self.chunk_spin, "tooltip_max_chunk"),
                     ("advanced_context", self.context_spin, "tooltip_context"),
                     ("advanced_quality", self.quality_spin, "tooltip_quality"),
                     ("advanced_stretch", self.stretch_spin, "tooltip_stretch"),
                     ("advanced_floor", self.floor_spin, "tooltip_floor"))]
        self.advanced_expander.setContentWidget(
            grid_holder(cells, 2, self.advanced_expander))
        layout.addWidget(self.advanced_expander)

    @staticmethod
    def _spin(low: float, high: float, value: float, step: float, suffix: str,
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
        return spin

    def _switch(self, parent, key: str) -> SwitchButton:
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

    def _field(self, parent, key: str, control: QWidget, tip_key: str | None = None) -> QWidget:
        field = SettingField(tr(key), control, parent)
        self._bind_tr(lambda field=field, key=key: field.setText(tr(key)))
        if tip_key:
            field.label().setToolTip(tr(tip_key))
            control.setToolTip(tr(tip_key))
            self._bind_tr(lambda: (field.label().setToolTip(tr(tip_key)),
                                   control.setToolTip(tr(tip_key))))
        return field

    def _setting_row(self, parent, key: str, control: QWidget, tip_key: str | None = None) -> QWidget:
        row = SettingRow(tr(key), control, parent)
        self._bind_tr(lambda row=row, key=key: row.setText(tr(key)))
        if tip_key:
            row.label().setToolTip(tr(tip_key))
            control.setToolTip(tr(tip_key))
            self._bind_tr(lambda: (row.label().setToolTip(tr(tip_key)),
                                   control.setToolTip(tr(tip_key))))
        return row

    def _switch_row(self, parent, key: str) -> QWidget:
        """A switch on the right of its sentence, so the state never reads as a prefix."""
        return self._setting_row(parent, key, self._switch(parent, key), f"tooltip_{key}")

    def _sync_modes(self) -> None:
        """Keep the mode combination legal; analysis and reuse cannot both be asked for."""
        analyze = self.analyze_check.isChecked()
        skip = self.skip_check.isChecked()
        self.skip_check.setEnabled(not analyze)
        self.analyze_check.setEnabled(not skip)
        self.verify_timing_check.setEnabled(not analyze)
        self.experiments_check.setEnabled(not analyze)
        self.export_opentune_check.setEnabled(not analyze)
        if analyze:
            self.verify_timing_check.setChecked(False)
            self.export_opentune_check.setChecked(False)

    def collect(self, request: JobRequest) -> JobRequest:
        request.onnx_device = self.onnx_combo.currentData() or self.onnx_combo.currentText()
        request.vocoder_device = self.vocoder_combo.currentData() or self.vocoder_combo.currentText()
        request.include_game = self.game_check.isChecked()
        request.experiments = self.experiments_check.isChecked()
        request.verify = self.verify_check.isChecked()
        request.verify_timing = self.verify_timing_check.isChecked()
        request.analyze_only = self.analyze_check.isChecked()
        request.skip_analysis = self.skip_check.isChecked()
        request.export_opentune = self.export_opentune_check.isChecked()
        request.allow_acoustic_fallback = self.fallback_check.isChecked()
        request.max_chunk_seconds = self.chunk_spin.value()
        request.context_seconds = self.context_spin.value()
        request.minimum_alignment_quality = self.quality_spin.value()
        request.maximum_stretch = self.stretch_spin.value()
        request.log_floor = self.floor_spin.value()
        return request

    def apply(self, request: JobRequest) -> None:
        self.onnx_combo.setCurrentIndex(max(0, self.onnx_combo.findData(request.onnx_device)))
        self.vocoder_combo.setCurrentIndex(max(0, self.vocoder_combo.findData(request.vocoder_device)))
        self.game_check.setChecked(request.include_game)
        self.experiments_check.setChecked(request.experiments)
        self.verify_check.setChecked(request.verify)
        self.verify_timing_check.setChecked(request.verify_timing)
        self.analyze_check.setChecked(request.analyze_only)
        self.skip_check.setChecked(request.skip_analysis)
        self.export_opentune_check.setChecked(request.export_opentune)
        self.fallback_check.setChecked(request.allow_acoustic_fallback)
        self.chunk_spin.setValue(request.max_chunk_seconds)
        self.context_spin.setValue(request.context_seconds)
        self.quality_spin.setValue(request.minimum_alignment_quality)
        self.stretch_spin.setValue(request.maximum_stretch)
        self.floor_spin.setValue(request.log_floor)
        self._sync_modes()

    def retranslate_ui(self) -> None:
        for fn in self._tr_bindings:
            fn()

    def _bind_tr(self, fn) -> None:
        self._tr_bindings.append(fn)
        fn()

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
    def export_opentune_check(self) -> SwitchButton:
        return self._switches["check_export_opentune"]

    @property
    def analyze_check(self) -> SwitchButton:
        return self._switches["check_analyze_only"]

    @property
    def skip_check(self) -> SwitchButton:
        return self._switches["check_skip_analysis"]

    @property
    def fallback_check(self) -> SwitchButton:
        return self._switches["check_fallback"]
