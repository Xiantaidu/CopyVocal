from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QMessageBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from application.artifacts import ROOT
from gui import text as T
from gui.fluent_compat import (BodyLabel, CardWidget, ComboBox, PushButton, ScrollArea, SubtitleLabel,
                               TextEdit, qconfig, theme_color)
from gui.i18n import tr
from gui.results import JobArtifacts


def _tabs_qss() -> str:
    """Flat tab strip: no pane border, an accent underline on the active tab."""
    hint = theme_color("text_secondary", "rgba(0, 0, 0, 0.6)")
    text = theme_color("text", "#000000")
    accent = theme_color("accent", "#009faa")
    hover = theme_color("hover_overlay", "rgba(0, 0, 0, 0.035)")
    return (
        "QTabWidget::pane { border: none; background: transparent; }"
        "QTabBar { background: transparent; }"
        "QTabBar::tab {"
        " background: transparent;"
        " border: none;"
        " border-bottom: 2px solid transparent;"
        f" color: {hint};"
        " padding: 6px 4px;"
        " margin-right: 22px;"
        "}"
        f"QTabBar::tab:hover {{ color: {text}; background-color: {hover}; }}"
        "QTabBar::tab:selected {"
        f" color: {text};"
        f" border-bottom: 2px solid {accent};"
        " font-weight: 600;"
        "}"
    )


def _table_qss() -> str:
    """Quiet table chrome: no grid, a flat header, rows that read as lines."""
    hint = theme_color("text_secondary", "rgba(0, 0, 0, 0.6)")
    border = theme_color("card_border", "rgba(0, 0, 0, 0.06)")
    text = theme_color("text", "#000000")
    return (
        "QTableWidget {"
        f" color: {text};"
        " background: transparent;"
        " border: none;"
        " gridline-color: transparent;"
        " outline: none;"
        "}"
        "QTableWidget::item { padding: 5px 8px; border: none; }"
        "QTableWidget::item:selected {"
        f" background-color: {theme_color('control_hover', 'rgba(0, 0, 0, 0.04)')};"
        f" color: {text};"
        "}"
        "QHeaderView { background: transparent; }"
        "QHeaderView::section {"
        " background: transparent;"
        " border: none;"
        f" border-bottom: 1px solid {border};"
        f" color: {hint};"
        " padding: 6px 8px;"
        " font-weight: 600;"
        "}"
    )


class ResultsInterface(ScrollArea):
    """Present one output directory's artifacts: metrics, correspondence, warnings."""

    playback_failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setObjectName("resultsInterface")
        self._tr_bindings: list = []
        self._output = str(ROOT / "outputs")
        self.player, self.audio_output = QMediaPlayer(self), QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        self._player_target = ""
        self._build()
        self._connect()
        self._restyle_tables()
        qconfig.themeChangedFinished.connect(self._restyle_tables)

    def _build(self) -> None:
        self.view = QWidget(self)
        self.view.setObjectName("view")
        self.vBoxLayout = QVBoxLayout(self.view)
        self.vBoxLayout.setContentsMargins(36, 16, 36, 28)
        self.vBoxLayout.setSpacing(16)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        self.title_label = SubtitleLabel(tr("nav_results"))
        self._bind_tr(lambda: self.title_label.setText(tr("nav_results")))
        self.vBoxLayout.addWidget(self.title_label)

        summary_card = CardWidget(self)
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(20, 14, 20, 14)
        summary_layout.setSpacing(6)
        self.summary_label = BodyLabel("", summary_card)
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.RichText)
        summary_layout.addWidget(self.summary_label)
        self.vBoxLayout.addWidget(summary_card)

        tables_card = CardWidget(self)
        tables_layout = QVBoxLayout(tables_card)
        tables_layout.setContentsMargins(20, 14, 20, 14)
        self.tables = QTabWidget(tables_card)
        self.metric_table = self._table((tr("header_measurement"), tr("header_value")))
        self.region_table = self._table(tuple(tr("headers_regions").split(",")))
        self.warning_view = TextEdit(tables_card)
        self.warning_view.setReadOnly(True)
        self.warning_view.setPlaceholderText(tr("warning_placeholder"))
        self._bind_tr(lambda: self.warning_view.setPlaceholderText(tr("warning_placeholder")))
        self.tables.addTab(self.metric_table, tr("table_metrics"))
        self.tables.addTab(self.region_table, tr("table_regions"))
        self.tables.addTab(self.warning_view, tr("table_warnings"))
        self._bind_tr(self._refresh_table_titles)
        tables_layout.addWidget(self.tables)
        self.vBoxLayout.addWidget(tables_card, 1)

        audio_card = CardWidget(self)
        audio_layout = QHBoxLayout(audio_card)
        audio_layout.setContentsMargins(20, 12, 20, 12)
        audio_layout.setSpacing(12)
        self.listening_label = BodyLabel(tr("label_listening"), audio_card)
        self._bind_tr(lambda: self.listening_label.setText(tr("label_listening")))
        self.audio_combo = ComboBox(audio_card)
        self.audio_combo.setMinimumWidth(225)
        self.play_button = PushButton(tr("button_play"), audio_card)
        self.stop_button = PushButton(tr("button_stop"), audio_card)
        self._bind_tr(lambda: (self.play_button.setText(tr("button_play")),
                               self.stop_button.setText(tr("button_stop"))))
        self.open_report_button = PushButton(tr("button_open_report"), audio_card)
        self.open_output_button = PushButton(tr("button_open_output"), audio_card)
        self.export_opentune_button = PushButton(tr("button_export_opentune"), audio_card)
        self._bind_tr(lambda: (self.open_report_button.setText(tr("button_open_report")),
                               self.open_output_button.setText(tr("button_open_output")),
                               self.export_opentune_button.setText(tr("button_export_opentune"))))
        for widget in (self.listening_label, self.audio_combo, self.play_button, self.stop_button):
            audio_layout.addWidget(widget)
        audio_layout.addStretch(1)
        audio_layout.addWidget(self.open_report_button)
        audio_layout.addWidget(self.open_output_button)
        audio_layout.addWidget(self.export_opentune_button)
        self.vBoxLayout.addWidget(audio_card)

    def _refresh_table_titles(self) -> None:
        self.tables.setTabText(0, tr("table_metrics"))
        self.tables.setTabText(1, tr("table_regions"))
        self.tables.setTabText(2, tr("table_warnings"))
        self.metric_table.setHorizontalHeaderLabels([tr("header_measurement"), tr("header_value")])
        self.region_table.setHorizontalHeaderLabels(tuple(tr("headers_regions").split(",")))

    def _restyle_tables(self) -> None:
        self.tables.setStyleSheet(_tabs_qss())
        for table in (self.metric_table, self.region_table):
            table.setStyleSheet(_table_qss())

    def _table(self, headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(list(headers))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setFixedHeight(32)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.setStyleSheet(_table_qss())
        return table

    def _connect(self) -> None:
        self.play_button.clicked.connect(self._play)
        self.stop_button.clicked.connect(self.player.stop)
        self.audio_combo.currentIndexChanged.connect(self._pick_audio)
        self.open_report_button.clicked.connect(self._open_report)
        self.open_output_button.clicked.connect(self._open_output)
        self.export_opentune_button.clicked.connect(self._export_opentune)
        self.player.errorOccurred.connect(self._on_player_error)

    def refresh(self, output: str) -> None:
        path = Path(output.strip() or ROOT / "outputs")
        artifacts = JobArtifacts(path)
        headline, note = artifacts.summary()
        self.summary_label.setText(self._summary_html(artifacts, headline, note))
        rows = artifacts.metrics() + artifacts.pitch_rows() + artifacts.timing_rows() + artifacts.diagnostics()
        self._fill_table(self.metric_table, [(key, value) for key, value in rows])
        self._fill_table(self.region_table, [
            (str(row["teacher"]), str(row["student"]), str(row["status"]), str(row["mode"]),
             "—" if row["quality"] is None else f"{float(row['quality']):.2f}",
             self._time(row["teacher_start"]), self._time(row["teacher_end"]), str(row["reason"]))
            for row in artifacts.regions()])
        warnings = artifacts.warnings()
        self.warning_view.setPlainText("\n\n".join(tr("bullet_item", item=item) for item in warnings)
                                       if warnings else tr("no_warnings"))
        self._reload_audio(artifacts)
        self.open_report_button.setEnabled(artifacts.report_path() is not None)
        self.open_output_button.setEnabled(path.is_dir())
        required = ("job.json", "teacher.analysis.json", "student.analysis.json",
                    "alignment.json", "target_pitch.npz")
        self.export_opentune_button.setEnabled(path.is_dir()
                                               and all((path / name).is_file() for name in required))

    @staticmethod
    def _summary_html(artifacts: JobArtifacts, headline: str, note: str) -> str:
        corrected = artifacts.corrected_path()
        line = (f"<br>{tr('summary_corrected_prefix')}<code>{corrected.name}</code>"
                if corrected is not None else "")
        return (f"<b>{headline}</b><br>{note}{line}<br>"
                f"<span style='color:#666'>{tr('summary_disclaimer')}</span>")

    def _reload_audio(self, artifacts: JobArtifacts) -> None:
        previous = self.audio_combo.currentData()
        available = artifacts.available_audio()
        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        for label, path in available:
            self.audio_combo.addItem(label, userData=str(path))
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
            return tr("value_seconds", value=float(value))
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
            self.playback_failed.emit(message)

    def stop_playback(self) -> None:
        self.player.stop()

    def _open_report(self) -> None:
        report = JobArtifacts(Path(self._output_path())).report_path()
        if report is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(report)))

    def _open_output(self) -> None:
        path = Path(self._output_path())
        if path.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _export_opentune(self) -> None:
        """Write <output>/opentune.cvf from the finished job's artifacts."""
        from application.opentune_bundle import export_bundle

        try:
            summary = export_bundle(Path(self._output_path()))
        except Exception as error:  # noqa: BLE001 - surfaced to the user verbatim
            QMessageBox.warning(self, tr("dialog_export_opentune_failed"), str(error))
            return
        QMessageBox.information(self, tr("dialog_export_opentune_done"),
                                tr("dialog_export_opentune_body", path=summary["bundle"]))

    def _output_path(self) -> str:
        return self._output.strip() or str(ROOT / "outputs")

    def set_output(self, output: str) -> None:
        self._output = output

    def retranslate_ui(self) -> None:
        for fn in self._tr_bindings:
            fn()

    def _bind_tr(self, fn) -> None:
        self._tr_bindings.append(fn)
        fn()
