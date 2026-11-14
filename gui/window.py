from __future__ import annotations

import json

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QMessageBox

from gui.correction_view import CorrectionInterface
from gui.fluent_compat import FluentIcon, FluentWindow, NavigationItemPosition
from gui.i18n import set_language, tr
from gui.job import PREFS_PATH, JobRequest
from gui.results_view import ResultsInterface
from gui.run_options import RunOptionsCard
from gui.settings_view import SettingsInterface


class MainWindow(FluentWindow):
    """Icon sidebar over three pages: correction, results, settings."""

    def __init__(self, version: str = "") -> None:
        super().__init__()
        self.request = JobRequest.from_defaults()
        self._load_prefs()
        set_language(self._prefs_language())
        # One options card, mounted on the settings page and read by the page that
        # builds the job, so the two can never disagree about what will run.
        self.options = RunOptionsCard(self)
        self.correction = CorrectionInterface(self.options, self)
        self.results = ResultsInterface(self)
        self.settings = SettingsInterface(version, self.options, self)
        self.runner = self.correction.runner
        self.setWindowTitle(tr("window_title"))
        self._build_navigation()
        self._connect()
        self.setAcceptDrops(True)
        self._apply_fixed_physical_size(1260, 1120)
        self.settings.apply_theme(self._prefs_theme())
        self.settings.apply_language(self._prefs_language())
        self.correction._apply(self.request)
        self.correction._sync_enabled()
        self.refresh_results()

    def _apply_fixed_physical_size(self, width: int, height: int) -> None:
        """Lock the window size in physical pixels: logical = physical / scale."""
        screen = self.screen() or QApplication.primaryScreen()
        ratio = screen.devicePixelRatio() if screen is not None else 1.0
        size = QSize(max(1, round(width / ratio)), max(1, round(height / ratio)))
        self.resize(size)
        self.setFixedSize(size)

    def _build_navigation(self) -> None:
        self.addSubInterface(self.correction, FluentIcon.MUSIC, tr("nav_correction"))
        self.addSubInterface(self.results, FluentIcon.FOLDER, tr("nav_results"))
        self.addSubInterface(self.settings, FluentIcon.SETTING, tr("nav_settings"),
                             position=NavigationItemPosition.BOTTOM)
        self.navigationInterface.setMenuButtonVisible(False)
        self.navigationInterface.setExpandWidth(50)
        self.navigationInterface.setMinimumExpandWidth(50)
        self.navigationInterface.setMaximumWidth(50)

    def _connect(self) -> None:
        self.settings.languageChanged.connect(self._on_language_changed)
        self.settings.themeChanged.connect(self._on_theme_changed)
        self.correction.runner.finished.connect(self._on_job_finished)
        self.correction.finished_signal.connect(self._on_job_finished_signal)
        self.results.playback_failed.connect(self._on_playback_failed)
        self._status_key, self._status_args = "statusbar_ready", {}
        self._render_status()

    def _render_status(self) -> None:
        self.statusBar().showMessage(tr(self._status_key, **self._status_args))

    def _show_status(self, key: str, **kwargs) -> None:
        """Remember the key, not the sentence, so a language switch can re-render it."""
        self._status_key, self._status_args = key, kwargs
        self._render_status()

    def _on_playback_failed(self, message: str) -> None:
        self._show_status("statusbar_playback_unavailable", message=message)

    def _on_theme_changed(self, value: str) -> None:
        self._save_prefs()

    def _on_language_changed(self, value: str) -> None:
        set_language(value)
        self.setWindowTitle(tr("window_title"))
        self._retranslate_navigation()
        self.correction.retranslate_ui()
        self.options.retranslate_ui()
        self.results.retranslate_ui()
        self.settings.retranslate_ui()
        self._render_status()
        self.refresh_results()
        self._save_prefs()

    def _retranslate_navigation(self) -> None:
        navigation = self.navigationInterface
        for page, key in ((self.correction, "nav_correction"),
                          (self.results, "nav_results"),
                          (self.settings, "nav_settings")):
            item = navigation.widget(page.objectName())
            if item is not None:
                item.setText(tr(key))

    def _on_job_finished(self, outcome: dict) -> None:
        self.refresh_results()
        if outcome.get("status") in {"succeeded", "failed", "cancelled"}:
            self.navigationInterface.setCurrentItem(self.results.objectName())
            self.stackedWidget.setCurrentWidget(self.results)

    def _on_job_finished_signal(self, status: str, headline: str) -> None:
        self._show_status("statusbar_job", status=headline)

    def refresh_results(self) -> None:
        self.results.set_output(self.correction.output_edit.text().strip())
        self.results.refresh(self.correction.output_edit.text().strip())

    def closeEvent(self, event) -> None:
        if self.correction.is_running():
            if QMessageBox.question(self, tr("dialog_job_running"), tr("dialog_job_running_body")) \
                    != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.correction.request_stop()
            self.correction.wait_for_worker(8000)
        self.results.stop_playback()
        self._save_prefs()
        event.accept()

    def _prefs_theme(self) -> str:
        return self._prefs.get("theme", "light")

    def _prefs_language(self) -> str:
        # Old preference files wrote the interface language under "language";
        # keep reading it as the UI choice while "language" now means the
        # recording language (the pipeline's own key).
        return self._prefs.get("ui_language", self._prefs.get("language", "zh"))

    def _load_prefs(self) -> None:
        self._prefs: dict = {}
        try:
            values = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(values, dict):
            self._prefs = values
            self.request = JobRequest.from_prefs(values, self.request)

    def _save_prefs(self) -> None:
        try:
            PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
            values = self.correction._collect().to_prefs()
            values["theme"] = self.settings.theme()
            # "language" is the recording language and comes from to_prefs();
            # the interface language is a separate preference.
            values["ui_language"] = self.settings.language()
            PREFS_PATH.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
        except OSError:
            self._show_status("statusbar_prefs_failed")

    # The probes and tests drive the window through these names.
    @property
    def teacher_edit(self):
        return self.correction.teacher_edit

    @property
    def student_edit(self):
        return self.correction.student_edit

    @property
    def output_edit(self):
        return self.correction.output_edit

    @property
    def onnx_combo(self):
        return self.options.onnx_combo

    @property
    def vocoder_combo(self):
        return self.options.vocoder_combo

    @property
    def language_combo(self):
        return self.correction.language_combo

    @property
    def game_check(self):
        return self.options.game_check

    @property
    def experiments_check(self):
        return self.options.experiments_check

    @property
    def verify_check(self):
        return self.options.verify_check

    @property
    def verify_timing_check(self):
        return self.options.verify_timing_check

    @property
    def export_copyvocal_file_check(self):
        return self.options.export_copyvocal_file_check

    @property
    def analyze_check(self):
        return self.options.analyze_check

    @property
    def skip_check(self):
        return self.options.skip_check

    @property
    def fallback_check(self):
        return self.options.fallback_check

    @property
    def timing_check(self):
        return self.correction.timing_check

    @property
    def pitch_combo(self):
        return self.correction.pitch_combo

    @property
    def pitch_map_check(self):
        return self.correction.pitch_map_check

    @property
    def chunk_spin(self):
        return self.options.chunk_spin

    @property
    def context_spin(self):
        return self.options.context_spin

    @property
    def quality_spin(self):
        return self.options.quality_spin

    @property
    def stretch_spin(self):
        return self.options.stretch_spin

    @property
    def floor_spin(self):
        return self.options.floor_spin

    @property
    def run_button(self):
        return self.correction.run_button

    @property
    def cancel_button(self):
        return self.correction.cancel_button

    @property
    def progress_bar(self):
        return self.correction.progress_bar

    @property
    def stage_list(self):
        return self.correction.stage_list

    @property
    def detail_label(self):
        return self.correction.detail_label

    @property
    def log_view(self):
        return self.correction.log_view

    @property
    def summary_label(self):
        return self.results.summary_label

    @property
    def tables(self):
        return self.results.tables

    @property
    def metric_table(self):
        return self.results.metric_table

    @property
    def region_table(self):
        return self.results.region_table

    @property
    def warning_view(self):
        return self.results.warning_view

    @property
    def audio_combo(self):
        return self.results.audio_combo

    @property
    def play_button(self):
        return self.results.play_button

    @property
    def stop_button(self):
        return self.results.stop_button

    @property
    def open_report_button(self):
        return self.results.open_report_button

    @property
    def open_output_button(self):
        return self.results.open_output_button

    @property
    def export_copyvocal_file_button(self):
        return self.results.export_copyvocal_file_button

    @property
    def tabs(self):
        return self.results.tables

    def start_job(self) -> None:
        self.correction.start_job()

    def _collect(self) -> JobRequest:
        return self.correction._collect()

    def _pitch_mode(self) -> str:
        return self.correction._pitch_mode()

    def _append_log(self, text: str) -> None:
        self.correction._append_log(text)

    def _update_stage(self, stage: str, state: str) -> None:
        self.correction._update_stage(stage, state)

    def _update_progress(self, value: int) -> None:
        self.correction._update_progress(value)
