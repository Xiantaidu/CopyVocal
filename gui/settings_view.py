from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from gui.fluent_compat import (BodyLabel, CardWidget, ComboBox, HintLabel, ScrollArea, SubtitleLabel,
                               Theme, setTheme)
from gui.i18n import tr


class SettingsInterface(ScrollArea):
    """Appearance and about: theme, interface language, version."""

    languageChanged = Signal(str)
    themeChanged = Signal(str)

    def __init__(self, version: str = "", parent=None) -> None:
        super().__init__(parent=parent)
        self.setObjectName("settingsInterface")
        self._tr_bindings: list = []
        self._version = version
        self._build()

    def _build(self) -> None:
        self.view = QWidget(self)
        self.view.setObjectName("view")
        self.vBoxLayout = QVBoxLayout(self.view)
        self.vBoxLayout.setContentsMargins(36, 16, 36, 28)
        self.vBoxLayout.setSpacing(16)
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.enableTransparentBackground()

        self.title_label = SubtitleLabel(tr("nav_settings"))
        self._bind_tr(lambda: self.title_label.setText(tr("nav_settings")))
        self.vBoxLayout.addWidget(self.title_label)

        appearance, layout = self._card("card_appearance")
        self.theme_combo = ComboBox(appearance)
        self.theme_combo.setFixedWidth(220)
        self.language_combo = ComboBox(appearance)
        self.language_combo.setFixedWidth(220)
        self._fill_theme_combo()
        self._fill_language_combo()
        self._bind_tr(self._fill_theme_combo)
        self._bind_tr(self._fill_language_combo)
        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(8)
        self.theme_label = BodyLabel(tr("label_theme"), appearance)
        self.language_label = BodyLabel(tr("label_ui_language"), appearance)
        self._bind_tr(lambda: (self.theme_label.setText(tr("label_theme")),
                               self.language_label.setText(tr("label_ui_language"))))
        grid.addWidget(self.theme_label, 0, 0)
        grid.addWidget(self.theme_combo, 0, 1)
        grid.addWidget(self.language_label, 0, 2)
        grid.addWidget(self.language_combo, 0, 3)
        grid.setColumnStretch(4, 1)
        layout.addLayout(grid)
        self.vBoxLayout.addWidget(appearance)

        about, about_layout = self._card("card_about")
        self.about_label = HintLabel(tr("about_body"), about)
        self.about_label.setWordWrap(True)
        self._bind_tr(lambda: self.about_label.setText(tr("about_body")))
        self.version_label = HintLabel(tr("version_info", version=self._version), about)
        self._bind_tr(lambda: self.version_label.setText(tr("version_info", version=self._version)))
        about_layout.addWidget(self.about_label)
        about_layout.addWidget(self.version_label)
        self.vBoxLayout.addWidget(about)
        self.vBoxLayout.addStretch(1)

        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)

    def _card(self, title_key: str) -> tuple[CardWidget, QVBoxLayout]:
        card = CardWidget(self)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(10)
        label = BodyLabel(tr(title_key), card)
        label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._bind_tr(lambda: label.setText(tr(title_key)))
        layout.addWidget(label)
        return card, layout

    def _fill_theme_combo(self) -> None:
        current = self.theme_combo.currentData() or "light"
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        for value, key in (("light", "theme_light"), ("dark", "theme_dark"), ("auto", "theme_auto")):
            self.theme_combo.addItem(tr(key), userData=value)
        index = self.theme_combo.findData(current)
        self.theme_combo.setCurrentIndex(index if index >= 0 else 0)
        self.theme_combo.blockSignals(False)

    def _fill_language_combo(self) -> None:
        current = self.language_combo.currentData() or "zh"
        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        for value in ("zh", "en"):
            self.language_combo.addItem(tr(f"language_{value}"), userData=value)
        index = self.language_combo.findData(current)
        self.language_combo.setCurrentIndex(index if index >= 0 else 0)
        self.language_combo.blockSignals(False)

    def theme(self) -> str:
        return self.theme_combo.currentData() or "light"

    def language(self) -> str:
        return self.language_combo.currentData() or "zh"

    def apply_theme(self, value: str) -> None:
        index = self.theme_combo.findData(value)
        if index >= 0 and index != self.theme_combo.currentIndex():
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentIndex(index)
            self.theme_combo.blockSignals(False)
        setTheme({"dark": Theme.DARK, "auto": Theme.AUTO}.get(value, Theme.LIGHT))

    def apply_language(self, value: str) -> None:
        index = self.language_combo.findData(value)
        if index >= 0:
            self.language_combo.blockSignals(True)
            self.language_combo.setCurrentIndex(index)
            self.language_combo.blockSignals(False)

    def _on_theme_changed(self) -> None:
        value = self.theme()
        setTheme({"dark": Theme.DARK, "auto": Theme.AUTO}.get(value, Theme.LIGHT))
        self.themeChanged.emit(value)

    def _on_language_changed(self) -> None:
        self.languageChanged.emit(self.language())

    def retranslate_ui(self) -> None:
        for fn in self._tr_bindings:
            fn()

    def _bind_tr(self, fn) -> None:
        self._tr_bindings.append(fn)
        fn()
