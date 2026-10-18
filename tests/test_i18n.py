import unittest

from gui import i18n
from gui.i18n import _STRINGS, set_language, tr


class TableTests(unittest.TestCase):
    """Every key must carry both languages, and lookups must never raise."""

    def setUp(self):
        self.addCleanup(set_language, "zh")

    def test_every_key_has_both_languages(self):
        for key, entry in _STRINGS.items():
            self.assertIn("zh", entry, key)
            self.assertIn("en", entry, key)
            self.assertTrue(entry["zh"].strip(), key)
            self.assertTrue(entry["en"].strip(), key)

    def test_unknown_key_returns_the_key(self):
        self.assertEqual(tr("no_such_key"), "no_such_key")

    def test_placeholders_are_consistent_across_languages(self):
        import re

        pattern = re.compile(r"\{(\w+)")
        for key, entry in _STRINGS.items():
            self.assertEqual(set(pattern.findall(entry["zh"])), set(pattern.findall(entry["en"])), key)


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_language, "zh")

    def test_language_switch_round_trips(self):
        set_language("zh")
        self.assertEqual(tr("button_run"), "开始")
        set_language("en")
        self.assertEqual(tr("button_run"), "Run")
        set_language("zh")
        self.assertEqual(tr("button_run"), "开始")

    def test_english_aliases_and_unknown_values(self):
        set_language("English")
        self.assertEqual(i18n.current_language, "en")
        set_language("klingon")
        self.assertEqual(i18n.current_language, "zh")

    def test_format_arguments_are_applied(self):
        set_language("en")
        self.assertEqual(tr("value_seconds", value=1.5), "1.500 s")
        set_language("zh")
        self.assertEqual(tr("value_seconds", value=1.5), "1.500 秒")


class PipelineTextTests(unittest.TestCase):
    """Pipeline sentences translate in Chinese and pass through in English."""

    def setUp(self):
        self.addCleanup(set_language, "zh")

    def test_warnings_translate_only_in_chinese(self):
        from gui import text as T

        sentence = "Output waveform verification was skipped for this job."
        set_language("zh")
        self.assertEqual(T.warning_text(sentence), "本次任务跳过了输出波形校验。")
        set_language("en")
        self.assertEqual(T.warning_text(sentence), sentence)

    def test_unknown_sentence_passes_through_in_both_languages(self):
        from gui import text as T

        sentence = "A sentence the table does not know."
        for language in ("zh", "en"):
            set_language(language)
            self.assertEqual(T.warning_text(sentence), sentence)

    def test_region_labels_follow_the_language(self):
        from gui import text as T

        set_language("zh")
        self.assertEqual(T.status_label("homophone"), "同音字")
        self.assertEqual(T.policy_label("none"), "不改变音高（学生原样）")
        set_language("en")
        self.assertEqual(T.status_label("homophone"), "homophone")
        self.assertEqual(T.policy_label("none"), "Pitch unchanged (student as sung)")
        self.assertEqual(T.status_label("invented_status"), "invented_status")

    def test_stage_labels_follow_the_language(self):
        from gui import text as T

        set_language("zh")
        self.assertEqual(T.stage_labels()[0][1], "分析：歌词、音素、音高与音符")
        set_language("en")
        self.assertEqual(T.stage_labels()[0][1], "Analysis: lyrics, phonemes, pitch and notes")
        self.assertEqual(len(T.stage_labels()), 6)


if __name__ == "__main__":
    unittest.main()
