import unittest
from unittest.mock import patch

from inference.alignment import build_alignment
from inference.alignment.planner import _japanese_signature, _units
from inference.timing_verification import compare_word_timing


def analysis(texts, *, moras=False, phones=None, language="ja"):
    words = []
    for index, text in enumerate(texts):
        start, end = index * 0.4, (index + 1) * 0.4
        word = {"text": text, "start": start, "end": end, "quality": 0.95}
        if moras:
            word.update(language="ja", unit_kind="mora", identity_basis="phonetic_mora")
        if phones:
            labels = phones[index]
            word["phonemes"] = [
                {"text": label, "start": start + position * 0.4 / len(labels),
                 "end": start + (position + 1) * 0.4 / len(labels)}
                for position, label in enumerate(labels)
            ]
        words.append(word)
    result = {"duration": len(words) * 0.4, "words": words, "pitch": {},
              "completed_stages": ["lyrics"], "notes": []}
    if language:
        result["config"] = {"language": language}
    return result


class JapaneseCorrespondenceTests(unittest.TestCase):
    def test_contracted_kana_retains_tifa_mora_and_phone_boundaries(self):
        track = analysis(["きゃ"], phones=[["k", "y", "a"]])
        units, warnings = _units(track, track["duration"])
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["text"], "きゃ")
        self.assertEqual(len(units[0]["phonemes"]), 3)
        self.assertFalse(units[0]["estimated_timing"])
        self.assertFalse(warnings)
        self.assertEqual(build_alignment(track, track)["regions"][0]["anchor_strategy"], "vowel_weighted")

    def test_japanese_kanji_word_is_not_split_into_chinese_characters(self):
        track = analysis(["世界"], phones=[["s", "e", "k", "a", "i"]])
        with patch("inference.alignment.planner._pinyin", side_effect=AssertionError("Chinese reader used")):
            units, _ = _units(track, track["duration"])
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0]["text"], "世界")
        self.assertEqual(units[0]["pronunciation"], "s e k a i")
        self.assertEqual(len(units[0]["phonemes"]), 5)

    def test_known_kanji_and_kana_reading_are_only_homophones(self):
        teacher = analysis(["世界"], phones=[["s", "e", "k", "a", "i"]])
        student = analysis(["せかい"], phones=[["s", "e", "k", "a", "i"]])
        plan = build_alignment(teacher, student)
        self.assertEqual(plan["matches"][0]["relation"], "homophone")
        self.assertEqual(plan["regions"][0]["mode"], "synthesize")

    def test_katakana_and_hiragana_have_equivalent_readings(self):
        plan = build_alignment(analysis(["キャ", "ッ", "ト"]), analysis(["きゃ", "っ", "と"]))
        self.assertEqual([match["relation"] for match in plan["matches"]], ["homophone"] * 3)
        self.assertEqual(plan["metrics"]["teacher_unit_count"], 3)

    def test_kana_and_romaji_moras_align_as_phonetic_evidence(self):
        teacher = analysis(["キャ", "ッ", "ト"], moras=True)
        student = analysis(["kya", "cl", "to"], moras=True,
                           phones=[["ky", "a"], ["cl"], ["t", "o"]])
        plan = build_alignment(teacher, student)
        self.assertEqual([match["relation"] for match in plan["matches"]], ["homophone"] * 3)
        self.assertEqual(plan["metrics"]["phonetic_only_match_count"], 3)
        self.assertEqual(plan["metrics"]["semantic_match_rate"], 0)
        self.assertTrue(all(match["phonetic_only"] for match in plan["matches"]))
        self.assertTrue(any("phonetic only" in warning for warning in plan["warnings"]))

    def test_long_vowel_preserves_extra_mora(self):
        self.assertEqual(_japanese_signature("コーヒー"), _japanese_signature("ko o hi i"))
        self.assertNotEqual(_japanese_signature("コーヒー"), _japanese_signature("ko hi"))

    def test_moraic_n_is_not_equated_with_n_onset(self):
        self.assertEqual(_japanese_signature("んい"), "N i")
        self.assertEqual(_japanese_signature("に"), "n i")
        self.assertNotEqual(_japanese_signature("N i", True), _japanese_signature("n i", True))

    def test_unknown_kanji_reading_is_not_invented(self):
        teacher, student = analysis(["生"]), analysis(["なま"])
        with patch("inference.alignment.planner._pinyin", side_effect=AssertionError("Chinese reader used")):
            plan = build_alignment(teacher, student)
        self.assertEqual(plan["matches"][0]["relation"], "substitution")
        self.assertEqual(plan["matches"][0]["reason"], "possible_lyric_mismatch_or_asr_error")

    def test_missing_mora_stays_an_explicit_muted_barrier(self):
        plan = build_alignment(analysis(["ki", "mi", "e"], moras=True),
                               analysis(["ki", "e"], moras=True))
        missing = [region for region in plan["regions"] if region["status"] == "missing"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["teacher_text"], "mi")
        self.assertEqual(missing[0]["mode"], "mute")
        self.assertTrue(missing[0]["unsupported"])

    def test_wrong_mora_is_not_a_homophone(self):
        plan = build_alignment(analysis(["ki", "mi", "e"], moras=True),
                               analysis(["ki", "ni", "e"], moras=True))
        middle = plan["matches"][1]
        self.assertEqual(middle["relation"], "substitution")
        self.assertEqual(middle["reason"], "possible_lyric_mismatch_or_asr_error")

    def test_equal_romaji_does_not_certify_lexical_timing(self):
        teacher = analysis(["shi", "a"], moras=True)
        output = analysis(["shi", "a"], moras=True)
        report = compare_word_timing(teacher, output)
        observations = report["word_observations"]
        self.assertTrue(all(item["relation"] == "homophone" for item in observations))
        self.assertTrue(all(not item["reliable_for_timing"] for item in observations))
        self.assertTrue(all("exact_lexical_identity_not_established" in item["exclusion_reasons"]
                            for item in observations))

    def test_kana_evidence_prevents_chinese_reading_for_neighboring_kanji(self):
        track = analysis(["世界", "へ"], language=None)
        with patch("inference.alignment.planner._pinyin", side_effect=AssertionError("Chinese reader used")):
            units, _ = _units(track, track["duration"])
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["language"], "ja")
        self.assertEqual(units[0]["pronunciation"], "")

    def test_chinese_and_english_tokenization_stays_unchanged(self):
        chinese = analysis(["世界"], language="zh")
        english = analysis(["don't stop"], language="en")
        self.assertEqual([unit["text"] for unit in _units(chinese, chinese["duration"])[0]], ["世", "界"])
        self.assertEqual([unit["text"] for unit in _units(english, english["duration"])[0]], ["don't", "stop"])


if __name__ == "__main__":
    unittest.main()
