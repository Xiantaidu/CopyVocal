import copy
import json
import unittest

import numpy as np

from inference.alignment import build_alignment, map_times


def analysis(texts, spans=None, duration=None, quality=0.95):
    if spans is None:
        spans = [(i * 0.3, (i + 1) * 0.3) for i in range(len(texts))]
    return {"duration": duration if duration is not None else (spans[-1][1] if spans else 0),
            "words": [{"text": text, "start": span[0], "end": span[1], "quality": quality}
                      for text, span in zip(texts, spans)], "pitch": {}, "notes": []}


class AlignmentTests(unittest.TestCase):
    def test_known_correspondence_and_teacher_clock(self):
        teacher = analysis("我爱你", [(0.12, 0.32), (0.32, 0.48), (0.49, 0.59)], 0.7)
        student = analysis("我爱你", [(0.12, 0.32), (0.31, 0.45), (0.47, 0.67)], 0.8)
        plan = build_alignment(teacher, student)
        self.assertEqual([m["status"] for m in plan["matches"]], ["matched"] * 3)
        valid = [r for r in plan["regions"] if r["mode"] == "synthesize"]
        self.assertEqual([(r["teacher_start"], r["teacher_end"]) for r in valid],
                         [(0.12, 0.32), (0.32, 0.48), (0.49, 0.59)])
        self.assertAlmostEqual(plan["metrics"]["lyric_coverage"], 1.0)
        self.assertAlmostEqual(map_times(plan, [0.54])[0], 0.57)
        self.assertEqual(plan["time_map"]["direction"], "teacher_to_student")
        json.dumps(plan, allow_nan=False)

    def test_insertions_and_missing_words_are_not_fabricated(self):
        teacher = analysis("我爱你啊")
        student = analysis("我爱呀你")
        plan = build_alignment(teacher, student)
        self.assertEqual([(m["teacher_text"], m["student_text"], m["status"])
                          for m in plan["matches"]],
                         [("我", "我", "matched"), ("爱", "爱", "matched"),
                          (None, "呀", "extra"), ("你", "你", "matched"),
                          ("啊", None, "missing")])
        missing = next(r for r in plan["regions"] if r["status"] == "missing")
        self.assertEqual(missing["mode"], "mute")
        self.assertTrue(missing["unsupported"])
        self.assertEqual(plan["metrics"]["lyric_coverage"], 0.75)

    def test_repeated_text_uses_context_instead_of_greedy_first_match(self):
        teacher = analysis("我爱你我爱你")
        student = analysis("我爱你你")
        plan = build_alignment(teacher, student)
        pairs = [(m["teacher_index"], m["student_index"]) for m in plan["matches"]
                 if m.get("relation") == "matched"]
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2), (5, 3)])

    def test_pitch_errors_do_not_change_correspondence(self):
        teacher, student = analysis("我爱你"), analysis("我爱你")
        baseline = build_alignment(teacher, student)
        teacher["pitch"] = {"times": [0.0, 0.2], "f0_hz": [440, 220], "voiced": [True, True]}
        student["pitch"] = {"times": [0.0, 0.2], "f0_hz": [55, 1760], "voiced": [False, True]}
        student["notes"] = [{"start": 0, "end": 0.9, "midi": 100}]
        changed = build_alignment(teacher, student)
        self.assertEqual(baseline, changed)

    def test_vowel_takes_stretch_while_consonant_stays_natural(self):
        teacher = analysis(["我"], [(0, 0.6)])
        student = analysis(["我"], [(0, 0.3)])
        teacher["words"][0]["phonemes"] = [
            {"text": "w", "start": 0, "end": 0.1},
            {"text": "o", "start": 0.1, "end": 0.6}]
        student["words"][0]["phonemes"] = [
            {"text": "w", "start": 0, "end": 0.06},
            {"text": "o", "start": 0.06, "end": 0.3}]
        plan = build_alignment(teacher, student)
        region = plan["regions"][0]
        self.assertEqual(region["anchor_strategy"], "vowel_weighted")
        self.assertAlmostEqual(region["teacher_anchors"][1], 0.06)
        self.assertAlmostEqual(map_times(plan, [0.03])[0], 0.03)
        self.assertAlmostEqual(map_times(plan, [0.33])[0], 0.18)

    def test_homophones_and_uncertain_substitutions(self):
        teacher, student = analysis(["在", "你"]), analysis(["再", "他"])
        teacher["words"][0]["pinyin"] = student["words"][0]["pinyin"] = "zai4"
        plan = build_alignment(teacher, student)
        self.assertEqual([m["status"] for m in plan["matches"]], ["homophone", "substitution"])
        self.assertEqual(plan["matches"][1]["reason"], "possible_lyric_mismatch_or_asr_error")
        student["words"][1]["quality"] = 0.05
        uncertain = build_alignment(teacher, student)
        self.assertEqual(uncertain["matches"][1]["status"], "unresolved")
        self.assertEqual(uncertain["regions"][-1]["mode"], "mute")

    def test_tifa_phonemes_supply_homophones_without_pypinyin(self):
        teacher, student = analysis(["在"]), analysis(["再"])
        teacher["words"][0]["phonemes"] = [
            {"text": "z", "start": 0, "end": 0.05},
            {"text": "ai4", "start": 0.05, "end": 0.3}]
        student["words"][0]["phonemes"] = [
            {"text": "z", "start": 0, "end": 0.06},
            {"text": "ai2", "start": 0.06, "end": 0.3}]
        # Use non-Han text with identical phone signatures so this also passes
        # on installations where an optional Chinese reader is present.
        teacher["words"][0]["text"] = "here"
        student["words"][0]["text"] = "hear"
        plan = build_alignment(teacher, student)
        self.assertEqual(plan["matches"][0]["status"], "homophone")

    def test_untranscribed_voicing_is_never_reported_as_silence(self):
        teacher = analysis(["我"], [(0, 0.3)], duration=1)
        student = analysis(["我"], [(0, 0.3)], duration=1)
        for track in (teacher, student):
            track["pitch"] = {"times": [0.4, 0.5, 0.6], "voiced": [True, True, True]}
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        regions = [r for r in plan["regions"] if r["status"] == "unresolved"]
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["mode"], "acoustic_fallback")
        strict = build_alignment(teacher, student, allow_acoustic_fallback=False)
        muted = next(r for r in strict["regions"] if r["status"] == "unresolved")
        self.assertEqual(muted["mode"], "mute")

    def test_monotonic_map_cuts_extra_source_material(self):
        teacher = analysis("我你", [(0, 0.3), (0.3, 0.6)], 0.7)
        student = analysis("我爱你", [(0, 0.3), (0.3, 0.6), (0.6, 0.9)], 1.0)
        plan = build_alignment(teacher, student)
        targets = np.linspace(-0.1, 0.8, 3001)
        sources = map_times(plan, targets)
        self.assertTrue(np.all(np.diff(sources) >= -1e-12))
        self.assertTrue(np.all(np.diff(plan["time_map"]["teacher_times"]) > 0))
        self.assertTrue(np.all(np.diff(plan["time_map"]["student_times"]) >= 0))
        self.assertTrue(np.all((sources >= 0) & (sources <= 1.0)))
        self.assertAlmostEqual(map_times(plan, [0.299])[0], 0.299)
        self.assertAlmostEqual(map_times(plan, [0.301])[0], 0.601)

    def test_empty_and_silent_data_do_not_claim_success(self):
        empty = build_alignment({}, {})
        self.assertEqual(empty["metrics"]["lyric_coverage"], 0)
        np.testing.assert_array_equal(map_times(empty, [0, 1]), [0, 0])
        silent = build_alignment({"duration": 1}, {"duration": 2})
        self.assertEqual(silent["regions"][0]["mode"], "silence")
        self.assertFalse(silent["metrics"]["acoustic_fallback"])
        missing = build_alignment(analysis("我"), {"duration": 1})
        self.assertEqual(missing["matches"][0]["status"], "missing")
        self.assertEqual(missing["regions"][0]["mode"], "mute")

    def test_activity_fallback_explicitly_has_no_semantic_coverage(self):
        teacher = {"duration": 1.0, "pitch": {"times": [0.1, 0.2, 0.3], "voiced": [True] * 3}}
        student = {"duration": 1.0, "pitch": {"times": [0.2, 0.3, 0.4], "voiced": [True] * 3}}
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        self.assertTrue(plan["metrics"]["acoustic_fallback"])
        self.assertEqual(plan["metrics"]["semantic_match_rate"], 0)
        fallback = next(r for r in plan["regions"] if r["mode"] == "acoustic_fallback")
        self.assertEqual(fallback["quality"], 0.15)
        self.assertIn("no_lyric_correspondence", fallback["reason"])
        disabled = build_alignment(teacher, student, allow_acoustic_fallback=False)
        self.assertFalse(disabled["metrics"]["acoustic_fallback"])

    def test_extreme_stretch_is_reported_and_muted(self):
        plan = build_alignment(analysis(["我"], [(0, 2)]), analysis(["我"], [(0, 0.1)]))
        self.assertEqual(plan["metrics"]["extreme_stretch_count"], 1)
        self.assertEqual(plan["regions"][0]["mode"], "mute")

    def test_inputs_remain_unchanged_and_segmentation_is_audited(self):
        teacher, student = analysis(["我爱你"], [(0, 0.9)]), analysis("我爱你")
        original = copy.deepcopy(teacher)
        plan = build_alignment(teacher, student)
        self.assertEqual(teacher, original)
        self.assertEqual(plan["metrics"]["teacher_unit_count"], 3)
        self.assertTrue(all(m["timing_estimated"] for m in plan["matches"]))
        self.assertTrue(any("estimated internal" in warning for warning in plan["warnings"]))


    def test_maximum_stretch_boundary_is_inclusive(self):
        # 0.2 / 0.8 evaluates to 0.24999999999999997; the old strict comparison
        # muted a word that the user's 4x limit explicitly allows.
        teacher = {"duration": 0.4, "words": [
            {"text": "我", "start": 0.1, "end": 0.3, "quality": 0.95}], "pitch": {}, "notes": []}
        student = {"duration": 0.9, "words": [
            {"text": "我", "start": 0.0, "end": 0.8, "quality": 0.95}], "pitch": {}, "notes": []}
        plan = build_alignment(teacher, student, max_stretch=4.0)
        region = next(r for r in plan["regions"] if r.get("status") == "matched")
        self.assertFalse(region["extreme_stretch"])
        self.assertEqual(region["mode"], "synthesize")
        self.assertEqual(plan["metrics"]["extreme_stretch_count"], 0)

    def test_repeat_snap_preserves_an_extra_student_word(self):
        # The snap chooses a contiguous teacher window without turning an
        # inserted student word into a forced positional match.
        teacher = analysis(["我", "爱", "你", "啊", "我", "爱", "你", "啊", "我", "爱"])
        student = analysis(["我", "爱", "呀", "你"])
        plan = build_alignment(teacher, student)
        extra = next(m for m in plan["matches"] if m["student_text"] == "呀")
        self.assertIsNone(extra["teacher_index"])
        self.assertEqual(extra["status"], "extra")
        matched = [(m["teacher_text"], m["student_text"]) for m in plan["matches"]
                   if m["status"] == "matched"]
        self.assertEqual(matched, [("我", "我"), ("爱", "爱"), ("你", "你")])
        self.assertGreaterEqual(plan["metrics"]["status_counts"].get("extra", 0), 1)

    def test_muted_word_audio_is_not_reused_by_a_later_fallback(self):
        teacher = analysis(["我", "你", "他"], [(0, 0.5), (0.5, 1.0), (1.5, 2.0)], 2.0)
        student = analysis(["我", "你", "他"], [(0, 0.5), (0.5, 1.0), (1.5, 2.0)], 2.0)
        student["words"][1]["quality"] = 0.05
        times = np.arange(0, 2.0, 0.05)
        for track, voiced_from, voiced_to in ((teacher, 1.0, 1.5), (student, 0.5, 1.5)):
            voiced = [bool(voiced_from <= t < voiced_to) for t in times]
            track["pitch"] = {"times": times.tolist(), "voiced": voiced,
                              "f0_hz": [200.0 if value else 0.0 for value in voiced]}
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        muted = [r for r in plan["regions"] if r["mode"] == "mute"
                 and r.get("student_end", 0) > r.get("student_start", 0)]
        fallback = [r for r in plan["regions"] if r["mode"] == "acoustic_fallback"]
        self.assertTrue(muted)
        self.assertTrue(fallback)
        for region in fallback:
            for blocked in muted:
                self.assertFalse(region["student_start"] < blocked["student_end"]
                                 and blocked["student_start"] < region["student_end"])

    def test_one_phone_per_token_uses_the_aligner_boundaries(self):
        from inference.alignment.planner import _units
        payload = {"duration": 1.0, "words": [{
            "text": "君・きみ", "start": 0.0, "end": 0.6, "quality": 1.0,
            "phonemes": [{"text": "ki", "start": 0.0, "end": 0.25},
                         {"text": "mi", "start": 0.3, "end": 0.6}]}],
            "pitch": {}, "notes": []}
        units, _ = _units(payload, 1.0)
        self.assertEqual([unit["text"] for unit in units], ["君", "きみ"])
        self.assertAlmostEqual(units[0]["start"], 0.0)
        self.assertAlmostEqual(units[0]["end"], 0.3)
        self.assertAlmostEqual(units[1]["start"], 0.3)
        self.assertAlmostEqual(units[1]["end"], 0.6)
        self.assertFalse(any(unit["estimated_timing"] for unit in units))
        self.assertEqual([unit["phonemes"][0]["text"] for unit in units], ["ki", "mi"])


if __name__ == "__main__":
    unittest.main()
