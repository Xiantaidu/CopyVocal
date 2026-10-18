"""Adversarial lyric editing and CPU-only analysis/renderer contract checks."""

from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from inference.alignment import build_alignment, map_times
from inference.correction.renderer import render_student


def track(tokens, *, spans=None, quality=1.0, sample_rate=1000, tail=0.1):
    if spans is None:
        spans = [(i * 0.4, i * 0.4 + 0.3) for i in range(len(tokens))]
    duration = spans[-1][1] + tail if spans else tail
    count = int(round(duration * sample_rate))
    duration = count / sample_rate
    times = np.arange(math.ceil(duration / 0.01)) * 0.01
    voiced = np.zeros(times.shape, dtype=bool)
    words = []
    for text, (start, end) in zip(tokens, spans):
        voiced |= (times >= start) & (times < end)
        words.append({"text": text, "start": start, "end": end, "quality": quality,
                      "phonemes": []})
    return {"sample_rate": sample_rate, "sample_count": count, "duration": duration,
            "words": words, "pitch": {"times": times.tolist(), "voiced": voiced.tolist(),
                                       "f0_hz": np.where(voiced, 220.0, 0.0).tolist()}, "notes": []}


class CapturingVocoder:
    """Make waveform masking observable without a neural model or a GPU."""

    sample_rate = 1000
    hop_size = 10
    log_floor = 1e-5

    def mel(self, audio):
        return np.ones((2, math.ceil(len(audio) / self.hop_size)), dtype=np.float32)

    def synthesize(self, mel, f0, *, sample_count, chunk_frames):
        self.last_mel = mel.copy()
        self.last_f0 = f0.copy()
        return np.ones(sample_count, dtype=np.float32)


class AlignmentStressTests(unittest.TestCase):
    def assert_safe_plan(self, plan):
        regions = plan["regions"]
        previous_teacher_end = 0.0
        previous_student_end = 0.0
        for region in regions:
            self.assertGreaterEqual(region["teacher_start"], previous_teacher_end - 1e-12)
            self.assertGreater(region["teacher_end"], region["teacher_start"])
            previous_teacher_end = region["teacher_end"]
            if region["mode"] not in {"synthesize", "acoustic_fallback"}:
                continue
            self.assertGreaterEqual(region["student_start"], previous_student_end - 1e-12)
            previous_student_end = region["student_end"]
            self.assertTrue(np.all(np.diff(region["teacher_anchors"]) > 0))
            self.assertTrue(np.all(np.diff(region["student_anchors"]) > 0))
        times = np.linspace(0.0, plan["teacher_duration"], 10001)
        source = map_times(plan, times)
        self.assertTrue(np.isfinite(source).all())
        self.assertTrue(np.all(np.diff(source) >= -1e-10))
        self.assertTrue(np.all((source >= 0) & (source <= plan["student_duration"])))

    def test_repeated_chorus_with_omitted_words_and_an_extra_word(self):
        teacher_text = list("春风吹过我爱你春风吹过我爱你")
        # Distinct positions inside each chorus retain sufficient context.
        removed = {2, 11}
        student_text = [text for i, text in enumerate(teacher_text) if i not in removed]
        student_text.insert(7, "啊")
        teacher, student = track(teacher_text), track(student_text)
        plan = build_alignment(teacher, student)
        actual = {m["teacher_index"] for m in plan["matches"] if m["status"] == "missing"}
        self.assertEqual(actual, removed)
        self.assertEqual(plan["metrics"]["status_counts"]["extra"], 1)
        self.assertAlmostEqual(plan["metrics"]["lyric_coverage"], 12 / 14)
        self.assert_safe_plan(plan)

    def test_same_character_repetition_and_one_syllable_many_notes(self):
        teacher = track(list("啦啦啦啦"), spans=[(0, 0.2), (0.25, 0.65), (0.7, 1.6), (1.7, 2.0)])
        student = track(list("啦啦啦啦"), spans=[(0, 0.3), (0.4, 0.7), (0.8, 1.1), (1.2, 1.5)])
        teacher["notes"] = [{"start": 0.7, "end": 1.0, "midi": 60},
                            {"start": 1.0, "end": 1.3, "midi": 64},
                            {"start": 1.3, "end": 1.6, "midi": 67}]
        plan = build_alignment(teacher, student)
        self.assertEqual([(m["teacher_index"], m["student_index"]) for m in plan["matches"]],
                         [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(plan["metrics"]["lyric_coverage"], 1.0)
        self.assertEqual(plan["metrics"]["unsupported_region_count"], 0)
        self.assert_safe_plan(plan)
        changed = copy.deepcopy(student)
        changed["notes"] = [{"start": 0, "end": 1.5, "midi": 100}]
        self.assertEqual(build_alignment(teacher, changed), plan)

    def test_uncertain_asr_mutes_only_the_uncertain_word(self):
        teacher, student = track(list("我爱你的声音")), track(list("我爱他的声音"))
        # Remove the artificial shared vowel signature for this mismatch.
        teacher["words"][2]["phonemes"] = []
        student["words"][2]["phonemes"] = []
        student["words"][2]["quality"] = 0.1
        plan = build_alignment(teacher, student, allow_acoustic_fallback=False)
        muted = [r for r in plan["regions"] if r["mode"] == "mute"]
        self.assertEqual(len(muted), 1)
        self.assertEqual(muted[0]["teacher_index"], 2)
        self.assertEqual(muted[0]["status"], "unresolved")
        self.assertAlmostEqual(plan["metrics"]["lyric_coverage"], 5 / 6)
        self.assert_safe_plan(plan)

    def test_shallow_overlap_keeps_high_quality_words_editable(self):
        teacher = track(list("你你的的声音"), spans=[(i * 0.3, i * 0.3 + 0.31) for i in range(6)])
        student = track(list("你你的的声音"), spans=[(i * 0.29, i * 0.29 + 0.30) for i in range(6)])
        plan = build_alignment(teacher, student)
        self.assertEqual(plan["metrics"]["unsupported_region_count"], 0)
        self.assertEqual(plan["metrics"]["lyric_coverage"], 1.0)
        self.assertTrue(any("Overlapping" in warning for warning in plan["warnings"]))
        self.assert_safe_plan(plan)

    def test_random_insert_delete_sequences_keep_local_maps_ordered(self):
        rng = np.random.default_rng(20261015)
        for _ in range(20):
            tokens = [f"word{i}" for i in range(24)]
            removed = set(rng.choice(len(tokens), size=4, replace=False).tolist())
            kept = [token for i, token in enumerate(tokens) if i not in removed]
            for insertion in sorted(rng.choice(len(kept), size=3, replace=False).tolist(), reverse=True):
                kept.insert(insertion, f"extra{insertion}")
            teacher, student = track(tokens), track(kept)
            plan = build_alignment(teacher, student)
            # Adjacent insertion/deletion can equally represent a substitution.
            # What must remain exact is the correspondence of surviving words.
            pairs = {m["teacher_index"]: m["student_index"] for m in plan["matches"]
                     if m.get("relation") == "matched"}
            self.assertEqual(pairs, {i: kept.index(token) for i, token in enumerate(tokens) if i not in removed})
            self.assert_safe_plan(plan)

    def test_all_empty_transcripts_never_claim_semantic_success(self):
        teacher, student = track(list("声音")), track(list("声音"))
        teacher["words"], student["words"] = [], []
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        self.assertEqual(plan["metrics"]["semantic_match_rate"], 0.0)
        self.assertEqual(plan["metrics"]["lyric_coverage"], 0.0)
        self.assertTrue(plan["metrics"]["acoustic_fallback"])
        self.assert_safe_plan(plan)

    def test_missing_student_transcript_has_symmetric_opt_in_fallback(self):
        teacher, student = track(list("我爱你")), track(list("我爱你"))
        student["words"] = []
        strict = build_alignment(teacher, student)
        self.assertFalse(strict["metrics"]["acoustic_fallback"])
        self.assertTrue(strict["metrics"]["all_lyric_material_muted"])
        self.assertTrue(all(m["status"] == "unresolved" for m in strict["matches"]))
        self.assertTrue(any("All timed teacher" in warning for warning in strict["warnings"]))
        fallback = build_alignment(teacher, student, allow_acoustic_fallback=True)
        self.assertTrue(fallback["metrics"]["acoustic_fallback"])
        self.assertEqual(fallback["metrics"]["semantic_match_rate"], 0.0)
        self.assertEqual(fallback["metrics"]["lyric_coverage"], 0.0)
        self.assertFalse(fallback["metrics"]["all_lyric_material_muted"])
        self.assertTrue(any("student lyrics" in warning for warning in fallback["warnings"]))
        self.assert_safe_plan(fallback)

    def test_whole_repeated_chorus_omission_is_explicitly_ambiguous(self):
        teacher, student = track(list("春风吹过春风吹过")), track(list("春风吹过"))
        strict = build_alignment(teacher, student)
        self.assertEqual(strict["metrics"]["ambiguous_match_count"], 4)
        self.assertTrue(strict["metrics"]["all_lyric_material_muted"])
        self.assertTrue(any("Ambiguous repeated" in warning for warning in strict["warnings"]))
        selected = [m for m in strict["matches"] if m.get("ambiguous")]
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(m["status"] == "unresolved" and m["quality"] <= 0.25 for m in selected))
        self.assertTrue(all(len(m["ambiguity"]["candidate_teacher_unit_indices"]) > 1 for m in selected))
        fallback = build_alignment(teacher, student, allow_acoustic_fallback=True)
        self.assertTrue(fallback["metrics"]["acoustic_fallback"])
        self.assert_safe_plan(strict)
        self.assert_safe_plan(fallback)

    def test_cpu_renderer_respects_missing_word_and_output_clock(self):
        teacher = track(list("我爱你"), spans=[(0, 0.3), (0.4, 0.7), (0.8, 1.1)])
        student = track(list("我你"), spans=[(0, 0.3), (0.4, 0.7)])
        plan = build_alignment(teacher, student)
        vocoder = CapturingVocoder()
        waveform, diagnostics = render_student(np.ones(student["sample_count"], dtype=np.float32),
                                               teacher, student, plan, vocoder, mode="full")
        self.assertEqual(len(waveform), teacher["sample_count"])
        self.assertTrue(np.all(waveform[400:700] == 0))
        self.assertGreater(np.mean(waveform[810:1090]), 0.9)
        self.assertEqual(diagnostics["muted_region_count"], 1)
        self.assert_safe_plan(plan)


if __name__ == "__main__":
    unittest.main()
