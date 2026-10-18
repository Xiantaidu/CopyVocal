"""Regression tests for bounded acoustic context around lyric islands."""

from copy import deepcopy
import math
import unittest

from inference.correction.boundaries import build_boundary_context


def supported(start=1.0, end=2.0, source_start=1.0, source_end=2.0, **extra):
    return {"teacher_start": start, "teacher_end": end,
            "student_start": source_start, "student_end": source_end,
            "mode": "synthesize", "quality": 1.0, **extra}


def gap(start, end, **extra):
    return {"teacher_start": start, "teacher_end": end,
            "student_start": 0.0, "student_end": 0.0,
            "mode": "silence", **extra}


class BoundaryContextTests(unittest.TestCase):
    def test_preserves_final_release_opportunity_without_changing_support(self):
        regions = [gap(0, 1), supported(), gap(2, 3)]
        snapshot = deepcopy(regions)
        contexts = build_boundary_context(regions, [{"start": 1, "end": 2}], 3, 3)
        attack, release = contexts
        self.assertAlmostEqual(attack["teacher_start"], 0.98)
        self.assertAlmostEqual(release["teacher_end"], 2.06)
        self.assertEqual(release["source_strategy"], "continuation")
        self.assertAlmostEqual(release["student_end"], 2.06)
        self.assertTrue(all(context["semantic_support"] is False for context in contexts))
        self.assertEqual(regions, snapshot)

    def test_source_extra_word_bounds_continuation(self):
        regions = [supported(), gap(2, 3)]
        words = [{"start": 1, "end": 2}, {"start": 2.03, "end": 2.8}]
        release, = build_boundary_context(regions, words, 3, 3)
        self.assertAlmostEqual(release["student_end"], 2.03)
        self.assertAlmostEqual(release["student_limit_end"], 2.03)

    def test_no_source_room_uses_short_internal_held_release_and_no_attack(self):
        regions = [supported(), gap(2, 2.3), supported(2.3, 3, 2, 2.7)]
        words = [{"start": 1, "end": 2}, {"start": 2, "end": 2.7}]
        release, = build_boundary_context(regions, words, 3, 3)
        self.assertEqual(release["kind"], "release")
        self.assertEqual(release["source_strategy"], "hold")
        self.assertEqual(release["student_start"], release["student_end"])
        self.assertLess(release["student_end"], 2)
        self.assertGreater(release["student_end"], 1.99)
        self.assertAlmostEqual(release["teacher_end"], 2.04)
        self.assertEqual(release["envelope_strategy"], "full_cos_squared_release")

    def test_missing_region_is_a_barrier_even_without_word_index(self):
        missing = gap(2.04, 2.08, mode="mute", status="missing",
                      reason="no_corresponding_student_lyric_unit")
        del missing["student_start"], missing["student_end"]
        regions = [supported(), gap(2, 2.1), missing, supported(2.1, 3, 2.4, 3.3)]
        contexts = build_boundary_context(regions, [], 3, 4)
        release = next(context for context in contexts if context["kind"] == "release")
        attack = next(context for context in contexts if context["kind"] == "attack")
        self.assertAlmostEqual(release["teacher_end"], 2.04)
        self.assertGreaterEqual(attack["teacher_start"], 2.08)
        self.assertTrue(all(c["teacher_end"] <= 2.04 or c["teacher_start"] >= 2.08 for c in contexts))

    def test_gap_run_mute_allows_context_but_lexical_rejection_does_not(self):
        acoustic = gap(2, 2.2, mode="mute", unsupported=True,
                       reason="voiced_teacher_region_without_lyric_correspondence")
        release, = build_boundary_context([supported(), acoustic], [], 3, 3)
        self.assertAlmostEqual(release["teacher_end"], 2.06)
        for reason in ("possible_lyric_mismatch_or_asr_error", "ambiguous_repeated_lyrics", "matched", None):
            rejected = {**acoustic, "reason": reason}
            self.assertEqual(build_boundary_context([supported(), rejected], [], 3, 3), [])

    def test_low_quality_and_disabled_fallback_are_hard_barriers(self):
        for rejected in (supported(2, 2.1, 2, 2.1, quality=0.1),
                         supported(2, 2.1, 2, 2.1, mode="acoustic_fallback", quality=0.9)):
            contexts = build_boundary_context([supported(), rejected, gap(2.1, 3)], [],
                                              3, 3, minimum_quality=0.5)
            self.assertEqual(contexts, [])

    def test_enabled_fallback_obeys_quality_and_merges_with_supported_island(self):
        fallback = supported(2, 2.1, 2, 2.1, mode="acoustic_fallback", quality=0.9)
        release, = build_boundary_context([supported(), fallback, gap(2.1, 3)], [], 3, 3,
                                          allow_fallback=True, minimum_quality=0.5)
        self.assertAlmostEqual(release["boundary"], 2.1)

    def test_short_gap_splits_contexts_without_overlap_and_prefers_release(self):
        regions = [supported(), gap(2, 2.03), supported(2.03, 3, 2.3, 3.27)]
        contexts = build_boundary_context(regions, [], 3, 4)
        release, attack = contexts
        self.assertEqual([c["kind"] for c in contexts], ["release", "attack"])
        self.assertAlmostEqual(release["teacher_end"], attack["teacher_start"])
        self.assertGreater(release["teacher_end"] - release["teacher_start"],
                           attack["teacher_end"] - attack["teacher_start"])
        self.assertAlmostEqual(release["student_end"] - release["student_start"],
                               release["teacher_end"] - release["teacher_start"])
        self.assertAlmostEqual(attack["student_end"] - attack["student_start"],
                               attack["teacher_end"] - attack["teacher_start"])

    def test_touching_words_form_one_island_without_internal_context(self):
        regions = [gap(0, 1), supported(), supported(2, 2.5, 2.1, 2.6), gap(2.5, 3)]
        contexts = build_boundary_context(regions, [], 3, 3)
        self.assertEqual([(c["kind"], c["boundary"]) for c in contexts], [("attack", 1), ("release", 2.5)])

    def test_source_lexical_region_limits_context_without_word_metadata(self):
        regions = [supported(), gap(2, 2.4), supported(2.4, 3, 2.01, 2.61)]
        contexts = build_boundary_context(regions, [], 3, 3)
        release = next(c for c in contexts if c["kind"] == "release")
        self.assertLessEqual(release["student_end"], 2.01)
        attack = next(c for c in contexts if c["kind"] == "attack")
        self.assertGreaterEqual(attack["student_start"], 2)

    def test_track_bounds_clip_context_and_unknown_gaps_are_not_assumed_safe(self):
        regions = [gap(-1, 0.01), supported(0.01, 0.99, 0.01, 0.99), gap(0.99, 2)]
        for context in build_boundary_context(regions, [], 1, 1):
            self.assertTrue(0 <= context["teacher_start"] < context["teacher_end"] <= 1)
            self.assertTrue(0 <= context["student_start"] <= context["student_end"] <= 1)
        self.assertEqual(build_boundary_context([supported()], [], 3, 3), [])
        with self.assertRaises(ValueError):
            build_boundary_context(regions, [], math.inf, 1)


if __name__ == "__main__":
    unittest.main()
