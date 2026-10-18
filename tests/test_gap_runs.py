import unittest

import numpy as np

from inference.alignment import build_alignment
from inference.alignment.planner import _gap_regions


def activity_track(runs, *, duration=16.0, with_word=True):
    times = np.arange(int(round(duration / 0.01))) * 0.01
    voiced = np.zeros(times.size, dtype=bool)
    for start, end in runs:
        voiced |= (times >= start - 1e-8) & (times < end - 1e-8)
    return {"duration": duration,
            "words": [{"text": "a", "start": 0.0, "end": 0.3, "quality": 1.0}]
            if with_word else [],
            "pitch": {"times": times.tolist(), "voiced": voiced.tolist()}}


class GapRunTests(unittest.TestCase):
    def test_pitch_unvoiced_gap_does_not_confirm_acoustic_silence(self):
        teacher = activity_track([], duration=2.0)
        student = activity_track([], duration=2.0)
        regions = _gap_regions(teacher, student, 0.5, 1.5, 0.25, 1.25, False, 4.0)
        self.assertEqual(len(regions), 1)
        gap = regions[0]
        self.assertEqual(gap["status"], "unvoiced_gap")
        self.assertEqual(gap["mode"], "silence")
        self.assertEqual(gap["reason"], "pitch_unvoiced_not_acoustic_silence")
        self.assertEqual(gap["acoustic_classification"], "unknown")
        self.assertFalse(gap["silence_confirmed"])
        self.assertEqual((gap["gap_source_start"], gap["gap_source_end"]), (0.25, 1.25))
        # Annotation changes must not silently enable new synthesis or alter maps.
        self.assertEqual(gap["teacher_anchors"], [0.5, 1.5])
        self.assertEqual(gap["student_anchors"], [0.25, 0.25])
        self.assertEqual(gap["quality"], 1.0)
        self.assertIn("structural", gap["quality_definition"])
        self.assertIn("not acoustic silence", gap["quality_definition"])

    def test_mixed_activity_marks_all_unvoiced_subintervals_unknown(self):
        teacher = activity_track([(0.7, 0.8), (1.0, 1.1)], duration=2.0)
        student = activity_track([(0.7, 0.8), (1.0, 1.1)], duration=2.0)
        for allow_fallback in (False, True):
            with self.subTest(allow_fallback=allow_fallback):
                regions = _gap_regions(teacher, student, 0.5, 1.5, 0.5, 1.5,
                                       allow_fallback, 4.0)
                gaps = [region for region in regions if region["mode"] == "silence"]
                self.assertEqual(len(gaps), 3)
                np.testing.assert_allclose(
                    [(gap["teacher_start"], gap["teacher_end"]) for gap in gaps],
                    [(0.5, 0.7), (0.8, 1.0), (1.1, 1.5)], atol=1e-8, rtol=0)
                for gap in gaps:
                    self.assertEqual(gap["status"], "unvoiced_gap")
                    self.assertFalse(gap["silence_confirmed"])
                    self.assertEqual(gap["acoustic_classification"], "unknown")
                    self.assertEqual(gap["reason"], "pitch_unvoiced_not_acoustic_silence")
                    self.assertEqual((gap["gap_source_start"], gap["gap_source_end"]), (0.5, 1.5))
                    self.assertEqual(gap["student_anchors"][0], gap["student_anchors"][1])
                activity = [region for region in regions if region["mode"] != "silence"]
                self.assertEqual(len(activity), 2)
                self.assertTrue(all(region["status"] == "unresolved" for region in activity))
                self.assertTrue(all(region["mode"] == ("acoustic_fallback" if allow_fallback else "mute")
                                    for region in activity))

    def test_sparse_frames_do_not_claim_continuous_missing_vocals(self):
        teacher = activity_track([(1.0, 1.02), (4.0, 4.01), (11.0, 11.01), (14.0, 14.01)])
        student = activity_track([])
        plan = build_alignment(teacher, student)
        gaps = [r for r in plan["regions"] if r["status"] == "unresolved"]
        self.assertEqual(len(gaps), 4)
        self.assertTrue(all(r["mode"] == "mute" and r["unsupported"] for r in gaps))
        self.assertAlmostEqual(sum(r["teacher_end"] - r["teacher_start"] for r in gaps), 0.05)
        self.assertAlmostEqual(sum(r["teacher_end"] - r["teacher_start"]
                                   for r in plan["regions"]), teacher["duration"])
        # Lyric editing remains separate from sparse out-of-lyric activity.
        self.assertEqual(plan["metrics"]["lyric_coverage"], 1.0)
        self.assertEqual([(r["teacher_start"], r["teacher_end"])
                          for r in plan["regions"] if r["mode"] == "synthesize"], [(0.0, 0.3)])

    def test_matching_activity_runs_never_warp_across_silence(self):
        teacher = activity_track([(1.0, 1.2), (14.0, 14.2)])
        student = activity_track([(1.0, 1.2), (14.0, 14.2)])
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        allowed = [r for r in plan["regions"] if r["mode"] == "acoustic_fallback"]
        self.assertEqual(len(allowed), 2)
        self.assertTrue(all(r["teacher_end"] - r["teacher_start"] < 0.21 for r in allowed))
        middle = next(r for r in plan["regions"] if r["teacher_start"] <= 5 < r["teacher_end"])
        self.assertEqual(middle["mode"], "silence")

    def test_missing_source_run_rejects_activity_pairing(self):
        teacher = activity_track([(1.0, 1.2), (14.0, 14.2)])
        student = activity_track([(1.0, 1.2)])
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        self.assertFalse(plan["metrics"]["acoustic_fallback"])
        self.assertEqual(sum(r["mode"] == "mute" for r in plan["regions"]), 2)

    def test_equal_run_count_with_incompatible_progress_is_rejected(self):
        teacher = activity_track([(1.0, 1.2), (14.0, 14.2)])
        student = activity_track([(5.0, 5.2), (14.0, 14.2)])
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        self.assertFalse(plan["metrics"]["acoustic_fallback"])

    def test_no_transcript_fallback_uses_the_same_run_policy(self):
        teacher = activity_track([(1.0, 1.2), (14.0, 14.2)], with_word=False)
        student = activity_track([(1.0, 1.2), (14.0, 14.2)], with_word=False)
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        allowed = [r for r in plan["regions"] if r["mode"] == "acoustic_fallback"]
        self.assertEqual(len(allowed), 2)
        self.assertAlmostEqual(sum(r["teacher_end"] - r["teacher_start"] for r in allowed), 0.4)
        self.assertEqual(plan["metrics"]["semantic_match_rate"], 0.0)

    def test_extreme_run_ratio_does_not_enable_fallback(self):
        teacher = activity_track([(1.0, 3.0)])
        student = activity_track([(1.0, 1.1)])
        plan = build_alignment(teacher, student, allow_acoustic_fallback=True)
        self.assertFalse(plan["metrics"]["acoustic_fallback"])
        self.assertEqual(sum(r["mode"] == "mute" for r in plan["regions"]), 1)


if __name__ == "__main__":
    unittest.main()
