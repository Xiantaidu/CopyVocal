import unittest

import numpy as np

from inference.correction.durations import (
    DurationBudgetError, classify_phone, constrain_window_mapping,
    monotone_segment_map, plan_segment_durations, segments_for_window,
)
from inference.correction.renderer import _crossfade_splices


def segment(start, end, kind, native=None, energy=None):
    item = {"start": start, "end": end, "kind": kind, "label": kind,
            "native": (end - start) if native is None else native}
    if energy is not None:
        item["energy"] = energy
    return item


class ClassifyPhoneTests(unittest.TestCase):
    def test_vowels_consonants_and_silence_are_distinguished(self):
        for label in ("ir", "i", "ai", "ang", "a", "ua", "ing"):
            self.assertEqual(classify_phone(label), "vowel", label)
        for label in ("z", "sh", "f", "h", "g", "b", "x", "zh", "ch", "k"):
            self.assertEqual(classify_phone(label), "consonant", label)
        for label in ("", "sp", "sil", "<blank>"):
            self.assertEqual(classify_phone(label), "span", label)


class DurationBudgetTests(unittest.TestCase):
    def test_fricative_keeps_sung_length_when_the_gap_is_covered_by_a_vowel(self):
        # The reported regression: 110 ms of /z/ asked to fill a 220 ms rest.
        segments = [segment(0.0, 0.24, "vowel", native=0.098),
                    segment(0.24, 0.35, "consonant", native=0.0975)]
        desired = np.array([0.1093, 0.2970])
        duration, info = plan_segment_durations(segments, desired, 0.4155)
        self.assertAlmostEqual(float(duration.sum()), 0.4155, places=9)
        self.assertLessEqual(info["maximum_consonant_source_rate"], 1.0 + 1e-9)
        self.assertFalse(info["consonant_stretched"])
        self.assertAlmostEqual(float(duration[1]), 0.11, places=9)
        self.assertGreater(float(duration[0]), 0.24 * 0.80)

    def test_quiet_rest_absorbs_time_before_a_vowel_is_stretched(self):
        segments = [segment(0.0, 0.2, "vowel", native=0.2),
                    segment(0.2, 0.6, "quiet_span", native=0.4, energy=-20.0)]
        desired = np.array([0.2, 0.4])
        duration, info = plan_segment_durations(segments, desired, 1.2)
        self.assertAlmostEqual(float(duration.sum()), 1.2, places=9)
        self.assertGreater(float(duration[1]), 0.4 * 2.0)
        self.assertAlmostEqual(info["residual_seconds"], 0.0, places=9)

    def test_a_window_with_no_non_consonant_material_is_refused(self):
        segments = [segment(0.0, 0.05, "consonant", native=0.05)]
        with self.assertRaises(DurationBudgetError):
            plan_segment_durations(segments, np.array([0.05]), 0.40)

    def test_consonants_are_never_used_to_absorb_a_remainder(self):
        segments = [segment(0.0, 0.20, "vowel", native=0.20),
                    segment(0.20, 0.25, "consonant", native=0.05)]
        with self.assertRaises(DurationBudgetError):
            plan_segment_durations(segments, np.array([0.20, 0.05]), 1.50)

    def test_short_target_compresses_a_consonant_but_never_below_the_floor(self):
        segments = [segment(0.0, 0.20, "consonant", native=0.20),
                    segment(0.20, 0.60, "vowel", native=0.40)]
        desired = np.array([0.20, 0.40])
        duration, info = plan_segment_durations(segments, desired, 0.30)
        self.assertAlmostEqual(float(duration.sum()), 0.30, places=9)
        self.assertLess(float(duration[0]), 0.20)
        self.assertGreaterEqual(float(duration[0]), 0.55 * 0.20 - 1e-9)
        self.assertFalse(info["consonant_stretched"])

    def test_budget_ignores_an_unreasonably_long_searched_consonant(self):
        # The searched warp wants 3x the fricative; the budget keeps it sung-length.
        segments = [segment(0.0, 0.30, "vowel", native=0.30),
                    segment(0.30, 0.40, "consonant", native=0.10)]
        desired = np.array([0.30, 0.30])
        duration, info = plan_segment_durations(segments, desired, 0.70)
        self.assertAlmostEqual(float(duration[1]), 0.10, places=9)
        self.assertFalse(info["consonant_stretched"])

    def test_target_below_all_lower_bounds_is_refused_not_negative(self):
        # Regression: the vowel used to be shrunk to -175 ms, then the
        # constrained mapping raised a bare ValueError and aborted the render.
        segments = [segment(0.0, 0.5, "consonant", native=0.5),
                    segment(0.5, 1.0, "vowel", native=0.5)]
        with self.assertRaises(DurationBudgetError):
            plan_segment_durations(segments, np.array([0.05, 0.05]), 0.1)

    def test_constrained_window_refuses_an_unfittable_compression(self):
        # The caller turns this into "constraint not applied" and keeps the
        # searched warp instead of aborting the render.
        segments = [segment(0.0, 0.5, "consonant", native=0.5),
                    segment(0.5, 1.0, "vowel", native=0.5)]
        with self.assertRaises(DurationBudgetError):
            constrain_window_mapping(np.linspace(0.0, 0.1, 11), np.array([0.0, 0.1]),
                                     np.array([0.0, 1.0]), segments, window=(0.0, 0.1))

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            plan_segment_durations([], np.array([]), 1.0)
        with self.assertRaises(ValueError):
            plan_segment_durations([segment(0, 1, "vowel")], np.array([1.0]), 0.0)
        with self.assertRaises(ValueError):
            plan_segment_durations([segment(0, 1, "vowel")], np.array([1.0, 2.0]), 1.0)


class SegmentPartitionTests(unittest.TestCase):
    def analysis(self, phones, start=0.0, end=1.0):
        return {"words": [{"text": "x", "start": start, "end": end,
                           "phonemes": [{"text": label, "start": a, "end": b}
                                        for label, a, b in phones]}]}

    def regions(self):
        return [{"teacher_start": 0.0, "teacher_end": 0.4,
                 "teacher_anchors": [0.0, 0.2, 0.4], "student_anchors": [0.0, 0.2, 0.4]},
                {"teacher_start": 0.6, "teacher_end": 1.0,
                 "teacher_anchors": [0.6, 0.8, 1.0], "student_anchors": [0.4, 0.6, 0.8]}]

    def test_window_is_partitioned_into_phones_and_the_rest_between_them(self):
        analysis = self.analysis([("ir", 0.20, 0.40), ("z", 0.44, 0.55)])
        segments = segments_for_window(analysis, 0.2, 0.6, self.regions(), 0, 1, mel=None)
        self.assertEqual([item["kind"] for item in segments],
                         ["vowel", "quiet_span", "consonant", "quiet_span"])
        self.assertAlmostEqual(segments[0]["start"], 0.2)
        self.assertAlmostEqual(segments[-1]["end"], 0.6)

    def test_phone_native_duration_comes_from_its_owning_region(self):
        # The left region compresses 0.5x and the right region stretches 1.5x.
        regions = [{"teacher_start": 0.0, "teacher_end": 0.2,
                    "teacher_anchors": [0.0, 0.2], "student_anchors": [0.0, 0.4]},
                   {"teacher_start": 0.6, "teacher_end": 0.75,
                    "teacher_anchors": [0.6, 0.75], "student_anchors": [0.4, 0.5]}]
        analysis = self.analysis([("z", 0.20, 0.30), ("ai", 0.44, 0.50)])
        segments = segments_for_window(analysis, 0.2, 0.5, regions, 0, 1, mel=None)
        by_label = {item["label"]: item for item in segments}
        self.assertAlmostEqual(by_label["z"]["native"], 0.05, places=9)
        self.assertAlmostEqual(by_label["ai"]["native"], 0.09, places=9)

    def test_a_window_without_phone_labels_is_left_to_the_searched_warp(self):
        analysis = self.analysis([("sp", 0.20, 0.60)])
        self.assertEqual(segments_for_window(analysis, 0.2, 0.6, self.regions(), 0, 1, mel=None), [])

    def test_audible_rest_is_not_treated_as_a_quiet_span(self):
        mel = np.full((8, 100), -20.0)
        centers = (np.arange(mel.shape[1]) + 0.5) * 0.01
        mel[:, (centers >= 0.2) & (centers < 0.4)] = -1.0
        mel[:, (centers >= 0.4) & (centers < 0.6)] = -0.6  # breath, only 0.4 nats down
        analysis = self.analysis([("ir", 0.20, 0.40)])
        segments = segments_for_window(analysis, 0.2, 0.6, self.regions(), 0, 1, mel=mel, step=0.01)
        self.assertEqual([item["kind"] for item in segments], ["vowel", "span"])


class ConstrainedMappingTests(unittest.TestCase):
    def test_mapping_pins_boundaries_and_stays_strictly_increasing(self):
        segments = [segment(0.0, 0.24, "vowel", native=0.098),
                    segment(0.24, 0.35, "consonant", native=0.0975)]
        query = np.linspace(0.0, 0.4155, 416)
        warp_target = np.array([0.0, 0.2, 0.4155])
        warp_source = np.array([0.0, 0.16, 0.35])
        mapped, info = constrain_window_mapping(query, warp_target, warp_source, segments,
                                                window=(0.0, 0.4155))
        self.assertTrue(np.all(np.diff(mapped) > 0))
        self.assertAlmostEqual(float(mapped[0]), 0.0, places=6)
        self.assertAlmostEqual(float(mapped[-1]), 0.35, places=6)
        for boundary, planned in zip(info["boundaries"][1:-1], info["knot_target"][1:-1]):
            self.assertAlmostEqual(float(np.interp(planned, query, mapped)), boundary, places=3)

    def test_consonant_span_keeps_its_sung_seconds_in_the_final_mapping(self):
        segments = [segment(0.0, 0.10, "consonant", native=0.10),
                    segment(0.10, 0.40, "vowel", native=0.30)]
        query = np.linspace(0.0, 0.40, 801)
        warp_target = np.array([0.0, 0.15, 0.40])
        warp_source = np.array([0.0, 0.10, 0.40])
        mapped, info = constrain_window_mapping(query, warp_target, warp_source, segments,
                                                window=(0.0, 0.40))
        self.assertAlmostEqual(float(info["planned_seconds"][0]), 0.10, places=6)
        self.assertFalse(info["consonant_stretched"])
        self.assertAlmostEqual(float(np.interp(0.10, query, mapped)), 0.10, places=3)
        self.assertAlmostEqual(float(mapped[-1]), 0.40, places=6)

    def test_context_knots_blend_the_rate_across_a_window_edge(self):
        target = [0.0, 0.2, 0.3]
        source = [0.0, 0.10, 0.30]
        query = np.linspace(0.0, 0.3, 301)
        without = monotone_segment_map(query, target, source)
        with_context = monotone_segment_map(query, target, source, context=[(-0.2, -0.30)])
        # Both pin every window knot, so the durational budget is unaffected.
        for point, expected in zip(target, source):
            self.assertAlmostEqual(float(np.interp(point, query, without)), expected, places=6)
            self.assertAlmostEqual(float(np.interp(point, query, with_context)), expected, places=6)
        # The neighbouring anchor bends the rate reaching into the window instead
        # of letting it step at the seam.
        self.assertFalse(np.allclose(without[1:-1], with_context[1:-1]))

    def test_invalid_warp_knots_are_rejected(self):
        segments = [segment(0.0, 0.5, "vowel"), segment(0.5, 1.0, "consonant")]
        with self.assertRaises(ValueError):
            constrain_window_mapping(np.arange(0, 1, 0.01), np.array([0.0, 1.0]),
                                     np.array([0.0, 1.0, 2.0]), segments, window=(0.0, 1.0))
        with self.assertRaises(ValueError):
            constrain_window_mapping(np.arange(0, 1, 0.01), np.array([0.0, 1.0]),
                                     np.array([0.0, 1.0]), [], window=(0.0, 1.0))
        with self.assertRaises(ValueError):
            constrain_window_mapping(np.arange(0, 1, 0.01), np.array([0.0, 1.0]),
                                     np.array([0.0, 1.0]), segments, window=(1.0, 1.0))


class SpliceCrossfadeTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0.005, 1.0, 0.0116)
        # A texture that changes over time so a blend is observable.
        self.mel = np.add.outer(np.arange(8.0), self.times * 40.0)
        self.transition = {"teacher_start": 0.4, "teacher_end": 0.7,
                           "duration_constraint": {"applied": True}}

    def test_discontinuous_splice_is_blended(self):
        source = self.times.copy()
        source[self.times >= 0.4] += 0.06
        mel = self.mel.copy()
        edits = _crossfade_splices(mel, self.times, source, [self.transition])
        self.assertTrue(edits)
        self.assertFalse(np.allclose(mel, self.mel))
        for edit in edits:
            self.assertEqual(edit["policy"],
                             "equal_power_mel_blend_across_discontinuous_constrained_splice")

    def test_continuous_source_needs_no_blend(self):
        source = self.times.copy()
        mel = self.mel.copy()
        edits = _crossfade_splices(mel, self.times, source, [self.transition])
        self.assertEqual(edits, [])
        np.testing.assert_array_equal(mel, self.mel)

    def test_unconstrained_transition_is_left_alone(self):
        source = self.times.copy()
        source[self.times >= 0.4] += 0.06
        mel = self.mel.copy()
        transition = {"teacher_start": 0.4, "teacher_end": 0.7,
                      "duration_constraint": {"applied": False}}
        self.assertEqual(_crossfade_splices(mel, self.times, source, [transition]), [])
        np.testing.assert_array_equal(mel, self.mel)


if __name__ == "__main__":
    unittest.main()
