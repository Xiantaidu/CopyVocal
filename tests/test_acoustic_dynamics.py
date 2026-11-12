import unittest

import numpy as np

from inference.correction.dynamics import reference_transition_gain


class AcousticDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.sr = 8000
        self.times = np.arange(0.0, 1.0001, 0.001)
        sample_times = np.arange(self.sr) / self.sr
        self.teacher = (0.1 * np.sin(2 * np.pi * 220 * sample_times)).astype(np.float32)
        self.student = (0.2 * np.sin(2 * np.pi * 247 * sample_times)).astype(np.float32)
        self.source_times = self.times.copy()
        self.regions = [
            {"teacher_start": 0.0, "teacher_end": 0.4, "student_start": 0.0, "student_end": 0.4,
             "teacher_anchors": [0.0, 0.2, 0.4], "student_anchors": [0.0, 0.2, 0.4], "mode": "synthesize"},
            {"teacher_start": 0.4, "teacher_end": 0.6, "mode": "silence"},
            {"teacher_start": 0.6, "teacher_end": 1.0, "student_start": 0.6, "student_end": 1.0,
             "teacher_anchors": [0.6, 0.8, 1.0], "student_anchors": [0.6, 0.8, 1.0], "mode": "synthesize"},
        ]
        self.transitions = [{"teacher_start": 0.2, "teacher_end": 0.8,
                             "gap_teacher_start": 0.4, "gap_teacher_end": 0.6,
                             "left_region_index": 0, "right_region_index": 2}]

    def compute(self, teacher=None, student=None, transitions=None, regions=None):
        return reference_transition_gain(
            self.transitions if transitions is None else transitions,
            self.regions if regions is None else regions, self.times, self.source_times,
            self.teacher if teacher is None else teacher,
            self.student if student is None else student, self.sr,
        )

    def test_true_silence_suppresses_carried_source_vowel_without_zeroing(self):
        teacher = self.teacher.copy()
        teacher[int(0.4 * self.sr):int(0.6 * self.sr)] = 0
        gain, report = self.compute(teacher=teacher)
        middle = (self.times >= 0.44) & (self.times <= 0.56)
        self.assertLess(gain[middle].max(), 0.002)
        self.assertTrue((gain > 0).all())
        self.assertEqual(report[0]["status"], "constrained")
        self.assertGreater(report[0]["attenuated_frame_count"], 100)

    def test_real_unvoiced_noise_is_preserved_without_pitch_metadata(self):
        rng = np.random.default_rng(8)
        teacher, student = self.teacher.copy(), self.student.copy()
        sl = slice(int(0.4 * self.sr), int(0.6 * self.sr))
        teacher[sl] = rng.normal(0, 0.025, sl.stop - sl.start)
        student[sl] = rng.normal(0, 0.05, sl.stop - sl.start)
        gain, report = self.compute(teacher=teacher, student=student)
        np.testing.assert_array_equal(gain, np.ones(len(self.times), np.float32))
        self.assertEqual(report[0]["status"], "preserved")

    def test_local_loudness_normalization_is_invariant_to_speaker_gain(self):
        teacher = self.teacher.copy()
        teacher[int(0.4 * self.sr):int(0.6 * self.sr)] *= 0.01
        base, _ = self.compute(teacher=teacher)
        scaled, _ = self.compute(teacher=teacher * 0.1, student=self.student * 3)
        np.testing.assert_allclose(base, scaled, atol=1e-6)
        # Different recording levels alone never attenuate an energetic gap.
        unchanged, _ = self.compute(teacher=self.teacher * 0.001, student=self.student * 2)
        np.testing.assert_array_equal(unchanged, np.ones(len(self.times), np.float32))

    def test_quiet_but_matching_nonperiodic_audio_is_not_silenced(self):
        rng = np.random.default_rng(13)
        teacher, student = self.teacher.copy(), self.student.copy()
        sl = slice(int(0.4 * self.sr), int(0.6 * self.sr))
        texture = rng.normal(0, 0.001, sl.stop - sl.start)
        teacher[sl], student[sl] = texture, texture * 2
        gain, _ = self.compute(teacher=teacher, student=student)
        self.assertGreater(gain.min(), 0.999)

    def test_unrelated_and_missing_regions_are_unchanged(self):
        teacher = self.teacher.copy()
        teacher[int(0.4 * self.sr):int(0.6 * self.sr)] = 0
        gain, _ = self.compute(teacher=teacher)
        np.testing.assert_array_equal(gain[self.times <= 0.375], 1)
        np.testing.assert_array_equal(gain[self.times >= 0.625], 1)
        regions = [dict(r) for r in self.regions]
        regions[0]["unsupported"] = True
        unchanged, report = self.compute(teacher=teacher, regions=regions)
        np.testing.assert_array_equal(unchanged, 1)
        self.assertEqual(report, [])
        unchanged, report = self.compute(teacher=teacher, transitions=[])
        np.testing.assert_array_equal(unchanged, 1)
        self.assertEqual(report, [])
        regions = [dict(r) for r in self.regions]
        regions[1].update(mode="mute", reason="no_corresponding_student_lyric_unit", unsupported=True)
        unchanged, report = self.compute(teacher=teacher, regions=regions)
        np.testing.assert_array_equal(unchanged, 1)
        self.assertEqual(report, [])

    def test_gain_is_finite_continuous_and_has_unity_endpoints(self):
        teacher = self.teacher.copy()
        teacher[int(0.4 * self.sr):int(0.6 * self.sr)] = 0
        transition = {**self.transitions[0], "teacher_start": 0.4, "teacher_end": 0.6}
        gain, _ = self.compute(teacher=teacher, transitions=[transition])
        self.assertTrue(np.isfinite(gain).all())
        self.assertTrue(((gain > 0) & (gain <= 1)).all())
        self.assertEqual(gain[400], 1)
        self.assertEqual(gain[600], 1)
        self.assertLess(np.max(np.abs(np.diff(gain))), 0.16)

    def test_long_gap_uses_flanking_phones_instead_of_gap_percentile(self):
        teacher = self.teacher.copy()
        teacher[int(0.2 * self.sr):int(0.8 * self.sr)] *= 0.01
        regions = [
            {**self.regions[0], "teacher_end": 0.2, "student_end": 0.2,
             "teacher_anchors": [0, 0.1, 0.2], "student_anchors": [0, 0.1, 0.2]},
            self.regions[1],
            {**self.regions[2], "teacher_start": 0.8, "student_start": 0.8,
             "teacher_anchors": [0.8, 0.9, 1.0], "student_anchors": [0.8, 0.9, 1.0]},
        ]
        transition = {**self.transitions[0], "teacher_start": 0.2, "teacher_end": 0.8,
                      "gap_teacher_start": 0.2, "gap_teacher_end": 0.8}
        gain, report = self.compute(teacher=teacher, transitions=[transition], regions=regions)
        self.assertLess(gain[500], 0.03)
        self.assertGreater(report[0]["teacher_phone_reference_rms"], 0.05)

    def test_a_gap_that_rounds_to_zero_is_skipped_not_rejected(self):
        # Adjacent regions share one boundary time, and the two ends can disagree by
        # a few femtoseconds of float round-off. That gap holds no audio, so it must
        # be dropped rather than raised on: a real 100 s job died here.
        transition = {**self.transitions[0],
                      "gap_teacher_start": 0.5 + 1.4210854715202004e-14,
                      "gap_teacher_end": 0.5}
        gain, report = self.compute(transitions=[transition])
        np.testing.assert_array_equal(gain, np.ones(len(self.times), np.float32))
        self.assertEqual(report, [])

    def test_a_misordered_gap_is_still_rejected(self):
        transition = {**self.transitions[0], "gap_teacher_start": 0.7, "gap_teacher_end": 0.3}
        with self.assertRaises(ValueError):
            self.compute(transitions=[transition])


if __name__ == "__main__":
    unittest.main()
