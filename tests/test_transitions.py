"""Acoustic transition regression tests independent of pitch and the vocoder."""

import unittest
from copy import deepcopy

import numpy as np

from inference.correction.transitions import (
    _aperiodic_onset, _feature_cost, _noise_onset, _waveform_features, refine_transition_map,
)


def word(start, end, source_start, source_end, index=0, **extra):
    return {"teacher_start": start, "teacher_end": end,
            "student_start": source_start, "student_end": source_end,
            "teacher_anchors": [start, (start + end) / 2, end],
            "student_anchors": [source_start, (source_start + source_end) / 2, source_end],
            "teacher_index": index, "student_index": index,
            "mode": "synthesize", "quality": 1.0, **extra}


def gap(start, end, **extra):
    return {"teacher_start": start, "teacher_end": end, "mode": "silence", **extra}


class TransitionMapTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0.005, 2, 0.01)
        bands = np.arange(16)[:, None]
        time = self.times[None, :]
        self.mel = -3 + np.cos(bands * 0.7 + time * 13) + 0.3 * np.sin(time * 31)

    def render(self, regions, student_words=None, teacher_mel=None, student_mel=None, **kwargs):
        source = np.full(self.times.shape, np.nan)
        active = np.zeros(self.times.shape, bool)
        for region in regions:
            if region["mode"] != "synthesize" or region.get("unsupported"):
                continue
            selected = ((self.times >= region["teacher_start"]) &
                        (self.times < region["teacher_end"]))
            source[selected] = np.interp(self.times[selected], region["teacher_anchors"], region["student_anchors"])
            active[selected] = True
        return refine_transition_map(
            regions, {}, {"words": student_words or []},
            self.mel if teacher_mel is None else teacher_mel,
            self.mel if student_mel is None else student_mel,
            self.times, source, active, sample_rate=1000, hop_size=10, **kwargs,
        ), source, active

    def test_zero_source_gap_borrows_changing_phone_material_without_freeze(self):
        regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.62), word(0.62, 1, 0.4, 0.8, 1)]
        snapshot = deepcopy(regions)
        (mapped, active, mask, diagnostics), original, _ = self.render(regions)
        in_gap = (self.times >= 0.4) & (self.times < 0.62)
        self.assertTrue(active[in_gap].all())
        self.assertTrue(mask[in_gap].all())
        self.assertTrue(np.isfinite(mapped[mask]).all())
        self.assertTrue((np.diff(mapped[mask]) > 0).all())
        self.assertGreater(np.ptp(mapped[in_gap]), 0.05)
        self.assertEqual(diagnostics[0]["strategy"], "local_acoustic_dtw")
        self.assertFalse(diagnostics[0]["semantic_support"])
        np.testing.assert_equal(mapped[~mask], original[~mask])
        self.assertEqual(regions, snapshot)

    def test_noise_and_unvoiced_material_do_not_require_pitch_support(self):
        rng = np.random.default_rng(7)
        noise = rng.normal(-5, 0.8, self.mel.shape)
        regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.6), word(0.6, 1, 0.4, 0.8, 1)]
        (mapped, active, mask, diagnostics), _, _ = self.render(regions, teacher_mel=noise, student_mel=noise)
        self.assertTrue(active[(self.times >= 0.4) & (self.times < 0.6)].all())
        self.assertTrue(np.all(np.diff(mapped[mask]) > 0))
        self.assertEqual(len(diagnostics), 1)

    def test_real_long_quiet_gap_retains_all_source_material_continuously(self):
        regions = [word(0, 0.3, 0, 0.3), gap(0.3, 1.3), word(1.3, 1.7, 1.2, 1.6, 1)]
        (mapped, active, mask, diagnostics), _, _ = self.render(regions)
        selected = (self.times >= 0.3) & (self.times < 1.3)
        self.assertTrue(active[selected].all())
        np.testing.assert_allclose(mapped[selected], 0.3 + (self.times[selected] - 0.3) * 0.9)
        self.assertEqual(diagnostics[0]["strategy"], "continuous_gap_linear")
        np.testing.assert_equal(mask, selected)

    def test_missing_or_rejected_target_word_prevents_bridge_without_source_metadata(self):
        for reason in ("no_corresponding_student_lyric_unit", "possible_lyric_mismatch_or_asr_error", "ambiguous_repeated_lyrics"):
            regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.6),
                       gap(0.45, 0.5, mode="mute", reason=reason), word(0.6, 1, 0.4, 0.8, 1)]
            (mapped, active, mask, diagnostics), original, before = self.render(regions)
            self.assertEqual(diagnostics, [])
            self.assertFalse(mask.any())
            np.testing.assert_equal(mapped, original)
            np.testing.assert_equal(active, before)

    def test_gap_run_activity_is_safe_but_unclassified_holes_are_not(self):
        left, right = word(0, 0.4, 0, 0.4), word(0.6, 1, 0.4, 0.8, 1)
        activity = gap(0.4, 0.6, mode="mute", reason="voiced_teacher_region_without_lyric_correspondence")
        result, _, _ = self.render([left, activity, right])
        self.assertEqual(len(result[-1]), 1)
        result, _, _ = self.render([left, right])
        self.assertEqual(result[-1], [])

    def test_source_extra_and_nonconsecutive_lexical_indices_are_barriers(self):
        left, right = word(0, 0.4, 0, 0.4), word(0.6, 1, 0.6, 1, 1)
        words = [{"start": 0, "end": 0.4}, {"start": 0.6, "end": 1}, {"start": 0.43, "end": 0.57}]
        result, _, _ = self.render([left, gap(0.4, 0.6), right], words)
        self.assertEqual(result[-1], [])
        for key in ("student_index", "teacher_index"):
            result, _, _ = self.render([left, gap(0.4, 0.6), {**right, key: 2}])
            self.assertEqual(result[-1], [])

    def test_fixed_outer_anchors_give_endpoint_continuity(self):
        self.times = np.arange(0, 1.001, 0.005)
        regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.6), word(0.6, 1, 0.4, 0.8, 1)]
        (mapped, _, mask, diagnostics), _, _ = self.render(regions)
        transition, = diagnostics
        self.assertAlmostEqual(transition["teacher_start"], 0.2)
        self.assertAlmostEqual(transition["teacher_end"], 0.8)
        self.assertAlmostEqual(mapped[np.argmin(abs(self.times - 0.2))], 0.2)
        self.assertAlmostEqual(mapped[np.argmin(abs(self.times - 0.8))], 0.6)
        self.assertTrue(np.all(np.diff(mapped[np.isfinite(mapped)]) > 0))

    def test_adjacent_transition_windows_never_overlap(self):
        regions = [word(0, 0.3, 0, 0.3), gap(0.3, 0.45), word(0.45, 0.75, 0.3, 0.6, 1),
                   gap(0.75, 0.9), word(0.9, 1.2, 0.6, 0.9, 2)]
        (mapped, _, mask, diagnostics), _, _ = self.render(regions)
        self.assertEqual(len(diagnostics), 2)
        self.assertLessEqual(diagnostics[0]["teacher_end"], diagnostics[1]["teacher_start"])
        self.assertTrue(np.all(np.diff(mapped[mask]) > 0))

    def test_extreme_ratio_uses_auditable_monotone_phone_window_fallback(self):
        regions = [word(0, 0.1, 0, 0.055), gap(0.1, 0.15), word(0.15, 0.25, 0.055, 0.11, 1)]
        (mapped, active, mask, diagnostics), _, _ = self.render(regions)
        self.assertEqual(diagnostics[0]["strategy"], "phone_window_linear")
        self.assertTrue(active[(self.times >= 0.1) & (self.times < 0.15)].all())
        self.assertTrue(np.all(np.diff(mapped[mask]) > 0))

    def test_long_unsourced_or_tiny_source_gap_does_not_create_a_drone(self):
        for source_gap in (0, 0.01):
            regions = [word(0, 0.1, 0, 0.1), gap(0.1, 1.5),
                       word(1.5, 1.6, 0.1 + source_gap, 0.2 + source_gap, 1)]
            (_, active, mask, diagnostics), _, _ = self.render(regions)
            self.assertEqual(diagnostics, [])
            self.assertFalse(mask.any())
            self.assertFalse(active[(self.times >= 0.1) & (self.times < 1.5)].any())

    def test_different_duration_fricative_uses_auditable_noise_landmark(self):
        bands = np.arange(16)[:, None]
        vowel = np.broadcast_to(-1 - bands * 0.5, self.mel.shape).copy()
        noise = np.broadcast_to(-4 + bands * 0.003, self.mel.shape)
        teacher = vowel.copy()
        student = vowel.copy()
        teacher[:, (self.times >= 0.4) & (self.times < 0.75)] = noise[:, (self.times >= 0.4) & (self.times < 0.75)]
        student[:, (self.times >= 0.48) & (self.times < 0.58)] = noise[:, (self.times >= 0.48) & (self.times < 0.58)]
        regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.62), word(0.62, 1, 0.4, 0.8, 1)]
        (mapped, _, mask, diagnostics), _, _ = self.render(regions, teacher_mel=teacher, student_mel=student)
        diagnostic, = diagnostics
        self.assertEqual(diagnostic["strategy"], "noise_anchored_acoustic_dtw")
        self.assertGreaterEqual(diagnostic["acoustic_landmark"]["minimum_duration_seconds"], 0.02)
        at_noise = np.interp(0.44, self.times, mapped)
        self.assertGreaterEqual(at_noise, 0.48)
        self.assertLess(at_noise, 0.58)
        self.assertTrue(np.all(np.diff(mapped[mask]) > 0))

    def test_tonal_to_silence_or_deep_noise_valley_cannot_establish_noise_anchor(self):
        bands = np.arange(16)[:, None]
        vowel = np.broadcast_to(-1 - bands * 0.5, self.mel.shape).copy()
        for silent_texture in (np.full(self.mel.shape, np.log(1e-5)),
                               np.broadcast_to(-9 + bands * 0.003, self.mel.shape)):
            teacher, student = vowel.copy(), vowel.copy()
            teacher_gap = (self.times >= 0.4) & (self.times < 0.75)
            student_gap = (self.times >= 0.48) & (self.times < 0.58)
            teacher[:, teacher_gap] = silent_texture[:, teacher_gap]
            student[:, student_gap] = silent_texture[:, student_gap]
            self.assertIsNone(_noise_onset(teacher[:, 20:80], self.times[20:80]))
            self.assertIsNone(_noise_onset(student[:, 20:60], self.times[20:60]))
            regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.62), word(0.62, 1, 0.4, 0.8, 1)]
            (_, _, _, diagnostics), _, _ = self.render(regions, teacher_mel=teacher, student_mel=student)
            self.assertIsNone(diagnostics[0]["acoustic_landmark"])

    def test_waveform_periodicity_distinguishes_weak_low_frequency_breath_from_vowel(self):
        from scipy.signal import butter, sosfilt

        sample_rate = 8000
        sample_times = np.arange(sample_rate) / sample_rate
        vowel = 0.12 * np.sin(2 * np.pi * 220 * sample_times)
        noise = sosfilt(butter(2, 1800, fs=sample_rate, output="sos"),
                       np.random.default_rng(123).normal(0, 0.025, sample_rate))
        audio = np.where(sample_times < 0.5, vowel, noise)
        query = np.arange(0.2, 0.9, 0.01)
        features = _waveform_features(audio, query, sample_rate)
        self.assertGreater(np.median(features["periodicity"][query < 0.45]), 0.95)
        self.assertLess(np.median(features["periodicity"][query > 0.55]), 0.4)
        onset = _aperiodic_onset(features, query)
        self.assertIsNotNone(onset)
        self.assertAlmostEqual(onset[0], 0.5, delta=0.025)
        scaled = _waveform_features(audio * 0.025, query, sample_rate)
        np.testing.assert_allclose(scaled["periodicity"], features["periodicity"], atol=1e-10)
        np.testing.assert_equal(scaled["audible"], features["audible"])

    def test_short_burst_and_silence_do_not_establish_sustained_waveform_landmarks(self):
        times = np.arange(0, 0.5, 0.01)
        for count in (0, 1, 2):
            periodicity = np.full(len(times), 0.98)
            periodicity[25:25 + count] = 0.2
            features = {"periodicity": periodicity, "audible": np.ones(len(times), bool)}
            self.assertIsNone(_aperiodic_onset(features, times))
        features = {"periodicity": np.r_[np.ones(25), np.full(25, 0.1)],
                    "audible": np.r_[np.ones(25, bool), np.zeros(25, bool)]}
        self.assertIsNone(_aperiodic_onset(features, times))

    def test_periodicity_landmark_can_follow_an_initial_noisy_consonant(self):
        times = np.arange(0, 0.8, 0.01)
        features = {"periodicity": np.r_[np.full(12, 0.2), np.full(35, 0.98), np.full(33, 0.25)],
                    "audible": np.ones(len(times), bool)}
        self.assertAlmostEqual(_aperiodic_onset(features, times)[0], 0.47)

    def test_absolute_periodicity_cost_prefers_noise_despite_identical_mel(self):
        mel = np.full((16, 3), -3.0)
        teacher = {"periodicity": np.full(3, 0.2), "audible": np.ones(3, bool)}
        student = {"periodicity": np.array([0.95, 0.2, 0.3]), "audible": np.ones(3, bool)}
        cost = _feature_cost(mel, mel, teacher, student)
        self.assertGreater(cost[0, 0], cost[0, 1] + 2)
        student["audible"][:] = False
        np.testing.assert_allclose(_feature_cost(mel, mel, teacher, student), 0)

    def test_weak_breath_waveform_landmark_aligns_when_mel_cannot_detect_noise(self):
        # Identical spectra cannot provide a mel rise, but waveform texture can.
        sample_times = np.arange(2000) / 1000
        vowel = 0.1 * np.sin(2 * np.pi * 125 * sample_times)
        noise = np.random.default_rng(66).normal(0, 0.015, len(vowel))
        teacher = np.where((sample_times >= 0.4) & (sample_times < 0.75), noise, vowel)
        student = np.where((sample_times >= 0.49) & (sample_times < 0.6), noise, vowel)
        mel = np.full(self.mel.shape, -3.0)
        regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.62), word(0.62, 1, 0.4, 0.8, 1)]
        (mapped, _, mask, diagnostics), _, _ = self.render(
            regions, teacher_mel=mel, student_mel=mel, teacher_audio=teacher, student_audio=student)
        landmark = diagnostics[0]["acoustic_landmark"]
        self.assertEqual(landmark["kind"], "sustained_aperiodic_onset")
        self.assertEqual(landmark["minimum_duration_seconds"], 0.030)
        self.assertEqual(landmark["student_cue"], "waveform_periodicity_drop")
        self.assertGreater(np.interp(0.5, self.times, mapped), 0.49)
        self.assertLess(np.interp(0.5, self.times, mapped), 0.6)
        self.assertTrue(np.all(np.diff(mapped[mask]) > 0))

    def test_waveform_cues_reject_incomplete_or_nonfinite_audio(self):
        regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.6), word(0.6, 1, 0.4, 0.8, 1)]
        for kwargs in ({"teacher_audio": np.zeros(2000)},
                       {"teacher_audio": np.zeros(2000), "student_audio": np.array([np.nan])}):
            with self.assertRaises(ValueError):
                self.render(regions, **kwargs)

    def test_unavailable_breath_remains_auditable_without_changing_lyric_semantics(self):
        sample_times = np.arange(2000) / 1000
        vowel = 0.1 * np.sin(2 * np.pi * 125 * sample_times)
        noise = np.random.default_rng(98).normal(0, 0.03, len(vowel))
        teacher = np.where((sample_times >= 0.4) & (sample_times < 0.6), noise, vowel)
        mel = np.full(self.mel.shape, -3.0)
        regions = [word(0, 0.4, 0, 0.4), gap(0.4, 0.6), word(0.6, 1, 0.4, 0.8, 1)]
        snapshot = deepcopy(regions)
        (_, active, _, diagnostics), _, _ = self.render(
            regions, teacher_mel=mel, student_mel=mel, teacher_audio=teacher, student_audio=vowel)
        mismatch = diagnostics[0]["source_acoustic_mismatch"]
        self.assertGreater(mismatch["mapped_periodic_frame_count"], 5)
        self.assertEqual(len(mismatch["mapped_periodic_target_times"]), mismatch["mapped_periodic_frame_count"])
        self.assertTrue(active[(self.times >= 0.4) & (self.times < 0.6)].all())
        self.assertEqual(regions, snapshot)


if __name__ == "__main__":
    unittest.main()
