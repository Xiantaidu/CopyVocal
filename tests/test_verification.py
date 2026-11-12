from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from inference.verification import compare_pitch, sample_measured_pitch, verify_output, yin_window


def track(frequencies, *, step=0.01):
    frequencies = np.asarray(frequencies, dtype=float)
    return {"times": (np.arange(len(frequencies)) * step).tolist(),
            "f0_hz": frequencies.tolist(), "voiced": (frequencies > 0).tolist()}


class PitchVerificationTests(unittest.TestCase):
    def test_missing_output_remains_in_reference_denominator(self):
        reference = track([440] * 100)
        actual = track([440] * 50)
        report = compare_pitch(actual, reference, actual_duration=0.5, reference_duration=1)
        self.assertEqual(report["reference_voiced_frames"], 100)
        self.assertEqual(report["missed_voiced_frames"], 50)
        self.assertEqual(report["voicing_recall"], 0.5)
        self.assertAlmostEqual(report["median_absolute_cents"], 0)
        self.assertEqual(report["correct_voiced_within_50_cents_ratio"], 0.5)

    def test_uv_precision_and_octave_errors(self):
        reference, actual = track([440, 440, 0, 0]), track([880, 0, 440, 0])
        report = compare_pitch(actual, reference)
        self.assertAlmostEqual(report["median_absolute_cents"], 1200)
        self.assertEqual(report["octave_error_ratio"], 1)
        self.assertEqual(report["voicing_precision"], 0.5)
        self.assertEqual(report["voicing_recall"], 0.5)
        self.assertEqual(report["false_voiced_frames"], 1)

    def test_missing_unvoiced_tail_is_not_a_correct_observation(self):
        report = compare_pitch(track([440, 440]), track([440, 440, 0, 0]))
        self.assertEqual(report["voicing_accuracy"], 0.5)
        self.assertEqual(report["output_availability_ratio"], 0.5)

    def test_sampler_does_not_bridge_unvoiced_region(self):
        f0, voiced, available = sample_measured_pitch(track([440, 0, 880]), np.array([0.001, 0.01, 0.019, 0.1]))
        np.testing.assert_array_equal(voiced, [True, False, True, False])
        self.assertEqual(f0[1], 0)
        self.assertFalse(available[-1])

    def test_yin_sine_frequency_and_silence(self):
        rate = 16000
        for frequency in (110, 220, 440, 880):
            frame = 0.2 * np.sin(np.arange(2048) * frequency * 2 * np.pi / rate)
            estimate, confidence = yin_window(frame, rate)
            error_cents = 1200 * np.log2(estimate / frequency)
            self.assertLess(abs(error_cents), 4, (frequency, estimate))
            self.assertGreater(confidence, 0.95)
        self.assertEqual(yin_window(np.zeros(2048), rate), (0, 0))


class VerificationArtifactTests(unittest.TestCase):
    def test_overall_metrics_retain_unsupported_material(self):
        rate, count = 16000, 16000
        reference = track([440] * 100)
        actual = track([440] * 50 + [0] * 50)
        teacher = {"sample_rate": rate, "sample_count": count, "duration": 1.0, "pitch": reference}
        plan = {"regions": [
            {"teacher_start": 0.0, "teacher_end": 0.5, "mode": "synthesize", "unsupported": False, "status": "matched"},
            {"teacher_start": 0.5, "teacher_end": 1.0, "mode": "mute", "unsupported": True, "status": "missing"},
        ], "metrics": {"lyric_coverage": 0.5, "status_counts": {"matched": 1, "missing": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = np.zeros(count)
            signal[:count // 2] = 0.2 * np.sin(np.arange(count // 2) * 440 * 2 * np.pi / rate)
            audio = root / "output.wav"
            target = root / "target.npz"
            sf.write(audio, signal, rate, subtype="FLOAT")
            np.savez(target, times=np.arange(100) * 0.01, f0_hz=np.array([440] * 50 + [0] * 50),
                     voiced=np.array([True] * 50 + [False] * 50), supported=np.array([True] * 50 + [False] * 50))
            with patch("inference.verification._measure_rmvpe", return_value=actual):
                report = verify_output(audio, teacher, plan, target, root=root, device="cpu")
            self.assertTrue(report["structural_checks_passed"])
            self.assertEqual(report["actual_vs_teacher"]["all_reference_frames"]["voicing_recall"], 0.5)
            self.assertEqual(report["actual_vs_teacher"]["semantically_supported_frames"]["voicing_recall"], 1)
            self.assertEqual(report["actual_vs_teacher"]["outside_semantic_support"]["voicing_recall"], 0)
            self.assertEqual(report["coverage"]["semantic_teacher_voiced_coverage"], 0.5)
            self.assertEqual(report["actual_vs_render_target"]["all_target_frames"]["voicing_accuracy"], 1)
            self.assertFalse(report["word_timing_validation"]["measured"])
            self.assertGreater(report["yin_diagnostic"]["accepted_windows"], 0)
            json.dumps(report, allow_nan=False)

    def test_a_relative_pitch_target_adds_a_register_matched_comparison(self):
        rate, count = 16000, 16000
        teacher_pitch = track([440] * 100)
        actual = track([220] * 100)
        teacher = {"sample_rate": rate, "sample_count": count, "duration": 1.0, "pitch": teacher_pitch}
        plan = {"regions": [{"teacher_start": 0.0, "teacher_end": 1.0, "mode": "synthesize",
                             "unsupported": False, "status": "matched"}],
                "metrics": {"lyric_coverage": 1.0, "status_counts": {"matched": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = 0.2 * np.sin(np.arange(count) * 220 * 2 * np.pi / rate)
            audio = root / "output.wav"
            target = root / "target.npz"
            sf.write(audio, signal, rate, subtype="FLOAT")
            np.savez(target, times=np.arange(100) * 0.01, f0_hz=np.array([220] * 100),
                     voiced=np.array([True] * 100), supported=np.array([True] * 100),
                     pitch_policy=np.array("teacher_contour_student_register"),
                     register_offset_cents=np.float64(-1200.0),
                     register_offset_available=np.bool_(True), timing_changed=np.bool_(True))
            with patch("inference.verification._measure_rmvpe", return_value=actual):
                report = verify_output(audio, teacher, plan, target, root=root, device="cpu")
            self.assertEqual(report["provenance"]["target_pitch_policy"],
                             "teacher_contour_student_register")
            self.assertAlmostEqual(
                report["actual_vs_teacher"]["all_reference_frames"]["median_absolute_cents"], 1200, delta=1)
            relative = report["actual_vs_teacher_relative"]["all_reference_frames"]
            self.assertAlmostEqual(relative["median_absolute_cents"], 0, delta=1)
            self.assertIn("-1200.0 cents", report["actual_vs_teacher_relative"]["reference"])
            self.assertTrue(any("register" in item for item in report["limitations"]))
            self.assertTrue(any("register offset" in item for item in report["warnings"]))
            json.dumps(report, allow_nan=False)

    def test_a_negligible_register_offset_needs_no_transposed_comparison(self):
        # With the singers already in one register the extra key would only be a
        # duplicate of the absolute numbers under another name.
        rate, count = 16000, 16000
        actual = track([440] * 100)
        teacher = {"sample_rate": rate, "sample_count": count, "duration": 1.0, "pitch": track([440] * 100)}
        plan = {"regions": [{"teacher_start": 0.0, "teacher_end": 1.0, "mode": "synthesize",
                             "unsupported": False, "status": "matched"}],
                "metrics": {"lyric_coverage": 1.0, "status_counts": {"matched": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = 0.2 * np.sin(np.arange(count) * 440 * 2 * np.pi / rate)
            audio = root / "output.wav"
            target = root / "target.npz"
            sf.write(audio, signal, rate, subtype="FLOAT")
            np.savez(target, times=np.arange(100) * 0.01, f0_hz=np.array([440] * 100),
                     voiced=np.array([True] * 100), supported=np.array([True] * 100),
                     pitch_policy=np.array("student_contour_student_register"),
                     register_offset_cents=np.float64(-0.0),
                     register_offset_available=np.bool_(True), timing_changed=np.bool_(True))
            with patch("inference.verification._measure_rmvpe", return_value=actual):
                report = verify_output(audio, teacher, plan, target, root=root, device="cpu")
            self.assertNotIn("actual_vs_teacher_relative", report)
            self.assertFalse(any("register offset" in item for item in report["warnings"]))

    def test_adaptive_placement_reports_that_the_words_moved_individually(self):
        # The register-shifted teacher curve approximates an adaptive render rather
        # than being its aim, so the verification has to say so and point at the
        # render target instead of letting the approximation pass as the reference.
        rate, count = 16000, 16000
        teacher_pitch = track([440] * 100)
        actual = track([220] * 100)
        teacher = {"sample_rate": rate, "sample_count": count, "duration": 1.0, "pitch": teacher_pitch}
        plan = {"regions": [{"teacher_start": 0.0, "teacher_end": 1.0, "mode": "synthesize",
                             "unsupported": False, "status": "matched"}],
                "metrics": {"lyric_coverage": 1.0, "status_counts": {"matched": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = 0.2 * np.sin(np.arange(count) * 220 * 2 * np.pi / rate)
            audio = root / "output.wav"
            target = root / "target.npz"
            sf.write(audio, signal, rate, subtype="FLOAT")
            np.savez(target, times=np.arange(100) * 0.01, f0_hz=np.array([220] * 100),
                     voiced=np.array([True] * 100), supported=np.array([True] * 100),
                     pitch_policy=np.array("student_contour_student_register"),
                     register_offset_cents=np.float64(-1200.0),
                     register_offset_available=np.bool_(True), note_shift_applied_count=np.int64(7),
                     timing_changed=np.bool_(True))
            with patch("inference.verification._measure_rmvpe", return_value=actual):
                report = verify_output(audio, teacher, plan, target, root=root, device="cpu")
            self.assertIn("actual_vs_teacher_relative", report)
            self.assertTrue(any("approximates the aim" in item for item in report["limitations"]))
            self.assertTrue(any("7 word(s)" in item for item in report["warnings"]))
            json.dumps(report, allow_nan=False)

    def test_a_constant_register_move_is_not_described_as_per_word(self):
        # The plain teacher-register move is one transposition, so the per-word
        # caveat must not appear and make a global move look like local editing.
        rate, count = 16000, 16000
        teacher_pitch = track([440] * 100)
        teacher = {"sample_rate": rate, "sample_count": count, "duration": 1.0, "pitch": teacher_pitch}
        plan = {"regions": [{"teacher_start": 0.0, "teacher_end": 1.0, "mode": "synthesize",
                             "unsupported": False, "status": "matched"}],
                "metrics": {"lyric_coverage": 1.0, "status_counts": {"matched": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = 0.2 * np.sin(np.arange(count) * 220 * 2 * np.pi / rate)
            audio = root / "output.wav"
            target = root / "target.npz"
            sf.write(audio, signal, rate, subtype="FLOAT")
            np.savez(target, times=np.arange(100) * 0.01, f0_hz=np.array([440] * 100),
                     voiced=np.array([True] * 100), supported=np.array([True] * 100),
                     pitch_policy=np.array("student_contour_teacher_register"),
                     register_offset_cents=np.float64(-1200.0),
                     register_offset_available=np.bool_(True), note_shift_applied_count=np.int64(0),
                     timing_changed=np.bool_(True))
            with patch("inference.verification._measure_rmvpe", return_value=track([440] * 100)):
                report = verify_output(audio, teacher, plan, target, root=root, device="cpu")
            self.assertFalse(any("approximates the aim" in item for item in report["limitations"]))
            self.assertFalse(any("word(s)" in item for item in report["warnings"]))

    def test_a_student_clock_render_is_measured_against_the_student_length(self):
        # The renderer can keep the student's timeline. Verification must then expect
        # the student's length, not the teacher's, or a correct waveform is reported
        # as a length mismatch and the job fails its structural check.
        rate = 16000
        teacher_count, student_count = 16000, 17000
        teacher = {"sample_rate": rate, "sample_count": teacher_count, "duration": 1.0,
                   "pitch": track([440] * 100)}
        student = {"sample_rate": rate, "sample_count": student_count, "duration": student_count / rate,
                   "pitch": track([220] * 100)}
        plan = {"regions": [{"teacher_start": 0.0, "teacher_end": 1.0, "student_start": 0.0,
                             "student_end": 1.0, "mode": "synthesize", "unsupported": False,
                             "status": "matched"}],
                "metrics": {"lyric_coverage": 1.0, "status_counts": {"matched": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = 0.2 * np.sin(np.arange(student_count) * 220 * 2 * np.pi / rate)
            audio = root / "output.wav"
            target = root / "target.npz"
            sf.write(audio, signal, rate, subtype="FLOAT")
            np.savez(target, times=np.arange(100) * 0.01, f0_hz=np.array([220] * 100),
                     voiced=np.array([True] * 100), supported=np.array([True] * 100),
                     pitch_policy=np.array("student_contour_teacher_register"),
                     register_offset_cents=np.float64(0.0),
                     register_offset_available=np.bool_(True), timing_changed=np.bool_(False))
            with patch("inference.verification._measure_rmvpe", return_value=track([220] * 100)):
                report = verify_output(audio, teacher, plan, target, root=root, device="cpu",
                                       student_analysis=student)
            self.assertEqual(report["provenance"]["output_clock"], "student")
            self.assertEqual(report["audio"]["expected_sample_count"], student_count)
            self.assertTrue(report["audio"]["sample_count_exact"])
            self.assertTrue(report["structural_checks_passed"])
            # The teacher-timeline comparisons cannot be made on this clock, so they
            # are absent rather than silently comparing unrelated positions.
            self.assertNotIn("actual_vs_teacher", report)
            self.assertNotIn("actual_vs_teacher_relative", report)
            self.assertIn("actual_vs_render_target", report)
            self.assertTrue(any("student's timeline" in item for item in report["limitations"]))
            self.assertEqual(report["yin_diagnostic"]["clock"], "student")
            self.assertNotIn("vs_teacher", report["yin_diagnostic"])
            json.dumps(report, allow_nan=False)

    def test_the_teacher_clock_default_is_unchanged(self):
        rate, count = 16000, 16000
        teacher = {"sample_rate": rate, "sample_count": count, "duration": 1.0,
                   "pitch": track([440] * 100)}
        plan = {"regions": [{"teacher_start": 0.0, "teacher_end": 1.0, "mode": "synthesize",
                             "unsupported": False, "status": "matched"}],
                "metrics": {"lyric_coverage": 1.0, "status_counts": {"matched": 1}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = 0.2 * np.sin(np.arange(count) * 440 * 2 * np.pi / rate)
            audio = root / "output.wav"
            target = root / "target.npz"
            sf.write(audio, signal, rate, subtype="FLOAT")
            np.savez(target, times=np.arange(100) * 0.01, f0_hz=np.array([440] * 100),
                     voiced=np.array([True] * 100), supported=np.array([True] * 100),
                     pitch_policy=np.array("teacher_contour_teacher_register"),
                     register_offset_cents=np.float64(0.0),
                     register_offset_available=np.bool_(True), timing_changed=np.bool_(True))
            with patch("inference.verification._measure_rmvpe", return_value=track([440] * 100)):
                report = verify_output(audio, teacher, plan, target, root=root, device="cpu")
            self.assertEqual(report["provenance"]["output_clock"], "teacher")
            self.assertIn("actual_vs_teacher", report)
            self.assertEqual(report["yin_diagnostic"]["clock"], "teacher")
            self.assertIn("vs_teacher", report["yin_diagnostic"])

    def test_the_clock_falls_back_safely(self):
        # Older renders left no clock flag. Treating them as teacher-clock keeps them
        # verifying exactly as before, and a declared student clock is honoured.
        from inference.verification import _target_clock
        rate = 16000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(_target_clock(None))
            legacy = root / "legacy.npz"
            np.savez(legacy, times=np.arange(10) * 0.01, f0_hz=np.zeros(10), voiced=np.zeros(10, dtype=bool))
            self.assertEqual(_target_clock(legacy), "teacher")
            declared = root / "declared.npz"
            np.savez(declared, times=np.arange(10) * 0.01, f0_hz=np.zeros(10),
                     voiced=np.zeros(10, dtype=bool), timing_changed=np.bool_(False))
            self.assertEqual(_target_clock(declared), "student")
            declared_true = root / "teacher.npz"
            np.savez(declared_true, times=np.arange(10) * 0.01, f0_hz=np.zeros(10),
                     voiced=np.zeros(10, dtype=bool), timing_changed=np.bool_(True))
            self.assertEqual(_target_clock(declared_true), "teacher")

    def test_student_clock_verification_requires_the_student_analysis(self):
        # Silently measuring against the wrong length is what caused the failure.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.wav"
            sf.write(path, np.zeros(1000, dtype=np.float32), 16000, subtype="FLOAT")
            with self.assertRaises(ValueError):
                verify_output(path, {"sample_rate": 16000, "sample_count": 1000, "duration": 0.0625},
                              {"regions": []}, None, root=directory, device="cpu", clock="student")

    def test_nonfinite_audio_skips_pitch_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.wav"
            sf.write(path, np.array([0, np.nan, 0], dtype=np.float32), 16000, subtype="FLOAT")
            teacher = {"sample_rate": 16000, "sample_count": 3, "duration": 3 / 16000}
            with patch("inference.verification._measure_rmvpe") as model:
                report = verify_output(path, teacher, {"regions": []}, root=directory, device="cpu")
            self.assertFalse(report["structural_checks_passed"])
            self.assertEqual(report["verification_status"], "invalid_audio")
            model.assert_not_called()
            json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
