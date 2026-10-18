"""Low-cost checks for timing, UV handling, and unsupported-word behavior."""

import unittest
import tempfile
from pathlib import Path
import numpy as np

from inference.correction import render_student, sample_pitch
from inference.correction.renderer import _region_map
from inference.vocoder import PcNsfVocoder


class RecordingVocoder:
    sample_rate = 44100
    hop_size = 512
    log_floor = 1e-5

    def mel(self, audio):
        count = (len(audio) + self.hop_size - 1) // self.hop_size
        return np.tile(np.arange(count, dtype=np.float32), (128, 1))

    def synthesize(self, mel, f0_hz, sample_count=None, **kwargs):
        self.last_mel, self.last_f0 = mel.copy(), f0_hz.copy()
        return np.full(sample_count, 0.25, dtype=np.float32)


def analysis(count, f0=220):
    times = (np.arange((count + 511) // 512) + 0.5) * 512 / 44100
    return {
        "sample_rate": 44100, "sample_count": count,
        "pitch": {"times": times.tolist(), "f0_hz": np.full(len(times), f0).tolist(), "voiced": np.full(len(times), f0 > 0).tolist()},
    }


class PitchSamplingTests(unittest.TestCase):
    def test_voiced_islands_never_bridge_silence(self):
        pitch = {"times": [0, 0.01, 0.02, 0.03, 0.04], "f0_hz": [100, 100, 0, 400, 400], "voiced": [True, True, False, True, True]}
        f0, voiced = sample_pitch(pitch, np.array([0.005, 0.014, 0.02, 0.026, 0.035]))
        np.testing.assert_allclose(f0, [100, 100, 0, 400, 400], rtol=1e-6)
        self.assertEqual(voiced.tolist(), [True, True, False, True, True])

    def test_octave_interpolation_uses_log_frequency(self):
        f0, _ = sample_pitch({"times": [0, 1], "f0_hz": [100, 400], "voiced": [True, True]}, np.array([0.5]))
        self.assertAlmostEqual(float(f0[0]), 200, places=4)

    def test_nan_and_out_of_range_are_unvoiced(self):
        f0, voiced = sample_pitch({"times": [0.01, 0.02], "f0_hz": [220, 220], "voiced": [True, True]}, np.array([np.nan, -1, 1]))
        self.assertFalse(voiced.any())
        self.assertFalse(f0.any())


class RendererTests(unittest.TestCase):
    def test_local_anchors_do_not_interpolate_across_missing_word(self):
        regions = [
            {"teacher_start": 0, "teacher_end": 1, "student_start": 0, "student_end": 2, "mode": "synthesize"},
            {"teacher_start": 1, "teacher_end": 2, "mode": "mute"},
            {"teacher_start": 2, "teacher_end": 3, "student_start": 4, "student_end": 5, "mode": "synthesize"},
        ]
        mapped, valid = _region_map(regions, np.array([0.5, 1.5, 2.5]), inverse=False, allow_fallback=False, minimum_quality=0)
        np.testing.assert_allclose(mapped[[0, 2]], [1, 4.5])
        self.assertTrue(np.isnan(mapped[1]))
        self.assertEqual(valid.tolist(), [True, False, True])

    def test_absolute_teacher_pitch_is_opt_in_and_mutes_missing(self):
        vocoder = RecordingVocoder()
        student, teacher = analysis(88200, 220), analysis(44101, 440)
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": 0.4, "student_start": 0.1, "student_end": 0.7, "mode": "synthesize"},
            {"teacher_start": 0.4, "teacher_end": 0.6, "mode": "mute", "status": "missing"},
            {"teacher_start": 0.6, "teacher_end": 1, "student_start": 1.1, "student_end": 1.8, "mode": "synthesize"},
        ]}
        # Teacher mode with the contour is the teacher's contour in the teacher's
        # register: the historical absolute behavior.
        wave, diagnostics = render_student(np.ones(88200, dtype=np.float32), teacher, student, plan,
                                          vocoder, pitch_mode="teacher", map_teacher_pitch=True)
        self.assertEqual(len(wave), 44101)
        self.assertTrue(np.all(wave[17640:26460] == 0))
        self.assertEqual(diagnostics["pitch_policy"], "teacher_contour_teacher_register")
        self.assertAlmostEqual(float(np.median(vocoder.last_f0[vocoder.last_f0 > 0])), 440, places=3)
        self.assertEqual(diagnostics["muted_region_count"], 1)

    def test_the_default_only_changes_the_timing_and_keeps_the_students_pitch(self):
        # The pitch switch is off by default, so even a student who sings an
        # octave below the teacher is left exactly as sung. Changing only the
        # timing must not silently correct the pitch.
        vocoder = RecordingVocoder()
        student, teacher = analysis(88200, 220), analysis(44101, 440)
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": 0.4, "student_start": 0.1, "student_end": 0.7, "mode": "synthesize"},
            {"teacher_start": 0.4, "teacher_end": 0.6, "mode": "mute", "status": "missing"},
            {"teacher_start": 0.6, "teacher_end": 1, "student_start": 1.1, "student_end": 1.8, "mode": "synthesize"},
        ]}
        wave, diagnostics = render_student(np.ones(88200, dtype=np.float32), teacher, student, plan, vocoder)
        self.assertEqual(len(wave), 44101)
        self.assertTrue(np.all(wave[17640:26460] == 0))
        self.assertEqual(diagnostics["pitch_policy"], "none")
        self.assertFalse(diagnostics["register_pitch"]["applied"])
        self.assertEqual(diagnostics["register_pitch"]["contour_shift_cents"], 0.0)
        self.assertEqual(diagnostics["register_pitch"]["region_shifts_cents"], [0.0, 0.0])
        self.assertAlmostEqual(float(np.median(vocoder.last_f0[vocoder.last_f0 > 0])), 220, places=3)
        self.assertEqual(diagnostics["muted_region_count"], 1)

    def test_adaptive_placement_keeps_the_students_register(self):
        # Adaptive mode puts each word on the teacher's note level but keeps the
        # melody where the student actually sings, an octave below the teacher.
        vocoder = RecordingVocoder()
        student, teacher = analysis(88200, 220), analysis(44101, 440)
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": 0.4, "student_start": 0.1, "student_end": 0.7, "mode": "synthesize"},
            {"teacher_start": 0.4, "teacher_end": 0.6, "mode": "mute", "status": "missing"},
            {"teacher_start": 0.6, "teacher_end": 1, "student_start": 1.1, "student_end": 1.8, "mode": "synthesize"},
        ]}
        wave, diagnostics = render_student(np.ones(88200, dtype=np.float32), teacher, student, plan, vocoder,
                                          pitch_mode="adaptive")
        self.assertEqual(len(wave), 44101)
        self.assertEqual(diagnostics["pitch_policy"], "student_contour_student_register")
        self.assertEqual(diagnostics["register_pitch"]["placement"], "per_word")
        self.assertTrue(diagnostics["register_pitch"]["offset_available"])
        self.assertAlmostEqual(diagnostics["register_pitch"]["offset_cents"], -1200.0, delta=1.0)
        # The student already sings the right note an octave down, so no word moves.
        self.assertEqual(diagnostics["register_pitch"]["region_shifts_cents"], [0.0, 0.0])
        self.assertEqual(diagnostics["register_pitch"]["note_shift_applied_count"], 0)
        self.assertAlmostEqual(float(np.median(vocoder.last_f0[vocoder.last_f0 > 0])), 220, places=3)

    def test_full_does_not_voice_student_consonants(self):
        vocoder = RecordingVocoder()
        student, teacher = analysis(44100, 0), analysis(44100, 440)
        plan = {"regions": [{"teacher_start": 0, "teacher_end": 1, "student_start": 0, "student_end": 1, "mode": "synthesize"}]}
        render_student(np.ones(44100, dtype=np.float32), teacher, student, plan, vocoder)
        self.assertFalse(vocoder.last_f0.any())

    def test_target_pitch_export_is_compact_and_reusable(self):
        vocoder = RecordingVocoder()
        student, teacher = analysis(44100), analysis(44100, 440)
        plan = {"regions": [{"teacher_start": 0, "teacher_end": 1, "student_start": 0, "student_end": 1, "mode": "synthesize"}]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target.npz"
            _, report = render_student(np.ones(44100, dtype=np.float32), teacher, student, plan, vocoder,
                                       target_pitch_path=target, pitch_mode="teacher",
                                       map_teacher_pitch=True)
            self.assertEqual(report["target_pitch_path"], str(target.resolve()))
            with np.load(target) as saved:
                self.assertEqual(saved["f0_hz"].dtype, np.float32)
                self.assertEqual(saved["voiced"].dtype, np.bool_)
                self.assertTrue(np.all(saved["f0_hz"][saved["voiced"]] == 440))
                # The policy is exported with the target, so verification can tell
                # which pitch reference the renderer aimed at.
                self.assertEqual(str(saved["pitch_policy"]), "teacher_contour_teacher_register")

    def test_experiments_keep_intended_output_timeline(self):
        vocoder = RecordingVocoder()
        student, teacher = analysis(88200), analysis(44100)
        plan = {"regions": [{"teacher_start": 0, "teacher_end": 1, "student_start": 0, "student_end": 2, "mode": "synthesize"}]}
        for mode in ("identity", "pitch_only", "time_only", "full"):
            wave, _ = render_student(np.ones(88200, dtype=np.float32), teacher, student, plan, vocoder, mode=mode)
            self.assertEqual(len(wave), 44100 if mode in {"time_only", "full"} else 88200)

    def test_phase_prefix_matches_source_trajectory(self):
        from types import SimpleNamespace
        vocoder = object.__new__(PcNsfVocoder)
        vocoder.model = SimpleNamespace(mini_nsf=True, upp=64, source_sr=5512.5)
        f0 = np.array([220, 330, 440, 0, 220], dtype=np.float32)
        prefix = vocoder._phase_prefix(f0)
        increments = []
        for i, frequency in enumerate(f0):
            following = f0[min(i + 1, len(f0) - 1)]
            steps = np.arange(1, 65)
            rad = frequency / 5512.5 * steps + 0.5 * (following - frequency) / 5512.5 * steps * (steps - 1) / 64
            increments.append(rad[-1])
        np.testing.assert_allclose(prefix, np.r_[0, np.cumsum(increments)] % 1, atol=1e-12)


class MelContractTests(unittest.TestCase):
    def test_feature_contract_matches_reference_complete_frames(self):
        try:
            import torch
            from librosa.filters import mel as mel_filter
        except ImportError:
            self.skipTest("Torch and librosa are required for the mel feature contract check.")
        vocoder = object.__new__(PcNsfVocoder)
        vocoder.sample_rate, vocoder.hop_size = 44100, 512
        vocoder.n_fft = vocoder.win_size = 2048
        vocoder.n_mels, vocoder.fmin, vocoder.fmax = 128, 40, 16000
        vocoder.log_floor = 1e-5
        vocoder._mel_basis = vocoder._window = None
        generator = np.random.default_rng(1234)
        audio = generator.normal(0, 0.1, size=16123).astype(np.float32)
        observed = vocoder.mel(audio)
        y = torch.from_numpy(audio)[None, None]
        y = torch.nn.functional.pad(y, (768, 768), mode="reflect").squeeze(1)
        spec = torch.stft(y, 2048, hop_length=512, win_length=2048, window=torch.hann_window(2048), center=False, return_complex=True).abs()[0]
        bank = torch.from_numpy(mel_filter(sr=44100, n_fft=2048, n_mels=128, fmin=40, fmax=16000))
        reference = torch.log(torch.clamp(bank @ spec, min=1e-5)).numpy()
        self.assertEqual(observed.shape, (128, 32))
        np.testing.assert_allclose(observed[:, :reference.shape[1]], reference, rtol=0, atol=1e-6)

    def test_subhop_audio_retains_one_frame(self):
        try:
            import torch
            import librosa
        except ImportError:
            self.skipTest("Torch and librosa are required for the mel feature contract check.")
        vocoder = object.__new__(PcNsfVocoder)
        vocoder.sample_rate, vocoder.hop_size = 44100, 512
        vocoder.n_fft = vocoder.win_size = 2048
        vocoder.n_mels, vocoder.fmin, vocoder.fmax = 128, 40, 16000
        vocoder.log_floor = 1e-5
        vocoder._mel_basis = vocoder._window = None
        for count in (1, 2, 511, 512):
            mel = vocoder.mel(np.ones(count, dtype=np.float32) * 0.1)
            self.assertEqual(mel.shape, (128, 1))
            self.assertTrue(np.isfinite(mel).all())


if __name__ == "__main__":
    unittest.main()
