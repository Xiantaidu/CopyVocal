from pathlib import Path
import tempfile
import unittest

import numpy as np

from inference.correction.renderer import render_student
from test_vocoder import RecordingVocoder, analysis


class ContinuousRenderTests(unittest.TestCase):
    def test_true_quiet_transition_attenuates_synthesis_without_a_new_mute(self):
        rate = 44100
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": .4, "student_start": 0, "student_end": .4,
             "teacher_anchors": [0, .2, .4], "student_anchors": [0, .2, .4],
             "teacher_index": 0, "student_index": 0, "mode": "synthesize"},
            {"teacher_start": .4, "teacher_end": .6, "mode": "silence"},
            {"teacher_start": .6, "teacher_end": 1, "student_start": .4, "student_end": 1,
             "teacher_anchors": [.6, .8, 1], "student_anchors": [.4, .7, 1],
             "teacher_index": 1, "student_index": 1, "mode": "synthesize"},
        ]}
        source = np.full(rate, .05, dtype=np.float32)
        reference = source.copy()
        reference[int(.4 * rate):int(.6 * rate)] = .0001
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pitch.npz"
            wave, report = render_student(source, analysis(rate, 0), analysis(rate, 0),
                                         plan, RecordingVocoder(), teacher_audio=reference,
                                         target_pitch_path=path)
            core = wave[int(.44 * rate):int(.56 * rate)]
            self.assertTrue(np.all(core > 0))
            self.assertLess(float(np.max(core)), .005)
            self.assertAlmostEqual(float(wave[int(.3 * rate)]), .25)
            self.assertAlmostEqual(float(wave[int(.7 * rate)]), .25)
            self.assertLess(float(np.max(np.abs(np.diff(wave[int(.35 * rate):int(.65 * rate)])))), .01)
            self.assertGreater(report["acoustic_energy_attenuated_frame_count"], 0)
            with np.load(path) as saved:
                gain = saved["acoustic_energy_gain"]
                self.assertTrue(np.all((gain > 0) & (gain <= 1)))
                gap = (saved["times"] > .44) & (saved["times"] < .56)
                self.assertTrue(saved["rendered"][gap].all())
                self.assertFalse(saved["f0_hz"][gap].any())

    def test_unvoiced_gap_uses_advancing_source_and_no_internal_mute(self):
        rate = 44100
        teacher, student = analysis(rate, 0), analysis(rate, 0)
        teacher["words"] = [{"text": "a", "start": 0, "end": .4},
                            {"text": "b", "start": .6, "end": 1}]
        student["words"] = [{"text": "a", "start": 0, "end": .5},
                            {"text": "b", "start": .5, "end": 1}]
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": .4, "student_start": 0, "student_end": .5,
             "teacher_anchors": [0, .2, .4], "student_anchors": [0, .25, .5],
             "teacher_index": 0, "student_index": 0, "mode": "synthesize"},
            {"teacher_start": .4, "teacher_end": .6, "mode": "silence",
             "status": "unvoiced_gap"},
            {"teacher_start": .6, "teacher_end": 1, "student_start": .5, "student_end": 1,
             "teacher_anchors": [.6, .8, 1], "student_anchors": [.5, .75, 1],
             "teacher_index": 1, "student_index": 1, "mode": "synthesize"},
        ]}
        signal = np.random.default_rng(33).normal(0, .03, rate).astype(np.float32)
        vocoder = RecordingVocoder()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pitch.npz"
            wave, report = render_student(signal, teacher, student, plan, vocoder,
                                         teacher_audio=signal, target_pitch_path=path)
            self.assertTrue(np.all(wave[int(.41 * rate):int(.59 * rate)] > 0))
            self.assertFalse(vocoder.last_f0.any())
            self.assertEqual(len(report["acoustic_transitions"]), 1)
            with np.load(path) as saved:
                gap = (saved["times"] >= .4) & (saved["times"] < .6)
                self.assertTrue(saved["rendered"][gap].all())
                self.assertFalse(saved["supported"][gap].any())
                self.assertTrue(np.all(np.diff(saved["source_times"][gap]) > 0))

    def test_teacher_audio_does_not_enable_a_missing_word(self):
        rate = 44100
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": .4, "student_start": 0, "student_end": .4,
             "mode": "synthesize", "teacher_index": 0, "student_index": 0},
            {"teacher_start": .4, "teacher_end": .6, "mode": "mute", "status": "missing"},
            {"teacher_start": .6, "teacher_end": 1, "student_start": .4, "student_end": 1,
             "mode": "synthesize", "teacher_index": 2, "student_index": 1},
        ]}
        waveform = np.ones(rate, dtype=np.float32)
        wave, report = render_student(waveform, analysis(rate), analysis(rate), plan,
                                     RecordingVocoder(), teacher_audio=waveform)
        self.assertTrue(np.all(wave[int(.4 * rate):int(.6 * rate)] == 0))
        self.assertEqual(report["acoustic_transitions"], [])


if __name__ == "__main__":
    unittest.main()
