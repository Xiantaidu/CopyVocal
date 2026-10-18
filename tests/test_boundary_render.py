import tempfile
import unittest
from pathlib import Path

import numpy as np

from inference.correction.renderer import (
    render_student, _smooth_source_jumps, _context_mel, _reference_release_gain,
)
from test_vocoder import RecordingVocoder, analysis


class BoundaryRenderingTests(unittest.TestCase):
    def test_release_has_mel_context_and_separate_semantic_support(self):
        rate = 44100
        vocoder = RecordingVocoder()
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": .4, "student_start": 0,
             "student_end": .4, "mode": "synthesize"},
            {"teacher_start": .4, "teacher_end": 1, "mode": "silence"},
        ]}
        source = analysis(rate)
        source["words"] = [{"start": 0, "end": .4, "text": "word"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npz"
            wave, report = render_student(np.ones(rate, np.float32), analysis(rate, 440),
                                         source, plan, vocoder, target_pitch_path=path)
            with np.load(path) as saved:
                context = saved["acoustic_context"]
                self.assertTrue(context.any())
                self.assertFalse(saved["supported"][context].any())
                self.assertTrue(saved["rendered"][context].all())
                self.assertTrue(np.all(saved["times"][context] < .521))
            self.assertGreater(float(wave[int(.405 * rate)]), 0)
            self.assertTrue(np.all(wave[int(.521 * rate):] == 0))
            self.assertTrue(np.all(vocoder.last_mel[:, context] >= np.log(vocoder.log_floor)))
            self.assertGreater(report["rendered_sample_ratio"], report["supported_sample_ratio"])

    def test_gap_activity_tail_cannot_enter_a_missing_word(self):
        rate = 44100
        plan = {"regions": [
            {"teacher_start": 0, "teacher_end": .4, "student_start": 0,
             "student_end": .4, "mode": "synthesize"},
            {"teacher_start": .4, "teacher_end": .45, "mode": "mute",
             "reason": "voiced_teacher_region_without_lyric_correspondence"},
            {"teacher_start": .45, "teacher_end": .6, "mode": "mute", "status": "missing"},
            {"teacher_start": .6, "teacher_end": 1, "student_start": .6,
             "student_end": 1, "mode": "synthesize"},
        ]}
        vocoder = RecordingVocoder()
        wave, _ = render_student(np.ones(rate, np.float32), analysis(rate), analysis(rate), plan, vocoder)
        self.assertGreater(float(wave[int(.41 * rate)]), 0)
        self.assertTrue(np.all(wave[int(.45 * rate):int(.6 * rate)] == 0))

    def test_only_discontinuous_source_edits_get_blended(self):
        times = (np.arange(100) + .5) * .01
        original = np.tile(np.where(times < .5, -2., -8.), (2, 1)).astype(np.float32)
        left = {"teacher_start": 0, "teacher_end": .5, "student_start": 0,
                "student_end": .5, "mode": "synthesize"}
        right = {"teacher_start": .5, "teacher_end": 1, "student_start": .5,
                 "student_end": 1, "mode": "synthesize"}
        continuous = original.copy()
        self.assertEqual(_smooth_source_jumps(continuous, times, [left, right], False, 0), [])
        np.testing.assert_array_equal(original, continuous)
        right.update(student_start=.7, student_end=1.2)
        edited = original.copy()
        transitions = _smooth_source_jumps(edited, times, [left, right], False, 0)
        self.assertEqual(len(transitions), 1)
        self.assertLess(float(np.max(np.abs(np.diff(edited[0])))), 6)
        np.testing.assert_array_equal(original[:, times < .48], edited[:, times < .48])
        np.testing.assert_array_equal(original[:, times > .52], edited[:, times > .52])

    def test_context_feature_window_cannot_borrow_next_lexical_word(self):
        class SourceCheckingVocoder(RecordingVocoder):
            win_size = 2048

            def mel(self, audio):
                self.asserted_source = np.asarray(audio).copy()
                return np.zeros((128, (len(audio) + 511) // 512), dtype=np.float32)

        rate = 44100
        audio = np.zeros(rate, dtype=np.float32)
        audio[int(.4 * rate):] = 10
        vocoder = SourceCheckingVocoder()
        regions = [{"student_start": .1, "student_end": .4}]
        contexts = [{"kind": "release", "teacher_start": .4, "teacher_end": .48,
                     "source_strategy": "hold", "source_region_index": 0,
                     "student_limit_start": .394, "student_limit_end": .394}]
        times, source = np.array([.41]), np.array([.394])
        mel = np.zeros((128, 1), np.float32)
        _context_mel(audio, vocoder, contexts, regions, times, source, np.array([False]), mel)
        self.assertGreater(len(vocoder.asserted_source), 0)
        self.assertEqual(float(np.max(np.abs(vocoder.asserted_source))), 0)

    def test_held_release_follows_reference_decay_without_changing_word_gain(self):
        rate = 44100
        teacher = np.ones(rate, dtype=np.float32)
        teacher[int(.4 * rate):] = .01
        times = np.array([.35, .399, .415, .43, .5])
        contexts = [{"kind": "release", "source_strategy": "hold", "boundary": .4,
                     "teacher_start": .4, "teacher_end": .44}]
        gain = _reference_release_gain(contexts, times, teacher, rate)
        np.testing.assert_allclose(gain[[0, 1, 4]], 1)
        np.testing.assert_allclose(gain[[2, 3]], .01, rtol=1e-5)
        constant = _reference_release_gain(contexts, times, np.ones(rate, np.float32), rate)
        np.testing.assert_allclose(constant, 1)


if __name__ == "__main__":
    unittest.main()
