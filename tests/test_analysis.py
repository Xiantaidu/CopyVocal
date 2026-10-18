"""CPU checks for analysis ownership, silence, provenance, and UV handling."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import wave

import numpy as np
import soundfile as sf

from inference.analysis import _FingerprintIndex, _deduplicate_words, analyze_pair, atomic_json, load_analysis, pitch_from_result
from inference.slicer.copyvocal_slicer import downmix_safe, slice_audio


class SlicerTests(unittest.TestCase):
    def test_hard_maximum_without_any_silence(self):
        rate = 1000
        audio = np.ones(69341, dtype=np.float32)
        chunks = slice_audio(audio, rate, max_seconds=12, min_seconds=3, context_seconds=0.75)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].core_start_sample, 0)
        self.assertEqual(chunks[-1].core_end_sample, len(audio))
        for left, right in zip(chunks, chunks[1:]):
            self.assertEqual(left.core_end_sample, right.core_start_sample)
        for chunk in chunks:
            self.assertLessEqual(chunk.end_sample - chunk.start_sample, 12000)
            self.assertLess(chunk.core_start_sample, chunk.core_end_sample)
            self.assertLessEqual(chunk.start_sample, chunk.core_start_sample)
            self.assertGreaterEqual(chunk.end_sample, chunk.core_end_sample)

    def test_silence_boundary_and_context(self):
        audio = np.ones(24000, dtype=np.float32)
        audio[8500:9100] = 0
        chunks = slice_audio(audio, 1000, max_seconds=12, min_seconds=3, context_seconds=0.5)
        self.assertGreaterEqual(chunks[0].core_end_sample, 8500)
        self.assertLessEqual(chunks[0].core_end_sample, 9100)
        self.assertGreater(chunks[0].end_sample, chunks[1].start_sample)

    def test_empty_and_all_silent(self):
        self.assertEqual(slice_audio(np.zeros(0), 44100), [])
        chunks = slice_audio(np.zeros(44100 * 65), 44100)
        self.assertEqual(sum(chunk.core_end_sample - chunk.core_start_sample for chunk in chunks), 44100 * 65)
        self.assertTrue(all(chunk.end - chunk.start <= 18.0 for chunk in chunks))

    def test_phase_cancellation_selects_a_real_channel(self):
        signal = np.sin(np.arange(1000) / 10).astype(np.float32)
        mono, warnings = downmix_safe(np.stack((signal, -signal), axis=1))
        np.testing.assert_array_equal(mono, signal)
        self.assertTrue(warnings)

    def test_invalid_audio_and_bounds(self):
        with self.assertRaises(ValueError):
            downmix_safe(np.asarray([np.nan]))
        with self.assertRaises(ValueError):
            slice_audio(np.ones(20), 10, max_seconds=1, context_seconds=0.6)


class AnalysisContractTests(unittest.TestCase):
    def test_uv_is_not_interpolated(self):
        result = SimpleNamespace(midi_pitch=np.array([69, 70, 71, 72, 73]),
                                 voiced_mask=np.array([True, False, False, True, False]), time_step_seconds=0.01)
        pitch = pitch_from_result(result, 0.041)
        self.assertEqual(pitch["voiced"], [True, False, False, True, False])
        self.assertEqual(pitch["f0_hz"][0], 440)
        self.assertEqual(pitch["f0_hz"][1:3], [0, 0])
        self.assertEqual(pitch["f0_hz"][-1], 0)
        self.assertFalse(pitch["uv_interpolated"])

    def test_missing_voicing_is_an_error(self):
        with self.assertRaises(RuntimeError):
            pitch_from_result(SimpleNamespace(midi_pitch=np.array([69]), voiced_mask=None), 1)

    def test_context_duplicate_keeps_best_span(self):
        words = [{"text": "a", "start": 1.0, "end": 1.4, "quality": 0.7},
                 {"text": "a", "start": 1.02, "end": 1.42, "quality": 1.0},
                 {"text": "b", "start": 1.5, "end": 1.7, "quality": 1.0}]
        result = _deduplicate_words(words, [])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["start"], 1.02)

    def test_atomic_json_and_content_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "data.json"
            atomic_json(path, {"valid": True})
            self.assertEqual(json.loads(path.read_text()), {"valid": True})
            with self.assertRaises(ValueError):
                atomic_json(path, {"invalid": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"valid": True})
            index = _FingerprintIndex(root)
            first = index.file(path)
            atomic_json(path, {"valid": False})
            second = index.file(path)
            self.assertNotEqual(first, second)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_tifa_raw_spans_preserve_gaps_and_reject_degenerate_tokens(self):
        from inference.TiFA.aligner import _build_word_list
        vocabulary = SimpleNamespace(decode=lambda token: ["zh/a"])
        lexicon = [[{"phonemes": ["zh/a"], "scripts": []}] for _ in range(3)]
        words, _ = _build_word_list(
            np.array([[1, 3], [5, 7], [8, 8]]), np.array([1, 1, 1]), np.array([1, 2, 3]),
            np.array([1, 2, 3]), np.array([1, 1, 1]), lexicon, ["a", "b", "c"], vocabulary,
            "zh", 0.01, 0.1, preserve_gaps=True, raw_spans=True,
        )
        self.assertEqual(len(words), 2)
        self.assertEqual(words[0].end, 0.03)
        self.assertEqual(words[1].start, 0.05)
        self.assertTrue(words.raw_alignment_issues)

    def test_no_stage_run_never_claims_model_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model in ("Qwen3-ASR-1.7B-dml", "tifa-1.0-onnx", "RMVPE"):
                folder = root / "models" / model
                folder.mkdir(parents=True)
                (folder / "fixture.txt").write_text("Fixture identity only; no model inference.")
            for relative in ("inference/slicer/copyvocal_slicer.py", "inference/TiFA/aligner.py",
                             "inference/TiFA/runtime.py", "inference/TiFA/decoding.py", "inference/TiFA/g2p_pipeline.py",
                             "inference/API/asr_api.py", "inference/API/rmvpe_api.py", "inference/API/game_api.py",
                             "inference/device_utils.py"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# Fingerprint fixture.\n")
            signal = 0.1 * np.sin(np.arange(8000, dtype=np.float32) * (440 * 2 * np.pi / 16000))
            teacher, student = root / "teacher.wav", root / "student.wav"
            sf.write(teacher, signal, 16000)
            sf.write(student, -signal, 16000)
            left, right = analyze_pair(teacher, student, root / "outputs", root,
                                       stages=(), device="cpu", include_game=False)
            self.assertEqual(left["completed_stages"], [])
            self.assertEqual(right["words"], [])
            self.assertEqual(left["sample_count"], 22050)
            self.assertNotEqual(left["input_sha256"], right["input_sha256"])
            self.assertTrue(left["warnings"])
            saved = load_analysis(root / "outputs" / "teacher.analysis.json")
            self.assertEqual(saved["input_sha256"], left["input_sha256"])
            self.assertTrue((root / "outputs" / "teacher.TextGrid").is_file())
            for chunk in (root / "cache" / "analysis").rglob("chunk_*.wav"):
                with wave.open(str(chunk), "rb") as handle:
                    self.assertEqual(handle.getsampwidth(), 2)
                    self.assertEqual(handle.getframerate(), 44100)
                self.assertEqual(sf.info(chunk).subtype, "PCM_16")


if __name__ == "__main__":
    unittest.main()
