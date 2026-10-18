"""CPU checks for the local Japanese text, pronunciation, and alignment path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from inference.analysis import (
    SAMPLE_RATE, _japanese_moras, _japanese_resources, _japanese_semantic_readings,
    _run_alignment, _run_japanese_pronunciation, analyze_pair, atomic_json,
)
from inference.slicer.copyvocal_slicer import AudioChunk


def japanese_resources(root: Path) -> None:
    for relative in ("romajiASR/model.onnx", "romajiASR/phoneme_vocab.json", "kashi-g2p-onnx/model.onnx",
                     "kashi-g2p-onnx/resources/lexicon.txz", "tifa-1.0-onnx/dictionaries/japanese_dict_full.txt"):
        path = root / "models" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Resource identity fixture, not a model.")


def track_fixture(root: Path, label: str, text: str = "君"):
    chunk = AudioChunk(0, 0, SAMPLE_RATE, 0, SAMPLE_RATE, SAMPLE_RATE)
    directory = root / label
    directory.mkdir()
    wav = directory / "chunk_0.wav"
    wav.touch()
    atomic_json(wav.with_suffix(".asr.json"), {"text": text})
    return SimpleNamespace(label=label, active=[chunk], chunks=[chunk], wav=lambda value: wav,
                           warnings=[], audio=np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1)


class JapaneseResourceTests(unittest.TestCase):
    def test_requires_project_local_japanese_models(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "romajiASR"):
                _japanese_resources(Path(directory))

    def test_accepts_physical_bundles_and_rejects_missing_lexicon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            japanese_resources(root)
            paths = _japanese_resources(root)
            self.assertEqual(paths["japanese_g2p"], root / "models/kashi-g2p-onnx")
            (paths["japanese_g2p"] / "resources/lexicon.txz").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "lexicon.txz"):
                _japanese_resources(root)

    def test_rejects_hard_linked_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            japanese_resources(root)
            source = root / "original.onnx"
            source.write_bytes(b"Fixture")
            destination = root / "models/romajiASR/model.onnx"
            destination.unlink()
            os.link(source, destination)
            with self.assertRaisesRegex(ValueError, "hard links"):
                _japanese_resources(root)

    def test_rejects_unsupported_language_before_loading_models(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Analysis language"):
                analyze_pair("unused.wav", "unused.wav", directory, directory, language="invalid")


class JapanesePronunciationTests(unittest.TestCase):
    def test_preserves_nasal_sokuon_yoon_devoiced_vowels_and_long_vowel(self):
        tokens, moras = _japanese_moras({"phonemes": ["k", "o", "N", "n", "i", "cl", "ky", "a", "a", "U", "SP"]})
        self.assertEqual(moras, ["ko", "n", "ni", "cl", "kya", "a", "u"])
        self.assertIn("N", tokens)
        self.assertIn("cl", tokens)

    def test_drops_only_explicit_nonlexical_tokens(self):
        tokens, moras = _japanese_moras({"phonemes": ["<blank>", "AP", "a", "EP", "sil", "br"]})
        self.assertEqual(tokens, ["AP", "a", "EP", "SP", "SP"])
        self.assertEqual(moras, ["a"])

    def test_rejects_missing_phoneme_stream(self):
        with self.assertRaisesRegex(RuntimeError, "phoneme token stream"):
            _japanese_moras({"text": "k i m i"})

    def test_semantic_candidates_are_untimed_and_not_lexical_certification(self):
        from inference.TiFA.g2p.converters.base import G2PGroup, G2PReading, G2PWord
        candidate = [G2PGroup("き", ["k", "i"]), G2PGroup("み", ["m", "i"])]
        pipeline = Mock()
        pipeline.convert.return_value = [G2PWord("君", readings=[G2PReading(paths=[candidate, candidate])])]
        result = _japanese_semantic_readings("君", pipeline)
        pipeline.convert.assert_called_once_with("君", languages=["ja"])
        self.assertEqual(result[0]["surface"], "君")
        self.assertEqual(len(result[0]["candidates"]), 1)
        self.assertEqual([mora["romaji"] for mora in result[0]["candidates"][0]], ["ki", "mi"])
        self.assertNotIn("start", result[0])
        self.assertNotIn("identity_basis", result[0])

    def test_dual_evidence_keeps_independent_student_lyrics_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracks = [track_fixture(root, "teacher", "君"), track_fixture(root, "student", "海")]
            model = Mock()
            model.transcribe.side_effect = [[{"phonemes": ["k", "i", "m", "i"]}],
                                           [{"phonemes": ["u", "m", "i"]}]]
            with patch("inference.romaji_asr.runtime.RomajiASROnnxModel.from_model_path", return_value=model) as load, \
                 patch("inference.ja_onnx_g2p.get_ja_onnx_runtime", return_value=object()), \
                 patch("inference.TiFA.g2p_pipeline.build_g2p_pipeline", return_value=Mock()), \
                 patch("inference.analysis._japanese_semantic_readings", return_value=[]) as semantic:
                for _ in range(2):
                    _run_japanese_pronunciation(tracks, root / "models/romajiASR", root / "models/kashi-g2p-onnx",
                                               root / "models/tifa-1.0-onnx", "cpu", None, lambda message: None)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(model.transcribe.call_count, 2)
            self.assertEqual([call.args[0] for call in semantic.call_args_list], ["君", "海"])
            for track, text, alignment in zip(tracks, ["君", "海"], ["ki mi", "u mi"]):
                transcript = json.loads(track.wav(None).with_suffix(".asr.json").read_text(encoding="utf-8"))
                self.assertEqual(transcript["text"], text)
                self.assertEqual(transcript["alignment_text"], alignment)
                self.assertEqual(transcript["asr_engine"], "qwen+romaji")
                self.assertEqual(track.wav(None).with_suffix(".txt").read_text(encoding="utf-8"), alignment)

    def test_missing_contextual_runtime_fails_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = track_fixture(root, "student")
            atomic_json(track.wav(None).with_suffix(".asr.json"), {"text": "君", "mora_tokens": ["ki", "mi"],
                        "alignment_text": "ki mi"})
            with patch("inference.ja_onnx_g2p.get_ja_onnx_runtime", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "contextual G2P"):
                    _run_japanese_pronunciation([track], root, root / "local-kashi", root, "cpu", None, print)

    def test_empty_direct_asr_does_not_force_qwen_lyrics_into_student(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = track_fixture(root, "student")
            model = Mock()
            model.transcribe.return_value = [{"phonemes": []}]
            with patch("inference.romaji_asr.runtime.RomajiASROnnxModel.from_model_path", return_value=model), \
                 patch("inference.ja_onnx_g2p.get_ja_onnx_runtime", return_value=object()), \
                 patch("inference.TiFA.g2p_pipeline.build_g2p_pipeline", return_value=Mock()), \
                 patch("inference.analysis._japanese_semantic_readings", return_value=[]):
                _run_japanese_pronunciation([track], root, root, root, "cpu", None, lambda message: None)
            _run_alignment([track], root, "cpu", "ja", None, lambda message: None, root)
            aligned = json.loads(track.wav(None).with_suffix(".aligned.json").read_text(encoding="utf-8"))
            self.assertEqual(aligned["words"], [])

    def test_alignment_preserves_mora_units_phone_spans_and_kana(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = track_fixture(root, "student", "きゃっ")
            atomic_json(track.wav(None).with_suffix(".asr.json"), {"text": "きゃっ", "alignment_text": "kya cl",
                        "mora_tokens": ["kya", "cl"]})
            phone = lambda text, start, end: SimpleNamespace(text=text, start=start, end=end)
            words = [SimpleNamespace(text="kya", start=0.1, end=0.3,
                                     phonemes=[phone("ky", 0.1, 0.18), phone("a", 0.18, 0.3)]),
                     SimpleNamespace(text="cl", start=0.4, end=0.5, phonemes=[phone("cl", 0.4, 0.5)])]
            model = SimpleNamespace(sessions={})
            predictions = {"chunk_0": (None, 1.0, words)}
            with patch("inference.TiFA.runtime.load_tifa_model", return_value=model), \
                 patch("inference.TiFA.aligner.run_tifa_fa", return_value=(predictions, {"chunk_0": [("kya", "kya"), ("cl", "cl")]})) as align:
                _run_alignment([track], root, "cpu", "ja", None, lambda message: None, root / "local-kashi")
            aligned = json.loads(track.wav(None).with_suffix(".aligned.json").read_text(encoding="utf-8"))["words"]
            self.assertEqual([word["kana"] for word in aligned], ["きゃ", "っ"])
            self.assertEqual([word["text"] for word in aligned], ["kya", "cl"])
            self.assertTrue(all(word["identity_basis"] == "phonetic_mora" for word in aligned))
            self.assertTrue(all(word["unit_kind"] == "mora" for word in aligned))
            self.assertEqual(aligned[1]["start"], 0.4)
            self.assertEqual(aligned[0]["phonemes"][1]["end"], 0.3)
            self.assertEqual(align.call_args.kwargs["kashi_g2p_model_dir"], root / "local-kashi")


if __name__ == "__main__":
    unittest.main()
