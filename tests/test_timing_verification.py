"""CPU tests for measured timing, lexical exclusions, and model input isolation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from inference.timing_verification import compare_word_timing, diagnose_word_audio, verify_timing


def analysis(texts, spans=None, *, duration=2.0, quality=1.0):
    spans = spans or [(0.1 + index * 0.3, 0.3 + index * 0.3) for index in range(len(texts))]
    return {"source": "fixture.wav", "duration": duration, "sample_rate": 44100,
            "sample_count": int(duration * 44100), "completed_stages": ["lyrics"],
            "words": [{"text": text, "start": start, "end": end, "quality": quality, "issues": []}
                      for text, (start, end) in zip(texts, spans)], "pitch": {}, "warnings": []}


class WordTimingTests(unittest.TestCase):
    def test_exact_tolerance_is_inclusive_despite_float_subtraction(self):
        teacher = analysis(["one"], [(100., 100.2)], duration=110)
        exact = compare_word_timing(teacher, analysis(["one"], [(100.05, 100.25)], duration=110))
        self.assertEqual(exact["coverage"]["reliable_words_within_tolerance_count"], 1)
        outside = compare_word_timing(teacher, analysis(["one"], [(100.051, 100.251)], duration=110))
        self.assertEqual(outside["coverage"]["reliable_words_within_tolerance_count"], 0)

    def test_measured_shift_remains_visible_without_rewarping(self):
        teacher = analysis(["one", "two"], [(0.1, 0.3), (0.4, 0.7)])
        measured = analysis(["one", "two"], [(0.12, 0.34), (0.42, 0.74)])
        before = copy.deepcopy((teacher, measured))
        report = compare_word_timing(teacher, measured)
        self.assertEqual(report["coverage"]["reliable_exact_match_count"], 2)
        self.assertEqual(report["coverage"]["reliable_teacher_unit_coverage"], 1)
        self.assertAlmostEqual(report["reliable_boundary_errors"]["onset"]["mean_signed_ms"], 20)
        self.assertAlmostEqual(report["reliable_boundary_errors"]["offset"]["mean_signed_ms"], 40)
        self.assertAlmostEqual(report["reliable_boundary_errors"]["duration"]["mean_signed_ms"], 20)
        self.assertEqual(report["measurement"]["correspondence_time_prior"], "disabled")
        self.assertEqual((teacher, measured), before)
        json.dumps(report, allow_nan=False)

    def test_missing_units_stay_in_full_teacher_denominator(self):
        report = compare_word_timing(analysis(["one", "two", "three"]), analysis(["one", "three"], [(0.1, 0.3), (0.7, 0.9)]))
        self.assertEqual(report["coverage"]["relation_counts"]["missing"], 1)
        self.assertAlmostEqual(report["coverage"]["reliable_teacher_unit_coverage"], 2 / 3)
        self.assertAlmostEqual(report["coverage"]["within_tolerance_ratio_of_all_teacher_units"], 2 / 3)
        self.assertEqual(report["coverage"]["within_tolerance_ratio_of_reliable_matches"], 1)

    def test_repeated_missing_chorus_does_not_claim_precise_identity(self):
        report = compare_word_timing(analysis(["one", "two", "one", "two"]), analysis(["one", "two"]))
        self.assertGreater(report["coverage"]["ambiguous_paired_unit_count"], 0)
        self.assertEqual(report["coverage"]["reliable_exact_match_count"], 0)
        self.assertIsNone(report["reliable_boundary_errors"]["onset"]["mean_absolute_ms"])
        self.assertTrue(any("ambiguous_repeated_lyric_correspondence" in item["exclusion_reasons"] for item in report["word_observations"]))

    def test_substitution_homophone_and_estimated_boundaries_are_excluded(self):
        teacher = analysis(["here", "hello", "我爱你"])
        measured = analysis(["hear", "world", "我爱你"])
        teacher["words"][0]["pinyin"] = "same"
        measured["words"][0]["pinyin"] = "same"
        report = compare_word_timing(teacher, measured)
        self.assertEqual(report["coverage"]["relation_counts"]["homophone"], 1)
        self.assertEqual(report["coverage"]["relation_counts"]["substitution"], 1)
        self.assertEqual(report["coverage"]["reliable_exact_match_count"], 0)
        self.assertTrue(any("teacher_estimated_internal_boundaries" in item["exclusion_reasons"] for item in report["word_observations"]))

    def test_low_quality_and_clipped_overlaps_are_excluded(self):
        teacher = analysis(["one", "two"], [(0.1, 0.4), (0.4, 0.7)])
        measured = analysis(["one", "two"], [(0.1, 0.5), (0.4, 0.7)])
        measured["words"][0]["quality"] = 0.2
        report = compare_word_timing(teacher, measured)
        self.assertEqual(report["coverage"]["reliable_exact_match_count"], 0)
        self.assertIn("output_normalized_or_clipped_boundaries", report["word_observations"][1]["exclusion_reasons"])

    def test_valid_completed_chunk_survives_other_incomplete_chunks(self):
        teacher = analysis(["one"])
        measured = analysis(["one"])
        measured["completed_stages"] = []
        measured["chunks"] = [{"core_start": 0, "core_end": 1, "alignment_complete": True},
                              {"core_start": 1, "core_end": 2, "alignment_complete": False}]
        report = compare_word_timing(teacher, measured)
        self.assertEqual(report["coverage"]["reliable_exact_match_count"], 1)
        self.assertFalse(report["measurement"]["output_lyric_analysis_complete"])

    def test_empty_output_has_no_accuracy_score(self):
        report = compare_word_timing(analysis(["one"]), analysis([]))
        self.assertEqual(report["verification_status"], "insufficient_reliable_word_observations")
        self.assertEqual(report["coverage"]["reliable_teacher_unit_coverage"], 0)
        self.assertIsNone(report["reliable_boundary_errors"]["onset"]["mean_absolute_ms"])
        self.assertFalse(report["measurement"]["output_word_boundaries_remeasured"])

    def test_model_receives_exported_audio_and_own_text_only(self):
        teacher, measured = analysis(["teachersecret"]), analysis(["outputword"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planner = root / "inference" / "alignment" / "planner.py"
            planner.parent.mkdir(parents=True)
            planner.write_text("# Fixture code fingerprint.\n")
            audio = root / "corrected.wav"
            sf.write(audio, np.zeros(16000 * 2), 16000)
            with patch("inference.analysis.analyze_pair", return_value=(measured, measured)) as analyze:
                report = verify_timing(audio, teacher, root / "results", root=root, device="cpu")
            call = analyze.call_args
            self.assertEqual(call.args[0], audio)
            self.assertEqual(call.args[1], audio)
            self.assertNotIn("teachersecret", str(call))
            self.assertEqual(call.kwargs["stages"], ("lyrics",))
            self.assertFalse(call.kwargs["include_game"])
            self.assertFalse(report["measurement"]["teacher_boundaries_supplied_to_models"])
            self.assertTrue((root / "results" / "timing.verification.json").is_file())
            self.assertTrue((root / "results" / "output.analysis.json").is_file())
            json.dumps(report, allow_nan=False)

    def test_tifa_boundary_inside_silence_is_flagged_without_hiding_error(self):
        teacher = analysis(["one"], [(0.2, 0.4)])
        measured = analysis(["one"], [(0.1, 0.4)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.wav"
            audio = np.zeros(16000)
            audio[3200:6400] = 0.1
            sf.write(path, audio, 16000)
            annotated = diagnose_word_audio(path, measured)
        report = compare_word_timing(teacher, annotated)
        self.assertEqual(report["coverage"]["reliable_exact_match_count"], 1)
        self.assertAlmostEqual(report["reliable_boundary_errors"]["onset"]["mean_signed_ms"], -100)
        self.assertEqual(report["boundary_diagnostics"]["words_with_boundary_in_long_exact_silence"], 1)
        self.assertFalse(report["boundary_diagnostics"]["diagnostics_change_error_statistics"])
        self.assertNotIn("boundary_audio", measured["words"][0])


if __name__ == "__main__":
    unittest.main()
