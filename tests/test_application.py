from __future__ import annotations

import copy
import html
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from application.artifacts import ROOT, file_hash, read_json, write_json
from application.copyvocal import run_job
from application.report import write_report
from inference.alignment import build_alignment
from inference.timing_verification import compare_word_timing


def analysis(path, *, words=("one", "two"), shift=0.0):
    return {"source": str(path.resolve()), "input_sha256": file_hash(path),
            "sample_rate": 44100, "sample_count": 44100, "duration": 1.0,
            "completed_stages": ["lyrics", "pitch", "notes"], "warnings": [],
            "config": {"device": "cpu", "language": "en", "max_chunk_seconds": 18.0,
                       "context_seconds": 0.75, "include_game": True},
            "words": [{"text": text, "start": 0.1 + i * 0.4 + shift,
                       "end": 0.35 + i * 0.4 + shift, "quality": 1.0, "issues": [], "phonemes": []}
                      for i, text in enumerate(words)],
            "pitch": {"times": [0.15, 0.25, 0.55, 0.65], "f0_hz": [220] * 4, "voiced": [True] * 4},
            "chunks": [], "notes": []}


class PageData(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.polylines, self.audio = [], []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "polyline":
            self.polylines.append(values)
        if tag == "audio":
            self.audio.append(values.get("src"))


class ApplicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.teacher, self.student = self.directory / "teacher.wav", self.directory / "student.wav"
        self.teacher.write_bytes(b"teacher fixture")
        self.student.write_bytes(b"student fixture")
        self.output = self.directory / "output"
        self.output.mkdir()
        self.ta, self.sa = analysis(self.teacher), analysis(self.student)
        self.save_analysis()
        self.calls, self.timing = [], None
        self.structure_passed = True
        self.worker = patch("application.copyvocal.run_worker", side_effect=self.fake_worker)
        self.worker.start()
        self.addCleanup(self.worker.stop)
        fingerprint = patch("application.copyvocal.code_fingerprint", return_value="test-code")
        fingerprint.start()
        self.addCleanup(fingerprint.stop)

    def save_analysis(self):
        for label, value in (("teacher", self.ta), ("student", self.sa)):
            write_json(self.output / f"{label}.analysis.json", value)

    def fake_worker(self, script, arguments, output):
        self.calls.append((script, arguments))
        if script == "analysis_worker.py":
            self.save_analysis()
        elif script == "render_worker.py":
            for name in ("teacher.reference.wav", "student.reference.wav", "student.corrected.wav"):
                (output / name).write_bytes(b"wave fixture")
            write_json(output / "render.json", {"experiments": {"full": {"filename": "student.corrected.wav"}}})
        elif script == "verify_worker.py":
            write_json(output / "verification.measured_pitch.json", {"times": [0.15, 0.25, 0.55, 0.65], "f0_hz": [440] * 4, "voiced": [True] * 4})
            write_json(output / "verification.json", {"structural_checks_passed": self.structure_passed,
                "measured_pitch_path": str(output / "verification.measured_pitch.json"),
                "word_timing_validation": {"measured": False, "reason": "not remeasured"},
                "limitations": ["not remeasured", "timbre remains unverified"], "warnings": []})
        elif script == "timing_verify_worker.py":
            write_json(output / "timing.verification.json", self.timing or compare_word_timing(self.ta, self.ta))
        else:
            raise AssertionError(f"Unexpected worker: {script}")

    def run_job(self, **options):
        defaults = {"skip_analysis": True, "language": "en", "onnx_device": "cpu"}
        defaults.update(options)
        return run_job(self.teacher, self.student, self.output, **defaults)

    def test_skip_analysis_rejects_a_changed_input_and_records_failure(self):
        self.student.write_bytes(b"different student")
        with self.assertRaisesRegex(ValueError, "analysis does not match"):
            self.run_job()
        self.assertEqual(self.calls, [])
        job = read_json(self.output / "job.json")
        self.assertEqual(job["status"], "failed")
        self.assertIn("student", job["error"])
        self.assertIn("elapsed_seconds", job)

    def test_skip_analysis_with_same_input_uses_cached_stages(self):
        job = self.run_job(verify=False)
        self.assertEqual([item[0] for item in self.calls], ["render_worker.py"])
        self.assertEqual(job["stages"]["analysis"]["teacher"], self.ta["completed_stages"])
        self.assertEqual(job["status"], "completed")

    def test_analysis_only_does_not_render_or_verify(self):
        job = self.run_job(skip_analysis=False, analyze_only=True)
        self.assertEqual([item[0] for item in self.calls], ["analysis_worker.py"])
        self.assertEqual(job["status"], "analysis_complete")
        self.assertEqual(set(job["stages"]), {"analysis"})

    def test_render_failure_does_not_promote_a_stale_render_manifest(self):
        write_json(self.output / "render.json", {"experiments": {"full": {"stale_marker": True}}})
        with patch("application.copyvocal.run_worker", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                self.run_job(verify=False)
        job = read_json(self.output / "job.json")
        self.assertEqual(job["status"], "failed")
        self.assertNotIn("render", job["stages"])

    def test_structural_failure_fails_job_before_timing_or_report(self):
        self.structure_passed = False
        with self.assertRaisesRegex(RuntimeError, "structural verification"):
            self.run_job(verify_timing=True)
        job = read_json(self.output / "job.json")
        self.assertEqual(job["status"], "failed")
        self.assertFalse(job["stages"]["verification"]["structural_checks_passed"])
        self.assertNotIn("timing_verify_worker.py", [item[0] for item in self.calls])
        # A current failure status page is allowed; prior success audio cards
        # and an old completed status must not survive structural rejection.
        page = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("failed", page)
        self.assertEqual(PageData(page).audio, [])

    def test_no_verify_ignores_previous_verification_and_experiments(self):
        write_json(self.output / "verification.json", {"warnings": ["STALE_WARNING"]})
        write_json(self.output / "verification.measured_pitch.json", {"times": [0.2, 0.3], "f0_hz": [880, 880], "voiced": [True, True]})
        (self.output / "student.identity.wav").write_bytes(b"old audio")
        job = self.run_job(verify=False)
        page = (self.output / "report.html").read_text(encoding="utf-8")
        data = PageData(page)
        self.assertNotIn("verification", job["stages"])
        self.assertNotIn("STALE_WARNING", page)
        self.assertNotIn("student.identity.wav", data.audio)
        self.assertNotIn("#ffcb82", [item["stroke"] for item in data.polylines])
        self.assertIn("verification was skipped", page)

    def test_report_plots_fresh_measured_curve_and_real_uv_gaps(self):
        self.run_job()
        page = (self.output / "report.html").read_text(encoding="utf-8")
        curves = PageData(page).polylines
        teacher = next(item for item in curves if item["stroke"] == "#7cd8ff")
        measured = next(item for item in curves if item["stroke"] == "#ffcb82")
        self.assertNotEqual(teacher["points"], measured["points"])
        self.assertIn(",93.00", measured["points"])
        # A true UV frame must split a curve, so a line cannot cross the gap.
        self.ta["pitch"] = {"times": [0.1, 0.2, 0.3, 0.4, 0.5], "f0_hz": [220, 220, 0, 220, 220], "voiced": [True, True, False, True, True]}
        plan = build_alignment(self.ta, self.sa)
        write_report(self.output, self.ta, self.sa, plan, {"status": "test", "stages": {}})
        curves = PageData((self.output / "report.html").read_text(encoding="utf-8")).polylines
        self.assertEqual(len([item for item in curves if item["stroke"] == "#7cd8ff"]), 2)

    def test_long_pitch_chart_preserves_a_uv_gap_between_downsampled_points(self):
        # A short UV gap can fall between every globally sampled point.
        # The review still needs separate curves on either side of that gap.
        count = 5000
        times = [index / count for index in range(count)]
        f0, voiced = [220.0] * count, [True] * count
        f0[2501], voiced[2501] = 0.0, False
        self.ta["pitch"] = {"times": times, "f0_hz": f0, "voiced": voiced}
        plan = build_alignment(self.ta, self.sa)
        write_report(self.output, self.ta, self.sa, plan, {"status": "test", "stages": {}})
        page = PageData((self.output / "report.html").read_text(encoding="utf-8"))
        curves = [item for item in page.polylines if item["stroke"] == "#7cd8ff"]
        self.assertEqual(len(curves), 2)
        before = [float(point.split(",")[0]) for point in curves[0]["points"].split()]
        after = [float(point.split(",")[0]) for point in curves[1]["points"].split()]
        gap_x = 45 + times[2501] * 910
        self.assertLess(max(before), gap_x)
        self.assertGreater(min(after), gap_x)

    def test_the_report_states_the_pitch_setting_it_rendered_with(self):
        # The option section is the only place a reader learns which pitch the file
        # carries; each mode must say something different, and the disabled switch
        # must not print its refusal twice.
        policies = (("none", "the student's own pitch, unchanged"),
                    ("teacher_contour_teacher_register", "exactly as analysed"),
                    ("student_contour_student_register", "adaptive placement"))
        for policy, phrase in policies:
            plan = build_alignment(self.ta, self.sa)
            job = {"status": "test", "stages": {"render": {"experiments": {"full": {
                "timing_changed": True, "pitch_policy": policy,
                "options": {"change_timing": True, "pitch_mode": "off", "map_teacher_pitch": False},
                "register_pitch": {"applied": policy != "none", "offset_available": False,
                                   "offset_reason": "Only 1 matched region(s) carry voiced pitch.",
                                   "placement": "per_word" if policy == "student_contour_student_register"
                                                else "constant",
                                   "note_shift_measured_count": 3, "note_shift_applied_count": 2}}}}}}
            write_report(self.output, self.ta, self.sa, plan, job)
            page = (self.output / "report.html").read_text(encoding="utf-8")
            section = html.unescape(page.split("<h2>Correction options</h2>", 1)[1].split("</ul>", 1)[0])
            self.assertIn(phrase, section, policy)
            self.assertEqual(section.count("<li>Pitch:"), 1, policy)
            self.assertEqual("Per-word placement" in section,
                             policy == "student_contour_student_register", policy)

    def test_suspected_substitution_is_completed_with_unresolved(self):
        self.sa["words"][1]["text"] = "three"
        self.save_analysis()
        job = self.run_job(verify=False)
        self.assertEqual(job["status"], "completed_with_unresolved")
        self.assertGreater(job["stages"]["alignment"]["status_counts"].get("substitution", 0), 0)

    def test_no_correspondence_fails_without_loading_vocoder(self):
        self.sa["words"] = []
        self.sa["pitch"]["voiced"] = [False] * 4
        self.save_analysis()
        with self.assertRaisesRegex(ValueError, "No supported vocal correspondences"):
            self.run_job(verify=False)
        self.assertEqual(self.calls, [])
        self.assertEqual(read_json(self.output / "job.json")["status"], "failed")

    def test_acoustic_fallback_is_not_fully_completed(self):
        self.ta["words"] = []
        self.sa["words"] = []
        self.save_analysis()
        job = self.run_job(verify=False, allow_acoustic_fallback=True)
        self.assertEqual(job["status"], "completed_with_unresolved")
        self.assertTrue(job["stages"]["alignment"]["acoustic_fallback"])
        self.assertIn("--allow-acoustic-fallback", self.calls[0][1])

    def test_independent_timing_runs_after_structure_and_updates_stage(self):
        job = self.run_job(verify_timing=True)
        self.assertEqual([item[0] for item in self.calls], ["render_worker.py", "verify_worker.py", "timing_verify_worker.py"])
        stage = job["stages"]["verification"]
        self.assertTrue(stage["word_timing_validation"]["measured"])
        self.assertNotIn("not remeasured", stage["limitations"])
        self.assertIn("timbre remains unverified", stage["limitations"])
        self.assertEqual(read_json(self.output / "verification.json"), stage)
        page = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("Reliable exact words: 2 / 2", page)
        self.assertIn("Fresh output ASR and TiFA receive no teacher words", page)
        args = self.calls[-1][1]
        self.assertEqual(args[args.index("--audio") + 1], str(self.output / "student.corrected.wav"))

    def test_timing_uncovered_words_prevent_complete_status_even_when_scored_words_pass(self):
        output_analysis = copy.deepcopy(self.ta)
        output_analysis["words"] = output_analysis["words"][:1]
        self.timing = compare_word_timing(self.ta, output_analysis)
        job = self.run_job(verify_timing=True)
        self.assertEqual(job["status"], "completed_with_unresolved")
        self.assertEqual(job["stages"]["timing_verification"]["coverage"]["within_tolerance_ratio_of_reliable_matches"], 1)
        self.assertEqual(job["stages"]["timing_verification"]["coverage"]["within_tolerance_ratio_of_all_teacher_units"], 0.5)

    def test_timing_only_is_current_stage_and_does_not_reuse_stale_pitch_metrics(self):
        write_json(self.output / "verification.json", {"warnings": ["OLD_PITCH_METRICS"]})
        job = self.run_job(verify=False, verify_timing=True)
        self.assertEqual([item[0] for item in self.calls], ["render_worker.py", "timing_verify_worker.py"])
        self.assertNotIn("verification", job["stages"])
        self.assertIn("timing_verification", job["stages"])
        self.assertNotIn("OLD_PITCH_METRICS", (self.output / "report.html").read_text(encoding="utf-8"))

    def test_reusing_analysis_with_another_language_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "analysis language does not match ja"):
            self.run_job(language="ja", verify=False)
        self.assertEqual(self.calls, [])
        self.assertEqual(read_json(self.output / "job.json")["status"], "failed")

    def test_skipped_analysis_source_must_refer_to_current_requested_input(self):
        previous = self.directory / "previous-student.wav"
        previous.write_bytes(self.student.read_bytes())
        self.sa["source"] = str(previous)
        self.save_analysis()
        # The hash is still for the requested input. The old source changes to
        # another waveform of equal byte count, evading a length-only check.
        previous.write_bytes(b"changed fixture")
        self.assertEqual(len(previous.read_bytes()), len(self.student.read_bytes()))
        self.run_job(verify=False)
        actual = Path(read_json(self.output / "student.analysis.json")["source"])
        self.assertEqual(actual.resolve(), self.student.resolve(), "Renderer will load a stale source instead of the input whose hash was checked.")

    def test_failed_rerun_invalidates_previous_success_review(self):
        self.run_job(verify=False)
        self.assertIn("completed", (self.output / "report.html").read_text(encoding="utf-8"))
        self.student.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.run_job(verify=False)
        if (self.output / "report.html").exists():
            self.assertIn("failed", (self.output / "report.html").read_text(encoding="utf-8"), "A report from a previous successful run remains authoritative after this run failed.")

    def test_missing_input_replaces_prior_success_with_current_failure(self):
        self.run_job(verify=False)
        self.assertIn("completed", (self.output / "report.html").read_text(encoding="utf-8"))
        self.student.unlink()
        calls_before = len(self.calls)
        with self.assertRaises(FileNotFoundError):
            self.run_job(verify=False)
        self.assertEqual(len(self.calls), calls_before)
        job = read_json(self.output / "job.json")
        self.assertEqual(job["status"], "failed")
        self.assertIn("student.wav", job["error"])
        page = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("failed", page)
        self.assertEqual(PageData(page).audio, [])


class CliConfigurationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("copyvocal_cli_test", ROOT / "scripts/copyvocal_cli.py")
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)

    def test_config_defaults_and_explicit_overrides_reach_job(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "custom.json"
            write_json(config, {"sample_rate": 44100, "onnx_device": "cpu", "language": "en", "max_chunk_seconds": 12,
                               "context_seconds": 0.4, "minimum_alignment_quality": 0.6, "maximum_stretch": 2.5,
                               "vocoder_log_floor": 1e-9, "allow_acoustic_fallback": False})
            with patch("sys.argv", ["copyvocal_cli.py", "--config", str(config), "--max-stretch", "3.5", "--skip-analysis", "--no-verify", "--verify-timing"]), patch("application.copyvocal.run_job", return_value={"status": "completed"}) as run:
                self.assertEqual(self.cli.main(), 0)
            options = run.call_args.kwargs
            self.assertEqual(options["onnx_device"], "cpu")
            self.assertEqual(options["language"], "en")
            self.assertEqual(options["max_chunk_seconds"], 12)
            self.assertEqual(options["context_seconds"], 0.4)
            self.assertEqual(options["min_quality"], 0.6)
            self.assertEqual(options["max_stretch"], 3.5)
            self.assertEqual(options["log_floor"], 1e-9)
            self.assertTrue(options["skip_analysis"])
            self.assertTrue(options["verify_timing"])
            self.assertFalse(options["verify"])

    def test_japanese_language_reaches_job_from_flag_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "japanese.json"
            write_json(config, {"sample_rate": 44100, "language": "ja"})
            for arguments in (["--language", "ja"], ["--config", str(config)]):
                with self.subTest(arguments=arguments), patch("sys.argv", ["copyvocal_cli.py", *arguments]), patch(
                        "application.copyvocal.run_job", return_value={"status": "completed_with_unresolved"}) as run:
                    self.assertEqual(self.cli.main(), 0)
                self.assertEqual(run.call_args.kwargs["language"], "ja")

    def test_incompatible_sample_rate_is_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.json"
            write_json(config, {"sample_rate": 48000})
            with patch("sys.argv", ["copyvocal_cli.py", "--config", str(config)]), patch("application.copyvocal.run_job") as run:
                with self.assertRaises(SystemExit) as raised:
                    self.cli.main()
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_runtime_failure_maps_to_nonzero_cli_exit(self):
        with patch("sys.argv", ["copyvocal_cli.py"]), patch("application.copyvocal.run_job", side_effect=RuntimeError("fixture failure")):
            self.assertEqual(self.cli.main(), 1)


if __name__ == "__main__":
    unittest.main()
