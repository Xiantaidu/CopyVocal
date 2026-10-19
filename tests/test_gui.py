from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QGridLayout, QHBoxLayout

from gui import correction_view as gui_correction
from gui import text as T
from gui import window as gui_window
from gui.i18n import tr, set_language
from gui.job import JobRequest, suggested_output
from gui.results import JobArtifacts
from gui.runner import STAGES, PipelineRunner

# Verbatim merged stdout of a finished run, copied from processing.log of a real
# job. Line shapes matter here: the runner must read what the workers write, not
# a format that only exists in this test.
WORKER_LOG = """\
2026-10-15 21:20:06,723 INFO Starting analysis_worker.py.
2026-10-15 21:20:07,954 INFO Transcribing teacher chunk 1/1.
2026-10-15 21:20:14,567 INFO Transcribing student chunk 1/1.
2026-10-15 21:20:17,958 INFO Aligning teacher chunk 1/1.
2026-10-15 21:20:20,458 INFO Aligning student chunk 1/1.
2026-10-15 21:20:22,100 INFO Extracting continuous pitch from teacher.
2026-10-15 21:20:23,075 INFO Extracting continuous pitch from student.
2026-10-15 21:20:25,211 INFO Detecting notes in teacher chunk 1/1.
2026-10-15 21:20:25,736 INFO Detecting notes in student chunk 1/1.
Analysis artifacts saved.
2026-10-15 21:12:45,442 INFO Starting render_worker.py.
2026-10-15 21:12:48,008 INFO Loading PC-NSF-HiFiGAN on cuda.
2026-10-15 21:12:48,411 INFO Rendering identity.
2026-10-15 21:12:49,381 INFO Rendering full.
2026-10-15 21:12:49,564 INFO Render artifacts saved.
2026-10-15 21:12:50,319 INFO Starting verify_worker.py.
2026-10-15 21:12:51,277 INFO [RMVPE ONNX] Using DirectML adapter 0.
Verification saved: D:\\CopyVocal\\outputs\\short\\verification.json
2026-10-15 21:12:53,000 INFO Starting timing_verify_worker.py.
Timing verification saved: D:\\CopyVocal\\outputs\\short\\timing.verification.json
2026-10-15 21:12:55,316 INFO Saved corrected vocal: D:\\CopyVocal\\outputs\\short\\student.corrected.wav
Job status: completed_with_unresolved
"""


class JobRequestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.teacher, self.student = self.directory / "teacher.wav", self.directory / "student.wav"
        self.teacher.write_bytes(b"teacher")
        self.student.write_bytes(b"student")

    def request(self, **overrides) -> JobRequest:
        values = {"teacher": str(self.teacher), "student": str(self.student),
                  "output": str(self.directory / "out")}
        values.update(overrides)
        return JobRequest(**values)

    def test_defaults_come_from_settings_file(self):
        from application.artifacts import read_json
        settings = read_json(ROOT / "settings" / "default.json")
        request = JobRequest.from_defaults()
        self.assertEqual(request.onnx_device, settings["onnx_device"])
        self.assertEqual(request.vocoder_device, settings["vocoder_device"])
        self.assertEqual(request.language, settings["language"])
        self.assertAlmostEqual(request.max_chunk_seconds, settings["max_chunk_seconds"])
        self.assertAlmostEqual(request.log_floor, settings["vocoder_log_floor"])

    def test_valid_request_reports_no_problems(self):
        self.assertEqual(self.request().validate(), [])

    def test_japanese_request_survives_config_and_preferences(self):
        request = self.request(language="ja")
        self.assertEqual(request.validate(), [])
        self.assertEqual(request.effective_config()["language"], "ja")
        restored = JobRequest.from_prefs(request.to_prefs(), JobRequest.from_defaults())
        self.assertEqual(restored.language, "ja")
        argv = restored.argv()
        self.assertEqual(argv[argv.index("--language") + 1], "ja")

    def test_missing_and_identical_inputs_are_rejected(self):
        problems = self.request(teacher="").validate()
        self.assertIn(tr("validate_choose_recording").format(label=tr("label_teacher_short")), problems)
        problems = self.request(student=str(self.teacher)).validate()
        self.assertIn(tr("validate_same_file"), problems)

    def test_out_of_range_option_is_rejected(self):
        problems = self.request(minimum_alignment_quality=1.5).validate()
        self.assertTrue(any(tr("advanced_quality") in item for item in problems), problems)

    def test_reuse_analysis_requires_stored_artifacts(self):
        problems = self.request(skip_analysis=True).validate()
        self.assertTrue(any("teacher.analysis.json" in item for item in problems), problems)
        output = self.directory / "stored"
        output.mkdir()
        for name in ("teacher.analysis.json", "student.analysis.json"):
            (output / name).write_text("{}", encoding="utf-8")
        self.assertEqual(self.request(skip_analysis=True, output=str(output)).validate(), [])

    def test_analyze_only_and_skip_analysis_conflict(self):
        problems = self.request(analyze_only=True, skip_analysis=True).validate()
        self.assertIn(tr("validate_conflict"), problems)

    def test_argv_maps_every_option_to_a_cli_flag(self):
        request = self.request(onnx_device="cpu", vocoder_device="cpu", language="en",
                               max_chunk_seconds=12.5, context_seconds=0.5,
                               minimum_alignment_quality=0.5, maximum_stretch=2.0, log_floor=1e-6,
                               include_game=False, allow_acoustic_fallback=True, experiments=True,
                               verify=False, verify_timing=True)
        argv = request.argv()
        for flag, value in (("--onnx-device", "cpu"), ("--vocoder-device", "cpu"), ("--language", "en"),
                            ("--max-chunk-seconds", "12.5"), ("--context-seconds", "0.5"),
                            ("--minimum-alignment-quality", "0.5"), ("--max-stretch", "2"),
                            ("--log-floor", "1e-06")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        for flag in ("--no-game", "--allow-acoustic-fallback", "--experiments", "--no-verify",
                     "--verify-timing"):
            self.assertIn(flag, argv)
        self.assertNotIn("--skip-analysis", argv)
        self.assertNotIn("--analyze-only", argv)

    def test_correction_switches_reach_the_cli_explicitly(self):
        # The pitch mode is always stated as its own value, so a stored settings file
        # can never silently decide which pitch the run uses.
        request = self.request(change_timing=False, pitch_mode="adaptive", map_teacher_pitch=True)
        argv = request.argv()
        self.assertIn("--no-change-timing", argv)
        self.assertEqual(argv[argv.index("--pitch-mode") + 1], "adaptive")
        self.assertIn("--map-teacher-pitch", argv)
        self.assertNotIn("--no-map-teacher-pitch", argv)
        self.assertNotIn("--no-change-pitch", argv)

    def test_every_pitch_mode_survives_the_round_trip(self):
        for mode in ("off", "teacher", "adaptive"):
            request = self.request(pitch_mode=mode)
            self.assertEqual(request.argv()[request.argv().index("--pitch-mode") + 1], mode)
            self.assertEqual(request.effective_config()["pitch_mode"], mode)
            restored = JobRequest.from_prefs(request.to_prefs(), JobRequest.from_defaults())
            self.assertEqual(restored.pitch_mode, mode, mode)

    def test_a_preferences_file_from_the_old_checkbox_still_applies_its_request(self):
        # The build before the switch had a 'change pitch' checkbox. A stored file
        # that asked for the teacher's register must keep asking for it instead of
        # silently reverting to leaving the pitch alone.
        base = JobRequest.from_defaults()
        self.assertEqual(JobRequest.from_prefs({"change_pitch": True}, base).pitch_mode, "teacher")
        self.assertEqual(JobRequest.from_prefs({"change_pitch": False}, base).pitch_mode, "off")
        # An explicit mode always wins over the legacy key.
        stored = {"change_pitch": True, "pitch_mode": "adaptive"}
        self.assertEqual(JobRequest.from_prefs(stored, base).pitch_mode, "adaptive")
        stored = {"change_pitch": True, "pitch_mode": "off"}
        self.assertEqual(JobRequest.from_prefs(stored, base).pitch_mode, "off")

    def test_an_unusable_pitch_mode_falls_back_to_off(self):
        # A stale preferences file naming a mode this build does not have must not
        # reach the pipeline as an unknown value.
        request = JobRequest.from_prefs({"pitch_mode": "louder"}, JobRequest.from_defaults())
        self.assertEqual(request.pitch_mode, "off")

    def test_correction_switches_reach_the_written_config(self):
        request = self.request(change_timing=False, pitch_mode="teacher", map_teacher_pitch=True)
        stored = request.effective_config()
        self.assertFalse(stored["change_timing"])
        self.assertEqual(stored["pitch_mode"], "teacher")
        self.assertTrue(stored["map_teacher_pitch"])
        # A stored config without the keys falls back to the dataclass defaults.
        defaults = JobRequest.from_prefs({}, JobRequest.from_defaults())
        self.assertTrue(defaults.change_timing)
        self.assertEqual(defaults.pitch_mode, "off")
        self.assertFalse(defaults.map_teacher_pitch)

    def test_written_config_carries_the_displayed_values(self):
        request = self.request(max_chunk_seconds=9.5, log_floor=1e-7, language="en")
        request.write_config(request.config_path())
        stored = json.loads(request.config_path().read_text(encoding="utf-8"))
        self.assertEqual(stored["sample_rate"], 44100)
        self.assertEqual(stored["language"], "en")
        self.assertAlmostEqual(stored["max_chunk_seconds"], 9.5)
        self.assertAlmostEqual(stored["vocoder_log_floor"], 1e-7)
        # The CLI reads the file named on the command line, not the default one.
        argv = request.argv()
        self.assertEqual(argv[argv.index("--config") + 1], str(request.config_path()))

    def test_prefs_round_trip_ignores_unknown_and_invalid_values(self):
        request = self.request(language="en", experiments=True, max_chunk_seconds=7.5)
        values = request.to_prefs()
        values["unknown_key"] = 1
        values["language"] = "fr"          # not a supported language
        values["max_chunk_seconds"] = "wide"
        restored = JobRequest.from_prefs(values, JobRequest.from_defaults())
        self.assertEqual(restored.language, JobRequest.from_defaults().language)
        self.assertAlmostEqual(restored.max_chunk_seconds, JobRequest.from_defaults().max_chunk_seconds)
        self.assertTrue(restored.experiments)
        self.assertFalse(hasattr(restored, "unknown_key"))

    def test_suggested_output_follows_the_student_name(self):
        self.assertEqual(suggested_output(r"D:\songs\Take 3.wav").name, "Take 3.corrected")


class StageParsingTests(unittest.TestCase):
    """The runner must read real worker output, not a simulated format."""

    def setUp(self):
        self.runner = PipelineRunner()
        self.states = []
        self.runner.stage_changed.connect(lambda stage, state: self.states.append((stage, state)))
        self.details = []
        self.runner.detail.connect(self.details.append)
        self.percentages = []
        self.runner.progress.connect(self.percentages.append)

    def consume(self, text: str) -> None:
        for line in text.splitlines():
            self.runner._consume(line)

    def announce(self, request) -> None:
        """Reproduce the stage seeding that start() performs without launching."""
        self.runner._stages = {key: ("pending" if self.runner.is_stage_enabled(key, request) else "skipped")
                               for key, _ in STAGES}
        if request.skip_analysis and not request.analyze_only:
            self.runner._set_stage("alignment", "running")

    def test_real_log_advances_every_stage(self):
        self.consume(WORKER_LOG)
        states = self.runner.stage_states()
        for stage in ("analysis", "alignment", "render", "verification",
                      "timing_verification", "review"):
            self.assertEqual(states[stage], "done", stage)

    def test_correspondence_is_derived_because_no_worker_logs_it(self):
        # Nothing in a real run announces correspondence: run_job performs it in
        # the parent process. It is inferred from the stages on either side.
        self.consume("2026-10-15 21:20:25,736 INFO Analysis artifacts saved.")
        self.assertEqual(self.runner.stage_states()["alignment"], "running")
        self.consume("2026-10-15 21:12:45,442 INFO Starting render_worker.py.")
        self.assertEqual(self.runner.stage_states()["alignment"], "done")
        self.assertEqual(self.runner.stage_states()["render"], "running")

    def test_stage_transitions_are_announced_in_order(self):
        self.consume(WORKER_LOG)
        running = [stage for stage, state in self.states if state == "running"]
        self.assertEqual(running, ["analysis", "alignment", "render", "verification",
                                   "timing_verification"])

    def test_activity_detail_reaches_the_window(self):
        self.consume(WORKER_LOG)
        self.assertIn(tr("detail_chunk").format(phase="歌词", track="教师", index=1, total=1), self.details)
        self.assertIn(tr("detail_chunk").format(phase="音素", track="学生", index=1, total=1), self.details)
        self.assertIn(tr("detail_pitch").format(track="教师"), self.details)
        self.assertIn(tr("detail_chunk").format(phase="音符", track="学生", index=1, total=1), self.details)
        self.assertIn(tr("detail_loading_vocoder").format(device="cuda"), self.details)
        self.assertIn(tr("detail_render").format(mode="full"), self.details)

    def test_progress_advances_monotonically_and_ends_at_complete(self):
        self.consume(WORKER_LOG)
        self.runner._set_percent(100)
        self.assertEqual(self.percentages, sorted(self.percentages))
        self.assertEqual(self.percentages[-1], 100)
        self.assertGreater(len(set(self.percentages)), 3)

    def test_unrelated_lines_do_not_advance_stages(self):
        self.consume("[RMVPE ONNX] Using DirectML adapter 0.\nSome other chatter.")
        self.assertEqual(set(self.runner.stage_states().values()), {"pending"})
        self.assertEqual(self.details, [])

    def test_failed_job_leaves_unfinished_stages_pending(self):
        self.consume("2026-10-15 21:12:48,411 INFO Starting render_worker.py.")
        self.assertEqual(self.runner.stage_states()["render"], "running")
        self.runner._active = True
        self.runner._emit_finish(1, "")
        self.assertEqual(self.runner.stage_states()["render"], "pending")
        self.assertEqual(self.percentages[-1], self.percentages[-2] if len(self.percentages) > 1
                         else self.percentages[-1])

    def test_cancelled_job_is_reported_as_cancelled(self):
        self.runner._active, self.runner._cancelled = True, True
        outcomes = []
        self.runner.finished.connect(outcomes.append)
        self.runner._emit_finish(1, "")
        self.assertEqual(outcomes[0]["status"], "cancelled")

    def test_finish_is_reported_once(self):
        outcomes = []
        self.runner.finished.connect(outcomes.append)
        self.runner._active = True
        self.runner._emit_finish(0, "")
        self.runner._emit_finish(0, "")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "succeeded")

    def test_skipped_stages_are_marked_for_the_request(self):
        request = JobRequest(analyze_only=True)
        self.announce(request)
        self.assertEqual(self.runner.stage_states()["analysis"], "pending")
        self.assertEqual(self.runner.stage_states()["render"], "skipped")
        self.assertEqual(self.runner.stage_states()["verification"], "skipped")
        self.assertEqual(self.runner.stage_states()["timing_verification"], "skipped")

    def test_stage_availability_follows_the_request(self):
        self.assertTrue(PipelineRunner.is_stage_enabled("verification", JobRequest(verify=True)))
        self.assertFalse(PipelineRunner.is_stage_enabled("verification", JobRequest(verify=False)))
        self.assertTrue(PipelineRunner.is_stage_enabled("timing_verification",
                                                        JobRequest(verify_timing=True)))
        self.assertFalse(PipelineRunner.is_stage_enabled("alignment", JobRequest(analyze_only=True)))
        self.assertTrue(PipelineRunner.is_stage_enabled("alignment", JobRequest(skip_analysis=True)))
        self.assertFalse(PipelineRunner.is_stage_enabled("analysis", JobRequest(skip_analysis=True)))

    def test_analysis_only_never_marks_render_or_correspondence(self):
        # An analysis-only job writes no alignment.json and renders nothing.
        self.announce(JobRequest(analyze_only=True))
        self.consume(WORKER_LOG)
        states = self.runner.stage_states()
        self.assertEqual(states["analysis"], "done")
        self.assertEqual(states["alignment"], "skipped")
        self.assertEqual(states["render"], "skipped")

    def test_reused_analysis_starts_at_correspondence(self):
        # With analysis skipped, correspondence is the first work the parent does,
        # so it must already look active when the job begins.
        self.announce(JobRequest(skip_analysis=True))
        self.assertEqual(self.runner.stage_states()["analysis"], "skipped")
        self.assertEqual(self.runner.stage_states()["alignment"], "running")


class ResultsTests(unittest.TestCase):
    """Displayed values are copied from the artifacts, never recomputed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.write("job.json", {"status": "completed_with_unresolved", "elapsed_seconds": 12.5})
        self.write("alignment.json", {
            "metrics": {"teacher_unit_count": 17, "student_unit_count": 17,
                        "status_counts": {"matched": 16, "substitution": 1},
                        "lyric_coverage": 1.0, "semantic_match_rate": 0.9411764705882353,
                        "unsupported_region_count": 4, "ambiguous_match_count": 0},
            "warnings": ["Possible lyric mismatch or ASR error: '了' versus '啊'."],
            "regions": [{"teacher_text": "我", "student_text": "我", "status": "matched", "mode": "synthesize",
                         "quality": 1.0, "teacher_start": 0.0, "teacher_end": 0.4, "reason": "matched"},
                        {"teacher_text": "了", "student_text": "啊", "status": "substitution",
                         "mode": "synthesize", "quality": 0.48, "teacher_start": 1.0, "teacher_end": 1.5,
                         "reason": "possible_lyric_mismatch_or_asr_error"},
                        # A plain silence region is omitted from the correspondence table.
                        {"teacher_text": "", "student_text": "", "status": "silence", "mode": "silence",
                         "quality": 1.0, "teacher_start": 0.4, "teacher_end": 1.0, "reason": "rest"}]})
        self.write("verification.json", {
            "structural_checks_passed": True,
            "audio": {"sample_count": 322877, "sample_count_exact": True},
            "coverage": {"semantic_teacher_voiced_coverage": 0.978494623655914},
            "actual_vs_teacher": {"all_reference_frames": {"joint_voiced_frames": 552,
                                                           "median_absolute_cents": 5.83,
                                                           "p95_absolute_cents": 69.61,
                                                           "voicing_recall": 0.989247311827957}},
            "actual_vs_render_target": {"all_target_frames": {"joint_voiced_frames": 470,
                                                              "median_absolute_cents": 5.6,
                                                              "p95_absolute_cents": 62.74,
                                                              "voicing_recall": 1.0}},
            "warnings": ["Measurement limitation."],
            "limitations": ["Default verification does not remeasure rendered word boundaries."]})
        self.write("timing.verification.json", {
            "coverage": {"teacher_unit_count": 17, "reliable_exact_match_count": 16,
                         "within_tolerance_ratio_of_all_teacher_units": 0.9411764705882353},
            "reliable_boundary_errors": {"onset": {"mean_absolute_ms": 4.375, "p95_absolute_ms": 12.5},
                                         "offset": {"mean_absolute_ms": 5.59, "p95_absolute_ms": 18.3}}})
        self.write("render.json", {"device": "cuda", "peak_cuda_memory_bytes": 1048576,
                                   "experiments": {"full": {"rendered_sample_ratio": 0.998,
                                                            "voiced_frame_ratio": 0.61,
                                                            "muted_region_count": 2,
                                                            "acoustic_transitions": [{}, {}]}}})
        self.write("teacher.analysis.json", {"warnings": ["Teacher analysis warning."]})
        self.write("student.analysis.json", {"warnings": ["Teacher analysis warning."]})
        self.artifacts = JobArtifacts(self.output)

    def write(self, name: str, payload: dict) -> None:
        (self.output / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_unresolved_status_is_not_reported_as_plain_success(self):
        headline, note = self.artifacts.summary()
        self.assertEqual(headline, tr("status_unresolved"))
        self.assertIn("静音", note)

    def test_metrics_show_planned_and_measured_values_separately(self):
        values = dict(self.artifacts.metrics())
        self.assertEqual(values[tr("row_planned_coverage")], "100.00%")
        self.assertIn(f"{T.status_label('matched')} 16", values[tr("row_unit_status")])
        self.assertEqual(values[tr("row_unsupported_regions")], "4")
        self.assertIn("322877", values[tr("row_exported_length")])
        self.assertIn(tr("row_length_exact"), values[tr("row_exported_length")])
        self.assertEqual(values[tr("row_teacher_voiced_coverage")], "97.85%")

    def test_pitch_rows_carry_the_measured_numbers(self):
        values = dict(self.artifacts.pitch_rows())
        error_row = values[tr("pitch_row_error").format(label=tr("pitch_vs_teacher"))]
        self.assertIn("5.83 音分", error_row)
        self.assertIn("69.61 音分", error_row)
        self.assertEqual(values[tr("pitch_row_recall").format(label=tr("pitch_vs_teacher"))], "98.92%")

    def test_timing_rows_appear_only_when_remeasured(self):
        values = dict(self.artifacts.timing_rows())
        self.assertEqual(values[tr("timing_row_compared")], "16 / 17")
        self.assertIn("0.004 秒", values[tr("timing_row_onset")])
        (self.output / "timing.verification.json").unlink()
        self.assertEqual(self.artifacts.timing_rows(), [])

    def test_warnings_are_deduplicated_and_include_limitations(self):
        warnings = self.artifacts.warnings()
        self.assertEqual(warnings.count("Teacher analysis warning."), 1)
        self.assertTrue(any(item.startswith(tr("warning_limit_prefix")) for item in warnings))
        self.assertTrue(any("歌词可能不一致" in item for item in warnings))

    def test_failed_structural_check_is_surfaced_first(self):
        self.write("verification.json", {"structural_checks_passed": False, "warnings": ["other"]})
        self.assertEqual(self.artifacts.warnings()[0], tr("warning_structural_failure"))

    def test_correspondence_rows_skip_plain_silence(self):
        rows = self.artifacts.regions()
        self.assertEqual([row["teacher"] for row in rows], ["我", "了"])
        self.assertEqual(rows[1]["quality"], 0.48)

    def test_artifacts_without_a_manifest_are_still_reported(self):
        # A directory can hold real artifacts with no job.json (a partial run, or
        # files copied out of one). Saying only "no artifacts" would deny the
        # alignment and verification tabs' own contents.
        empty = JobArtifacts(self.output / "no_manifest")
        (self.output / "no_manifest").mkdir(exist_ok=True)
        self.assertEqual(empty.summary(), (tr("summary_no_job"), tr("summary_no_job_note")))
        (self.output / "no_manifest" / "alignment.json").write_text("{}", encoding="utf-8")
        headline, note = empty.summary()
        self.assertEqual(headline, tr("status_no_manifest_with_artifacts"))
        self.assertIn("job.json", note)

    def test_missing_artifacts_are_reported_not_invented(self):
        empty = JobArtifacts(self.output / "not-there")
        self.assertEqual(empty.metrics(), [])
        self.assertEqual(empty.warnings(), [])
        self.assertEqual(empty.available_audio(), [])
        self.assertIsNone(empty.corrected_path())
        self.assertIsNone(empty.report_path())
        self.assertEqual(empty.summary(), (tr("summary_no_job"), tr("summary_no_job_note")))

    def test_corrupt_artifact_does_not_break_the_panel(self):
        (self.output / "alignment.json").write_text("{not json", encoding="utf-8")
        # The unreadable file is skipped; values from the intact ones still show.
        values = dict(self.artifacts.metrics())
        self.assertNotIn("Planned lyric coverage", values)
        self.assertEqual(values[tr("row_teacher_voiced_coverage")], "97.85%")
        self.assertEqual(self.artifacts.regions(), [])


class TranslationTests(unittest.TestCase):
    """The Chinese interface text must cover real artifacts without hiding any."""

    def test_unknown_sentences_are_shown_unchanged(self):
        # A warning the interface has not been taught about must still be readable,
        # so an unmatched sentence is returned verbatim rather than replaced.
        unknown = "A brand new warning that no translation table knows about."
        self.assertEqual(T.warning_text(unknown), unknown)

    def test_every_warning_in_real_artifacts_is_translated(self):
        # The shipped outputs are the only real artifacts available, so they are the
        # evidence that the table covers what the pipeline actually writes.
        sources = [
            ROOT / "outputs" / "short" / "verification.json",
            ROOT / "outputs" / "short" / "teacher.analysis.json",
            ROOT / "outputs" / "short" / "student.analysis.json",
            ROOT / "outputs" / "short" / "alignment.json",
            ROOT / "outputs" / "short" / "timing.verification.json",
        ]
        seen, untranslated = 0, []
        for path in sources:
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for item in payload.get("warnings", []) + payload.get("limitations", []):
                seen += 1
                if T.warning_text(item) == item:
                    untranslated.append(item)
        self.assertGreater(seen, 0, "no warnings found to check")
        self.assertEqual(untranslated, [], f"{len(untranslated)} sentences have no translation")

    def test_every_reason_in_real_alignment_is_translated(self):
        # The reason column showed raw English for statuses the pipeline reuses as
        # reasons, so the coverage is asserted against a real alignment file.
        path = ROOT / "outputs" / "long_relative" / "alignment.json"
        if not path.is_file():
            path = ROOT / "outputs" / "short" / "alignment.json"
        reasons = {region.get("reason", "") for region in
                   json.loads(path.read_text(encoding="utf-8"))["regions"]}
        reasons.discard("")
        untranslated = [value for value in reasons if T.reason_label(value) == value]
        self.assertEqual(untranslated, [], f"{len(untranslated)} reasons have no label")
        self.assertTrue(reasons, "no reasons found to check")

    def test_region_labels_cover_the_values_the_pipeline_emits(self):
        for status in ("matched", "homophone", "substitution", "missing", "unresolved",
                       "unvoiced_gap", "phonetic_only"):
            self.assertNotEqual(T.status_label(status), status, status)
        for mode in ("synthesize", "mute", "silence", "acoustic_fallback"):
            self.assertNotEqual(T.mode_label(mode), mode, mode)
        for reason in ("voiced_teacher_region_without_lyric_correspondence",
                       "teacher_lyrics_unavailable",
                       "untranscribed_activity_no_lyric_correspondence"):
            self.assertNotEqual(T.reason_label(reason), reason, reason)
        for policy in ("none", "student_contour_student_register", "teacher_contour_student_register",
                       "student_contour_teacher_register", "teacher_contour_teacher_register"):
            self.assertNotEqual(T.policy_label(policy), policy, policy)
        # An unrecognized value stays visible instead of becoming a blank cell.
        self.assertEqual(T.status_label("something_new"), "something_new")

    def test_value_carrying_warnings_keep_their_numbers(self):
        text = T.warning_text("Possible lyric mismatch or ASR error: 'le' versus 'a'.")
        self.assertIn("le", text)
        self.assertIn("a", text)
        text = T.warning_text("Unsupported stretch ratio 5.20 for teacher word 'x'.")
        self.assertIn("5.20", text)
        self.assertIn("x", text)
        text = T.warning_text("ASR failed for chunk 3; retry analysis before correction.")
        self.assertIn("3", text)
        text = T.warning_text("7 unsupported or low-quality regions remain unresolved; "
                              "acoustic rendering does not establish lyric correspondence.")
        self.assertIn("7", text)

    def test_job_errors_are_translated_and_unknown_ones_kept(self):
        # A failed job records its reason in job.json; the dialog must show it in
        # Chinese, and must not silently drop a reason it does not recognize.
        self.assertEqual(T.job_error_text("Pitch analysis is missing; correction cannot run."),
                         "缺少音高分析，无法进行校正。")
        translated = T.job_error_text(
            "analysis_worker.py failed with exit code 1. See " + "D:" + chr(92) + "out" + chr(92) + "processing.log")
        self.assertIn("退出码 1", translated)
        self.assertIn("processing.log", translated)
        self.assertIn("教师", T.job_error_text("teacher analysis does not match the current input file."))
        self.assertEqual(T.job_error_text("An entirely new failure."), "An entirely new failure.")

    def test_the_window_shows_no_english_interface_labels(self):
        set_language("zh")
        for value in (tr("window_title"), tr("button_run"), tr("button_cancel"), tr("tab_log"),
                      tr("tab_results"), tr("card_recordings"), tr("card_options"), tr("check_timing"),
                      tr("label_pitch_mode"), tr("check_pitch_map"), tr("detail_no_job"),
                      tr("warning_placeholder"), *T.pitch_mode_labels()):
            self.assertRegex(value, r"[一-鿿]", value)


class WindowSmokeTests(unittest.TestCase):
    """The window must build and run headless on the example artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from gui.window import MainWindow
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)

    def test_window_builds_with_expected_controls(self):
        self.assertEqual(self.window.stage_list.count(), 6)
        self.assertFalse(self.window.cancel_button.isEnabled())
        self.assertTrue(self.window.run_button.isEnabled())

    def test_japanese_selection_reaches_command_preview(self):
        self.assertGreaterEqual(self.window.language_combo.findText("ja"), 0)
        self.window.language_combo.setCurrentText("ja")
        self.assertIn("--language ja", self.window.command_label.text())

    def test_command_preview_matches_the_collected_options(self):
        self.window.teacher_edit.setText(r"D:\a\teacher.wav")
        self.window.student_edit.setText(r"D:\a\student.wav")
        self.window.language_combo.setCurrentText("en")
        self.window.experiments_check.setChecked(True)
        preview = self.window.command_label.text()
        self.assertIn("--teacher D:\\a\\teacher.wav", preview)
        self.assertIn("--language en", preview)
        self.assertIn("--experiments", preview)
        self.assertIn("--config", preview)
        # The preview is the equivalent run.bat call, not the raw interpreter argv.
        self.assertTrue(preview.startswith("run.bat"))

    def test_correction_axes_are_independent_and_previewed(self):
        window = self.window
        # The window restores the user's stored preferences, so pin the states
        # under test instead of asserting whatever a previous session saved.
        # Defaults themselves are covered by JobRequest.from_defaults().
        window.timing_check.setChecked(True)
        window.pitch_combo.setCurrentIndex(T.PITCH_MODE_VALUES.index("off"))
        window.pitch_map_check.setChecked(False)
        preview = window.command_label.text()
        self.assertIn("--change-timing", preview)
        self.assertIn("--pitch-mode off", preview)
        self.assertIn("--no-map-teacher-pitch", preview)
        # Turning one axis off must not disturb the others.
        window.timing_check.setChecked(False)
        self.assertFalse(window.timing_check.isChecked())
        self.assertEqual(window._pitch_mode(), "off")
        self.assertIn("--no-change-timing", window.command_label.text())
        # Every pitch mode stays selectable in either timing state, because the
        # three axes are independent.
        window.pitch_combo.setCurrentIndex(T.PITCH_MODE_VALUES.index("adaptive"))
        self.assertEqual(window._pitch_mode(), "adaptive")
        self.assertIn("--pitch-mode adaptive", window.command_label.text())
        self.assertFalse(window.timing_check.isChecked())

    def test_the_contour_checkbox_is_enabled_wherever_there_is_a_contour_to_choose(self):
        # Both 'teacher' and 'adaptive' sing a contour inside the student's or the
        # teacher's register, so the checkbox is a real choice there. 'Off' does not
        # touch the pitch at all, so it is disabled rather than offering a no-op.
        window = self.window
        for mode in ("teacher", "adaptive"):
            window.pitch_combo.setCurrentIndex(T.PITCH_MODE_VALUES.index(mode))
            self.assertTrue(window.pitch_map_check.isEnabled(), mode)
        window.pitch_combo.setCurrentIndex(T.PITCH_MODE_VALUES.index("off"))
        self.assertFalse(window.pitch_map_check.isEnabled())

    def test_quoting_keeps_paths_with_spaces_together(self):
        request = JobRequest(teacher=r"D:\my songs\teacher.wav", student=r"D:\my songs\student.wav",
                             output=r"D:\my songs\out")
        self.assertIn(r'--teacher "D:\my songs\teacher.wav"', request.command_line())

    def test_conflicting_options_are_mutually_disabled(self):
        self.window.analyze_check.setChecked(True)
        self.assertFalse(self.window.skip_check.isEnabled())
        self.assertFalse(self.window.verify_timing_check.isChecked())
        self.window.skip_check.setChecked(False)
        self.window.analyze_check.setChecked(False)
        self.window.skip_check.setChecked(True)
        self.assertFalse(self.window.analyze_check.isEnabled())

    def test_invalid_input_is_reported_without_starting_a_job(self):
        self.window.teacher_edit.setText("")
        self.window.student_edit.setText("")
        from unittest.mock import patch
        with patch("gui.window.QMessageBox.warning") as warning:
            self.window.start_job()
        self.assertTrue(warning.called)
        self.assertFalse(self.window.runner.running)

    def test_results_panel_reflects_an_existing_output_directory(self):
        self.window.output_edit.setText(str(ROOT / "outputs" / "short"))
        self.window.refresh_results()
        metrics = {self.window.metric_table.item(row, 0).text()
                   for row in range(self.window.metric_table.rowCount())}
        self.assertIn(tr("row_planned_coverage"), metrics)
        self.assertTrue(self.window.region_table.rowCount() > 0)
        self.window.output_edit.setText(str(ROOT / "outputs" / "long_hard"))
        self.window.refresh_results()

    def test_existing_output_is_reviewable_without_running_anything(self):
        # Opening the window on a finished job must show it; the pipeline is untouched.
        self.window.output_edit.setText(str(ROOT / "outputs" / "short"))
        self.window.refresh_results()
        self.assertIn(tr("summary_corrected_prefix"), self.window.summary_label.text())
        self.assertGreater(self.window.audio_combo.count(), 0)
        self.assertTrue(self.window.open_report_button.isEnabled())
        self.assertEqual(self.window.audio_combo.itemText(2), tr("audio_corrected"))

    def test_log_view_shows_worker_output(self):
        self.window._append_log("2026-10-15 21:20:07,954 INFO Transcribing teacher chunk 1/1.")
        self.assertIn("Transcribing teacher chunk 1/1", self.window.log_view.toPlainText())

    def test_progress_bar_reflects_reported_progress(self):
        self.window._update_progress(63)
        self.assertEqual(self.window.progress_bar.value(), 63)
        # A negative value (unknown progress) must not show as a negative bar.
        self.window._update_progress(-1)
        self.assertEqual(self.window.progress_bar.value(), 0)


class NavigationTests(unittest.TestCase):
    """The icon sidebar must expose the three pages and switch between them."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from gui.window import MainWindow
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)

    def test_three_pages_are_registered(self):
        names = [page.objectName() for page in
                 (self.window.correction, self.window.results, self.window.settings)]
        self.assertEqual(names, ["correctionInterface", "resultsInterface", "settingsInterface"])
        for name in names:
            self.assertIsNotNone(self.window.navigationInterface.widget(name))

    def test_selecting_a_page_switches_the_stack(self):
        self.window.navigationInterface.setCurrentItem(self.window.settings.objectName())
        self.assertIs(self.window.stackedWidget.currentWidget(), self.window.settings)
        self.window.navigationInterface.setCurrentItem(self.window.correction.objectName())
        self.assertIs(self.window.stackedWidget.currentWidget(), self.window.correction)

    def test_theme_switch_keeps_the_window_alive(self):
        for value in ("dark", "light", "auto"):
            self.window.settings.apply_theme(value)
            self.assertEqual(self.window.settings.theme(), value)

    def test_language_switch_retranslates_the_interface(self):
        self.window.settings.language_combo.setCurrentIndex(
            self.window.settings.language_combo.findData("en"))
        self.assertEqual(tr("button_run"), "Run")
        self.assertEqual(self.window.run_button.text(), "Run")
        self.assertEqual(tr("window_title"), self.window.windowTitle())
        self.window.settings.language_combo.setCurrentIndex(
            self.window.settings.language_combo.findData("zh"))
        self.assertEqual(self.window.run_button.text(), "开始")


class ProgressiveDisclosureTests(unittest.TestCase):
    """Secondary controls stay folded until a job needs them."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from gui.window import MainWindow
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)

    def test_secondary_sections_start_collapsed(self):
        for name in ("run_expander", "advanced_expander", "command_expander", "log_expander"):
            self.assertFalse(getattr(self.window.correction, name).isExpanded(), name)

    def test_switches_show_on_off_rather_than_their_caption(self):
        switch = self.window.game_check
        self.assertEqual((switch.getOnText(), switch.getOffText()),
                         (tr("switch_on"), tr("switch_off")))
        # The caption beside the switch carries the meaning, never the state text.
        self.assertNotIn(tr("check_game"), (switch.getOnText(), switch.getOffText()))

    def test_stage_list_appears_only_once_a_job_starts(self):
        self.assertFalse(self.window.stage_list.isVisible())
        self.assertEqual(self.window.stage_list.count(), 6)

    def test_expanding_a_section_reveals_its_controls(self):
        expander = self.window.correction.advanced_expander
        expander.expand()
        self.assertTrue(expander.isExpanded())
        self.assertTrue(self.window.chunk_spin.isVisibleTo(expander))
        expander.collapse()
        self.assertFalse(self.window.chunk_spin.isVisibleTo(expander))

    def test_advanced_values_survive_being_folded(self):
        self.window.chunk_spin.setValue(12.0)
        self.assertEqual(self.window._collect().max_chunk_seconds, 12.0)


class LayoutCompositionTests(unittest.TestCase):
    """Rows are composed, not dumped: captions lead, controls trail, hints are quiet."""

    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from gui.window import MainWindow
        # Language and theme switches persist preferences; keep the user's file out of it.
        target = Path(tempfile.mkdtemp()) / "gui_prefs.json"
        for module in (gui_window, gui_correction):
            self.addCleanup(setattr, module, "PREFS_PATH", module.PREFS_PATH)
            module.PREFS_PATH = target
        self.window = MainWindow()
        self.addCleanup(self.window.deleteLater)

    @staticmethod
    def _row_of(widget) -> QHBoxLayout:
        parent = widget.parentWidget()
        while parent is not None:
            layout = parent.layout()
            if isinstance(layout, QHBoxLayout):
                return layout
            parent = parent.parentWidget()
        raise AssertionError(f"{widget} is not inside a horizontal row")

    @staticmethod
    def _cell_of(widget):
        """The SettingField/SettingRow that holds one caption and its control."""
        parent = widget.parentWidget()
        while parent is not None:
            if parent.__class__.__name__ in {"SettingField", "SettingRow"}:
                return parent
            parent = parent.parentWidget()
        raise AssertionError(f"{widget} is not inside a setting cell")

    @staticmethod
    def _grid_of(cell):
        """The QGridLayout that holds a setting cell, found by searching upwards."""
        parent = cell.parentWidget()
        while parent is not None:
            for grid in parent.findChildren(QGridLayout):
                for index in range(grid.count()):
                    if grid.itemAt(index).widget() is cell:
                        return grid
            parent = parent.parentWidget()
        raise AssertionError(f"{cell} is not in a grid")

    def test_a_switch_sits_after_its_caption(self):
        row = self._row_of(self.window.game_check)
        order = [row.itemAt(index).widget() for index in range(row.count())]
        caption = next(w for w in order if w is not self.window.game_check)
        self.assertLess(order.index(caption), order.index(self.window.game_check))
        self.assertIn(tr("check_game"), caption.text())

    def test_the_correction_axes_share_one_row_of_their_own_card(self):
        # A misplaced switch once drew on top of the card title, and the axes were
        # stacked as three full-width rows; they belong side by side in one card.
        from gui.fluent_compat import CardWidget
        cards = self.window.correction.view.findChildren(CardWidget)
        axes = (self.window.timing_check, self.window.pitch_combo, self.window.pitch_map_check)
        for widget in axes:
            owner = self._cell_of(widget)
            while owner is not None and not isinstance(owner, CardWidget):
                owner = owner.parentWidget()
            self.assertIn(owner, cards, f"{widget} is not inside a card")
        grid = self._grid_of(self._cell_of(axes[0]))
        for widget in axes:
            cell = self._cell_of(widget)
            rows = {grid.getItemPosition(i)[0] for i in range(grid.count())
                    if grid.itemAt(i).widget() is cell}
            self.assertEqual(rows, {0}, "the three correction axes must share one row")

    def test_the_run_options_are_packed_sideways(self):
        # The devices and the two top-level toggles each fill one row, and no two
        # settings share a cell.
        for group in ((self.window.onnx_combo, self.window.vocoder_combo, self.window.language_combo),
                      (self.window.game_check, self.window.verify_check)):
            cells = [self._cell_of(w) for w in group]
            self.assertEqual(len({id(c) for c in cells}), len(cells))
            grid = self._grid_of(cells[0])
            for cell in cells:
                rows = {grid.getItemPosition(i)[0] for i in range(grid.count())
                        if grid.itemAt(i).widget() is cell}
                self.assertEqual(rows, {0}, f"{cell} must share one row")

    def test_hints_use_the_secondary_text_label(self):
        from gui.fluent_compat import HintLabel
        for hint in (self.window.correction.axis_hint, self.window.correction.hint_label):
            self.assertIsInstance(hint, HintLabel)
            self.assertIn("font-size: 12px", hint.styleSheet())

    def test_the_progress_bar_only_shows_for_a_running_job(self):
        self.assertFalse(self.window.progress_bar.isVisibleTo(self.window.correction))

    def test_a_switch_caption_follows_the_language(self):
        from gui.i18n import set_language
        set_language("zh")
        switch = self.window.game_check
        switch.setChecked(False)
        try:
            self.assertEqual("关", switch.text())
            set_language("en")
            self.window._on_language_changed("en")
            # A retranslated caption must rewrite the text already on screen, not
            # just the pair it will use the next time the switch is toggled.
            self.assertEqual("Off", switch.text())
            self.assertEqual(("On", "Off"), (switch.getOnText(), switch.getOffText()))
            switch.setChecked(True)
            self.assertEqual("On", switch.text())
        finally:
            set_language("zh")
            self.window._on_language_changed("zh")

    def test_the_status_bar_follows_the_language(self):
        from gui.i18n import set_language
        self.window._show_status("statusbar_job", status="done")
        self.assertIn("done", self.window.statusBar().currentMessage())
        set_language("en")
        self.window._on_language_changed("en")
        try:
            self.assertEqual("Job done.", self.window.statusBar().currentMessage())
        finally:
            set_language("zh")
            self.window._on_language_changed("zh")

    def test_wrapped_text_is_not_clipped(self):
        # A wrapped label reports one line as its minimum unless height-for-width
        # is enabled, which silently cut the last line of the hint sentences.
        for label in (self.window.correction.axis_hint, self.window.correction.hint_label,
                      self.window.results.summary_label, self.window.settings.about_label):
            self.assertTrue(label.sizePolicy().hasHeightForWidth(), label.text()[:24])


if __name__ == "__main__":
    unittest.main()
