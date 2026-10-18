from __future__ import annotations

from pathlib import Path

from application.artifacts import read_json
from gui import text as T
from gui.i18n import tr

# Output files offered for playback, in the order the report presents them.
AUDIO_FILES: tuple[tuple[str, str], ...] = (
    ("audio_teacher", "teacher.reference.wav"),
    ("audio_student", "student.reference.wav"),
    ("audio_corrected", "student.corrected.wav"),
    ("audio_identity", "student.identity.wav"),
    ("audio_pitch_only", "student.pitch_only.wav"),
    ("audio_time_only", "student.time_only.wav"),
)

STATUS_KEYS = {
    "completed": "status_completed",
    "completed_with_unresolved": "status_unresolved",
    "analysis_complete": "status_analysis_only",
    "failed": "status_failed",
    "cancelled": "status_cancelled",
    "running": "status_running",
}


def _ratio(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _cents(value) -> str:
    try:
        return f"{float(value):.2f} 音分"
    except (TypeError, ValueError):
        return "—"


def _seconds(value) -> str:
    try:
        return tr("value_seconds").format(value=float(value))
    except (TypeError, ValueError):
        return "—"


def _read(path: Path) -> dict:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


# Nothing here recomputes a measurement: every number is copied from the files
# the pipeline wrote, with the pipeline's own caveats.
class JobArtifacts:
    """A read-only view of one output directory."""

    def __init__(self, output: Path) -> None:
        self.output = Path(output)

    def available_audio(self) -> list[tuple[str, Path]]:
        return [(tr(key), self.output / name) for key, name in AUDIO_FILES
                if (self.output / name).is_file()]

    def report_path(self) -> Path | None:
        candidate = self.output / "report.html"
        return candidate if candidate.is_file() else None

    def corrected_path(self) -> Path | None:
        candidate = self.output / "student.corrected.wav"
        return candidate if candidate.is_file() else None

    def job(self) -> dict:
        return _read(self.output / "job.json")

    def summary(self) -> tuple[str, str]:
        job = self.job()
        if not job:
            # A directory can hold analysis or render artifacts without a job
            # manifest (a partial run, or files copied out of one). Saying only
            # "no artifacts" would deny material the other tabs do show.
            readable = any((self.output / name).is_file() for name in (
                "alignment.json", "verification.json", "render.json",
                "teacher.analysis.json", "student.analysis.json"))
            return ((tr("status_no_manifest_with_artifacts") if readable
                     else tr("summary_no_job")),
                    tr("status_no_manifest_note") if readable else tr("summary_no_job_note"))
        status = job.get("status", "unknown")
        key = STATUS_KEYS.get(status)
        if key is None:
            return tr("status_unknown"), tr("status_unknown_note")
        note = tr("status_running_note") if status == "running" else tr(f"{key}_note")
        return tr(key), note

    def metrics(self) -> list[tuple[str, str]]:
        plan = _read(self.output / "alignment.json")
        verification = _read(self.output / "verification.json")
        metrics, counts = plan.get("metrics", {}), plan.get("metrics", {}).get("status_counts", {})
        rows: list[tuple[str, str]] = []
        if counts:
            rendered = "，".join(f"{T.status_label(key)} {value}"
                                    for key, value in sorted(counts.items()))
            rows.append((tr("row_lyric_units"), tr("row_lyric_units_value").format(
                teacher=metrics.get("teacher_unit_count", "—"),
                student=metrics.get("student_unit_count", "—"))))
            rows.append((tr("row_unit_status"), rendered))
        if metrics:
            rows.append((tr("row_planned_coverage"), _ratio(metrics.get("lyric_coverage"))))
            rows.append((tr("row_semantic_match"), _ratio(metrics.get("semantic_match_rate"))))
            rows.append((tr("row_unsupported_regions"),
                         str(metrics.get("unsupported_region_count", "—"))))
            rows.append((tr("row_ambiguous_matches"), str(metrics.get("ambiguous_match_count", "—"))))
        coverage = verification.get("coverage", {})
        if coverage:
            rows.append((tr("row_teacher_voiced_coverage"),
                         _ratio(coverage.get("semantic_teacher_voiced_coverage"))))
        audio = verification.get("audio", {})
        if audio:
            state = tr("row_length_exact") if audio.get("sample_count_exact") else tr("row_length_mismatch")
            rows.append((tr("row_exported_length"), tr("row_length_value").format(
                count=audio.get("sample_count", "—"), state=state)))
        return rows

    def pitch_rows(self) -> list[tuple[str, str]]:
        """Measured pitch agreement, kept separate from planned coverage.

        With register mapping, absolute teacher pitch is not the aimed reference."""
        verification = _read(self.output / "verification.json")
        sections = [(tr("pitch_vs_teacher"),
                     verification.get("actual_vs_teacher", {}).get("all_reference_frames", {}))]
        relative = verification.get("actual_vs_teacher_relative", {}).get("all_reference_frames", {})
        if relative:
            sections.append((tr("pitch_vs_teacher_register"), relative))
        sections.append((tr("pitch_vs_target"),
                         verification.get("actual_vs_render_target", {}).get("all_target_frames", {})))
        rows: list[tuple[str, str]] = []
        for label, frames in sections:
            if not frames:
                continue
            rows.append((tr("pitch_row_error").format(label=label),
                         tr("pitch_row_value").format(
                             median=_cents(frames.get("median_absolute_cents")),
                             p95=_cents(frames.get("p95_absolute_cents")),
                             frames=frames.get("joint_voiced_frames", "—"))))
            rows.append((tr("pitch_row_recall").format(label=label),
                         _ratio(frames.get("voicing_recall"))))
        return rows

    def timing_rows(self) -> list[tuple[str, str]]:
        timing = _read(self.output / "timing.verification.json")
        coverage = timing.get("coverage", {})
        if not coverage:
            return []
        rows = [(tr("timing_row_compared"), tr("timing_row_compared_value").format(
                     reliable=coverage.get("reliable_exact_match_count", "—"),
                     total=coverage.get("teacher_unit_count", "—"))),
                (tr("timing_row_coverage"),
                 _ratio(coverage.get("within_tolerance_ratio_of_all_teacher_units")))]
        errors = timing.get("reliable_boundary_errors", {})
        for edge, label in (("onset", tr("timing_row_onset")), ("offset", tr("timing_row_offset"))):
            if errors.get(edge):
                rows.append((label, tr("timing_row_error_value").format(
                    mean=_seconds(errors[edge].get("mean_absolute_ms", 0) / 1000),
                    p95=_seconds(errors[edge].get("p95_absolute_ms", 0) / 1000))))
        return rows

    def warnings(self) -> list[str]:
        """Every pipeline caveat, deduplicated and in a stable order."""
        plan = _read(self.output / "alignment.json")
        teacher = _read(self.output / "teacher.analysis.json")
        student = _read(self.output / "student.analysis.json")
        verification = _read(self.output / "verification.json")
        timing = _read(self.output / "timing.verification.json")
        collected: list[str] = []
        for source in (teacher, student, plan, verification, timing):
            for item in T.warning_list(source.get("warnings", [])):
                if item not in collected:
                    collected.append(item)
        if verification and not verification.get("structural_checks_passed", True):
            collected.insert(0, tr("warning_structural_failure"))
        if verification and verification.get("limitations"):
            for item in T.warning_list(verification["limitations"]):
                labelled = f"{tr("warning_limit_prefix")}{item}"
                if labelled not in collected:
                    collected.append(labelled)
        return collected

    def regions(self) -> list[dict]:
        plan = _read(self.output / "alignment.json")
        rows = []
        for region in plan.get("regions", []):
            if region.get("mode") == "silence" and region.get("status") != "unvoiced_gap":
                continue
            rows.append({"teacher": region.get("teacher_text", ""),
                         "student": region.get("student_text", ""),
                         "status": T.status_label(region.get("status", "")),
                         "mode": T.mode_label(region.get("mode", "")),
                         "quality": region.get("quality"),
                         "teacher_start": region.get("teacher_start"),
                         "teacher_end": region.get("teacher_end"),
                         "reason": T.reason_label(region.get("reason", ""))})
        return rows

    def diagnostics(self) -> list[tuple[str, str]]:
        render = _read(self.output / "render.json")
        experiment = render.get("experiments", {}).get("full", {})
        rows: list[tuple[str, str]] = []
        if render.get("device"):
            rows.append((tr("row_synthesis_device"), str(render["device"])))
        if experiment:
            rows.append((tr("row_rendered_ratio"), _ratio(experiment.get("rendered_sample_ratio"))))
            rows.append((tr("row_voiced_ratio"), _ratio(experiment.get("voiced_frame_ratio"))))
            rows.append((tr("row_timing_changed"),
                         tr("value_yes") if experiment.get("timing_changed") else tr("value_no")))
            policy = experiment.get("pitch_policy")
            if policy:
                rows.append((tr("row_pitch_policy"), T.policy_label(policy)))
            register = experiment.get("register_pitch", {})
            if register.get("placement") == "per_word":
                # Adaptive placement never moves the contour as a whole, so the
                # constant is not the interesting number: the measured register,
                # which anchors the note levels the words are placed on, is.
                rows.append((tr("row_register_offset"), tr("value_semitones").format(
                    value=float(register.get("offset_cents", 0)) / 100)))
                rows.append((tr("row_note_shifts"), tr("row_note_shifts_value").format(
                    moved=int(register.get("note_shift_applied_count", 0) or 0),
                    measured=int(register.get("note_shift_measured_count", 0) or 0))))
            elif register.get("applied"):
                rows.append((tr("row_register_offset"), tr("value_semitones").format(
                    value=float(register.get("contour_shift_cents", 0)) / 100)))
            rows.append((tr("row_muted_regions"), str(experiment.get("muted_region_count", "—"))))
            rows.append((tr("row_acoustic_transitions"),
                         str(len(experiment.get("acoustic_transitions", [])))))
        if render.get("peak_cuda_memory_bytes"):
            rows.append((tr("row_peak_memory"),
                         f"{render['peak_cuda_memory_bytes'] / (1024 ** 2):.1f} MiB"))
        job = self.job()
        if job.get("elapsed_seconds"):
            rows.append((tr("row_job_time"), _seconds(job["elapsed_seconds"])))
        return rows
