from __future__ import annotations

from pathlib import Path

from application.artifacts import read_json
from gui import text as T

# Output files offered for playback, in the order the report presents them.
AUDIO_FILES: tuple[tuple[str, str], ...] = (
    (T.AUDIO_TEACHER, "teacher.reference.wav"),
    (T.AUDIO_STUDENT, "student.reference.wav"),
    (T.AUDIO_CORRECTED, "student.corrected.wav"),
    (T.AUDIO_IDENTITY, "student.identity.wav"),
    (T.AUDIO_PITCH_ONLY, "student.pitch_only.wav"),
    (T.AUDIO_TIME_ONLY, "student.time_only.wav"),
)

STATUS_SUMMARY = {
    "completed": T.STATUS_COMPLETED,
    "completed_with_unresolved": T.STATUS_UNRESOLVED,
    "analysis_complete": T.STATUS_ANALYSIS_ONLY,
    "failed": T.STATUS_FAILED,
    "cancelled": T.STATUS_CANCELLED,
    "running": T.STATUS_RUNNING,
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
        return T.VALUE_SECONDS.format(value=float(value))
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
        return [(label, self.output / name) for label, name in AUDIO_FILES
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
            return ((T.STATUS_NO_MANIFEST_WITH_ARTIFACTS if readable
                     else T.SUMMARY_NO_JOB),
                    T.STATUS_NO_MANIFEST_NOTE if readable else T.SUMMARY_NO_JOB_NOTE)
        status = job.get("status", "unknown")
        headline, note = STATUS_SUMMARY.get(status, (T.STATUS_UNKNOWN, T.STATUS_UNKNOWN_NOTE))
        if status == "running":
            note = T.STATUS_RUNNING_NOTE
        return headline, note

    def metrics(self) -> list[tuple[str, str]]:
        plan = _read(self.output / "alignment.json")
        verification = _read(self.output / "verification.json")
        metrics, counts = plan.get("metrics", {}), plan.get("metrics", {}).get("status_counts", {})
        rows: list[tuple[str, str]] = []
        if counts:
            rendered = "，".join(f"{T.status_label(key)} {value}"
                                    for key, value in sorted(counts.items()))
            rows.append((T.ROW_LYRIC_UNITS, T.ROW_LYRIC_UNITS_VALUE.format(
                teacher=metrics.get("teacher_unit_count", "—"),
                student=metrics.get("student_unit_count", "—"))))
            rows.append((T.ROW_UNIT_STATUS, rendered))
        if metrics:
            rows.append((T.ROW_PLANNED_COVERAGE, _ratio(metrics.get("lyric_coverage"))))
            rows.append((T.ROW_SEMANTIC_MATCH, _ratio(metrics.get("semantic_match_rate"))))
            rows.append((T.ROW_UNSUPPORTED_REGIONS,
                         str(metrics.get("unsupported_region_count", "—"))))
            rows.append((T.ROW_AMBIGUOUS_MATCHES, str(metrics.get("ambiguous_match_count", "—"))))
        coverage = verification.get("coverage", {})
        if coverage:
            rows.append((T.ROW_TEACHER_VOICED_COVERAGE,
                         _ratio(coverage.get("semantic_teacher_voiced_coverage"))))
        audio = verification.get("audio", {})
        if audio:
            state = T.ROW_LENGTH_EXACT if audio.get("sample_count_exact") else T.ROW_LENGTH_MISMATCH
            rows.append((T.ROW_EXPORTED_LENGTH, T.ROW_LENGTH_VALUE.format(
                count=audio.get("sample_count", "—"), state=state)))
        return rows

    def pitch_rows(self) -> list[tuple[str, str]]:
        """Measured pitch agreement, kept separate from planned coverage.

        With register mapping, absolute teacher pitch is not the aimed reference."""
        verification = _read(self.output / "verification.json")
        sections = [(T.PITCH_VS_TEACHER,
                     verification.get("actual_vs_teacher", {}).get("all_reference_frames", {}))]
        relative = verification.get("actual_vs_teacher_relative", {}).get("all_reference_frames", {})
        if relative:
            sections.append((T.PITCH_VS_TEACHER_REGISTER, relative))
        sections.append((T.PITCH_VS_TARGET,
                         verification.get("actual_vs_render_target", {}).get("all_target_frames", {})))
        rows: list[tuple[str, str]] = []
        for label, frames in sections:
            if not frames:
                continue
            rows.append((T.PITCH_ROW_ERROR.format(label=label),
                         T.PITCH_ROW_VALUE.format(
                             median=_cents(frames.get("median_absolute_cents")),
                             p95=_cents(frames.get("p95_absolute_cents")),
                             frames=frames.get("joint_voiced_frames", "—"))))
            rows.append((T.PITCH_ROW_RECALL.format(label=label),
                         _ratio(frames.get("voicing_recall"))))
        return rows

    def timing_rows(self) -> list[tuple[str, str]]:
        timing = _read(self.output / "timing.verification.json")
        coverage = timing.get("coverage", {})
        if not coverage:
            return []
        rows = [(T.TIMING_ROW_COMPARED, T.TIMING_ROW_COMPARED_VALUE.format(
                     reliable=coverage.get("reliable_exact_match_count", "—"),
                     total=coverage.get("teacher_unit_count", "—"))),
                (T.TIMING_ROW_COVERAGE,
                 _ratio(coverage.get("within_tolerance_ratio_of_all_teacher_units")))]
        errors = timing.get("reliable_boundary_errors", {})
        for edge, label in (("onset", T.TIMING_ROW_ONSET), ("offset", T.TIMING_ROW_OFFSET)):
            if errors.get(edge):
                rows.append((label, T.TIMING_ROW_ERROR_VALUE.format(
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
            collected.insert(0, T.WARNING_STRUCTURAL_FAILURE)
        if verification and verification.get("limitations"):
            for item in T.warning_list(verification["limitations"]):
                labelled = f"{T.WARNING_LIMIT_PREFIX}{item}"
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
            rows.append((T.ROW_SYNTHESIS_DEVICE, str(render["device"])))
        if experiment:
            rows.append((T.ROW_RENDERED_RATIO, _ratio(experiment.get("rendered_sample_ratio"))))
            rows.append((T.ROW_VOICED_RATIO, _ratio(experiment.get("voiced_frame_ratio"))))
            rows.append((T.ROW_TIMING_CHANGED,
                         T.VALUE_YES if experiment.get("timing_changed") else T.VALUE_NO))
            policy = experiment.get("pitch_policy")
            if policy:
                rows.append((T.ROW_PITCH_POLICY, T.policy_label(policy)))
            register = experiment.get("register_pitch", {})
            if register.get("placement") == "per_word":
                # Adaptive placement never moves the contour as a whole, so the
                # constant is not the interesting number: the measured register,
                # which anchors the note levels the words are placed on, is.
                rows.append((T.ROW_REGISTER_OFFSET, T.VALUE_SEMITONES.format(
                    value=float(register.get("offset_cents", 0)) / 100)))
                rows.append((T.ROW_NOTE_SHIFTS, T.ROW_NOTE_SHIFTS_VALUE.format(
                    moved=int(register.get("note_shift_applied_count", 0) or 0),
                    measured=int(register.get("note_shift_measured_count", 0) or 0))))
            elif register.get("applied"):
                rows.append((T.ROW_REGISTER_OFFSET, T.VALUE_SEMITONES.format(
                    value=float(register.get("contour_shift_cents", 0)) / 100)))
            rows.append((T.ROW_MUTED_REGIONS, str(experiment.get("muted_region_count", "—"))))
            rows.append((T.ROW_ACOUSTIC_TRANSITIONS,
                         str(len(experiment.get("acoustic_transitions", [])))))
        if render.get("peak_cuda_memory_bytes"):
            rows.append((T.ROW_PEAK_MEMORY,
                         f"{render['peak_cuda_memory_bytes'] / (1024 ** 2):.1f} MiB"))
        job = self.job()
        if job.get("elapsed_seconds"):
            rows.append((T.ROW_JOB_TIME, _seconds(job["elapsed_seconds"])))
        return rows
