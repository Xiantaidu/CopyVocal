"""Generate a self-contained review page alongside the local audio files."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

from application.artifacts import read_json


def write_status_report(output: Path, job: dict) -> None:
    """Replace a previous review while the current job has no final result."""
    status = escape(job.get("status", "unknown"))
    error = escape(job.get("error", ""))
    content = (f'<!doctype html><html lang="en"><meta charset="utf-8">'
               f'<title>CopyVocal · {status}</title><body><h1>CopyVocal</h1>'
               f'<p>Current job: <strong>{status}</strong>.</p><p>{error}</p>'
               '<p>A final correction review is unavailable for this job. '
               'Files retained from earlier runs do not establish a current result.</p>'
               '<p><a href="job.json">Current job manifest</a> · '
               '<a href="processing.log">Processing log</a></p></body></html>')
    (output / "report.html").write_text(content, encoding="utf-8")


def _pitch_path(pitch: dict, duration: float, color: str, label: str) -> str:
    import numpy as np
    times = np.asarray(pitch.get("times", []), dtype=float)
    f0 = np.asarray(pitch.get("f0_hz", []), dtype=float)
    if not len(times) or times.shape != f0.shape:
        return ""
    voiced = np.asarray(pitch.get("voiced", f0 > 0), dtype=bool) & (f0 > 0)
    midi = np.zeros_like(f0)
    midi[voiced] = 69 + 12 * np.log2(f0[voiced] / 440)
    voiced &= np.isfinite(times) & np.isfinite(f0)
    stride = max(1, len(times) // 2500)
    edges = np.diff(np.pad(voiced.astype(np.int8), (1, 1)))
    segments = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        current = []
        indices = list(range(start, stop, stride))
        if indices[-1] != stop - 1:
            indices.append(stop - 1)
        for index in indices:
            x = 45 + min(1, times[index] / max(duration, 1e-8)) * 910
            y = 210 - (min(90, max(30, midi[index])) - 30) * 3
            current.append(f"{x:.2f},{y:.2f}")
        segments.append(current)
    return "".join(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"><title>{label}</title></polyline>'
                   for points in segments)


def write_report(output: Path, teacher: dict, student: dict, plan: dict, job: dict) -> None:
    verification = job.get("stages", {}).get("verification", {})
    metrics = plan["metrics"]
    timing = job.get("stages", {}).get("timing_verification", {})
    counts = metrics["status_counts"]
    warnings = list(dict.fromkeys(teacher.get("warnings", []) + student.get("warnings", []) +
                                 plan.get("warnings", []) + verification.get("warnings", [])))
    if not verification:
        warnings.append("Output waveform verification was skipped for this job.")
    warnings.extend(timing.get("warnings", []))
    audio_cards = []
    current_modes = job.get("stages", {}).get("render", {}).get("experiments", {})
    for label, filename in (("Teacher reference", "teacher.reference.wav"),
                            ("Original student", "student.reference.wav"),
                            ("Corrected student", "student.corrected.wav"),
                            ("Identity reconstruction", "student.identity.wav"),
                            ("Pitch only", "student.pitch_only.wav"),
                            ("Time only", "student.time_only.wav")):
        mode = filename.removeprefix("student.").removesuffix(".wav")
        if mode in {"identity", "pitch_only", "time_only"} and mode not in current_modes:
            continue
        if (output / filename).is_file():
            audio_cards.append(f'<div class="audio"><h3>{label}</h3><audio controls preload="metadata" src="{filename}"></audio><a href="{filename}" download>Download WAV</a></div>')
    word_rows = []
    for region in plan["regions"]:
        if region.get("mode") == "silence" and region.get("status") != "unvoiced_gap":
            continue
        word_rows.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in (
            region.get("teacher_text", ""), region.get("student_text", ""),
            f"{region['teacher_start']:.3f}–{region['teacher_end']:.3f}",
            f"{region.get('student_start', 0):.3f}–{region.get('student_end', 0):.3f}",
            region["status"], region["mode"], f"{region.get('quality', 0):.2f}")) + '</tr>')
    duration = teacher["duration"]
    render_full = job.get("stages", {}).get("render", {}).get("experiments", {}).get("full", {})
    # Which timeline the corrected vocal follows; every clock-dependent statement
    # below depends on this, and the render is the only thing that knows it.
    timing_changed = bool(render_full.get("timing_changed", True))
    register_html = ""
    register = render_full.get("register_pitch", {})
    options = render_full.get("options", {})
    if options or render_full.get("pitch_policy"):
        register_html = "<h2>Correction options</h2><ul>"
        register_html += (f"<li>Timing: {'warped onto the teacher' if render_full.get('timing_changed') else 'student timing kept'}</li>")
        policy = render_full.get("pitch_policy")
        descriptions = {
            "none": "the student's own pitch, unchanged",
            "student_contour_student_register":
                "adaptive placement: the student's own contour, with each word moved onto the teacher's "
                "note level inside the student's own register",
            "teacher_contour_student_register":
                "the teacher's continuous contour, moved into the student's own register -- in adaptive "
                "mode this carries the teacher's note levels there in one move",
            "student_contour_teacher_register":
                "the student's own contour, moved into the teacher's register",
            "teacher_contour_teacher_register":
                "the teacher's contour and register exactly as analysed",
        }
        if policy:
            register_html += f"<li>Pitch: {escape(descriptions.get(policy, policy))}</li>"
        moved = int(register.get("note_shift_applied_count", 0) or 0)
        measured = int(register.get("note_shift_measured_count", 0) or 0)
        if register.get("placement") == "per_word":
            register_html += (f"<li>Per-word placement: {moved} of {measured} measured word(s) were moved "
                              "onto their target note. The rest were already on it, carried voiced pitch on "
                              "only one side, or needed a move larger than the safe correction limit.</li>")
        if register.get("applied") and register.get("offset_available"):
            applied = float(register.get("contour_shift_cents", 0)) / 100
            raw = float(register.get("raw_offset_cents") or 0) / 100
            regions = register.get("comparable_regions")
            if abs(applied) < 0.005:
                register_html += (f"<li>Register move: none needed. The two singers are already in the same "
                                  f"register: the measured difference is {raw:+.2f} semitones over {regions} "
                                  "matched regions, which quantizes to zero, so the melody is left at its own "
                                  "level.</li>")
            else:
                register_html += (f"<li>Register move: {applied:+.2f} semitones applied to the chosen contour. "
                                  f"The measured difference is {raw:+.2f} semitones (from {regions} matched "
                                  "regions); it is quantized to whole semitones so the melody stays in a "
                                  "well-tempered key. A negative value means the student sings lower, so the "
                                  "melody moves down into their register instead of forcing them up.</li>")
        elif register.get("applied") and register.get("offset_reason"):
            register_html += f"<li>Register move: none. {escape(str(register['offset_reason']))}</li>"
        elif register.get("offset_reason") and policy != "none":
            register_html += f"<li>Pitch: {escape(str(register['offset_reason']))}</li>"
        register_html += "</ul>"
    transition_edits = render_full.get("acoustic_transitions", [])
    transition_html = ""
    if transition_edits:
        transition_html = ("<h2>Continuous acoustic transitions</h2>"
                           f"<p>{len(transition_edits)} interword gaps use continuous source audio "
                           "and local acoustic timing. Unvoiced intervals may contain consonants, "
                           "breaths, and room sound; they are not assumed silent. "
                           "Lyric annotations remain model estimates. Missing/rejected words "
                           "remain muted.</p><details><summary>Show acoustic timing changes</summary><pre>" +
                           escape(json.dumps(transition_edits, ensure_ascii=False, indent=2)) +
                           "</pre></details>")
        dynamics = render_full.get("acoustic_dynamics", [])
        if dynamics:
            constrained = sum(item.get("status") == "constrained" for item in dynamics)
            transition_html += (f"<p>{constrained} transitions use smooth level reductions "
                                "where the teacher is quiet relative to nearby sung phones. "
                                "Audible unvoiced consonants retain their source material.</p>"
                                "<details><summary>Show acoustic level changes</summary><pre>" +
                                escape(json.dumps(dynamics, ensure_ascii=False, indent=2)) +
                                "</pre></details>")
    pitch_svg = _pitch_path(teacher["pitch"], duration, "#7cd8ff", "Teacher F0")
    measured_name = Path(verification.get("measured_pitch_path", "verification.measured_pitch.json")).name
    measured_path = output / measured_name
    actual = read_json(measured_path) if verification and measured_path.is_file() else {}
    if actual:
        pitch_svg += _pitch_path(actual, duration, "#ffcb82", "Corrected F0")
    chart = f'<svg viewBox="0 0 1000 250" role="img" aria-label="Pitch over teacher time"><rect x="45" y="30" width="910" height="180" rx="4" fill="#111c2e"/>{pitch_svg}<text x="45" y="235" fill="#9dafc6">0 s</text><text x="900" y="235" fill="#9dafc6">{duration:.2f} s</text></svg>'
    details = escape(json.dumps({key: value for key, value in verification.items() if key != "output_pitch"}, ensure_ascii=False, indent=2))
    pitch_metrics = verification.get("actual_vs_teacher", {}).get("all_reference_frames", {})
    relative_metrics = verification.get("actual_vs_teacher_relative", {}).get("all_reference_frames", {})
    coverage_metrics = verification.get("coverage", {})
    diagnostic_cards = []
    for label, value, scale, suffix in (
        ("Median pitch error", pitch_metrics.get("median_absolute_cents"), 1, " cents"),
        ("Pitch error p95", pitch_metrics.get("p95_absolute_cents"), 1, " cents"),
        ("Teacher voiced recall", pitch_metrics.get("voicing_recall"), 100, "%"),
        ("Median error vs teacher in student register", relative_metrics.get("median_absolute_cents"), 1, " cents"),
        ("Planned semantic voiced coverage", coverage_metrics.get("semantic_teacher_voiced_coverage"), 100, "%"),
    ):
        if value is not None:
            diagnostic_cards.append(f'<span class="tag">{label}: {value * scale:.2f}{suffix}</span>')
    diagnostic_summary = ''.join(diagnostic_cards)
    register_note = (' Pitch was mapped relative to the student\'s own register, so the teacher-referenced error '
                     'includes the intended register offset; the register-matched row is the aimed accuracy.'
                     if relative_metrics else '')
    timing_html = '<p class="muted">Output word boundaries were not remeasured for this job.</p>'
    if timing:
        coverage = timing.get("coverage")
        errors = timing.get("reliable_boundary_errors")
        if coverage is None or errors is None:
            # Timing was kept on the student's clock, so the teacher's boundaries do
            # not describe the same moments; the comparison was not attempted and
            # the report says why instead of printing nothing or inventing numbers.
            reason = timing.get("measurement", {}).get("reason") or "no comparison was possible."
            timing_html = ('<p class="muted">Output word timing was not compared with the teacher: ' +
                           escape(str(reason)) + '</p>')
        else:
            onset = errors["onset"]["median_absolute_ms"]
            offset = errors["offset"]["median_absolute_ms"]
            timing_html = (f'<p>Reliable exact words: {coverage["reliable_exact_match_count"]} / '
                           f'{coverage["teacher_unit_count"]}. Median absolute onset / offset error: '
                           f'{onset:.1f} / {offset:.1f} ms.</p>' if onset is not None and offset is not None else
                           '<p>No reliable exact word observations are available for timing errors.</p>')
            timing_html += ('<p class="muted">Fresh output ASR and TiFA receive no teacher words or boundaries. '
                            'These are model estimates; uncovered words remain in the coverage denominator.</p>')
        timing_html += ('<details><summary>Show timing measurements</summary><pre>' +
                        escape(json.dumps(timing, ensure_ascii=False, indent=2)) + '</pre></details>')
    output_clock_note = ("The output follows the teacher clock." if timing_changed else
                        "The output follows the student clock, so its own times differ from the teacher's.")
    clock_heading = "Pitch on the teacher timeline" if timing_changed else "Pitch on the student timeline"
    clock_note = ("" if timing_changed else
                  " This render keeps the student's timeline, so the teacher's curve is drawn on the "
                  "teacher's own clock and the detected output on the student's; the two are not "
                  "sample-aligned and their horizontal positions are not comparable.")
    content = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CopyVocal review</title>
<style>body{{margin:0;background:#0b1220;color:#e8eff9;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:36px 24px}}h1{{font-size:36px;margin:0}}h2{{margin-top:34px}}h3{{margin:0 0 12px;font-size:16px}}.muted{{color:#9dafc6}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:16px}}.audio,.summary{{background:#152136;border:1px solid #283850;border-radius:12px;padding:18px}}audio{{width:100%;display:block;margin-bottom:9px}}a{{color:#7cd8ff}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #29364a;text-align:left}}.scroll{{overflow:auto}}pre{{overflow:auto;max-height:600px;background:#101b2e;padding:18px;border-radius:10px}}svg{{width:100%;height:auto}}li{{margin-bottom:6px}}.tag{{display:inline-block;margin:0 8px 6px 0;padding:3px 9px;border-radius:5px;background:#243851}}</style></head>
<body><main><h1>CopyVocal</h1><p class="muted">Student voice · Teacher timing and melody</p>
<div class="summary"><span class="tag">{escape(job.get('status', 'unknown'))}</span><span class="tag">Teacher {duration:.3f} s</span><span class="tag">Student {student['duration']:.3f} s</span><span class="tag">Supported lyric duration {100 * metrics['lyric_coverage']:.1f}%</span><span class="tag">Unresolved regions {metrics['unsupported_region_count']}</span>
<p>{escape(json.dumps(counts, ensure_ascii=False))}</p><p class="muted">Alignment quality is a structural score, not a calibrated model confidence. Unsupported material is reported explicitly.</p></div>
<h2>Listen and compare</h2><div class="grid">{''.join(audio_cards)}</div>
    <h2>{clock_heading}</h2><p class="muted">Blue: teacher. Orange: detected output when verification is available. Unvoiced intervals are omitted.{clock_note}</p>{chart}
    {register_html}
<h2>Correspondences</h2><p class="muted">Times are the planned teacher and student intervals. {output_clock_note}</p><div class="scroll"><table><thead><tr><th>Teacher</th><th>Student</th><th>Teacher time</th><th>Student time</th><th>Status</th><th>Action</th><th>Quality</th></tr></thead><tbody>{''.join(word_rows)}</tbody></table></div>
{transition_html}
<h2>Review notes</h2><ul>{''.join('<li>'+escape(warning)+'</li>' for warning in warnings) or '<li>No alignment warnings.</li>'}</ul>
<h2>Measured diagnostics</h2><div>{diagnostic_summary}</div><p class="muted">Pitch errors cover jointly voiced frames. Voiced recall retains missing material. Planned semantic coverage comes from the alignment plan. Pitch checks do not certify timbre preservation.{register_note}</p><details><summary>Show verification JSON</summary><pre>{details}</pre></details>
<h2>Remeasured word timing</h2>{timing_html}
<p><a href="alignment.json">Alignment JSON</a> · <a href="teacher.TextGrid">Teacher TextGrid</a> · <a href="student.TextGrid">Student TextGrid</a> · <a href="job.json">Job manifest</a></p>
</main></body></html>'''
    (output / "report.html").write_text(content, encoding="utf-8")
