"""Measure rendered audio without treating planned boundaries as observations."""
from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from inference.correction.register import STUDENT_REGISTER_POLICIES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    from scipy.signal import resample_poly
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(resample_poly(audio, target_rate // divisor, source_rate // divisor), dtype=np.float32)


def _pitch_arrays(pitch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(pitch.get("times", []), dtype=np.float64)
    f0 = np.asarray(pitch.get("f0_hz", []), dtype=np.float64)
    voiced = np.asarray(pitch.get("voiced", f0 > 0), dtype=bool)
    if times.ndim != 1 or f0.shape != times.shape or voiced.shape != times.shape:
        raise ValueError("Pitch arrays must be equally sized one-dimensional arrays.")
    if not np.isfinite(times).all() or not np.isfinite(f0).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Pitch values must be finite and timestamps strictly increasing.")
    return times, f0, voiced & (f0 > 0)


def _nearest(times: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(times):
        return np.zeros(query.shape, dtype=int), np.zeros(query.shape, dtype=bool)
    right = np.clip(np.searchsorted(times, query), 0, len(times) - 1)
    left = np.maximum(0, right - 1)
    nearest = np.where(np.abs(query - times[left]) <= np.abs(query - times[right]), left, right)
    half_left = (times[1] - times[0]) / 2 if len(times) > 1 else 0.005
    half_right = (times[-1] - times[-2]) / 2 if len(times) > 1 else 0.005
    valid = np.isfinite(query) & (query >= times[0] - half_left) & (query < times[-1] + half_right)
    return nearest, valid


def sample_measured_pitch(pitch: dict, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample observations with explicit availability and separate voicing.

    This implementation is separate from the rendering pitch sampler. It
    interpolates only when both bracketing observations are voiced.
    """
    query = np.asarray(query_times, dtype=np.float64)
    times, f0, voiced = _pitch_arrays(pitch)
    output = np.zeros(query.shape, dtype=np.float64)
    nearest, available = _nearest(times, query)
    if not len(times):
        return output, available.copy(), available
    output_voiced = available & voiced[nearest]
    output[output_voiced] = f0[nearest[output_voiced]]
    if len(times) > 1:
        right = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
        left = right - 1
        interpolated = output_voiced & voiced[left] & voiced[right]
        fraction = np.clip((query[interpolated] - times[left[interpolated]]) /
                           (times[right[interpolated]] - times[left[interpolated]]), 0, 1)
        output[interpolated] = np.exp((1 - fraction) * np.log(f0[left[interpolated]]) +
                                      fraction * np.log(f0[right[interpolated]]))
    return output, output_voiced, available


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _error_statistics(cents: np.ndarray) -> dict:
    absolute = np.abs(cents)
    if not len(absolute):
        return {"joint_voiced_frames": 0, "mean_absolute_cents": None, "median_absolute_cents": None,
                "p95_absolute_cents": None, "median_signed_cents": None, "within_50_cents_ratio": None,
                "within_100_cents_ratio": None, "octave_error_ratio": None}
    octave = np.rint(cents / 1200)
    return {"joint_voiced_frames": int(len(cents)), "mean_absolute_cents": float(np.mean(absolute)),
            "median_absolute_cents": float(np.median(absolute)), "p95_absolute_cents": float(np.percentile(absolute, 95)),
            "median_signed_cents": float(np.median(cents)), "within_50_cents_ratio": float(np.mean(absolute <= 50)),
            "within_100_cents_ratio": float(np.mean(absolute <= 100)),
            "octave_error_ratio": float(np.mean((octave != 0) & (np.abs(cents - octave * 1200) <= 100)))}


def compare_pitch(actual: dict, reference: dict, *, selection: np.ndarray | None = None,
                  actual_duration: float | None = None, reference_duration: float | None = None) -> dict:
    """Count missing output in the reference denominator, including UV misses."""
    times, expected_f0, expected_voiced = _pitch_arrays(reference)
    predicted_f0, predicted_voiced, available = sample_measured_pitch(actual, times)
    selected = np.ones(len(times), dtype=bool) if selection is None else np.asarray(selection, dtype=bool).copy()
    if selected.shape != times.shape:
        raise ValueError("Comparison selection must match reference timestamps.")
    if reference_duration is not None:
        selected &= (times >= 0) & (times < reference_duration)
    if actual_duration is not None:
        available &= (times >= 0) & (times < actual_duration)
        predicted_voiced &= available
    ref_voice = selected & expected_voiced
    out_voice = selected & predicted_voiced
    joint = ref_voice & out_voice
    cents = 1200 * np.log2(predicted_f0[joint] / expected_f0[joint])
    errors = _error_statistics(cents)
    ref_count, out_count, joint_count = int(ref_voice.sum()), int(out_voice.sum()), int(joint.sum())
    count = int(selected.sum())
    return {**errors, "evaluated_frames": count, "available_output_frames": int((selected & available).sum()),
            "reference_voiced_frames": ref_count, "output_voiced_frames": out_count,
            "output_availability_ratio": _ratio(int((selected & available).sum()), count),
            "voicing_precision": _ratio(joint_count, out_count), "voicing_recall": _ratio(joint_count, ref_count),
            "voicing_accuracy": _ratio(int((selected & available & (predicted_voiced == expected_voiced)).sum()), count),
            "missed_voiced_frames": int((ref_voice & ~predicted_voiced).sum()),
            "false_voiced_frames": int((out_voice & ~expected_voiced).sum()),
            "correct_voiced_within_50_cents_ratio": _ratio(int((np.abs(cents) <= 50).sum()), ref_count),
            "correct_voiced_within_100_cents_ratio": _ratio(int((np.abs(cents) <= 100).sum()), ref_count),
            "pitch_error_denominator": "Frames voiced in both output and reference; total voiced coverage is reported separately."}


def _shifted_pitch(pitch: dict, cents: float) -> dict:
    """One singer's pitch track transposed by a constant offset."""
    times, f0, voiced = _pitch_arrays(pitch)
    return {"times": times.tolist(), "f0_hz": (f0 * 2.0 ** (float(cents) / 1200.0)).tolist(),
            "voiced": voiced.tolist()}


def _region_mask(plan: dict, times: np.ndarray, *, semantic_only: bool = True) -> np.ndarray:
    selected = np.zeros(len(times), dtype=bool)
    for region in plan.get("regions", []):
        eligible = region.get("mode", "synthesize") == "synthesize" and not region.get("unsupported", False)
        if not semantic_only:
            eligible |= region.get("mode") == "acoustic_fallback"
        if eligible:
            selected |= (times >= float(region["teacher_start"])) & (times < float(region["teacher_end"]))
    return selected


def _audio_metrics(audio: np.ndarray, rate: int, expected: dict, plan: dict,
                   *, clock: str = "teacher") -> dict:
    """Measure the exported waveform against the clock it was rendered on.

    ``expected`` is the analysis whose sample count the output should match: the
    teacher's when the timing was changed, the student's when it was kept. The
    planned edit boundaries are only indexed as output times on the teacher
    clock; on the student clock they describe a different timeline, so the seam
    diagnostic is skipped rather than reported against the wrong positions.
    """
    finite = np.isfinite(audio)
    safe = np.where(finite, audio, 0).astype(np.float64)
    peak = float(np.max(np.abs(safe))) if safe.size else 0.0
    mono = np.mean(safe, axis=1)
    derivative = np.diff(mono)
    seams = []
    boundaries = [] if clock != "teacher" else sorted(
        {float(region[key]) for region in plan.get("regions", [])
         for key in ("teacher_start", "teacher_end") if key in region})
    radius = max(8, int(rate * 0.01))
    for time in boundaries:
        index = int(round(time * rate))
        if 0 < index < len(mono):
            local = derivative[max(0, index - radius):min(len(derivative), index + radius)]
            local_rms = float(np.sqrt(np.mean(local ** 2))) if len(local) else 0
            jump = float(abs(mono[index] - mono[index - 1]))
            seams.append({"time": time, "sample_jump": jump, "local_derivative_rms": local_rms,
                          "jump_to_local_derivative_rms": jump / local_rms if local_rms > 1e-12 else None})
    expected_rate = int(expected.get("sample_rate", rate))
    expected_count = int(expected.get("sample_count",
                                       round(float(expected.get("duration", 0)) * expected_rate)))
    return {"sample_rate": rate, "channels": int(audio.shape[1]), "sample_count": int(len(audio)),
            "output_clock": clock, "expected_sample_rate": expected_rate,
            "expected_sample_count": expected_count,
            "sample_rate_exact": rate == expected_rate, "sample_count_exact": len(audio) == expected_count,
            "sample_count_error": int(len(audio) - expected_count), "duration": len(audio) / rate,
            "expected_duration": expected_count / expected_rate,
            "duration_error_seconds": len(audio) / rate - expected_count / expected_rate,
            "finite": bool(finite.all()), "nonfinite_sample_count": int((~finite).sum()),
            "peak": peak, "rms": float(np.sqrt(np.mean(safe ** 2))) if safe.size else 0.0,
            "over_full_scale_sample_count": int((np.abs(safe) > 1).sum()),
            "near_full_scale_sample_count": int((np.abs(safe) >= 1 - 1 / 32768).sum()),
            "near_full_scale_sample_ratio": float(np.mean(np.abs(safe) >= 1 - 1 / 32768)) if safe.size else 0.0,
            "maximum_adjacent_sample_step": float(np.max(np.abs(derivative))) if len(derivative) else 0.0,
            "p99_9_adjacent_sample_step": float(np.percentile(np.abs(derivative), 99.9)) if len(derivative) else 0.0,
            "plan_boundary_seams": seams,
            "seam_interpretation": ("Sample discontinuity diagnostics at planned edit boundaries; natural "
                                    "transients can also produce large values." if clock == "teacher" else
                                    "Not measured: the planned boundaries are teacher times and this output "
                                    "keeps the student's timeline.")}


def _measure_rmvpe(audio: np.ndarray, rate: int, root: Path, device: str) -> dict:
    from inference.API.rmvpe_api import RmvpeTranscriber
    model = RmvpeTranscriber(root / "models" / "RMVPE", device=device, batch_size=1, threshold=0.03)
    try:
        result = model.infer(audio, rate)
        if result.voiced_mask is None:
            raise RuntimeError("Output RMVPE measurement requires an independent voiced mask.")
        midi = np.asarray(result.midi_pitch, dtype=np.float64)
        voiced = np.asarray(result.voiced_mask, dtype=bool) & np.isfinite(midi)
        if midi.shape != voiced.shape:
            raise RuntimeError("Output RMVPE returned inconsistent pitch and voicing lengths.")
        times = np.arange(len(midi), dtype=np.float64) * result.time_step_seconds
        valid = times < len(audio) / rate
        f0 = np.zeros(len(midi), dtype=np.float64)
        f0[voiced] = 440 * 2 ** ((midi[voiced] - 69) / 12)
        return {"times": times[valid].tolist(), "f0_hz": f0[valid].tolist(), "voiced": voiced[valid].tolist(),
                "method": "RMVPE remeasurement of exported audio", "threshold": 0.03,
                "model_sha256": _sha256(model.model_path),
                "time_step_seconds": float(result.time_step_seconds)}
    finally:
        model.shutdown()
        del model
        gc.collect()


def yin_window(frame: np.ndarray, rate: int, *, minimum_f0: float = 65.0,
               maximum_f0: float = 1100.0, threshold: float = 0.15) -> tuple[float, float]:
    """Independent YIN cumulative difference estimate for one real audio window."""
    frame = np.asarray(frame, dtype=np.float64)
    if not len(frame) or not np.isfinite(frame).all():
        return 0.0, 0.0
    frame = frame - np.mean(frame)
    if float(np.sqrt(np.mean(frame ** 2))) < 0.002:
        return 0.0, 0.0
    minimum_tau = max(2, int(rate / maximum_f0))
    maximum_tau = min(len(frame) // 2, int(math.ceil(rate / minimum_f0)))
    if maximum_tau <= minimum_tau:
        return 0.0, 0.0
    fft_size = 1 << (2 * len(frame) - 1).bit_length()
    spectrum = np.fft.rfft(frame, n=fft_size)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[:maximum_tau + 1]
    cumulative = np.concatenate(([0.0], np.cumsum(frame ** 2)))
    lags = np.arange(maximum_tau + 1)
    difference = cumulative[len(frame) - lags] + cumulative[-1] - cumulative[lags] - 2 * correlation
    difference = np.maximum(difference, 0)
    normalized = np.ones(maximum_tau + 1)
    normalized[1:] = difference[1:] * lags[1:] / np.maximum(np.cumsum(difference[1:]), 1e-15)
    candidates = np.flatnonzero(normalized[minimum_tau:maximum_tau] < threshold) + minimum_tau
    if not len(candidates):
        return 0.0, max(0.0, 1 - float(np.min(normalized[minimum_tau:maximum_tau + 1])))
    tau = int(candidates[0])
    while tau < maximum_tau and normalized[tau + 1] < normalized[tau]:
        tau += 1
    refined = float(tau)
    if 0 < tau < maximum_tau:
        left, center, right = normalized[tau - 1:tau + 2]
        denominator = left - 2 * center + right
        if abs(denominator) > 1e-12:
            refined += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return float(rate / refined), max(0.0, 1 - float(normalized[tau]))


def _yin_check(audio: np.ndarray, rate: int, actual: dict, teacher: dict, plan: dict,
               target: dict | None, *, clock: str = "teacher", maximum_windows: int = 256) -> dict:
    times, f0, voiced = _pitch_arrays(actual)
    if clock == "teacher":
        eligible = voiced & (f0 >= 65) & (f0 <= 1100) & _region_mask(plan, times)
    else:
        # The planned regions are teacher times; selecting output-clock frames with
        # them would pick windows unrelated to the plan. The diagnostic still runs on
        # the voiced frames the output actually has.
        eligible = voiced & (f0 >= 65) & (f0 <= 1100)
    # Avoid onsets and unstable observations; selection affects this diagnostic only.
    for shift in (-2, -1, 1, 2):
        shifted = np.clip(np.arange(len(f0)) + shift, 0, max(0, len(f0) - 1))
        ratio = np.ones(len(f0))
        valid = voiced & voiced[shifted]
        ratio[valid] = f0[valid] / f0[shifted[valid]]
        eligible &= valid & (np.abs(1200 * np.log2(ratio)) <= 150)
    indices = np.flatnonzero(eligible)
    candidate_count = len(indices)
    if candidate_count > maximum_windows:
        indices = indices[np.linspace(0, candidate_count - 1, maximum_windows).astype(int)]
    converted_rate = 16000
    waveform = _resample(audio, rate, converted_rate)
    values = []
    for index in indices:
        center = int(round(times[index] * converted_rate))
        begin, end = center - 1024, center + 1024
        if begin < 0 or end > len(waveform):
            continue
        estimate, periodicity = yin_window(waveform[begin:end], converted_rate)
        if estimate > 0 and periodicity >= 0.85:
            values.append((float(times[index]), estimate, periodicity))
    measured = {"times": [row[0] for row in values], "f0_hz": [row[1] for row in values], "voiced": [True] * len(values)}
    query = np.asarray(measured["times"])
    reports = {}
    references = [("vs_output_rmvpe", actual)]
    if clock == "teacher":
        references.append(("vs_teacher", teacher))
    for label, reference in references:
        reference_f0, reference_voiced, _ = sample_measured_pitch(reference, query)
        selected = reference_voiced
        cents = 1200 * np.log2(np.asarray(measured["f0_hz"])[selected] / reference_f0[selected])
        reports[label] = _error_statistics(cents)
    if target is not None:
        reference_f0, reference_voiced, _ = sample_measured_pitch(target, query)
        estimates = np.asarray(measured["f0_hz"])
        reports["vs_render_target"] = _error_statistics(1200 * np.log2(estimates[reference_voiced] / reference_f0[reference_voiced]))
    limitation = ("A selected stable and energetic voiced subset only; YIN can choose an octave or a strong "
                  "harmonic and does not validate words, consonants, or timbre.")
    if clock != "teacher":
        limitation += (" The teacher comparison is not measured: this output keeps the student's timeline, so "
                       "the teacher's pitch does not describe these timestamps.")
    return {"clock": clock, "method": "Independent YIN cumulative difference on 128 ms output windows at 16 kHz",
            "candidate_stable_voiced_frames": candidate_count, "attempted_windows": int(len(indices)),
            "accepted_windows": len(values), "maximum_windows": maximum_windows,
            "accepted_fraction_of_candidates": _ratio(len(values), candidate_count),
            "median_periodicity": float(np.median([row[2] for row in values])) if values else None,
            **reports,
            "limitations": limitation}


def _target_clock(target_pitch_path: str | Path | None) -> str | None:
    """Read the clock the render declared, or None when no target was written."""
    if target_pitch_path is None:
        return None
    with np.load(Path(target_pitch_path).resolve(strict=True), allow_pickle=False) as data:
        if "timing_changed" not in data:
            # An older render left no clock flag; those all warped the timing.
            return "teacher"
        return "teacher" if bool(np.asarray(data["timing_changed"]).reshape(-1)[0]) else "student"


def verify_output(audio_path: str | Path, teacher_analysis: dict, alignment_plan: dict,
                  target_pitch_path: str | Path | None = None, *, root: str | Path | None = None,
                  device: str = "dml", student_analysis: dict | None = None,
                  clock: str | None = None) -> dict:
    """Remeasure exported output after synthesis has released accelerator memory.

    ``clock`` is the timeline the output was rendered on; it defaults to what the
    target pitch file declares. On the teacher clock the output is expected to be
    exactly as long as the teacher after resampling, and every teacher-referenced
    comparison is meaningful. On the student clock the output is expected to match
    the student instead, and a teacher-timeline comparison would be comparing
    unrelated positions, so it is reported as not measured with the reason.
    """
    path = Path(audio_path).resolve(strict=True)
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if clock is None:
        clock = _target_clock(target_pitch_path) or "teacher"
    if clock not in {"teacher", "student"}:
        raise ValueError("The output clock must be the teacher or the student timeline.")
    if clock == "student" and student_analysis is None:
        raise ValueError("Student-clock verification requires the student analysis to size the output.")
    expected = teacher_analysis if clock == "teacher" else student_analysis
    audio_metrics = _audio_metrics(audio, rate, expected, alignment_plan, clock=clock)
    warnings = []
    limitations = [
        "RMVPE is also used for teacher analysis; agreement is not an independent proof of acoustic correctness.",
        "Word and phoneme timings have not been remeasured in the rendered waveform; planned boundaries are not measured output boundaries.",
        "No objective timbre-preservation score is claimed; listening to identity and corrected samples is required.",
    ]
    if clock == "student":
        limitations.append(
            "The output keeps the student's timeline, so it is not sample-aligned with the teacher and the "
            "teacher-timeline pitch and coverage metrics are not measured.")
    structural_passed = bool(len(audio) and audio_metrics["finite"] and audio_metrics["sample_rate_exact"] and
                             audio_metrics["sample_count_exact"] and not audio_metrics["over_full_scale_sample_count"])
    report = {"verification_status": "measured_with_limitations", "audio": audio_metrics,
              "structural_checks_passed": structural_passed,
              "provenance": {"audio_path": str(path), "audio_sha256": _sha256(path), "device": device,
                             "output_clock": clock,
                             "teacher_input_sha256": teacher_analysis.get("input_sha256"),
                             "alignment_sha256": hashlib.sha256(json.dumps(alignment_plan, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()},
              "word_timing_validation": {"measured": False, "reason": limitations[1]},
              "warnings": warnings, "limitations": limitations}
    if not len(audio) or not audio_metrics["finite"]:
        report["verification_status"] = "invalid_audio"
        warnings.append("Pitch verification was skipped because the waveform is empty or non-finite.")
        return report
    if audio_metrics["near_full_scale_sample_count"]:
        warnings.append("Output reaches digital full scale; inspect the reported full-scale sample count for clipping.")
    if not audio_metrics["sample_count_exact"] or not audio_metrics["sample_rate_exact"]:
        reference = "the teacher reference" if clock == "teacher" else "the student recording"
        warnings.append(f"Output sample count or sample rate differs from {reference}.")
    mono = np.mean(audio, axis=1)
    if audio.shape[1] > 1:
        channel_energy = np.mean(audio.astype(np.float64) ** 2, axis=0)
        if float(np.mean(mono.astype(np.float64) ** 2)) < 0.0625 * float(np.max(channel_energy)):
            mono = audio[:, int(np.argmax(channel_energy))]
            warnings.append("Output channels show phase cancellation; pitch measured on the strongest channel.")
    actual = _measure_rmvpe(mono, rate, root, device)
    report["provenance"]["verification_code_sha256"] = _sha256(Path(__file__))
    report["provenance"]["rmvpe_model_sha256"] = actual.get("model_sha256")
    teacher_pitch = teacher_analysis.get("pitch", {})
    times, _, teacher_voiced = _pitch_arrays(teacher_pitch)
    supported = _region_mask(alignment_plan, times)
    duration = len(audio) / rate
    expected_duration = float(expected.get("duration", audio_metrics["expected_duration"]))
    if clock == "teacher":
        report["actual_vs_teacher"] = {
            "all_reference_frames": compare_pitch(actual, teacher_pitch, actual_duration=duration, reference_duration=expected_duration),
            "semantically_supported_frames": compare_pitch(actual, teacher_pitch, selection=supported,
                                                           actual_duration=duration, reference_duration=expected_duration),
            "outside_semantic_support": compare_pitch(actual, teacher_pitch, selection=~supported,
                                                       actual_duration=duration, reference_duration=expected_duration),
        }
    in_duration = (times >= 0) & (times < expected_duration)
    total_voiced = int((teacher_voiced & in_duration).sum())
    support_voiced = int((teacher_voiced & supported & in_duration).sum())
    report["coverage"] = {"teacher_voiced_frames": total_voiced,
                          "semantically_supported_teacher_voiced_frames": support_voiced,
                          "semantic_teacher_voiced_coverage": _ratio(support_voiced, total_voiced),
                          "planned_lyric_coverage": alignment_plan.get("metrics", {}).get("lyric_coverage"),
                          "planned_status_counts": alignment_plan.get("metrics", {}).get("status_counts", {}),
                          "unsupported_regions": [{key: region.get(key) for key in
                              ("teacher_start", "teacher_end", "status", "mode", "reason")} for region in alignment_plan.get("regions", [])
                              if region.get("unsupported") or region.get("mode") in {"mute", "acoustic_fallback"}],
                          "boundary_source": "Alignment plan; not output word timing measurements."}
    target = None
    target_policy = None
    target_offset_cents = 0.0
    target_note_shifts = 0
    if target_pitch_path is not None:
        target_path = Path(target_pitch_path).resolve(strict=True)
        with np.load(target_path, allow_pickle=False) as data:
            target = {key: np.asarray(data[key]).tolist() for key in ("times", "f0_hz", "voiced")}
            target_times, _, target_voiced = _pitch_arrays(target)
            semantic_render = np.asarray(data["supported"], dtype=bool) if "supported" in data else np.ones(len(target_times), dtype=bool)
            rendered = np.asarray(data["rendered"], dtype=bool) if "rendered" in data else semantic_render
            context = np.asarray(data["acoustic_context"], dtype=bool) if "acoustic_context" in data else np.zeros(len(target_times), dtype=bool)
            if "pitch_policy" in data:
                target_policy = str(np.asarray(data["pitch_policy"]).reshape(-1)[0])
            if "register_offset_cents" in data:
                target_offset_cents = float(np.asarray(data["register_offset_cents"]).reshape(-1)[0])
            if "note_shift_applied_count" in data:
                target_note_shifts = int(np.asarray(data["note_shift_applied_count"]).reshape(-1)[0])
            if any(mask.shape != target_times.shape for mask in (semantic_render, rendered, context)):
                raise ValueError("Render target supported mask does not match pitch frames.")
        report["actual_vs_render_target"] = {
            "all_target_frames": compare_pitch(actual, target, actual_duration=duration, reference_duration=expected_duration),
            "render_supported_frames": compare_pitch(actual, target, selection=rendered,
                                                       actual_duration=duration, reference_duration=expected_duration),
            "semantically_supported_frames": compare_pitch(actual, target, selection=semantic_render,
                                                       actual_duration=duration, reference_duration=expected_duration),
            "acoustic_context_frames": compare_pitch(actual, target, selection=context,
                                                       actual_duration=duration, reference_duration=expected_duration),
            "outside_render_support": compare_pitch(actual, target, selection=~rendered,
                                                       actual_duration=duration, reference_duration=expected_duration),
        }
        if clock == "teacher":
            # These index the render target by teacher pitch times, which is only the
            # same timeline when the output really is on the teacher clock.
            rendered_on_teacher, valid = _nearest(target_times, times)
            rendered_teacher = rendered[rendered_on_teacher] & valid if len(target_times) else np.zeros(len(times), dtype=bool)
            report["coverage"]["render_supported_teacher_voiced_coverage"] = _ratio(int((rendered_teacher & teacher_voiced & in_duration).sum()), total_voiced)
            semantic_teacher = semantic_render[rendered_on_teacher] & valid if len(target_times) else np.zeros(len(times), dtype=bool)
            report["coverage"]["render_semantic_teacher_voiced_coverage"] = _ratio(int((semantic_teacher & teacher_voiced & in_duration).sum()), total_voiced)
            _, target_voice_at_teacher, _ = sample_measured_pitch(target, times)
            report["coverage"]["render_target_teacher_voiced_coverage"] = _ratio(int((target_voice_at_teacher & teacher_voiced & in_duration).sum()), total_voiced)
        report["coverage"]["acoustic_context_frame_count"] = int(context.sum())
        report["provenance"]["target_pitch_sha256"] = _sha256(target_path)
        report["provenance"]["target_pitch_path"] = str(target_path)
        if target_policy is not None:
            report["provenance"]["target_pitch_policy"] = target_policy
        if (clock == "teacher" and target_policy in STUDENT_REGISTER_POLICIES
                and abs(target_offset_cents) > 0.5):
            # The rendered melody deliberately sits in the student's register, so
            # an absolute comparison against the teacher would measure the
            # intended transposition as error. The aimed reference is the
            # teacher's pitch moved by the same register offset, so that is the
            # comparison reported here; target_pitch.npz carries the offset and
            # actual_vs_render_target still checks what was aimed at frame by frame.
            # A negligible offset is skipped: it is the same-number duplicate,
            # and the separate key would only invite a meaningless comparison.
            relative_reference = _shifted_pitch(teacher_pitch, target_offset_cents)
            report["actual_vs_teacher_relative"] = {
                "reference": f"teacher pitch moved into the student's register by {target_offset_cents:.1f} cents",
                "all_reference_frames": compare_pitch(
                    actual, relative_reference,
                    actual_duration=duration, reference_duration=expected_duration),
                "semantically_supported_frames": compare_pitch(
                    actual, relative_reference, selection=supported,
                    actual_duration=duration, reference_duration=expected_duration),
            }
            report["limitations"].append(
                "Pitch was mapped relative to the student's own register, so agreement with the teacher's "
                "absolute pitch measures the intended register offset, not an error. Use "
                "actual_vs_teacher_relative and actual_vs_render_target for the aimed pitch.")
            warnings.append(
                f"Register-relative pitch mapping used a {target_offset_cents:.1f}-cent register offset; "
                "the teacher's absolute pitch is not the intended output pitch.")
            if target_note_shifts:
                # Adaptive placement moves words individually, so the register-shifted
                # teacher curve is only an approximation of the aim; the render target
                # is the exact one. Saying so keeps a correct render from being read as
                # inaccurate against the wrong reference.
                report["limitations"].append(
                    f"{target_note_shifts} word(s) were placed individually onto the teacher's own note "
                    "levels, so the register-shifted teacher curve approximates the aim rather than being "
                    "it. actual_vs_render_target is the authoritative comparison for this render.")
                warnings.append(
                    f"{target_note_shifts} word(s) were placed individually onto the teacher's note level "
                    "inside the student's own register; the register-matched comparison is the aimed one, "
                    "and actual_vs_render_target records what each frame was aimed at.")
    else:
        warnings.append("No renderer target pitch was supplied; target execution and render support cannot be audited.")
    if support_voiced < total_voiced:
        warnings.append("Some teacher voiced frames lack semantic support; full reference metrics retain them in the denominator.")
    if not times.size:
        warnings.append("Teacher has no pitch frames; teacher accuracy metrics are unavailable.")
    report["yin_diagnostic"] = _yin_check(mono, rate, actual, teacher_pitch, alignment_plan, target,
                                          clock=clock)
    report["measured_pitch"] = actual
    return report
