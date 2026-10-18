"""Remeasure exported word timing with independent transcription and alignment.

The acoustic model receives only the exported audio and its own ASR text.
Teacher words and times enter the comparison after output analysis finishes.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from inference.alignment.planner import _correspond, _duration, _relation, _units

logger = logging.getLogger(__name__)


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _statistics(errors: list[float]) -> dict:
    values = np.asarray(errors, dtype=np.float64) * 1000
    if not len(values):
        return {"word_count": 0, "mean_signed_ms": None, "median_signed_ms": None,
                "mean_absolute_ms": None, "median_absolute_ms": None,
                "p95_absolute_ms": None, "maximum_absolute_ms": None}
    absolute = np.abs(values)
    return {"word_count": len(errors), "mean_signed_ms": float(np.mean(values)),
            "median_signed_ms": float(np.median(values)), "mean_absolute_ms": float(np.mean(absolute)),
            "median_absolute_ms": float(np.median(absolute)), "p95_absolute_ms": float(np.percentile(absolute, 95)),
            "maximum_absolute_ms": float(np.max(absolute))}


def diagnose_word_audio(audio_path: str | Path, output_analysis: dict) -> dict:
    """Flag aligned word edges inside exact silence without moving boundaries.

    This conservative check detects hard-muted spans. It does not classify
    low-energy consonants, estimate articulation onsets, or change scores.
    """
    analysis = copy.deepcopy(output_analysis)
    audio, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    amplitude = np.max(np.abs(audio), axis=1)
    for word in analysis.get("words", []):
        start = max(0, min(len(audio), int(round(float(word["start"]) * rate))))
        end = max(start, min(len(audio), int(round(float(word["end"]) * rate))))
        values = amplitude[start:end]
        active = np.flatnonzero(values > 1e-7)
        leading = int(active[0]) if len(active) else len(values)
        trailing = len(values) - int(active[-1]) - 1 if len(active) else len(values)
        word["boundary_audio"] = {
            "leading_exact_silence_ms": leading / rate * 1000,
            "trailing_exact_silence_ms": trailing / rate * 1000,
            "contains_nonzero_audio": bool(len(active)),
            "aligned_boundary_in_long_exact_silence": leading / rate >= 0.03 or trailing / rate >= 0.03,
            "threshold_amplitude": 1e-7,
            "interpretation": "Exact-silence evidence only; boundaries and timing error statistics are unchanged.",
        }
    return analysis


def _unit_reliability(unit: dict, analysis: dict, label: str, minimum_quality: float) -> list[str]:
    reasons = []
    if unit["quality"] < minimum_quality:
        reasons.append(f"{label}_low_structural_quality")
    if unit.get("estimated_timing"):
        reasons.append(f"{label}_estimated_internal_boundaries")
    raw = analysis.get("words", [])[unit["index"]]
    try:
        unchanged = math.isclose(float(raw["start"]), unit["start"], abs_tol=1e-7) and math.isclose(float(raw["end"]), unit["end"], abs_tol=1e-7)
    except (KeyError, ValueError, TypeError):
        unchanged = False
    if not unchanged:
        reasons.append(f"{label}_normalized_or_clipped_boundaries")
    midpoint = (unit["start"] + unit["end"]) / 2
    chunks = analysis.get("chunks") or []
    covering = [chunk for chunk in chunks if float(chunk.get("core_start", chunk.get("start", 0))) <= midpoint <
                float(chunk.get("core_end", chunk.get("end", 0)))]
    if covering:
        if not all(chunk.get("alignment_complete", False) for chunk in covering):
            reasons.append(f"{label}_chunk_alignment_incomplete")
    elif "lyrics" not in analysis.get("completed_stages", []):
        reasons.append(f"{label}_lyric_analysis_incomplete")
    if raw.get("issues"):
        reasons.append(f"{label}_alignment_quality_issues")
    return reasons


def compare_word_timing(teacher_analysis: dict, output_analysis: dict, *, minimum_quality: float = 0.6,
                        tolerance_seconds: float = 0.05) -> dict:
    """Compare fresh acoustic observations using monotonic lexical identity.

    Zero durations disable the planner's weak acoustic-time prior: matching
    must not select words simply because measured boundaries look correct.
    Repeated lexical paths remain ambiguous and cannot improve error metrics.
    """
    if not 0 <= minimum_quality <= 1 or not math.isfinite(tolerance_seconds) or tolerance_seconds <= 0:
        raise ValueError("minimum_quality must be in [0, 1] and tolerance_seconds must be positive.")
    teacher, teacher_warnings = _units(teacher_analysis, _duration(teacher_analysis))
    output, output_warnings = _units(output_analysis, _duration(output_analysis))
    pairs, ambiguities = _correspond(teacher, output, 0.0, 0.0)
    observations = []
    onset_errors, offset_errors, duration_errors = [], [], []
    reliable_teacher_duration = 0.0
    reliable_within = 0
    for teacher_index, output_index in pairs:
        reference = teacher[teacher_index] if teacher_index is not None else None
        measured = output[output_index] if output_index is not None else None
        relation = _relation(reference, measured) if reference and measured else "missing" if reference else "extra"
        reasons = []
        if reference:
            reasons.extend(_unit_reliability(reference, teacher_analysis, "teacher", minimum_quality))
        if measured:
            reasons.extend(_unit_reliability(measured, output_analysis, "output", minimum_quality))
        if relation != "matched":
            reasons.append("exact_lexical_identity_not_established")
        ambiguity = ambiguities.get((teacher_index, output_index))
        if ambiguity:
            reasons.append("ambiguous_repeated_lyric_correspondence")
        reliable = not reasons and reference is not None and measured is not None
        onset = measured["start"] - reference["start"] if reference and measured else None
        offset = measured["end"] - reference["end"] if reference and measured else None
        length = (measured["end"] - measured["start"]) - (reference["end"] - reference["start"]) if reference and measured else None
        # Timestamp subtraction can put an exact 50 ms error a few ULPs above
        # the limit. One nanosecond is far below the audio sample resolution.
        within = bool(abs(onset) <= tolerance_seconds + 1e-9 and abs(offset) <= tolerance_seconds + 1e-9) if reliable else None
        observation = {
            "teacher_unit_index": teacher_index, "output_unit_index": output_index,
            "teacher_word_index": reference["index"] if reference else None,
            "output_word_index": measured["index"] if measured else None,
            "teacher_text": reference["text"] if reference else None,
            "output_text": measured["text"] if measured else None,
            "teacher_start": reference["start"] if reference else None,
            "teacher_end": reference["end"] if reference else None,
            "measured_output_start": measured["start"] if measured else None,
            "measured_output_end": measured["end"] if measured else None,
            "relation": relation, "reliable_for_timing": reliable, "exclusion_reasons": reasons,
            "teacher_quality": reference["quality"] if reference else None,
            "output_quality": measured["quality"] if measured else None,
            "onset_error_ms": onset * 1000 if reliable else None,
            "offset_error_ms": offset * 1000 if reliable else None,
            "duration_error_ms": length * 1000 if reliable else None,
            "both_boundaries_within_tolerance": within,
        }
        if measured:
            observation["output_boundary_audio"] = output_analysis["words"][measured["index"]].get("boundary_audio")
        if ambiguity:
            observation["ambiguity"] = ambiguity
        observations.append(observation)
        if reliable:
            onset_errors.append(onset)
            offset_errors.append(offset)
            duration_errors.append(length)
            reliable_teacher_duration += reference["end"] - reference["start"]
            reliable_within += int(within)
    counts = Counter(observation["relation"] for observation in observations)
    for index, observation in enumerate(observations):
        neighbors = observations[max(0, index - 1):index] + observations[index + 1:index + 2]
        observation["neighboring_transcript_disagreement"] = any(neighbor["relation"] != "matched" for neighbor in neighbors)
    reliable_count = len(onset_errors)
    teacher_duration = sum(unit["end"] - unit["start"] for unit in teacher)
    warnings = teacher_warnings + output_warnings
    teacher_complete = "lyrics" in teacher_analysis.get("completed_stages", [])
    output_complete = "lyrics" in output_analysis.get("completed_stages", [])
    if not teacher_complete or not output_complete:
        warnings.append("Teacher or output lyric analysis is incomplete; valid completed chunks may still contribute measured spans.")
    if counts["missing"] or counts["extra"] or counts["substitution"]:
        warnings.append("Transcript differences can be ASR errors or rendered lyric problems; they do not prove omitted, added, or changed singing.")
    if reliable_count < len(teacher):
        warnings.append("Timing errors cover reliable exact lexical matches only; uncovered teacher units remain in the overall coverage denominator.")
    if not reliable_count:
        warnings.append("No reliable output word boundaries can be compared; timing accuracy is unavailable.")
    silence_boundaries = sum(bool(item.get("output_boundary_audio", {}).get("aligned_boundary_in_long_exact_silence"))
                             for item in observations if item.get("output_boundary_audio"))
    if silence_boundaries:
        warnings.append("Some TiFA boundaries include at least 30 ms of exact output silence; these are alignment estimates, not measured articulation onsets.")
    return {
        "verification_status": "remeasured_with_limitations" if reliable_count else "insufficient_reliable_word_observations",
        "measurement": {"output_word_boundaries_remeasured": bool(output),
                        "teacher_boundaries_supplied_to_models": False,
                        "method": "Independent output Qwen ASR and TiFA, followed by monotonic lexical comparison.",
                        "correspondence_time_prior": "disabled", "minimum_structural_quality": minimum_quality,
                        "tolerance_ms": tolerance_seconds * 1000,
                        "teacher_lyric_analysis_complete": teacher_complete, "output_lyric_analysis_complete": output_complete},
        "boundary_diagnostics": {"words_with_boundary_in_long_exact_silence": silence_boundaries,
                                 "paired_words_adjacent_to_transcript_disagreement": sum(item["neighboring_transcript_disagreement"] and item["teacher_unit_index"] is not None and item["output_unit_index"] is not None for item in observations),
                                 "diagnostics_change_error_statistics": False},
        "coverage": {"teacher_unit_count": len(teacher), "output_unit_count": len(output),
                     "reliable_exact_match_count": reliable_count,
                     "reliable_teacher_unit_coverage": _fraction(reliable_count, len(teacher)),
                     "teacher_lyric_duration": teacher_duration,
                     "reliably_compared_teacher_duration": reliable_teacher_duration,
                     "reliable_teacher_duration_coverage": _fraction(reliable_teacher_duration, teacher_duration),
                     "relation_counts": dict(counts),
                     "unreliable_paired_unit_count": sum(item["teacher_unit_index"] is not None and item["output_unit_index"] is not None and not item["reliable_for_timing"] for item in observations),
                     "ambiguous_paired_unit_count": len(ambiguities),
                     "reliable_words_within_tolerance_count": reliable_within,
                     "within_tolerance_ratio_of_reliable_matches": _fraction(reliable_within, reliable_count),
                     "within_tolerance_ratio_of_all_teacher_units": _fraction(reliable_within, len(teacher))},
        "reliable_boundary_errors": {"onset": _statistics(onset_errors), "offset": _statistics(offset_errors),
                                     "duration": _statistics(duration_errors),
                                     "signed_error_definition": "Remeasured output minus teacher; positive means later or longer."},
        "word_observations": observations,
        "warnings": list(dict.fromkeys(warnings)),
        "limitations": [
            "ASR and TiFA have recognition and boundary uncertainty; output measurements are model estimates, not human annotations.",
            "Teacher and output use the same model family, so correlated alignment errors can remain.",
            "Reliable means lexical and structural checks passed, not a calibrated acoustic boundary confidence; TiFA can absorb silence or missing ASR words into an adjacent matched word.",
            "Only exact, structurally valid, unambiguous lexical matches contribute timing errors; homophones and suspected substitutions are reported separately.",
            "Multi-unit words with estimated internal boundaries, low-quality spans, and clipped overlaps do not contribute precise timing metrics.",
            "Missing transcript units cannot establish whether the audio omitted a lyric or the recognizer failed.",
            "Word timing does not measure pitch, timbre, intelligibility, or perceptual audio quality.",
        ],
    }


def verify_timing(audio_path: str | Path, teacher_analysis: dict, output_dir: str | Path, *,
                  root: str | Path | None = None, device: str = "dml", language: str | None = None,
                  minimum_quality: float = 0.6, tolerance_seconds: float = 0.05,
                  max_chunk_seconds: float | None = None, cache_dir: str | Path | None = None,
                  clock: str = "teacher",
                  cancel_checker: Callable[[], bool] | None = None,
                  progress: Callable[[str], None] | None = None) -> dict:
    """Analyze exported audio alone, then compare its observations to teacher.

    The pair API receives the same output file twice, so its content cache
    performs ASR/TiFA once. Only lyrics are requested; RMVPE/GAME do not run.
    The teacher analysis is never passed to either acoustic model.

    The comparison subtracts teacher timestamps from the output's, which only
    describes the same events when the output was rendered on the teacher clock.
    An output that keeps the student's timeline is not compared at all: the
    absolute boundaries describe different points in the song, so the numbers
    would be meaningless. Such a job records the reason instead.
    """
    if clock not in {"teacher", "student"}:
        raise ValueError("The output clock must be the teacher or the student timeline.")
    if clock == "student":
        return {"verification_status": "not_measured", "clock": clock,
                "measurement": {"output_word_boundaries_remeasured": False,
                                "method": "Not attempted.",
                                "reason": "The output keeps the student's timeline."},
                "coverage": None, "reliable_boundary_errors": None,
                "warnings": ["Output word timing was not compared with the teacher: this render keeps the "
                             "student's timeline, so teacher boundaries do not describe the same moments."],
                "limitations": ["Timing accuracy against the teacher is only defined when the timing was "
                                "corrected; this job did not change the timing."],
                "provenance": {"audio_path": str(Path(audio_path)), "clock": clock}}
    from inference.analysis import analyze_pair, atomic_json
    audio_path = Path(audio_path).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    config = teacher_analysis.get("config", {})
    language = language or config.get("language", "zh")
    chunk_seconds = max_chunk_seconds if max_chunk_seconds is not None else float(config.get("max_chunk_seconds", 18.0))
    measured, _ = analyze_pair(audio_path, audio_path, output_dir / "output_analysis", root,
        device=device, language=language, max_chunk_seconds=chunk_seconds,
        min_chunk_seconds=float(config.get("min_chunk_seconds", 4.0)),
        context_seconds=float(config.get("context_seconds", 0.75)), include_game=False,
        stages=("lyrics",), cache_dir=cache_dir, cancel_checker=cancel_checker, progress=progress or logger.info)
    measured = diagnose_word_audio(audio_path, measured)
    report = compare_word_timing(teacher_analysis, measured, minimum_quality=minimum_quality,
                                 tolerance_seconds=tolerance_seconds)
    report["provenance"] = {
        "audio_path": str(audio_path), "audio_sha256": measured.get("input_sha256"),
        "teacher_input_sha256": teacher_analysis.get("input_sha256"),
        "teacher_analysis_sha256": _digest(teacher_analysis),
        "output_analysis_sha256": _digest(measured), "output_analysis_cache_key": measured.get("cache_key"),
        "output_model_fingerprints": measured.get("model_fingerprints"),
        "verification_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "correspondence_code_sha256": hashlib.sha256((root / "inference" / "alignment" / "planner.py").read_bytes()).hexdigest(),
        "language": language, "device": device,
    }
    report["artifacts"] = {"output_analysis": str(output_dir / "output.analysis.json"),
                           "output_textgrid": str(output_dir / "output_analysis" / "teacher.TextGrid")}
    report["output_analysis_warnings"] = measured.get("warnings", [])
    atomic_json(output_dir / "output.analysis.json", measured)
    atomic_json(output_dir / "timing.verification.json", report)
    return report
