from __future__ import annotations

import math
from pathlib import Path
import numpy as np

from .pitch import sample_pitch
from .boundaries import build_boundary_context
from .register import (PER_WORD_PLACEMENT_POLICIES, TEACHER_CONTOUR_POLICIES, frame_note_shifts,
                       register_pitch_plan, resolve_pitch_policy, shifted_f0)
from .transitions import refine_transition_map
from .dynamics import reference_transition_gain


def _register_diagnostics(plan: dict, limit: int = 8) -> dict:
    """The pitch plan for the log, without the full per-region measurement table.

    ``region_shifts_cents`` is kept: it is the record of what each word moved by."""
    summary = {key: value for key, value in plan.items() if key != "regions"}
    shifts = [(index, float(value)) for index, value in enumerate(plan.get("region_shifts_cents", []))
              if abs(float(value)) > 0.0]
    shifts.sort(key=lambda item: -abs(item[1]))
    summary["largest_note_shifts"] = [{"region_index": index, "shift_cents": value}
                                      for index, value in shifts[:limit]]
    return summary


def _anchors(region: dict) -> tuple[np.ndarray, np.ndarray]:
    teacher = np.asarray(region.get("teacher_anchors", [region["teacher_start"], region["teacher_end"]]), dtype=np.float64)
    student = np.asarray(region.get("student_anchors", [region["student_start"], region["student_end"]]), dtype=np.float64)
    if teacher.ndim != 1 or len(teacher) < 2 or teacher.shape != student.shape:
        raise ValueError("Each synthesis region requires equally sized teacher/student anchors.")
    if not np.isfinite(teacher).all() or not np.isfinite(student).all():
        raise ValueError("Region anchors must be finite.")
    if np.any(np.diff(teacher) <= 0) or np.any(np.diff(student) <= 0):
        raise ValueError("Region anchors must be strictly increasing on both timelines.")
    return teacher, student


def _allowed(region: dict, allow_fallback: bool, minimum_quality: float) -> bool:
    mode = region.get("mode", "synthesize")
    if mode == "acoustic_fallback":
        return bool(allow_fallback) and float(region.get("quality", 0)) >= minimum_quality
    return mode == "synthesize" and not region.get("unsupported", False) and float(region.get("quality", 1)) >= minimum_quality


def _warp_mel(mel: np.ndarray, source_times: np.ndarray, valid: np.ndarray, hop: int, sr: int, floor: float) -> np.ndarray:
    output = np.full((mel.shape[0], len(source_times)), math.log(floor), dtype=np.float32)
    if valid.any():
        coordinates = source_times[valid] * sr / hop - 0.5
        frame_index = np.arange(mel.shape[1], dtype=np.float64)
        for band in range(mel.shape[0]):
            output[band, valid] = np.interp(coordinates, frame_index, mel[band])
    return output


def _region_map(regions: list[dict], query: np.ndarray, *, inverse: bool, allow_fallback: bool, minimum_quality: float) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.full(query.shape, np.nan, dtype=np.float64)
    valid = np.zeros(query.shape, dtype=bool)
    for region in regions:
        if not _allowed(region, allow_fallback, minimum_quality):
            continue
        teacher, student = _anchors(region)
        source, target = (student, teacher) if inverse else (teacher, student)
        selected = (query >= source[0]) & (query < source[-1])
        if np.any(valid & selected):
            raise ValueError("Synthesis regions overlap on the selected timeline.")
        mapped[selected] = np.interp(query[selected], source, target)
        valid[selected] = True
    return mapped, valid


def _context_gain(context: dict, times: np.ndarray) -> np.ndarray:
    """Taper acoustic context without attenuating the original lyric interval."""
    start, stop = float(context["teacher_start"]), float(context["teacher_end"])
    if context["kind"] == "attack":
        progress = np.clip((times - start) / (stop - start), 0, 1)
        return np.sin(progress * np.pi / 2) ** 2
    release = stop - start if context["source_strategy"] == "hold" else min(0.035, stop - start)
    progress = np.clip((times - (stop - release)) / release, 0, 1)
    return np.cos(progress * np.pi / 2) ** 2


def _acoustic_map(contexts: list[dict], times: np.ndarray, source_times: np.ndarray,
                  supported: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add bounded tails separately from the immutable semantic support mask."""
    mapped, rendered = source_times.copy(), supported.copy()
    gain = np.ones(len(times), dtype=np.float32)
    for context in contexts:
        start, stop = float(context["teacher_start"]), float(context["teacher_end"])
        selected = (times >= start) & (times < stop) & ~supported
        if np.any(selected & rendered):
            raise ValueError("Acoustic contexts overlap.")
        mapped[selected] = np.interp(times[selected], [start, stop],
                                     [context["student_start"], context["student_end"]])
        gain[selected] = _context_gain(context, times[selected])
        rendered[selected] = True
    return mapped, rendered, gain


def _context_mel(audio: np.ndarray, vocoder, contexts: list[dict], regions: list[dict],
                 times: np.ndarray, source_times: np.ndarray, supported: np.ndarray,
                 output: np.ndarray) -> None:
    """Extract context locally so STFT lookahead cannot enter another word."""
    sr, hop = vocoder.sample_rate, vocoder.hop_size
    window = int(getattr(vocoder, "win_size", 2048))
    for context in contexts:
        selected = ((times >= context["teacher_start"]) &
                    (times < context["teacher_end"]) & ~supported)
        if not selected.any():
            continue
        region = regions[context["source_region_index"]]
        if context["kind"] == "release":
            lower = float(region["student_start"])
            upper = (float(region["student_end"]) if context["source_strategy"] == "hold"
                     else float(context["student_limit_end"]))
        else:
            lower, upper = float(context["student_limit_start"]), float(region["student_end"])
        # Keep the global frame grid where possible; clip and reflect at lexical
        # limits rather than letting a 46 ms STFT window borrow the next word.
        queries = source_times[selected]
        first = max(0, int(math.ceil(lower * sr)),
                    int(math.floor((float(queries.min()) * sr - window) / hop)) * hop)
        last = min(len(audio), int(math.floor(upper * sr)),
                   int(math.ceil((float(queries.max()) * sr + window) / hop)) * hop)
        if last <= first:
            raise ValueError("Acoustic context has no safe source samples.")
        local_mel = vocoder.mel(audio[first:last])
        output[:, selected] = _warp_mel(
            local_mel, queries - first / sr, np.ones(len(queries), dtype=bool),
            hop, sr, vocoder.log_floor,
        )


def _reference_release_gain(contexts: list[dict], times: np.ndarray, teacher_audio: np.ndarray,
                            sample_rate: int) -> np.ndarray:
    """Constrain held texture with teacher-relative energy, never teacher timbre."""
    gain = np.ones(len(times), dtype=np.float32)
    energy = np.r_[0.0, np.cumsum(teacher_audio.astype(np.float64) ** 2)]
    radius = max(1, int(round(0.010 * sample_rate)))

    def rms(query):
        centers = np.rint(np.asarray(query) * sample_rate).astype(np.int64)
        first = np.clip(centers - radius, 0, len(teacher_audio))
        last = np.clip(centers + radius, 0, len(teacher_audio))
        return np.sqrt((energy[last] - energy[first]) / np.maximum(1, last - first))

    for context in contexts:
        if context["kind"] != "release" or context["source_strategy"] != "hold":
            continue
        selected = (times >= context["teacher_start"]) & (times < context["teacher_end"])
        # A local ratio keeps the student's loudness while following how quickly
        # the teacher releases the vowel. No spectral features are transferred.
        reference = float(rms([context["boundary"] - 0.015])[0])
        if reference > 1e-6:
            gain[selected] = np.minimum(1.0, rms(times[selected]) / reference)
        context["reference_energy_policy"] = "20ms_RMS_relative_to_boundary_minus_15ms"
    return gain


def _smooth_source_jumps(mel: np.ndarray, times: np.ndarray, regions: list[dict],
                         allow_fallback: bool, minimum_quality: float) -> list[dict]:
    """Blend only genuine source edits, leaving continuous word maps intact."""
    allowed = sorted((region for region in regions if _allowed(region, allow_fallback, minimum_quality)),
                     key=lambda region: region["teacher_start"])
    transitions = []
    for left, right in zip(allowed, allowed[1:]):
        boundary = float(right["teacher_start"])
        jump = float(right["student_start"]) - float(left["student_end"])
        if abs(boundary - float(left["teacher_end"])) > 1 / 44100 or abs(jump) < 0.025:
            continue
        half_width = min(0.012, (boundary - float(left["teacher_start"])) / 4,
                         (float(right["teacher_end"]) - boundary) / 4)
        indices = np.flatnonzero((times >= boundary - half_width) & (times < boundary + half_width))
        if len(indices) < 2:
            continue
        first, last = int(indices[0]), int(indices[-1])
        progress = np.clip((times[indices] - (boundary - half_width)) / (2 * half_width), 0, 1)
        weight = np.sin(progress * np.pi / 2) ** 2
        left_texture, right_texture = mel[:, first].copy(), mel[:, last].copy()
        mel[:, indices] = left_texture[:, None] * (1 - weight) + right_texture[:, None] * weight
        transitions.append({"teacher_boundary": boundary, "source_jump_seconds": jump,
                            "blend_start": float(times[first]), "blend_end": float(times[last])})
    return transitions


def _crossfade_splices(mel: np.ndarray, times: np.ndarray, source_times: np.ndarray,
                       transitions: list[dict], *, window_seconds: float = 0.012) -> list[dict]:
    """Blend mel across a constrained window edge that meets a discontinuous source.

    A genuine source discontinuity would click; an equal-power blend removes it."""
    edits = []
    if not transitions or len(times) < 4:
        return edits
    step = max(float(np.median(np.diff(times))), 1e-9)
    half = max(2, int(round(window_seconds / step)))
    for transition in transitions:
        if not (transition.get("duration_constraint") or {}).get("applied"):
            continue
        for edge in (float(transition["teacher_start"]), float(transition["teacher_end"])):
            centre = int(np.argmin(np.abs(times - edge)))
            first, last = max(0, centre - half), min(len(times), centre + half)
            if last - first < 4:
                continue
            local = source_times[first:last]
            if not np.isfinite(local).all():
                continue
            steps = np.abs(np.diff(local))
            # A continuous trajectory needs no blend; the constrained mapping has
            # already smoothed its rate across this joint.
            if not len(steps) or float(np.max(steps)) < 0.020:
                continue
            progress = np.linspace(0.0, 1.0, last - first, endpoint=True)
            weight = np.sin(progress * np.pi / 2) ** 2
            left_texture, right_texture = mel[:, first].copy(), mel[:, last - 1].copy()
            mel[:, first:last] = (left_texture[:, None] * (1 - weight)
                                  + right_texture[:, None] * weight)
            edits.append({"teacher_edge": float(edge), "frame_start": int(first),
                          "frame_end": int(last),
                          "maximum_source_step_seconds": float(np.max(steps)),
                          "policy": "equal_power_mel_blend_across_discontinuous_constrained_splice"})
    return edits


def _valid_envelope(sample_count: int, sample_rate: int, regions: list[dict], allow_fallback: bool, minimum_quality: float, fade_seconds: float = 0.005, contexts: list[dict] | None = None) -> np.ndarray:
    """Apply exact lexical mutes and move safety fades outside preserved tails."""
    envelope = np.zeros(sample_count, dtype=np.float32)
    for region in regions:
        if _allowed(region, allow_fallback, minimum_quality):
            start = max(0, int(round(float(region["teacher_start"]) * sample_rate)))
            stop = min(sample_count, int(round(float(region["teacher_end"]) * sample_rate)))
            envelope[start:stop] = 1
    for context in contexts or []:
        start = max(0, int(round(float(context["teacher_start"]) * sample_rate)))
        stop = min(sample_count, int(round(float(context["teacher_end"]) * sample_rate)))
        envelope[start:stop] = 1
    edges = np.diff(np.pad(envelope.astype(np.int8), (1, 1)))
    fade_count = max(1, int(round(fade_seconds * sample_rate)))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        fade = min(fade_count, (stop - start) // 2)
        if fade:
            ramp = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float64)) ** 2
            envelope[start:start + fade] *= ramp
            envelope[stop - fade:stop] *= ramp[::-1]
    return envelope


def render_student(
    student_audio: np.ndarray,
    teacher_analysis: dict,
    student_analysis: dict,
    alignment_plan: dict,
    vocoder,
    *,
    mode: str = "full",
    change_timing: bool = True,
    pitch_mode: str = "off",
    map_teacher_pitch: bool = False,
    allow_acoustic_fallback: bool = False,
    minimum_quality: float = 0.0,
    chunk_frames: int = 768,
    target_pitch_path: str | Path | None = None,
    teacher_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Render student timbre against a teacher pitch/timing reference.

    Missing or uncertain lyrics are muted; acoustic fallbacks require opt-in."""
    if mode not in {"identity", "pitch_only", "time_only", "full"}:
        raise ValueError(f"Unknown rendering mode: {mode}")
    pitch_policy = resolve_pitch_policy(pitch_mode, map_teacher_pitch)
    audio = np.asarray(student_audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("student_audio must be a nonempty finite mono array.")
    sr, hop = vocoder.sample_rate, vocoder.hop_size
    if any(int(analysis["sample_rate"]) != sr for analysis in (teacher_analysis, student_analysis)):
        raise ValueError("Analysis and audio must use the vocoder sample rate.")
    if int(student_analysis["sample_count"]) != len(audio):
        raise ValueError("Student analysis sample_count does not match student_audio.")
    if teacher_audio is not None:
        teacher_audio = np.asarray(teacher_audio, dtype=np.float32)
        if (teacher_audio.ndim != 1 or len(teacher_audio) != int(teacher_analysis["sample_count"])
                or not np.isfinite(teacher_audio).all()):
            raise ValueError("teacher_audio must match the finite mono teacher analysis.")
    regions = alignment_plan.get("regions", [])
    warped = mode in {"time_only", "full"} and change_timing
    count = int(teacher_analysis["sample_count"]) if warped else len(audio)
    if count <= 0:
        raise ValueError("Output sample count must be positive.")
    frame_count = math.ceil(count / hop)
    times = (np.arange(frame_count, dtype=np.float64) + 0.5) * hop / sr
    source_mel = vocoder.mel(audio)
    warnings = []
    contexts, transitions, acoustic_transitions, acoustic_dynamics, crossfades = [], [], [], [], []
    refined = np.zeros(frame_count, dtype=bool)
    energy_gain = np.ones(frame_count, dtype=np.float32)

    if warped:
        source_times, valid = _region_map(
            regions, times, inverse=False, allow_fallback=allow_acoustic_fallback,
            minimum_quality=minimum_quality,
        )
        rendered = valid.copy()
        if teacher_audio is not None:
            teacher_mel = vocoder.mel(teacher_audio)
            source_times, rendered, refined, acoustic_transitions = refine_transition_map(
                regions, teacher_analysis, student_analysis, teacher_mel, source_mel,
                times, source_times, rendered, sample_rate=sr, hop_size=hop,
                allow_fallback=allow_acoustic_fallback, minimum_quality=minimum_quality,
                teacher_audio=teacher_audio, student_audio=audio,
            )
        contexts = build_boundary_context(
            regions, student_analysis.get("words", []), count / sr, len(audio) / sr,
            allow_fallback=allow_acoustic_fallback, minimum_quality=minimum_quality,
        )
        # Acoustic transitions own their gaps; do not layer held release/attack
        # context over those trajectories.
        contexts = [context for context in contexts if not any(
            min(context["teacher_end"], edit["teacher_end"]) >
            max(context["teacher_start"], edit["teacher_start"]) for edit in acoustic_transitions)]
        source_times, rendered, context_gain = _acoustic_map(contexts, times, source_times, rendered)
        if teacher_audio is not None:
            context_gain *= _reference_release_gain(contexts, times, teacher_audio, sr)
            energy_gain, acoustic_dynamics = reference_transition_gain(
                acoustic_transitions, regions, times, source_times, teacher_audio, audio, sr,
            )
        mel = _warp_mel(source_mel, source_times, rendered, hop, sr, vocoder.log_floor)
        context_mask = rendered & ~valid
        _context_mel(audio, vocoder, contexts, regions, times, source_times, valid | refined, mel)
        mel[:, context_mask] = np.maximum(
            math.log(vocoder.log_floor),
            mel[:, context_mask] + np.log(np.maximum(context_gain[context_mask], 1e-8)),
        )
        transitions = _smooth_source_jumps(mel, times, regions, allow_acoustic_fallback, minimum_quality)
        crossfades = _crossfade_splices(mel, times, source_times, acoustic_transitions)
    else:
        source_times = times
        valid = np.ones(frame_count, dtype=bool)
        rendered = valid.copy()
        context_mask = np.zeros(frame_count, dtype=bool)
        mel = source_mel
    student_f0, student_voiced = sample_pitch(student_analysis["pitch"], source_times)
    # identity and time_only are the pitch-preserving halves of the A/B set, so
    # their pitch is fixed by the mode name rather than by the switches.
    effective_policy = "none" if mode in {"identity", "time_only"} else pitch_policy
    allowed = [bool(_allowed(region, allow_acoustic_fallback, minimum_quality)) for region in regions]
    pitch_regions = [region for region, keep in zip(regions, allowed)
                     if keep and "student_start" in region and "student_end" in region]
    pitch_plan = register_pitch_plan(pitch_regions, [True] * len(pitch_regions),
                                     teacher_analysis["pitch"], student_analysis["pitch"], effective_policy)
    teacher_contour = effective_policy in TEACHER_CONTOUR_POLICIES
    per_word = effective_policy in PER_WORD_PLACEMENT_POLICIES
    if effective_policy == "none":
        f0 = student_f0
    elif per_word:
        # Adaptive placement moves each word onto the teacher's note level inside
        # the student's own register, so the melody stays in the student's range.
        shifts = frame_note_shifts(pitch_regions, pitch_plan["region_shifts_cents"], times,
                                  clock="teacher" if warped else "student")
        f0 = np.where(student_voiced, shifted_f0(student_f0, shifts), 0).astype(np.float32)
    elif teacher_contour:
        if warped:
            teacher_times, matched = times, np.ones(len(times), dtype=bool)
        else:
            # Without a timing change the output is on the student's clock, so the
            # teacher's curve is read through the inverse map.
            teacher_times, matched = _region_map(
                regions, times, inverse=True, allow_fallback=allow_acoustic_fallback,
                minimum_quality=minimum_quality,
            )
        teacher_f0, teacher_voiced = sample_pitch(teacher_analysis["pitch"], teacher_times)
        # The contour option puts the teacher's pitch on the student; the register
        # option then moves it into the student's range as one constant.
        teacher_f0 = shifted_f0(teacher_f0, pitch_plan["contour_shift_cents"])
        target = np.where(teacher_voiced & student_voiced, teacher_f0, 0).astype(np.float32)
        if warped:
            f0 = target
        else:
            # Nothing is muted without a timing change: an unmatched word is the
            # student's own material, so it keeps the student's own pitch.
            f0 = np.where(matched, target, np.where(student_voiced, student_f0, 0)).astype(np.float32)
    else:
        f0 = np.where(student_voiced,
                      shifted_f0(student_f0, pitch_plan["contour_shift_cents"]),
                      0).astype(np.float32)
    f0[~rendered] = 0
    if (effective_policy in ("student_contour_teacher_register", "teacher_contour_student_register")
            and not pitch_plan["offset_available"] and pitch_plan.get("offset_reason")):
        # The move the user asked for needs a measured register difference, and
        # the plan refused to measure one; say so rather than let the render look
        # like the move was applied.
        warnings.append(f"The requested pitch move was not applied: {pitch_plan['offset_reason']}")
    if per_word and pitch_plan["applied"]:
        measured = pitch_plan["note_shift_measured_count"]
        moved = pitch_plan["note_shift_applied_count"]
        if not pitch_plan["offset_available"]:
            warnings.append("Adaptive pitch placement could not measure a register "
                            f"difference, so no word was moved: {pitch_plan['offset_reason']}")
        else:
            warnings.append(f"Adaptive pitch placement moved {moved} of {measured} measured word(s) onto "
                            "the teacher's note level inside the student's own register; the rest were "
                            "already in place or beyond the safe correction limit.")

    saved_pitch = None
    if target_pitch_path is not None:
        pitch_path = Path(target_pitch_path).resolve()
        pitch_path.parent.mkdir(parents=True, exist_ok=True)
        with pitch_path.open("wb") as handle:
            np.savez_compressed(
                handle, times=times, f0_hz=f0, voiced=f0 > 0,
                source_times=source_times, source_voiced=student_voiced,
                supported=valid, rendered=rendered, acoustic_context=context_mask,
                time_map_refined=refined,
                acoustic_energy_gain=energy_gain,
                # Metadata keeps the target self-describing so verification knows
                # which pitch reference was aimed at.
                pitch_policy=np.array(effective_policy),
                register_offset_cents=np.float64(pitch_plan["offset_cents"]),
                register_offset_available=np.bool_(bool(pitch_plan["offset_available"])),
                note_shift_applied_count=np.int64(pitch_plan["note_shift_applied_count"]),
                timing_changed=np.bool_(bool(warped)),
            )
        saved_pitch = str(pitch_path)
    waveform = vocoder.synthesize(mel, f0, sample_count=count, chunk_frames=chunk_frames)
    waveform = np.asarray(waveform, dtype=np.float32)
    # Apply once after synthesis so neural excitation cannot defeat a quiet
    # reference; linear interpolation keeps the envelope continuous.
    if np.any(energy_gain < 1):
        for first in range(0, count, 1 << 20):
            last = min(count, first + (1 << 20))
            sample_times = np.arange(first, last, dtype=np.float64) / sr
            waveform[first:last] *= np.interp(sample_times, times, energy_gain)
    render_intervals = [{"teacher_start": edit["teacher_start"], "teacher_end": edit["teacher_end"],
                         "mode": "synthesize"} for edit in acoustic_transitions]
    envelope = _valid_envelope(count, sr, regions + render_intervals, allow_acoustic_fallback, minimum_quality, contexts=contexts) if warped else np.ones(count, dtype=np.float32)
    waveform = np.asarray(waveform * envelope, dtype=np.float32)
    rejected = [region for region in regions if not _allowed(region, allow_acoustic_fallback, minimum_quality)]
    unsupported = [region for region in rejected if region.get("mode") != "silence"]
    with_context = [region for region in unsupported if any(
        min(float(region["teacher_end"]), context["teacher_end"]) >
        max(float(region["teacher_start"]), context["teacher_start"]) for context in contexts + render_intervals)]
    if unsupported and warped:
        warnings.append(f"{len(unsupported)} unsupported or low-quality regions remain unresolved; acoustic rendering does not establish lyric correspondence.")
    if not np.any(f0 > 0):
        warnings.append("No voiced F0 frames remain after voicing and correspondence checks.")
    diagnostics = {
        "mode": mode,
        "options": {"change_timing": bool(change_timing), "pitch_mode": pitch_mode,
                    "map_teacher_pitch": bool(map_teacher_pitch)},
        "timing_changed": bool(warped),
        "pitch_policy": effective_policy,
        "register_pitch": _register_diagnostics(pitch_plan),
        "sample_rate": sr,
        "sample_count": count,
        "duration": count / sr,
        "frame_count": frame_count,
        "frame_center_offset_seconds": 0.5 * hop / sr,
        "target_pitch_path": saved_pitch,
        "supported_sample_ratio": float(np.mean(_valid_envelope(count, sr, regions, allow_acoustic_fallback, minimum_quality) > 0)) if warped else 1.0,
        "rendered_sample_ratio": float(np.mean(envelope > 0)),
        "acoustic_context_frame_count": int(context_mask.sum()),
        "acoustic_contexts": contexts,
        "acoustic_transitions": acoustic_transitions,
        "acoustic_splice_crossfades": crossfades,
        "acoustic_dynamics": acoustic_dynamics,
        "acoustic_energy_attenuated_frame_count": int(np.sum(energy_gain < 0.999)),
        "time_map_refined_frame_count": int(refined.sum()),
        "source_edit_transitions": transitions,
        "boundary_policy": "Continuous acoustic transitions in safe lyric gaps; hard lexical barriers remain muted. Unvoiced does not imply silence.",
        "voiced_frame_ratio": float(np.mean(f0 > 0)),
        "unresolved_region_count": len(unsupported) if warped else 0,
        "muted_region_count": len(unsupported) - len(with_context) if warped else 0,
        "partially_rendered_unresolved_region_count": len(with_context) if warped else 0,
        "unsupported_regions": [{key: region.get(key) for key in ("teacher_start", "teacher_end", "status", "mode", "quality")} for region in unsupported],
        "peak": float(np.max(np.abs(waveform))),
        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
        "warnings": warnings,
    }
    return waveform, diagnostics

