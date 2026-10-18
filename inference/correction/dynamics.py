"""Reference-relative level ceilings for continuous acoustic transitions.

Only waveform energy is compared. Pitch voicing is deliberately absent: a
fricative or a breath can be unvoiced and still need its full acoustic level.
"""

from __future__ import annotations

import numpy as np


_RMS_RADIUS_SECONDS = 0.010
_SHOULDER_SECONDS = 0.025
_QUIET_BEGIN = 0.100
_QUIET_FULL = 0.032
_CEILING_HEADROOM = 2.0
_MINIMUM_GAIN = 0.001


def _energy_integral(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Acoustic dynamics requires finite mono waveforms.")
    energy = np.empty(len(samples) + 1, dtype=np.float64)
    energy[0] = 0.0
    np.cumsum(np.square(samples, dtype=np.float64), out=energy[1:])
    return energy


def _rms(energy: np.ndarray, query: np.ndarray, sample_rate: int) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    finite = np.isfinite(query)
    centers = np.rint(np.where(finite, query, 0.0) * sample_rate).astype(np.int64)
    radius = max(1, int(round(_RMS_RADIUS_SECONDS * sample_rate)))
    first = np.clip(centers - radius, 0, len(energy) - 1)
    last = np.clip(centers + radius, 0, len(energy) - 1)
    result = np.sqrt(np.maximum(0.0, energy[last] - energy[first]) / np.maximum(1, last - first))
    return np.where(finite, result, 0.0)


def _smooth_rms(energy: np.ndarray, query: np.ndarray, sample_rate: int) -> np.ndarray:
    """Round the end of an RMS window so exact silence has no gain cliff."""
    result = np.zeros(np.asarray(query).shape, dtype=np.float64)
    offsets = np.arange(-10, 11, dtype=np.float64) / 1000
    weights = 11 - np.abs(np.arange(-10, 11))
    for offset, weight in zip(offsets, weights):
        result += weight * _rms(energy, np.asarray(query) + offset, sample_rate)
    return result / weights.sum()


def _phone_reference(region: dict, side: str, energy: np.ndarray, sample_rate: int,
                     timeline: str) -> tuple[float, dict]:
    """Use the original neighboring phone, not the contents of the whole gap."""
    anchors = np.asarray(region.get(f"{timeline}_anchors", [
        region[f"{timeline}_start"], region[f"{timeline}_end"],
    ]), dtype=np.float64)
    if anchors.ndim != 1 or len(anchors) < 2 or not np.isfinite(anchors).all() or np.any(np.diff(anchors) <= 0):
        raise ValueError("Acoustic dynamics requires increasing neighboring phone anchors.")
    first, last = (anchors[-2], anchors[-1]) if side == "left" else (anchors[0], anchors[1])
    # Sample the phone itself. A short boundary silence cannot dominate a robust
    # upper-middle percentile, and a long intervening rest is never a reference.
    count = max(3, int(np.ceil((last - first) / 0.005)))
    query = np.linspace(first, last, count + 2)[1:-1]
    reference = float(np.percentile(_rms(energy, query, sample_rate), 75))
    return reference, {"start": float(first), "end": float(last), "rms_p75": reference}


def _is_supported(region: dict) -> bool:
    return (region.get("mode", "synthesize") in ("synthesize", "acoustic_fallback")
            and not region.get("unsupported", False))


def _smooth_gain(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Smooth amplitude over 18 ms per side without creating a zero floor."""
    offsets = np.arange(-6, 7, dtype=np.float64) * 0.003
    weights = np.exp(-0.5 * (offsets / 0.006) ** 2)
    output = np.zeros(values.shape, dtype=np.float64)
    for offset, weight in zip(offsets, weights):
        output += weight * np.interp(times + offset, times, values, left=1.0, right=1.0)
    return output / weights.sum()


def reference_transition_gain(
    transitions: list[dict], regions: list[dict], times: np.ndarray,
    source_times: np.ndarray, teacher_audio: np.ndarray, student_audio: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, list[dict]]:
    """Suppress source carryover only where the teacher is relatively quiet.

    Each singer's original flanking phones establish an independent local level.
    At teacher levels below -20 dB relative to those phones, the mapped source is
    capped with 6 dB headroom. The ceiling reaches full strength below -30 dB.
    Continuous teacher consonants retain their energy; no pitch/UV labels enter
    this calculation. Gain is never boosted or set to zero, and its short outer
    shoulders remain inside the transition's fixed phone anchors.

    Semantic support, time maps and F0 remain untouched. The caller applies the
    returned frame gain to student-derived synthesis features or waveform.
    """
    times = np.asarray(times, dtype=np.float64)
    source_times = np.asarray(source_times, dtype=np.float64)
    if (sample_rate <= 0 or times.ndim != 1 or times.shape != source_times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Acoustic dynamics requires matching arrays and increasing finite query times.")
    gain = np.ones(times.shape, dtype=np.float32)
    if not transitions or not len(times):
        return gain, []
    teacher_energy = _energy_integral(teacher_audio)
    student_energy = _energy_integral(student_audio)
    teacher_rms = _smooth_rms(teacher_energy, times, sample_rate)
    source_rms = _smooth_rms(student_energy, source_times, sample_rate)
    diagnostics = []
    for transition in transitions:
        left_index = int(transition["left_region_index"])
        right_index = int(transition["right_region_index"])
        if not (0 <= left_index < len(regions) and 0 <= right_index < len(regions)):
            raise ValueError("Acoustic dynamics transition references an unknown region.")
        left, right = regions[left_index], regions[right_index]
        if not _is_supported(left) or not _is_supported(right):
            continue
        gap_start = float(transition["gap_teacher_start"])
        gap_end = float(transition["gap_teacher_end"])
        outer_start = float(transition["teacher_start"])
        outer_end = float(transition["teacher_end"])
        if not (np.isfinite([gap_start, gap_end, outer_start, outer_end]).all()
                and outer_start <= gap_start <= gap_end <= outer_end):
            raise ValueError("Acoustic dynamics requires a gap inside its outer phone anchors.")
        if gap_end <= gap_start:
            continue
        blocked = any(
            float(region["teacher_start"]) < gap_end - 1e-8
            and float(region["teacher_end"]) > gap_start + 1e-8
            and region.get("mode", "synthesize") != "silence"
            and not (region.get("mode") == "mute"
                     and region.get("reason") == "voiced_teacher_region_without_lyric_correspondence")
            for region in regions
        )
        if blocked:
            # Transition maps normally enforce this already. Keep the dynamics
            # helper harmless if called with an obsolete or external plan.
            continue
        start = max(outer_start, gap_start - _SHOULDER_SECONDS)
        end = min(outer_end, gap_end + _SHOULDER_SECONDS)
        selected = (times > start) & (times < end) & np.isfinite(source_times)
        if not selected.any():
            continue
        teacher_phones = [_phone_reference(region, side, teacher_energy, sample_rate, "teacher")
                          for region, side in ((left, "left"), (right, "right"))]
        student_phones = [_phone_reference(region, side, student_energy, sample_rate, "student")
                          for region, side in ((left, "left"), (right, "right"))]
        teacher_reference = max(item[0] for item in teacher_phones)
        student_reference = max(item[0] for item in student_phones)
        diagnostic = {
            "teacher_start": start, "teacher_end": end,
            "gap_teacher_start": gap_start, "gap_teacher_end": gap_end,
            "teacher_phone_reference_rms": teacher_reference,
            "student_phone_reference_rms": student_reference,
            "teacher_phone_references": [item[1] for item in teacher_phones],
            "student_phone_references": [item[1] for item in student_phones],
            "policy": "20ms_waveform_rms_local_phone_relative_quiet_ceiling",
            "rms_smoothing_support_seconds": 0.020,
            "gain_smoothing_support_seconds": 0.036,
            "quiet_begin_relative_db": -20.0, "quiet_full_relative_db": float(20 * np.log10(_QUIET_FULL)),
            "ceiling_headroom_db": float(20 * np.log10(_CEILING_HEADROOM)),
            "minimum_gain": 1.0, "attenuated_frame_count": 0,
        }
        if min(teacher_reference, student_reference) < 1e-7:
            diagnostic["status"] = "insufficient_flanking_phone_energy"
            diagnostics.append(diagnostic)
            continue
        target_level = teacher_rms[selected] / teacher_reference
        source_level = source_rms[selected] / student_reference
        ceiling = np.minimum(1.0, _CEILING_HEADROOM * target_level / np.maximum(source_level, 1e-10))
        # Log-level interpolation gives a soft transition into the quiet regime.
        quiet_weight = np.clip(np.log(_QUIET_BEGIN / np.maximum(target_level, 1e-10)) /
                               np.log(_QUIET_BEGIN / _QUIET_FULL), 0.0, 1.0)
        quiet_weight = quiet_weight ** 2 * (3.0 - 2.0 * quiet_weight)
        attenuation_db = np.log(np.maximum(_MINIMUM_GAIN, ceiling)) * quiet_weight
        selected_times = times[selected]
        raw_gain = _smooth_gain(np.exp(attenuation_db), selected_times)
        # Extend by at most 25 ms on either side, or taper inside a transition
        # whose outer anchors coincide with the gap boundaries.
        attack = min(_SHOULDER_SECONDS, (end - start) / 2)
        release = attack
        shoulder = np.sin(np.clip((selected_times - start) / attack, 0.0, 1.0) * np.pi / 2) ** 2
        shoulder *= np.sin(np.clip((end - selected_times) / release, 0.0, 1.0) * np.pi / 2) ** 2
        local_gain = np.exp(np.log(raw_gain) * shoulder).astype(np.float32)
        gain[selected] = np.minimum(gain[selected], local_gain)
        diagnostic.update({
            "status": "constrained" if np.any(local_gain < 0.999) else "preserved",
            "minimum_gain": float(local_gain.min()),
            "attenuated_frame_count": int(np.sum(local_gain < 0.999)),
            "teacher_gap_relative_rms_min": float(target_level.min()),
            "source_gap_relative_rms_max": float(source_level.max()),
        })
        diagnostics.append(diagnostic)
    return gain, diagnostics
