"""Continuous acoustic maps across safe, semantically matched word transitions."""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import dct
from scipy.signal import fftconvolve
from scipy.special import logsumexp

from .durations import DurationBudgetError, constrain_window_mapping, segments_for_window


_EPSILON = 1e-8
_GAP_REASON = "voiced_teacher_region_without_lyric_correspondence"
_MAX_UNSOURCED_GAP_SECONDS = 0.600
_MAX_PHONE_WINDOW_STRETCH = 3.0
_MAX_REAL_GAP_STRETCH = 4.0
_NOISE_ENERGY_DROP_LOG = 3.45
_MAX_NOISE_SEGMENT_STRETCH = 6.0


def _allowed(region: dict, allow_fallback: bool, minimum_quality: float) -> bool:
    mode = region.get("mode", "synthesize")
    if mode == "acoustic_fallback":
        return bool(allow_fallback) and float(region.get("quality", 0)) >= minimum_quality
    return mode == "synthesize" and not region.get("unsupported", False) and float(region.get("quality", 1)) >= minimum_quality


def _safe(region: dict) -> bool:
    return region.get("mode") == "silence" or (
        region.get("mode") == "mute" and region.get("reason") == _GAP_REASON
    )


def _anchors(region: dict) -> tuple[np.ndarray, np.ndarray]:
    teacher = np.asarray(region.get("teacher_anchors", [region["teacher_start"], region["teacher_end"]]), dtype=float)
    student = np.asarray(region.get("student_anchors", [region["student_start"], region["student_end"]]), dtype=float)
    if (teacher.ndim != 1 or len(teacher) < 2 or teacher.shape != student.shape
            or not np.isfinite(teacher).all() or not np.isfinite(student).all()
            or np.any(np.diff(teacher) <= 0) or np.any(np.diff(student) <= 0)):
        raise ValueError("Transition anchors must be finite and strictly increasing.")
    return teacher, student


def _edge_anchor(region: dict, side: str) -> tuple[float, float]:
    teacher, student = _anchors(region)
    if len(teacher) > 2:
        index = -2 if side == "left" else 1
        return float(teacher[index]), float(student[index])
    margin = min(0.060, float(teacher[-1] - teacher[0]) / 3)
    target = teacher[-1] - margin if side == "left" else teacher[0] + margin
    return float(target), float(np.interp(target, teacher, student))


def _context_knots(regions: list[dict], left_index: int, right_index: int,
                   start: float, end: float, source_left: float, source_right: float,
                   ) -> list[tuple[float, float]]:
    """Return one knot pair outside each window edge, when the neighbours provide one.

    Pinning a neighbouring anchor keeps the constrained cubic's rate at the edge
    close to the region it joins, so tempo changes gradually across the seam
    instead of stepping at it. That rate blending is the warp-domain crossfade:
    the source position itself is already continuous at these anchors.
    """
    context = []
    for index, side, edge, limit in ((left_index, "left", start, source_left),
                                     (right_index, "right", end, source_right)):
        if not 0 <= index < len(regions):
            continue
        region = regions[index]
        teacher = np.asarray(region.get("teacher_anchors", [region.get("teacher_start"), region.get("teacher_end")]), dtype=float)
        student = np.asarray(region.get("student_anchors", [region.get("student_start"), region.get("student_end")]), dtype=float)
        if (teacher.ndim != 1 or teacher.shape != student.shape or len(teacher) < 2
                or not np.isfinite(teacher).all() or not np.isfinite(student).all()
                or np.any(np.diff(teacher) <= 0) or np.any(np.diff(student) <= 0)):
            continue
        if side == "left":
            candidates = np.flatnonzero((teacher < edge - _EPSILON) & (student < limit - _EPSILON))
            index_to_use = int(candidates[-1]) if len(candidates) else None
        else:
            candidates = np.flatnonzero((teacher > edge + _EPSILON) & (student > limit + _EPSILON))
            index_to_use = int(candidates[0]) if len(candidates) else None
        if index_to_use is not None:
            context.append((float(teacher[index_to_use]), float(student[index_to_use])))
    return context


def _consecutive(left: dict, right: dict, key: str) -> bool:
    a, b = left.get(key), right.get(key)
    return a is None or b is None or b == a + 1


def _gap_is_safe(regions: list[dict], start: float, end: float,
                 allow_fallback: bool, minimum_quality: float) -> bool:
    cursor = start
    for region in sorted(regions, key=lambda item: item["teacher_start"]):
        a, b = float(region["teacher_start"]), float(region["teacher_end"])
        if b <= start + _EPSILON or a >= end - _EPSILON:
            continue
        if not _safe(region):
            return False
        if a > cursor + _EPSILON:
            return False
        cursor = max(cursor, b)
    return cursor >= end - _EPSILON


def _source_gap_is_safe(left: dict, right: dict, words: list[dict]) -> bool:
    start, end = float(left["student_end"]), float(right["student_start"])
    for index, word in enumerate(words):
        if index in (left.get("student_index"), right.get("student_index")):
            continue
        a, b = float(word["start"]), float(word["end"])
        if a < end - _EPSILON and b > start + _EPSILON:
            return False
    return True


def _sample_mel(mel: np.ndarray, start: float, end: float, step: float) -> tuple[np.ndarray, np.ndarray]:
    times = np.linspace(start, end, max(3, int(math.ceil((end - start) / step)) + 1))
    centers = (np.arange(mel.shape[1]) + 0.5) * step
    sampled = np.asarray([np.interp(times, centers, band) for band in mel])
    return times, sampled


def _feature_cost(teacher: np.ndarray, student: np.ndarray,
                  teacher_acoustics: dict | None = None,
                  student_acoustics: dict | None = None) -> np.ndarray:
    count = min(12, teacher.shape[0] - 1)
    tf = dct(teacher, type=2, axis=0, norm="ortho")[1:count + 1].T
    sf = dct(student, type=2, axis=0, norm="ortho")[1:count + 1].T
    if count:
        # Remove local speaker offsets, then share a robust scale per coefficient.
        tf -= np.median(tf, axis=0)
        sf -= np.median(sf, axis=0)
        pooled = np.concatenate([tf, sf])
        scale = np.maximum(0.35, (np.percentile(pooled, 75, axis=0) - np.percentile(pooled, 25, axis=0)) / 1.349)
        difference = (tf[:, None, :] - sf[None, :, :]) / scale
        cost = np.mean(np.minimum(difference ** 2, 9.0), axis=2)
    else:
        cost = np.zeros((teacher.shape[1], student.shape[1]))
    te = logsumexp(teacher, axis=0) - math.log(teacher.shape[0])
    se = logsumexp(student, axis=0) - math.log(student.shape[0])
    tflat, sflat = np.mean(teacher, axis=0) - te, np.mean(student, axis=0) - se
    te -= np.percentile(te, 85)
    se -= np.percentile(se, 85)
    # Energy remains weaker than spectral shape; no pitch/UV identity assumption.
    cost += 0.15 * np.minimum(((te[:, None] - se[None, :]) / 2) ** 2, 9.0)
    # Broadband noise is an acoustic cue even when a pitch tracker incorrectly
    # reports voicing. Keep this cue independent of speaker cepstral offsets.
    cost += 0.35 * np.minimum(((tflat[:, None] - sflat[None, :]) / 0.8) ** 2, 9.0)
    if teacher_acoustics is not None and student_acoustics is not None:
        # Absolute waveform periodicity survives local cepstral normalization.
        # A weak breath can be aperiodic without having a bright spectrum.
        tp, sp = teacher_acoustics["periodicity"], student_acoustics["periodicity"]
        ta, sa = teacher_acoustics["audible"], student_acoustics["audible"]
        cost += 0.8 * np.minimum(((tp[:, None] - sp[None, :]) / 0.35) ** 2, 9.0) * (ta[:, None] & sa[None, :])
    return cost


def _waveform_features(audio: np.ndarray, times: np.ndarray, sample_rate: int) -> dict:
    """Measure local periodicity directly, without a pitch or voicing model.

    Overlap-normalized autocorrelation searches 70--1100 Hz in 30 ms windows.
    Short noisy breaths and low-frequency /h/ remain distinguishable from a
    vowel, even when a pitch tracker hallucinates a stable F0 in that breath.
    """
    radius = max(2, int(round(0.015 * sample_rate)))
    size = 2 * radius
    first_lag = max(1, int(round(sample_rate / 1100)))
    last_lag = min(size // 2, int(round(sample_rate / 70)))
    lags = np.arange(first_lag, max(first_lag + 1, last_lag + 1))
    periodicity, rms = np.zeros(len(times)), np.zeros(len(times))
    offsets = np.arange(-radius, radius)
    for index, time in enumerate(times):
        indices = np.clip(int(round(time * sample_rate)) + offsets, 0, len(audio) - 1)
        frame = audio[indices].astype(np.float64)
        frame -= np.mean(frame)
        energy = np.r_[0.0, np.cumsum(frame * frame)]
        rms[index] = math.sqrt(energy[-1] / size)
        if energy[-1] <= 1e-18:
            continue
        acf = fftconvolve(frame, frame[::-1], mode="full")[size - 1:]
        denominator = np.sqrt(np.maximum(1e-24, energy[size - lags] * (energy[-1] - energy[lags])))
        periodicity[index] = np.clip(np.max(acf[lags] / denominator), 0, 1)
    log_rms = np.log(np.maximum(rms, 1e-12))
    audible = (log_rms >= np.percentile(log_rms, 85) - _NOISE_ENERGY_DROP_LOG) & (rms > 1e-8)
    return {"periodicity": periodicity, "rms": rms, "audible": audible}


def _aperiodic_onset(features: dict, times: np.ndarray) -> tuple[float, float, float] | None:
    """Find sustained audible noise following a sustained tonal phone.

    The tonal reference may lie after an initial noisy consonant. A minimum
    30 ms noise run avoids expanding a single release burst into a long rest.
    """
    if len(times) < 10:
        return None
    step = float(np.median(np.diff(times)))
    tonal_run = max(2, int(math.ceil(0.020 / step)))
    noise_run = max(3, int(math.ceil(0.030 / step)))
    periodicity, audible = features["periodicity"], features["audible"]
    tonal = (periodicity >= 0.75) & audible
    noise = (periodicity <= 0.55) & audible
    last_tonal = None
    for index in range(len(times) - noise_run + 1):
        if index >= tonal_run and tonal[index - tonal_run:index].all():
            last_tonal = index - 1
        if (last_tonal is None or index <= last_tonal or times[index] - times[0] < 0.035
                or times[-1] - times[index] < 0.035 or not noise[index:index + noise_run].all()):
            continue
        stop = index + noise_run
        while stop < len(times) and noise[stop]:
            stop += 1
        duration = float(times[min(stop, len(times) - 1)] - times[index])
        contrast = float(np.median(periodicity[max(0, last_tonal - tonal_run + 1):last_tonal + 1]) -
                         np.median(periodicity[index:stop]))
        return float(times[index]), contrast, duration
    return None


def _noise_onset(mel: np.ndarray, times: np.ndarray) -> tuple[float, float] | None:
    """Find sustained broadband noise after an initially more tonal phone.

    Both spectral flatness and high/low frequency energy must rise. Every frame
    in the qualifying run must remain within about 30 dB of local p85 energy;
    a silent valley becoming spectrally flat cannot establish a noise anchor.
    A local reference makes this useful across different singers without assuming that
    every unvoiced frame is silence, or that breath must equal a consonant.
    """
    if len(times) < 10 or mel.shape[0] < 4:
        return None
    energy = logsumexp(mel, axis=0) - math.log(mel.shape[0])
    flatness = np.mean(mel, axis=0) - energy
    middle = mel.shape[0] // 2
    high = logsumexp(mel[middle:], axis=0) - logsumexp(mel[:middle], axis=0)
    reference = min(4, len(times) // 4)
    flat_base, high_base = np.median(flatness[:reference]), np.median(high[:reference])
    flat_rise = np.percentile(flatness, 85) - flat_base
    high_rise = np.percentile(high, 85) - high_base
    if flat_rise < 0.6 or high_rise < 0.8:
        return None
    activity = ((flatness >= flat_base + 0.55 * flat_rise) &
                (high >= high_base + 0.35 * high_rise) &
                (energy >= np.percentile(energy, 85) - _NOISE_ENERGY_DROP_LOG) &
                (np.ptp(mel, axis=0) > 1e-6))
    run = max(2, int(math.ceil(0.020 / np.median(np.diff(times)))))
    for index in range(reference, len(times) - run + 1):
        if (activity[index:index + run].all() and times[index] - times[0] >= 0.035
                and times[-1] - times[index] >= 0.035):
            return float(times[index]), float(min(flat_rise / 0.6, high_rise / 0.8))
    return None


def _dtw(teacher: np.ndarray, student: np.ndarray, teacher_times: np.ndarray,
         student_times: np.ndarray, teacher_acoustics: dict | None = None,
         student_acoustics: dict | None = None) -> tuple[np.ndarray, float] | None:
    cost = _feature_cost(teacher, student, teacher_acoustics, student_acoustics)
    rows, columns = cost.shape
    if not 0.5 <= (columns - 1) / (rows - 1) <= 2.0:
        return None
    accumulated = np.full((rows, columns), np.inf)
    previous = np.zeros((rows, columns), dtype=np.int8)
    accumulated[0, 0] = cost[0, 0]
    expected = np.linspace(student_times[0], student_times[-1], rows)
    for i in range(1, rows):
        for j in range(1, columns):
            if abs(student_times[j] - expected[i]) > 0.300:
                continue
            candidates = [(accumulated[i - 1, j - 1], 1, 1.0)]
            if j > 1:
                candidates.append((accumulated[i - 1, j - 2], 2, 1.5))
            if i > 1:
                candidates.append((accumulated[i - 2, j - 1], 3, 1.5))
            value, code, weight = min((value + weight * cost[i, j], code, weight)
                                      for value, code, weight in candidates)
            accumulated[i, j], previous[i, j] = value, code
    if not np.isfinite(accumulated[-1, -1]):
        return None
    i, j = rows - 1, columns - 1
    path = [(i, j)]
    while i or j:
        code = previous[i, j]
        if code == 1:
            i, j = i - 1, j - 1
        elif code == 2:
            i, j = i - 1, j - 2
        elif code == 3:
            i, j = i - 2, j - 1
        else:
            return None
        path.append((i, j))
    path.reverse()
    target_path = teacher_times[[point[0] for point in path]]
    source_path = student_times[[point[1] for point in path]]
    mapping = np.interp(teacher_times, target_path, source_path)
    if len(mapping) >= 5:
        first_step, last_step = mapping[1] - mapping[0], mapping[-1] - mapping[-2]
        padded = np.r_[mapping[0] - 2 * first_step, mapping[0] - first_step,
                       mapping, mapping[-1] + last_step, mapping[-1] + 2 * last_step]
        mapping = np.convolve(padded, np.array([1, 2, 3, 2, 1]) / 9, mode="valid")
        mapping = student_times[0] + ((mapping - mapping[0]) / (mapping[-1] - mapping[0])) * (student_times[-1] - student_times[0])
    return mapping, float(accumulated[-1, -1] / (rows + columns))


def _constrain(grid: np.ndarray, mapping: np.ndarray, start: float, end: float,
               source_left: float, source_right: float,
               student_analysis: dict | None, student_mel: np.ndarray,
               regions: list[dict] | None, left_index: int | None, right_index: int | None,
               step: float) -> tuple[np.ndarray, dict | None]:
    """Re-time a searched warp so consonants keep the length the student sang.

    Without forced-alignment phones inside the window there is nothing to
    protect, so the searched warp stands and the constraint is reported as not
    applied rather than guessed at. A window that cannot be timed without
    lengthening a consonant is also left to the searched warp and reported, so a
    stretched fricative is never produced silently.
    """
    if (student_analysis is None or regions is None or left_index is None or right_index is None):
        return mapping, None
    segments = segments_for_window(student_analysis, source_left, source_right,
                                   regions, left_index, right_index, student_mel, step)
    if not segments:
        return mapping, {"applied": False, "reason": "no_phone_timeline_in_window"}
    context = _context_knots(regions, left_index, right_index, start, end, source_left, source_right)
    try:
        constrained, info = constrain_window_mapping(grid, grid, mapping, segments,
                                                     window=(start, end), context=context)
    except DurationBudgetError as error:
        return mapping, {"applied": False, "reason": "consonant_limits_infeasible",
                         "detail": str(error)}
    info["applied"] = True
    return constrained, info


def _align_window(teacher_mel: np.ndarray, student_mel: np.ndarray,
                  start: float, end: float, source_left: float, source_right: float,
                  step: float, teacher_audio: np.ndarray | None = None,
                  student_audio: np.ndarray | None = None, sample_rate: int = 44100,
                  *, student_analysis: dict | None = None, regions: list[dict] | None = None,
                  left_index: int | None = None, right_index: int | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, float | None, str, dict | None, dict | None]:
    teacher_grid, teacher_features = _sample_mel(teacher_mel, start, end, step)
    student_grid, student_features = _sample_mel(student_mel, source_left, source_right, step)
    teacher_acoustics = (_waveform_features(teacher_audio, teacher_grid, sample_rate)
                         if teacher_audio is not None else None)
    student_acoustics = (_waveform_features(student_audio, student_grid, sample_rate)
                         if student_audio is not None else None)
    teacher_noise = _noise_onset(teacher_features, teacher_grid)
    student_noise = _noise_onset(student_features, student_grid)
    teacher_cue, student_cue = "mel_noise_rise", "mel_noise_rise"
    teacher_support, student_support = None, None
    if teacher_acoustics is not None and student_acoustics is not None:
        teacher_wave_noise = _aperiodic_onset(teacher_acoustics, teacher_grid)
        student_wave_noise = _aperiodic_onset(student_acoustics, student_grid)
        if teacher_wave_noise is not None and (teacher_noise is None or
                teacher_wave_noise[0] + 0.040 < teacher_noise[0] < teacher_wave_noise[0] + teacher_wave_noise[2]):
            teacher_noise = teacher_wave_noise[:2]
            teacher_cue = "waveform_periodicity_drop"
        if student_wave_noise is not None and (student_noise is None or
                student_wave_noise[0] + 0.040 < student_noise[0] < student_wave_noise[0] + student_wave_noise[2]):
            student_noise = student_wave_noise[:2]
            student_cue = "waveform_periodicity_drop"
        if teacher_wave_noise is not None:
            teacher_support = teacher_wave_noise[2]
        if student_wave_noise is not None:
            student_support = student_wave_noise[2]
    if teacher_noise is not None and student_noise is not None:
        target_onset, target_strength = teacher_noise
        source_onset, source_strength = student_noise
        target_progress = (target_onset - start) / (end - start)
        source_progress = (source_onset - source_left) / (source_right - source_left)
        ratios = ((target_onset - start) / (source_onset - source_left),
                  (end - target_onset) / (source_right - source_onset))
        # A sustained source breath can cover a longer teacher breath, while
        # brief bursts retain the conservative fourfold segment limit.
        maximum_ratio = (_MAX_NOISE_SEGMENT_STRETCH if student_support is not None
                         and student_support >= 0.060 else 4.0)
        if abs(target_progress - source_progress) >= 0.08 and 0.25 <= ratios[0] <= 4 and 1 / maximum_ratio <= ratios[1] <= maximum_ratio:
            target_knots, source_knots = [start, target_onset, end], [source_left, source_onset, source_right]
            grids, maps, costs = [], [], []
            for index in range(2):
                tg, tf = _sample_mel(teacher_mel, target_knots[index], target_knots[index + 1], step)
                sg, sf = _sample_mel(student_mel, source_knots[index], source_knots[index + 1], step)
                ta = _waveform_features(teacher_audio, tg, sample_rate) if teacher_audio is not None else None
                sa = _waveform_features(student_audio, sg, sample_rate) if student_audio is not None else None
                result = _dtw(tf, sf, tg, sg, ta, sa) if max(len(tg), len(sg)) <= 240 else None
                mapping = result[0] if result is not None else np.interp(tg, [tg[0], tg[-1]], [sg[0], sg[-1]])
                grids.append(tg if index == 0 else tg[1:])
                maps.append(mapping if index == 0 else mapping[1:])
                if result is not None:
                    costs.append(result[1])
            waveform_cue = "waveform_periodicity_drop" in (teacher_cue, student_cue)
            landmark = {"kind": ("sustained_aperiodic_onset" if waveform_cue else "sustained_broadband_noise_onset"), "teacher_time": target_onset,
                        "student_time": source_onset, "teacher_strength": target_strength,
                        "student_strength": source_strength,
                        "minimum_duration_seconds": 0.030 if waveform_cue else 0.020,
                        "teacher_minimum_duration_seconds": 0.030 if teacher_cue == "waveform_periodicity_drop" else 0.020,
                        "student_minimum_duration_seconds": 0.030 if student_cue == "waveform_periodicity_drop" else 0.020,
                        "minimum_relative_logmel_energy": -_NOISE_ENERGY_DROP_LOG,
                        "minimum_relative_waveform_log_rms": -_NOISE_ENERGY_DROP_LOG if waveform_cue else None,
                        "teacher_cue": teacher_cue, "student_cue": student_cue,
                        "teacher_aperiodic_support_seconds": teacher_support,
                        "student_aperiodic_support_seconds": student_support,
                        "maximum_post_onset_stretch": maximum_ratio}
            mapping = np.concatenate(maps)
            grid = np.concatenate(grids)
            constrained, constraint = _constrain(grid, mapping, start, end, source_left, source_right,
                                                 student_analysis, student_mel, regions,
                                                 left_index, right_index, step)
            return grid, constrained, float(np.mean(costs)) if costs else None, "noise_anchored_acoustic_dtw", landmark, constraint
    result = _dtw(teacher_features, student_features, teacher_grid, student_grid,
                  teacher_acoustics, student_acoustics) if max(len(teacher_grid), len(student_grid)) <= 240 else None
    if result is None:
        return (np.array([start, end]), np.array([source_left, source_right]), None,
                "phone_window_linear", None, None)
    constrained, constraint = _constrain(teacher_grid, result[0], start, end, source_left, source_right,
                                         student_analysis, student_mel, regions,
                                         left_index, right_index, step)
    return teacher_grid, constrained, result[1], "local_acoustic_dtw", None, constraint


def refine_transition_map(
    regions: list[dict], teacher_analysis: dict, student_analysis: dict,
    teacher_mel: np.ndarray, student_mel: np.ndarray, query_times: np.ndarray,
    source_times: np.ndarray, rendered: np.ndarray, *, sample_rate: int = 44100,
    hop_size: int = 512, allow_fallback: bool = False, minimum_quality: float = 0.0,
    teacher_audio: np.ndarray | None = None, student_audio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Retain entire safe interword material using locally anchored acoustic maps.

    Word correspondence remains an immutable semantic annotation. Interior phone
    anchors bound acoustic refinements, which can borrow a changing vowel or
    consonant trajectory when a target gap has no dedicated source samples.
    Missing words, rejected alignments and unmatched source words are barriers.
    """
    teacher_mel, student_mel = np.asarray(teacher_mel, dtype=float), np.asarray(student_mel, dtype=float)
    times = np.asarray(query_times, dtype=float)
    mapped, active = np.asarray(source_times, dtype=float).copy(), np.asarray(rendered, dtype=bool).copy()
    if (times.ndim != 1 or mapped.shape != times.shape or active.shape != times.shape
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Transition query arrays must have equal shapes and increasing finite times.")
    if (teacher_mel.ndim != 2 or student_mel.ndim != 2 or teacher_mel.shape[0] != student_mel.shape[0]
            or not teacher_mel.shape[1] or not student_mel.shape[1]
            or not np.isfinite(teacher_mel).all() or not np.isfinite(student_mel).all()
            or sample_rate <= 0 or hop_size <= 0):
        raise ValueError("Transition mel arrays and frame settings must be finite and valid.")
    if (teacher_audio is None) != (student_audio is None):
        raise ValueError("Transition waveform cues require both teacher and student audio.")
    if teacher_audio is not None:
        teacher_audio, student_audio = np.asarray(teacher_audio), np.asarray(student_audio)
        if any(audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all()
               for audio in (teacher_audio, student_audio)):
            raise ValueError("Transition waveforms must be nonempty finite mono arrays.")
    mask = np.zeros(times.shape, dtype=bool)
    diagnostics = []
    allowed = sorted(((index, region) for index, region in enumerate(regions)
                      if _allowed(region, allow_fallback, minimum_quality)),
                     key=lambda item: item[1]["teacher_start"])
    step = hop_size / sample_rate
    for (left_index, left), (right_index, right) in zip(allowed, allowed[1:]):
        gap_start, gap_end = float(left["teacher_end"]), float(right["teacher_start"])
        source_start, source_end = float(left["student_end"]), float(right["student_start"])
        target_gap, source_gap = gap_end - gap_start, source_end - source_start
        if (target_gap < -_EPSILON or source_gap < -_EPSILON
                or (target_gap <= _EPSILON and source_gap <= _EPSILON)
                or not _consecutive(left, right, "teacher_index")
                or not _consecutive(left, right, "student_index")
                or not _gap_is_safe(regions, gap_start, gap_end, allow_fallback, minimum_quality)
                or not _source_gap_is_safe(left, right, student_analysis.get("words", []))):
            continue
        start, source_left = _edge_anchor(left, "left")
        end, source_right = _edge_anchor(right, "right")
        if (source_right <= source_left + _EPSILON
                or (end - start) / (source_right - source_left) > _MAX_PHONE_WINDOW_STRETCH
                or (source_gap < max(0.030, 2 * step) and target_gap > _MAX_UNSOURCED_GAP_SECONDS)):
            # A long target rest without source material must remain unresolved;
            # spreading a tiny phone trajectory over it would create a drone.
            continue
        strategy, normalized_cost, landmark, constraint = "local_acoustic_dtw", None, None, None
        if source_gap > _EPSILON and target_gap > _EPSILON and (target_gap > 0.8 or source_gap > 0.8):
            # Preserve long real rests/breaths directly rather than forcing a
            # large spectral alignment with weak acoustic identity evidence.
            if not 1 / _MAX_REAL_GAP_STRETCH <= target_gap / source_gap <= _MAX_REAL_GAP_STRETCH:
                continue
            start, end, source_left, source_right = gap_start, gap_end, source_start, source_end
            strategy = "continuous_gap_linear"
        if end <= start + _EPSILON or source_right <= source_left + _EPSILON:
            continue
        selected = (times >= start) & (times < end)
        if not selected.any() or np.any(mask & selected):
            continue
        local_times = np.array([start, end])
        local_mapping = np.array([source_left, source_right])
        if strategy == "local_acoustic_dtw":
            local_times, local_mapping, normalized_cost, strategy, landmark, constraint = _align_window(
                teacher_mel, student_mel, start, end, source_left, source_right, step,
                teacher_audio, student_audio, sample_rate,
                student_analysis=student_analysis, regions=regions,
                left_index=left_index, right_index=right_index,
            )
        replacement = np.interp(times[selected], local_times, local_mapping)
        acoustic_mismatch = None
        if teacher_audio is not None and target_gap > _EPSILON:
            in_gap = (times[selected] >= gap_start) & (times[selected] < gap_end)
            gap_queries = times[selected][in_gap]
            if len(gap_queries):
                ta = _waveform_features(teacher_audio, gap_queries, sample_rate)
                sa = _waveform_features(student_audio, replacement[in_gap], sample_rate)
                target_noise = (ta["periodicity"] < 0.4) & ta["audible"]
                mapped_tonal = target_noise & (sa["periodicity"] > 0.6) & sa["audible"]
                acoustic_mismatch = {
                    "aperiodic_target_frame_count": int(target_noise.sum()),
                    "mapped_periodic_frame_count": int(mapped_tonal.sum()),
                    "mapped_periodic_target_times": gap_queries[mapped_tonal].tolist(),
                    "policy": "waveform_texture_mismatch_is_diagnostic_only_not_lyric_evidence",
                }
        old = mapped[selected]
        changed = ~np.isfinite(old) | (np.abs(old - replacement) > 1e-7)
        mapped[selected], active[selected], mask[selected] = replacement, True, True
        diagnostics.append({
            "teacher_start": start, "teacher_end": end,
            "student_start": source_left, "student_end": source_right,
            "gap_teacher_start": gap_start, "gap_teacher_end": gap_end,
            "gap_student_start": source_start, "gap_student_end": source_end,
            "source_strategy": strategy, "strategy": strategy,
            "normalized_cost": normalized_cost,
            "acoustic_landmark": landmark,
            "source_acoustic_mismatch": acoustic_mismatch,
            "duration_constraint": constraint,
            "changed_source_frame_count": int(changed.sum()),
            "transition_frame_count": int(selected.sum()),
            "left_region_index": left_index, "right_region_index": right_index,
            "semantic_support": False,
            "policy": "continuous_safe_interword_acoustics_with_fixed_outer_phone_anchors",
        })
    return mapped, active, mask, diagnostics
