from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp

from inference.alignment.planner import _SILENCE, _is_vowel, _phone


_EPSILON = 1e-9
_TOLERANCE = 1e-7
_MINIMUM_SEGMENT_SECONDS = 0.004
# A resting span this far below the window's phones is a rest rather than
# breath or fricative material worth preserving at its sung length.
_SPAN_AUDIBLE_DROP_LOG = 2.30

# Preferred, extended and hard rate limits per tier; consonants are bounded
# against their own sung length because the singer already pronounced them.
_LIMITS = {
    "vowel": ((0.80, 2.00), (0.65, 3.00), (0.45, 4.00)),
    "span": ((0.70, 1.60), (0.55, 2.20), (0.40, 3.00)),
    "quiet_span": ((0.40, 4.00), (0.25, 6.00), (0.15, 8.00)),
}
# Weighted projection: a high weight resists leaving the searched duration.
# Consonants are stiff, quiet rests are the most elastic.
_WEIGHTS = {"vowel": 1.0, "consonant": 6.0, "span": 2.0, "quiet_span": 0.35}
# Order in which an unfittable remainder is handed out; consonants are absent on
# purpose so an infeasible budget reports itself instead of stretching one.
_RESIDUAL_ORDER = (("quiet_span",), ("span",), ("vowel",))


class DurationBudgetError(ValueError):
    """Raised when a window cannot be timed without stretching a consonant."""


def classify_phone(label: str) -> str:
    """Return ``vowel``, ``consonant`` or ``span`` for an aligned phone label."""
    phone = _phone(label)
    if not phone or phone in _SILENCE:
        return "span"
    return "vowel" if _is_vowel(label) else "consonant"


def _tier_limits(kind: str, source: float, native: float, tier: int) -> tuple[float, float]:
    """Cumulative rate limits, so an expanded tier always contains the earlier ones.

    A consonant is bounded above by the length the student sang in every tier."""
    if kind == "consonant":
        candidates = [(0.85 * source, source),
                      (0.70 * source, source),
                      (0.55 * source, source)]
    else:
        candidates = [(_lower * source, _upper * source) for _lower, _upper in _LIMITS[kind]]
    lower, upper = candidates[0]
    for index in range(1, min(tier, len(candidates) - 1) + 1):
        lower = min(lower, candidates[index][0])
        upper = max(upper, candidates[index][1])
    return lower, upper


def _preferred(desired: np.ndarray, kinds: list[str], source: np.ndarray,
               native: np.ndarray) -> np.ndarray:
    """Prefer the planner's budget for consonants, the searched warp elsewhere.

    The warp's spectral-continuity cost makes stretching a fricative look free."""
    result = np.maximum(np.asarray(desired, dtype=float), _MINIMUM_SEGMENT_SECONDS)
    for index, kind in enumerate(kinds):
        if kind != "consonant":
            continue
        reference = native[index] if native[index] > 0 else source[index]
        if reference > 0:
            result[index] = min(result[index], reference)
    return np.maximum(result, _MINIMUM_SEGMENT_SECONDS)


def _project(desired: np.ndarray, lower: np.ndarray, upper: np.ndarray,
             target: float, weights: np.ndarray) -> np.ndarray:
    """Weighted least-change projection of ``desired`` onto the bounded sum."""
    lower = np.maximum(np.asarray(lower, dtype=float), 0.0)
    upper = np.maximum(np.asarray(upper, dtype=float), lower)
    if float(lower.sum()) >= target:
        return lower.copy()
    if float(upper.sum()) <= target:
        return upper.copy()
    low_lambda = float(np.min(weights * (lower - desired)))
    high_lambda = float(np.max(weights * (upper - desired)))
    for _ in range(80):
        middle = 0.5 * (low_lambda + high_lambda)
        if float(np.clip(desired + middle / weights, lower, upper).sum()) < target:
            low_lambda = middle
        else:
            high_lambda = middle
    return np.clip(desired + 0.5 * (low_lambda + high_lambda) / weights, lower, upper)


def _absorb(duration: np.ndarray, kinds: list[str], energy: np.ndarray,
            hard_upper: np.ndarray, remainder: float) -> tuple[np.ndarray, int | None, float]:
    """Grow an unfittable remainder into the most elastic non-consonant segments.

    A negative remainder means the target is below the sum of the hard lower
    bounds, which the projection already returned. Shrinking past those bounds
    would produce non-physical segments (even negative durations), so it is
    left to the caller to report the window as infeasible. Anything still
    unfitted is returned so the caller can keep the searched warp."""
    duration, chosen = duration.copy(), None
    if remainder < -_TOLERANCE:
        return duration, None, remainder
    for group in _RESIDUAL_ORDER:
        if remainder <= _TOLERANCE:
            break
        indices = [index for index, kind in enumerate(kinds) if kind in group]
        if not indices:
            continue
        indices.sort(key=lambda index: float(energy[index]))
        while remainder > _TOLERANCE:
            room = [max(0.0, float(hard_upper[index]) - duration[index]) for index in indices]
            if sum(room) <= _EPSILON:
                break
            share = remainder / max(1, sum(1 for value in room if value > _EPSILON))
            for index, available in zip(list(indices), room):
                if available <= _EPSILON:
                    continue
                applied = min(available, share)
                duration[index] += applied
                remainder -= applied
                if remainder <= _TOLERANCE:
                    break
            indices = [index for index in indices
                       if duration[index] < float(hard_upper[index]) - _EPSILON]
        if chosen is None and indices:
            chosen = indices[0]
    return duration, chosen, remainder


def plan_segment_durations(segments: list[dict], desired: np.ndarray, target: float, *,
                           minimum_seconds: float = _MINIMUM_SEGMENT_SECONDS) -> tuple[np.ndarray, dict]:
    """Budget target time so consonants never stretch and rests flex first.

    Segments carry ``start``, ``end``, ``kind``, ``native`` and ``energy``."""
    if not segments or not math.isfinite(target) or target <= 0:
        raise ValueError("Duration planning requires segments and a positive finite target.")
    desired = np.asarray(desired, dtype=float)
    if desired.shape != (len(segments),) or not np.isfinite(desired).all():
        raise ValueError("Desired durations must be finite and match the segment count.")
    source = np.array([max(0.0, float(item["end"]) - float(item["start"])) for item in segments])
    kinds = [str(item["kind"]) for item in segments]
    native = np.array([max(0.0, float(item.get("native") or 0.0)) for item in segments])
    energy = np.array([float(item.get("energy", 1.0)) for item in segments])
    weights = np.array([_WEIGHTS.get(kind, 1.0) for kind in kinds])
    desired = _preferred(desired, kinds, source, native)

    duration, tier_used = None, 0
    hard_upper = None
    for tier in range(3):
        lower, upper = [], []
        for index, kind in enumerate(kinds):
            low, high = _tier_limits(kind, float(source[index]), float(native[index]), tier)
            lower.append(max(minimum_seconds, low))
            upper.append(max(lower[-1], high))
        lower, upper = np.array(lower), np.array(upper)
        hard_upper = upper
        duration, tier_used = _project(desired, lower, upper, target, weights), tier
        if abs(float(duration.sum()) - target) <= _TOLERANCE:
            break
    assert hard_upper is not None

    remainder = float(target - duration.sum())
    residual_policy = "none"
    residual_segment = None
    if abs(remainder) > _TOLERANCE:
        # Consonants are never asked to absorb time; when only they remain, the
        # window is infeasible and the caller keeps the searched warp.
        duration, residual_segment, remainder = _absorb(
            duration, kinds, energy, hard_upper, remainder)
        residual_policy = "quiet_rest_then_span_then_vowel"
    if abs(remainder) > _TOLERANCE:
        raise DurationBudgetError(
            f"Window cannot be timed within phoneme limits; {remainder:.6f} s remains unfitted.")
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise DurationBudgetError("Window duration planning produced non-positive segments.")

    rates = duration / np.maximum(source, _EPSILON)
    reference = np.where(native > 0, native, source)
    consonant = [index for index, kind in enumerate(kinds) if kind == "consonant"]
    vowel = [index for index, kind in enumerate(kinds) if kind == "vowel"]
    info = {
        "segment_count": len(segments),
        "kinds": kinds,
        "tier_used": int(tier_used),
        "source_seconds": [float(value) for value in source],
        "native_seconds": [float(value) for value in native],
        "planned_seconds": [float(value) for value in duration],
        "source_rates": [float(value) for value in rates],
        "native_rates": [float(value) for value in duration / np.maximum(reference, _EPSILON)],
        "consonant_stretched": bool(any(rates[index] > 1.0 + 1e-9 for index in consonant)),
        "maximum_consonant_source_rate": float(max((rates[index] for index in consonant), default=0.0)),
        "maximum_vowel_source_rate": float(max((rates[index] for index in vowel), default=0.0)),
        "residual_seconds": float(target - duration.sum()),
        "residual_segment": residual_segment,
        "residual_kind": kinds[residual_segment] if residual_segment is not None else None,
        "residual_unresolved": False,
        "residual_policy": residual_policy,
        "policy": "consonants_keep_sung_length_vowels_and_rests_absorb_time",
    }
    return duration, info


def monotone_segment_map(query: np.ndarray, knot_target: list[float], knot_source: list[float],
                         context: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Evaluate a monotone cubic that pins every knot exactly.

    ``context`` adds knots outside the window so the rate blends into neighbours."""
    target = [float(value) for value in knot_target]
    source = [float(value) for value in knot_source]
    for point_target, point_source in sorted(context or [], key=lambda item: float(item[0])):
        point_target, point_source = float(point_target), float(point_source)
        if point_target <= target[0] - _EPSILON:
            target.insert(0, point_target)
            source.insert(0, point_source)
        elif point_target >= target[-1] + _EPSILON:
            target.append(point_target)
            source.append(point_source)
    target_array, source_array = np.asarray(target, dtype=float), np.asarray(source, dtype=float)
    keep = np.r_[True, np.diff(target_array) > _EPSILON]
    target_array, source_array = target_array[keep], source_array[keep]
    if (len(target_array) < 2 or np.any(np.diff(target_array) <= _EPSILON)
            or np.any(np.diff(source_array) <= _EPSILON)):
        raise ValueError("Constrained mappings require increasing knots on both timelines.")
    query = np.asarray(query, dtype=float)
    if not np.isfinite(query).all():
        raise ValueError("Constrained mapping queries must be finite.")
    clipped = np.clip(query, target_array[0], target_array[-1])
    mapped = np.asarray(PchipInterpolator(target_array, source_array)(clipped), dtype=float)
    if not np.isfinite(mapped).all():
        raise ValueError("Constrained mapping produced non-finite source times.")
    return mapped


def constrain_window_mapping(query: np.ndarray, warp_target: np.ndarray, warp_source: np.ndarray,
                             segments: list[dict], *, window: tuple[float, float],
                             context: list[tuple[float, float]] | None = None,
                             ) -> tuple[np.ndarray, dict]:
    """Re-time a searched warp so consonants keep their sung length.

    Boundaries are pinned on both timelines and joined by a monotone cubic."""
    warp_target, warp_source = np.asarray(warp_target, dtype=float), np.asarray(warp_source, dtype=float)
    if (warp_target.ndim != 1 or warp_target.shape != warp_source.shape or len(warp_target) < 2
            or not np.isfinite(warp_target).all() or np.any(np.diff(warp_target) <= 0)
            or not np.isfinite(warp_source).all() or np.any(np.diff(warp_source) <= 0)):
        raise ValueError("Constrained windows require matching increasing finite warp knots.")
    start, end = float(window[0]), float(window[1])
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Constrained windows require a finite increasing window.")
    if not segments:
        raise ValueError("Constrained windows require at least one segment.")
    boundaries = np.array([float(item["start"]) for item in segments]
                          + [float(segments[-1]["end"])], dtype=float)
    if np.any(np.diff(boundaries) <= _EPSILON):
        raise ValueError("Constrained windows require increasing segment boundaries.")
    desired = (np.interp(boundaries[1:], warp_source, warp_target)
               - np.interp(boundaries[:-1], warp_source, warp_target))
    duration, info = plan_segment_durations(segments, desired, end - start)
    knot_target = start + np.concatenate(([0.0], np.cumsum(duration)))
    # Pin the outer knots so the constrained window still spans its anchors.
    knot_target[-1] = end
    if np.any(np.diff(knot_target) <= _EPSILON):
        raise ValueError("Constrained window durations collapsed.")
    mapped = monotone_segment_map(query, knot_target.tolist(), boundaries.tolist(), context)
    info.update({
        "window_start": start,
        "window_end": end,
        "mapping": "monotone_cubic_segment_pinned",
        "boundaries": [float(value) for value in boundaries],
        "knot_target": [float(value) for value in knot_target],
        "context_knot_count": int(len(context or [])),
        "rate_crossfade": "shape_preserving_cubic_blends_rate_across_joints",
    })
    return mapped, info


def segments_for_window(student_analysis: dict, source_left: float, source_right: float,
                        regions: list[dict], left_index: int, right_index: int,
                        mel: np.ndarray | None = None, step: float = 512 / 44100) -> list[dict]:
    """Partition a source window into phones and the rests between them.

    An empty result means no phoneme evidence, so the caller keeps the warp."""
    if source_right <= source_left + _EPSILON:
        return []
    phones: list[list[float | str]] = []
    for word in student_analysis.get("words") or []:
        for phone in word.get("phonemes") or []:
            if not isinstance(phone, dict):
                continue
            start = max(float(phone.get("start", 0.0)), source_left)
            end = min(float(phone.get("end", 0.0)), source_right)
            if end - start > _EPSILON:
                phones.append([start, end, str(phone.get("text", phone.get("phoneme", "")))])
    phones.sort(key=lambda item: (item[0], item[1]))
    clean: list[list[float | str]] = []
    for start, end, label in phones:
        if clean and start < clean[-1][1] - _EPSILON:
            start = clean[-1][1]
            if end - start <= _EPSILON:
                continue
        clean.append([start, end, label])
    if not any(classify_phone(str(label)) != "span" for _, _, label in clean):
        return []

    owners = [(int(left_index), regions[left_index]), (int(right_index), regions[right_index])]
    segments: list[dict] = []
    cursor = float(source_left)
    for start, end, label in clean:
        start, end = float(start), float(end)
        if start > cursor + _EPSILON:
            segments.append(_segment(cursor, start, "span", "<rest>", None, owners))
        segments.append(_segment(start, end, classify_phone(str(label)), str(label), None, owners))
        cursor = end
    if cursor < source_right - _EPSILON:
        segments.append(_segment(cursor, source_right, "span", "<rest>", None, owners))
    _mark_spans(segments, mel, step)
    return segments


def _segment(start: float, end: float, kind: str, label: str,
             _native: float | None, owners: list[tuple[int, dict]]) -> dict:
    native = float(end - start)
    owner_index = None
    for index, region in owners:
        anchors_s = np.asarray(region.get("student_anchors", [region.get("student_start"), region.get("student_end")]), dtype=float)
        anchors_t = np.asarray(region.get("teacher_anchors", [region.get("teacher_start"), region.get("teacher_end")]), dtype=float)
        if (len(anchors_s) < 2 or len(anchors_s) != len(anchors_t)
                or not np.isfinite(anchors_s).all() or not np.isfinite(anchors_t).all()
                or np.any(np.diff(anchors_s) <= 0) or np.any(np.diff(anchors_t) <= 0)):
            continue
        if anchors_s[0] - 1e-6 <= start and end <= anchors_s[-1] + 1e-6:
            native = float(np.interp(end, anchors_s, anchors_t) - np.interp(start, anchors_s, anchors_t))
            owner_index = index
            break
    return {"start": float(start), "end": float(end), "kind": kind, "label": label,
            "native": max(0.0, native), "region": owner_index}


def _span_level(segment: dict, mel: np.ndarray, step: float) -> float:
    centers = (np.arange(mel.shape[1]) + 0.5) * step
    selected = np.flatnonzero((centers >= segment["start"]) & (centers < segment["end"]))
    if not len(selected):
        return float("nan")
    frame = np.asarray(mel, dtype=float)[:, selected]
    return float(np.mean(logsumexp(frame, axis=0) - math.log(frame.shape[0])))


def _mark_spans(segments: list[dict], mel: np.ndarray | None, step: float) -> None:
    for segment in segments:
        segment.setdefault("energy", 1.0)
    if mel is None or not getattr(mel, "shape", None) or mel.shape[1] == 0:
        for segment in segments:
            if segment["kind"] == "span":
                segment["kind"] = "quiet_span"
        return
    levels = {id(segment): _span_level(segment, mel, step) for segment in segments}
    phone_levels = [levels[id(segment)] for segment in segments
                    if segment["kind"] != "span" and math.isfinite(levels[id(segment)])]
    reference = float(np.percentile(phone_levels, 75)) if phone_levels else None
    for segment in segments:
        level = levels[id(segment)]
        segment["energy"] = float(level) if math.isfinite(level) else 1.0
        if segment["kind"] != "span":
            continue
        audible = (reference is not None and math.isfinite(level)
                   and level > reference - _SPAN_AUDIBLE_DROP_LOG)
        segment["audible"] = bool(audible)
        if not audible:
            segment["kind"] = "quiet_span"
