"""Bounded acoustic context outside semantically supported lyric regions."""

from __future__ import annotations

import math


_EPSILON = 1e-9
_ATTACK_SECONDS = 0.020
_RELEASE_SECONDS = 0.060
_HELD_RELEASE_SECONDS = 0.040
_HALF_HOP_SECONDS = 256 / 44100
_GAP_REASON = "voiced_teacher_region_without_lyric_correspondence"


def _allowed(region: dict, allow_fallback: bool, minimum_quality: float) -> bool:
    """Match the renderer policy without importing the renderer."""
    mode = region.get("mode", "synthesize")
    if mode == "acoustic_fallback":
        return bool(allow_fallback) and float(region.get("quality", 0)) >= minimum_quality
    return mode == "synthesize" and not region.get("unsupported", False) and float(region.get("quality", 1)) >= minimum_quality


def _safe_gap(region: dict) -> bool:
    return region.get("mode") == "silence" or (
        region.get("mode") == "mute" and region.get("reason") == _GAP_REASON
    )


def _span(item: dict, prefix: str, duration: float) -> tuple[float, float]:
    start, end = float(item[prefix + "start"]), float(item[prefix + "end"])
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError("Boundary intervals must have finite, ordered endpoints.")
    return max(0.0, min(duration, start)), max(0.0, min(duration, end))


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + _EPSILON:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _safe_spans(regions: list[dict], teacher_duration: float, allow_fallback: bool,
                minimum_quality: float) -> list[tuple[float, float]]:
    safe, barriers = [], []
    for region in regions:
        span = _span(region, "teacher_", teacher_duration)
        if not _allowed(region, allow_fallback, minimum_quality):
            (safe if _safe_gap(region) else barriers).append(span)
    # A lexical rejection wins even if a coarse silence annotation overlaps it.
    result = _merge(safe)
    for left, right in _merge(barriers):
        clipped = []
        for start, end in result:
            if end <= left or start >= right:
                clipped.append((start, end))
            else:
                if start < left:
                    clipped.append((start, left))
                if end > right:
                    clipped.append((right, end))
        result = clipped
    return result


def _source_blockers(regions: list[dict], words: list[tuple[int, float, float]],
                     region_index: int, student_duration: float) -> list[tuple[float, float]]:
    region = regions[region_index]
    start, end = _span(region, "student_", student_duration)
    own_index = region.get("student_index")
    own_word = next((index for index, left, right in words
                     if index == own_index and min(end, right) > max(start, left)), None)
    if own_word is None:
        # Synthetic fixtures and activity regions may not carry original indices.
        candidates = [(min(end, right) - max(start, left), index)
                      for index, left, right in words if min(end, right) > max(start, left)]
        if candidates:
            own_word = max(candidates)[1]
    blockers = [(left, right) for index, left, right in words if index != own_word]
    for index, other in enumerate(regions):
        if index == region_index or _safe_gap(other):
            continue
        # Include rejected lexical ranges even when original word indices are absent.
        # A missing target word can have no source interval at all.
        if "student_start" not in other or "student_end" not in other:
            continue
        left, right = _span(other, "student_", student_duration)
        if right > left:
            blockers.append((left, right))
    return _merge(blockers)


def _context(kind: str, edge: float, target_limit: float, region_index: int,
             regions: list[dict], words: list[tuple[int, float, float]],
             student_duration: float) -> dict | None:
    region = regions[region_index]
    left, right = _span(region, "student_", student_duration)
    if right <= left:
        return None
    blockers = _source_blockers(regions, words, region_index, student_duration)
    source_edge = right if kind == "release" else left
    source_limit = student_duration if kind == "release" else 0.0
    for start, end in blockers:
        if kind == "release" and end > source_edge + _EPSILON:
            source_limit = min(source_limit, max(source_edge, start))
        elif kind == "attack" and start < source_edge - _EPSILON:
            source_limit = max(source_limit, min(source_edge, end))
    target_length = min(abs(target_limit - edge),
                        _RELEASE_SECONDS if kind == "release" else _ATTACK_SECONDS)
    source_room = abs(source_limit - source_edge)
    strategy = "continuation"
    if source_room <= _EPSILON:
        if kind == "attack":
            return None
        strategy = "hold"
        target_length = min(target_length, _HELD_RELEASE_SECONDS)
        point = right - min(_HALF_HOP_SECONDS, (right - left) / 2)
        if any(start <= point < end for start, end in blockers):
            return None
        source_start = source_end = source_min = source_max = point
    elif kind == "release":
        source_start, source_end = source_edge, min(source_limit, source_edge + target_length)
        source_min, source_max = source_edge, source_limit
    else:
        source_start, source_end = max(source_limit, source_edge - target_length), source_edge
        source_min, source_max = source_limit, source_edge
    if target_length <= _EPSILON:
        return None
    return {
        "kind": kind,
        "teacher_start": edge if kind == "release" else edge - target_length,
        "teacher_end": edge + target_length if kind == "release" else edge,
        "student_start": source_start,
        "student_end": source_end,
        "boundary": edge,
        "policy": "bounded_nonlexical_acoustic_context",
        "source_strategy": strategy,
        "student_limit_start": source_min,
        "student_limit_end": source_max,
        "semantic_support": False,
        "source_region_index": region_index,
        "envelope_strategy": ("full_cos_squared_release" if strategy == "hold" else
                              "tail_sin_squared_release" if kind == "release" else
                              "full_sin_squared_attack"),
    }


def _trim(context: dict, start: float, end: float) -> None:
    old_start, old_end = context["teacher_start"], context["teacher_end"]
    source_start, source_end = context["student_start"], context["student_end"]
    slope = (source_end - source_start) / (old_end - old_start)
    context["student_start"] = source_start + (start - old_start) * slope
    context["student_end"] = source_start + (end - old_start) * slope
    context["teacher_start"], context["teacher_end"] = start, end


def build_boundary_context(regions: list[dict], student_words: list[dict],
                           teacher_duration: float, student_duration: float,
                           allow_fallback: bool = False, minimum_quality: float = 0.0) -> list[dict]:
    """Add acoustic edges without changing lyric support or crossing rejected words.

    Only explicitly classified nonlexical gaps can receive context. Attacks use
    at most 20 ms of preceding source audio. Releases prefer up to 60 ms of
    real continuation; without source room they use at most 40 ms of smoothly
    decaying terminal texture. The renderer must apply the indicated envelope.
    """
    teacher_duration, student_duration = float(teacher_duration), float(student_duration)
    if any(not math.isfinite(value) or value < 0 for value in (teacher_duration, student_duration)):
        raise ValueError("Track durations must be finite and nonnegative.")
    if not math.isfinite(minimum_quality):
        raise ValueError("minimum_quality must be finite.")
    safe = _safe_spans(regions, teacher_duration, allow_fallback, minimum_quality)
    words = []
    for index, word in enumerate(student_words):
        start, end = _span(word, "", student_duration)
        if end > start:
            words.append((index, start, end))
    allowed = []
    for index, region in enumerate(regions):
        if _allowed(region, allow_fallback, minimum_quality):
            start, end = _span(region, "teacher_", teacher_duration)
            if end > start:
                allowed.append((start, end, index))
    islands = []
    for start, end, index in sorted(allowed):
        if islands and start <= islands[-1][1] + _EPSILON:
            if end > islands[-1][1]:
                islands[-1][1], islands[-1][3] = end, index
        else:
            islands.append([start, end, index, index])
    contexts = []
    for island_index, (start, end, first, last) in enumerate(islands):
        previous_end = islands[island_index - 1][1] if island_index else 0.0
        next_start = islands[island_index + 1][0] if island_index + 1 < len(islands) else teacher_duration
        for kind, edge, region_index in (("attack", start, first), ("release", end, last)):
            for safe_start, safe_end in safe:
                if kind == "release" and safe_start <= edge + _EPSILON and safe_end > edge + _EPSILON:
                    limit = min(safe_end, next_start)
                elif kind == "attack" and safe_start < edge - _EPSILON and safe_end >= edge - _EPSILON:
                    limit = max(safe_start, previous_end)
                else:
                    continue
                context = _context(kind, edge, limit, region_index, regions, words, student_duration)
                if context is not None:
                    contexts.append(context)
                break
    contexts.sort(key=lambda context: (context["teacher_start"], context["teacher_end"]))
    for left, right in zip(contexts, contexts[1:]):
        if left["teacher_end"] > right["teacher_start"]:
            # Reserve at most one third of a short gap for the next attack.
            gap = right["boundary"] - left["boundary"]
            attack = min(right["teacher_end"] - right["teacher_start"], gap / 3)
            split = min(left["teacher_end"], max(right["teacher_start"], right["boundary"] - attack))
            _trim(left, left["teacher_start"], split)
            _trim(right, split, right["teacher_end"])
            left["overlap_policy"] = right["overlap_policy"] = "release_priority_split"
    return [context for context in contexts if context["teacher_end"] > context["teacher_start"] + _EPSILON]
