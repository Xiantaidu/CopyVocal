from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math
import re
import unicodedata
from typing import Any

import numpy as np


_HAN = re.compile(r"[\u3400-\u9fff]")
_KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa\u31f0-\u31ff]")
_TOKENS = re.compile(r"[\u3400-\u9fff]|[a-z0-9]+(?:'[a-z]+)?|[^\W_]", re.UNICODE)
_JAPANESE_TOKENS = re.compile(
    r"[\u3041-\u3096\u3099-\u309f\u30a1-\u30fa\u30fc-\u30ff\u31f0-\u31ff\u3400-\u9fff々〆]+|[a-z0-9]+(?:'[a-z]+)?",
    re.UNICODE,
)
_SILENCE = {"", "sp", "ap", "sil", "pau", "br", "<blank>", "<eps>"}
_INITIALS = {"b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
             "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "y", "w", "ng"}
_ENGLISH_VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey",
                   "ih", "iy", "ow", "oy", "uh", "uw", "ax", "ix", "ux"}
# Ratio limits are inclusive; without a tolerance, 0.2/0.8 evaluates a few ULPs
# below 0.25 and a legal 4x boundary would be reported as an extreme stretch.
_STRETCH_EPSILON = 1e-9


def _ratio_within_stretch(ratio: float, maximum: float) -> bool:
    return (ratio <= maximum * (1.0 + _STRETCH_EPSILON)
            and ratio >= (1.0 / maximum) * (1.0 - _STRETCH_EPSILON))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _duration(analysis: dict) -> float:
    duration = _finite(analysis.get("duration"), -1.0)
    if duration < 0:
        rate = _finite(analysis.get("sample_rate"))
        duration = _finite(analysis.get("sample_count")) / rate if rate > 0 else 0.0
    return max(0.0, duration)


def _normalize(text: Any) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold().strip()


def _phone(text: Any) -> str:
    return re.sub(r"[0-9ˈˌ ]", "", _normalize(text)).replace("ü", "v")


def _is_vowel(label: str) -> bool:
    phone = _phone(label)
    if phone in _SILENCE or phone in _INITIALS:
        return False
    return phone in _ENGLISH_VOWELS or bool(re.search(r"[aeiouvɑɐɒæəɛɜɞɪɔœøɵʊɯɤʌɨʉɚɝ]", phone))


@lru_cache(maxsize=4096)
def _pinyin(text: str) -> str:
    """Use the installed lightweight Chinese reader, without loading TiFA."""
    if not _HAN.search(text):
        return ""
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return ""
    result = lazy_pinyin(text, style=Style.NORMAL, errors="ignore")
    return " ".join(result).replace("ü", "v")


def _analysis_language(analysis: dict) -> str:
    config = analysis.get("config") or {}
    value = _normalize(analysis.get("language") or config.get("language"))
    if value in {"ja", "jpn", "japanese"}:
        return "ja"
    if value:
        return value
    if any(_KANA.search(str(word.get("text", "")))
           for word in analysis.get("words") or [] if isinstance(word, dict)):
        return "ja"
    return ""


@lru_cache(maxsize=4096)
def _japanese_signature(text: str, phone_input: bool = False) -> str:
    """Canonicalize a known Japanese reading without inventing kanji readings.

    Moraic N and closure stay distinct from an onset n."""
    from inference.HubertFA.tools.g2p import JapanesePhonemeMoraG2P
    from inference.TiFA.g2p.converters.japanese import _KANA_TO_ROMAJI, _kata_to_hira
    from inference.TiFA.g2p.converters.text import split_words

    value = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not value or _HAN.search(value):
        return ""
    if _KANA.search(value) or "ー" in value:
        moras, last_vowel = [], ""
        for token in split_words(value):
            if token == "ー":
                if not last_vowel:
                    return ""
                mora = last_vowel
            else:
                mora = _KANA_TO_ROMAJI.get(_kata_to_hira(token))
                if mora is None:
                    return ""
            moras.append(mora)
            last_vowel = mora[-1] if mora[-1:] in "aeiou" else ""
        tokens = moras
        phone_input = False
    else:
        tokens = value.split()
    phones = []
    for token in tokens:
        token = token.rsplit("/", 1)[-1]
        if _normalize(token) in _SILENCE or _normalize(token) in {"ep", "gs"}:
            continue
        if phone_input:
            parts = [token if token == "N" else token.lower()]
        else:
            parts = JapanesePhonemeMoraG2P._split_mora_token(token.lower())
            if not parts:
                if token.lower() == "cl":
                    parts = ["cl"]
                else:
                    return ""
        for part in parts:
            # The aligner accepts both palatal onset spellings (ky / k y).
            if len(part) > 1 and part.endswith("y") and part != "cl":
                phones.extend((part[:-1], "y"))
            else:
                phones.append(part)
    return " ".join(phones)


def _japanese_pronunciation(word: dict, token: str, phones: list[dict], estimated: bool) -> str:
    if phones:
        return _japanese_signature(" ".join(p["text"] for p in phones), True)
    if not estimated:
        for key in ("romaji", "pronunciation", "reading", "kana"):
            if word.get(key):
                signature = _japanese_signature(str(word[key]))
                if signature:
                    return signature
    return _japanese_signature(token)


def _clean_phones(word: dict, start: float, end: float) -> list[dict]:
    phones = []
    previous = start
    for phone in word.get("phonemes") or []:
        if not isinstance(phone, dict):
            continue
        a = min(end, max(previous, _finite(phone.get("start"), start)))
        b = min(end, max(a, _finite(phone.get("end"), a)))
        if b > a:
            phones.append({"text": str(phone.get("text", phone.get("phoneme", ""))),
                           "start": a, "end": b})
            previous = b
    return phones


def _units(analysis: dict, duration: float) -> tuple[list[dict], list[str]]:
    result, warnings = [], []
    previous_start = 0.0
    analysis_language = _analysis_language(analysis)
    for index, word in enumerate(analysis.get("words") or []):
        if not isinstance(word, dict):
            continue
        start = np.clip(_finite(word.get("start")), 0.0, duration)
        end = np.clip(_finite(word.get("end"), start), start, duration)
        language = _normalize(word.get("language") or analysis_language)
        japanese = language in {"ja", "jpn", "japanese"}
        tokenizer = _JAPANESE_TOKENS if japanese else _TOKENS
        # Japanese word/mora boundaries come from TiFA; splitting kanji into
        # characters would discard that alignment.
        tokens = tokenizer.findall(_normalize(word.get("text", "")))
        if not tokens or end <= start:
            continue
        if start < previous_start:
            warnings.append("Word order was normalized by time; inspect overlapping ASR timestamps.")
        previous_start = start
        quality = np.clip(_finite(word.get("quality", word.get("confidence", 0.85)), 0.0), 0.0, 1.0)
        phones = _clean_phones(word, float(start), float(end))
        estimated = len(tokens) > 1
        # An aligner can supply exactly one phone per token (typical for Japanese
        # mora units). Those phone times are real evidence, so use them instead
        # of the equal split that made them unusable.
        phone_aligned = bool(estimated and phones and len(phones) == len(tokens))
        for position, token in enumerate(tokens):
            if phone_aligned:
                a = float(start) if position == 0 else phones[position]["start"]
                b = float(end) if position == len(tokens) - 1 else phones[position + 1]["start"]
                local_phones = [phones[position]]
            else:
                # A multi-character ASR word lacks internal boundaries; equal
                # spans are an estimate, not a forced alignment.
                a = float(start + (end - start) * position / len(tokens))
                b = float(start + (end - start) * (position + 1) / len(tokens))
                local_phones = phones if not estimated else []
            unit_estimated = estimated and not phone_aligned
            if japanese:
                pronunciation = _japanese_pronunciation(word, token, local_phones, unit_estimated)
            else:
                # Word-level pinyin describes the whole word, so a multi-token
                # split must derive pronunciation from the phones or the token.
                pronunciation = _phone(word.get("pinyin", word.get("pronunciation", ""))) if not estimated else ""
                if not pronunciation and local_phones:
                    pronunciation = " ".join(_phone(p["text"]) for p in local_phones
                                             if _phone(p["text"]) not in _SILENCE)
                if not pronunciation:
                    pronunciation = _pinyin(token)
            unit = {"text": token, "start": a, "end": b, "index": index,
                           "unit_index": len(result), "quality": float(quality * (0.8 if unit_estimated else 1.0)),
                           "pronunciation": pronunciation, "phonemes": local_phones,
                           "estimated_timing": unit_estimated}
            if japanese:
                unit.update(language="ja", unit_kind=word.get("unit_kind", "word"),
                            identity_basis=word.get("identity_basis", "surface_text"))
                if unit["unit_kind"] == "mora" and "identity_basis" not in word:
                    unit["identity_basis"] = "phonetic_mora"
            result.append(unit)
    result.sort(key=lambda word: (word["start"], word["end"], word["unit_index"]))
    # Do not silently generate a backwards warp from malformed word overlaps.
    previous_end = 0.0
    clean = []
    for word in result:
        if word["start"] < previous_end - 1e-7:
            word["start"] = previous_end
            word["quality"] *= 0.6
            word["phonemes"] = _clean_phones(word, previous_end, word["end"])
            warnings.append("Overlapping word boundaries were clipped and their confidence reduced.")
        if word["end"] > word["start"]:
            clean.append(word)
            previous_end = word["end"]
    return clean, warnings


def _relation(teacher: dict, student: dict) -> str:
    phonetic_only = any(word.get("identity_basis") == "phonetic_mora" for word in (teacher, student))
    if teacher["text"] == student["text"] and not phonetic_only:
        return "matched"
    if teacher["pronunciation"] and teacher["pronunciation"] == student["pronunciation"]:
        return "homophone"
    return "substitution"


_REPEAT_MATERIAL_RATIO = 2.5


def _correspond(teacher: list[dict], student: list[dict], teacher_duration: float,
                student_duration: float) -> tuple[list[tuple[int | None, int | None]], dict]:
    n, m = len(teacher), len(student)
    scores = np.empty((n + 1, m + 1), dtype=np.float64)
    lexical_scores = np.empty_like(scores)
    costs = np.empty((n, m), dtype=np.float64)
    paths = np.zeros((n + 1, m + 1), dtype=np.uint8)
    scores[:, 0] = np.arange(n + 1)
    scores[0, :] = np.arange(m + 1)
    lexical_scores[:, 0] = np.arange(n + 1)
    lexical_scores[0, :] = np.arange(m + 1)
    paths[1:, 0] = 1
    paths[0, 1:] = 2
    for i, tw in enumerate(teacher, 1):
        for j, sw in enumerate(student, 1):
            relation = _relation(tw, sw)
            lexical = {"matched": 0.0, "homophone": 0.16, "substitution": 1.30}[relation]
            progress = 0.08 * abs((i - 0.5) / max(n, 1) - (j - 0.5) / max(m, 1))
            if teacher_duration > 0 and student_duration > 0:
                progress += 0.025 * abs(tw["start"] / teacher_duration - sw["start"] / student_duration)
            # Low confidence makes an uncertain transcription less decisive.
            lexical *= 0.7 + 0.3 * math.sqrt(tw["quality"] * sw["quality"])
            costs[i - 1, j - 1] = lexical
            lexical_scores[i, j] = min(lexical_scores[i - 1, j - 1] + lexical,
                                      lexical_scores[i - 1, j] + 1.0,
                                      lexical_scores[i, j - 1] + 1.0)
            choices = (scores[i - 1, j - 1] + lexical + progress,
                       scores[i - 1, j] + 1.0, scores[i, j - 1] + 1.0)
            action = int(np.argmin(choices))
            scores[i, j], paths[i, j] = choices[action], action
    pairs = []
    i, j = n, m
    while i or j:
        action = paths[i, j]
        if i and j and action == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and (not j or action == 1):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    if not n or not m:
        return pairs, {}
    # The progress prior picks a deterministic path but is not independent
    # evidence of which identical chorus was omitted; forward/backward lexical
    # costs reveal equally plausible identities on either side of that path.
    backwards = np.empty_like(lexical_scores)
    backwards[:, m] = np.arange(n, -1, -1)
    backwards[n, :] = np.arange(m, -1, -1)
    for ti in range(n - 1, -1, -1):
        for si in range(m - 1, -1, -1):
            backwards[ti, si] = min(costs[ti, si] + backwards[ti + 1, si + 1],
                                     1.0 + backwards[ti + 1, si], 1.0 + backwards[ti, si + 1])
    optimum = lexical_scores[n, m]
    ambiguities = {}
    for ti, si in pairs:
        if ti is None or si is None or _relation(teacher[ti], student[si]) == "substitution":
            continue
        teacher_candidates, student_candidates = [], []
        for alternate_t in range(n):
            if _relation(teacher[alternate_t], student[si]) == "substitution":
                continue
            through = lexical_scores[alternate_t, si] + costs[alternate_t, si] + backwards[alternate_t + 1, si + 1]
            if through <= optimum + 0.04:
                teacher_candidates.append(alternate_t)
        for alternate_s in range(m):
            if _relation(teacher[ti], student[alternate_s]) == "substitution":
                continue
            through = lexical_scores[ti, alternate_s] + costs[ti, alternate_s] + backwards[ti + 1, alternate_s + 1]
            if through <= optimum + 0.04:
                student_candidates.append(alternate_s)
        if len(teacher_candidates) > 1 or len(student_candidates) > 1:
            ambiguities[(ti, si)] = {
                "candidate_teacher_unit_indices": teacher_candidates,
                "candidate_student_unit_indices": student_candidates,
                "reason": "multiple_lexically_equivalent_paths_progress_is_only_weak_evidence"}
    return pairs, ambiguities


def _snap_to_contiguous_window(teacher: list[dict], student: list[dict],
                               pairs: list[tuple[int | None, int | None]],
                               ambiguities: dict
                               ) -> tuple[list[tuple[int | None, int | None]], dict]:
    """Re-align student units onto one uninterrupted run of teacher units.

    A teacher recording often repeats a chorus while the student sings it once.
    Lexical matching cannot tell the repetitions apart, so the minimum-cost path
    may begin in one chorus and hop to a later one partway through. That path is
    a valid edit script, but it maps a short student span across a large stretch
    of teacher time and turns every skipped repetition into a silent hole in the
    middle of the song.

    This only applies when the teacher carries far more material than the student
    sang. With comparable word counts the edit script's own gaps are real --
    dropped or inserted words -- and rewriting them would hide genuine
    misalignment, so the pairs are returned unchanged.
    """
    n, m = len(teacher), len(student)
    if n < m or m < 2 or n < m * _REPEAT_MATERIAL_RATIO:
        return pairs, ambiguities

    weights = {"matched": 0.0, "homophone": 0.16, "substitution": 1.30}
    student_span = student[-1]["end"] - student[0]["start"]
    best: tuple[float, int, float] | None = None
    for start_index in range(0, n - m + 1):
        lexical = 0.0
        for offset in range(m):
            lexical += weights[_relation(teacher[start_index + offset], student[offset])]
        span = teacher[start_index + m - 1]["end"] - teacher[start_index]["start"]
        ratio = span / student_span if student_span > 0 else 1.0
        # A performance can be retimed, but not by an order of magnitude.
        penalty = 0.0 if 0.5 <= ratio <= 2.0 else abs(ratio - 1.0)
        total = lexical + penalty
        if best is None or total < best[0]:
            best = (total, start_index, ratio)

    if best is None:
        return pairs, ambiguities

    _, start_index, _ = best
    # The positional score above only chooses the window. Inside it, run the
    # real edit script so insertions and deletions stay insertions/deletions
    # instead of being force-matched by position. Time priors are disabled:
    # the whole point of the snap is that global progress is not evidence.
    local_pairs, local_ambiguities = _correspond(
        teacher[start_index:start_index + m], student, 0.0, 0.0)
    result: list[tuple[int | None, int | None]] = [(index, None) for index in range(start_index)]
    result.extend(((start_index + ti) if ti is not None else None, si)
                  for ti, si in local_pairs)
    result.extend((index, None) for index in range(start_index + m, n))
    remapped = {}
    for (ti, si), info in local_ambiguities.items():
        if ti is None or si is None:
            continue
        item = dict(info)
        item["candidate_teacher_unit_indices"] = [
            start_index + value for value in info.get("candidate_teacher_unit_indices", [])]
        remapped[(start_index + ti, si)] = item
    return result, remapped


def _local_anchors(teacher: dict, student: dict) -> tuple[list[float], list[float], str]:
    """Keep source consonants close to natural length; allocate stretch to vowels."""
    ts, te = teacher["start"], teacher["end"]
    ss, se = student["start"], student["end"]
    phones = student["phonemes"]
    if not phones or not any(_is_vowel(p["text"]) for p in phones):
        return [ts, te], [ss, se], "word_linear"
    boundaries = sorted({ss, se, *(max(ss, min(se, p["start"])) for p in phones),
                         *(max(ss, min(se, p["end"])) for p in phones)})
    if len(boundaries) < 3:
        return [ts, te], [ss, se], "word_linear"
    source_lengths = np.diff(boundaries)
    labels = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        middle = (left + right) / 2
        label = next((p["text"] for p in phones if p["start"] <= middle < p["end"]), "SP")
        labels.append(label)
    vowels = np.array([_is_vowel(label) for label in labels], dtype=bool)
    if not vowels.any():
        return [ts, te], [ss, se], "word_linear"
    target_length = te - ts
    source_consonants = float(source_lengths[~vowels].sum())
    # Compression can force consonants shorter; reserve at least 35% of the
    # target syllable for voiced nuclei to avoid collapsed durations.
    consonant_scale = min(1.0, 0.65 * target_length / max(source_consonants, 1e-9))
    target_lengths = source_lengths * consonant_scale
    remaining = target_length - float(target_lengths[~vowels].sum())
    vowel_weights = source_lengths[vowels].copy()
    teacher_vowels = [p for p in teacher["phonemes"] if _is_vowel(p["text"])]
    source_vowels = [p for p in phones if _is_vowel(p["text"])]
    # Matched vowel identities let the teacher supply relative rhythm, including
    # syllables with several nuclei.
    if (len(teacher_vowels) == len(source_vowels)
            and [_phone(p["text"]) for p in teacher_vowels]
            == [_phone(p["text"]) for p in source_vowels]):
        for interval, is_vowel in enumerate(vowels):
            if is_vowel:
                midpoint = (boundaries[interval] + boundaries[interval + 1]) / 2
                for source_phone, teacher_phone in zip(source_vowels, teacher_vowels):
                    if source_phone["start"] <= midpoint < source_phone["end"]:
                        vowel_weights[np.count_nonzero(vowels[:interval])] = (
                            (teacher_phone["end"] - teacher_phone["start"])
                            * source_lengths[interval] / (source_phone["end"] - source_phone["start"]))
                        break
    target_lengths[vowels] = remaining * vowel_weights / max(float(vowel_weights.sum()), 1e-9)
    target_boundaries = np.concatenate(([ts], ts + np.cumsum(target_lengths)))
    target_boundaries[-1] = te
    return target_boundaries.tolist(), [float(x) for x in boundaries], "vowel_weighted"


def _region(ts: float, te: float, ss: float, se: float, status: str, mode: str,
            quality: float, unsupported: bool = False, **kwargs: Any) -> dict:
    return {"teacher_start": float(ts), "teacher_end": float(te),
            "student_start": float(ss), "student_end": float(se), "status": status,
            "mode": mode, "quality": float(quality), "unsupported": bool(unsupported),
            "teacher_anchors": [float(ts), float(te)],
            "student_anchors": [float(ss), float(se)],
            "stretch_ratio": float((te - ts) / (se - ss)) if se > ss else None,
            **kwargs}


def _voice_extent(analysis: dict, duration: float) -> tuple[float, float] | None:
    """Activity-only fallback; never inspect F0 values to establish identity."""
    pitch = analysis.get("pitch") or {}
    times = np.asarray(pitch.get("times", []), dtype=float)
    voiced = np.asarray(pitch.get("voiced", []), dtype=bool)
    if times.ndim != 1 or voiced.ndim != 1 or times.size != voiced.size or not times.size:
        return None
    active = times[voiced & np.isfinite(times)]
    if not active.size:
        return None
    step = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    return max(0.0, float(active.min())), min(duration, float(active.max()) + max(0.0, step))


def _voice_in_interval(analysis: dict, start: float, end: float) -> tuple[float, float] | None:
    """Return the bounding extent of voiced frames in an interval.

    Kept for whole-interval checks; gap planning uses ``_voice_runs`` instead."""
    runs = _voice_runs(analysis, start, end)
    if not runs:
        return None
    return runs[0][0], runs[-1][1]


def _voice_runs(analysis: dict, start: float, end: float) -> list[tuple[float, float]]:
    pitch = analysis.get("pitch") or {}
    times = np.asarray(pitch.get("times", []), dtype=float)
    voiced = np.asarray(pitch.get("voiced", []), dtype=bool)
    if times.ndim != 1 or voiced.ndim != 1 or times.size != voiced.size or not times.size:
        return []
    selected = voiced & np.isfinite(times) & (times >= start) & (times < end)
    active = times[selected]
    if not active.size:
        return []
    step = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    step = max(1e-4, step)
    # A missing voiced frame separates runs; UV alone is not acoustic silence.
    split = np.flatnonzero(np.diff(active) > step * 1.5)
    starts = np.r_[0, split + 1]
    stops = np.r_[split, active.size - 1]
    return [(float(active[a]), min(end, float(active[b]) + step))
            for a, b in zip(starts, stops) if active[b] >= active[a]]


def _gap_regions(teacher: dict, student: dict, ts: float, te: float, ss: float,
                 se: float, allow_fallback: bool, max_stretch: float) -> list[dict]:
    """Classify pitch-only gaps without asserting acoustic silence."""
    def unvoiced_region(start: float, end: float, source_cursor: float) -> dict:
        # Legacy rendering modes and anchors are unchanged; the score describes
        # pitch-gap structure, not acoustic silence or semantic confidence.
        return _region(start, end, source_cursor, source_cursor,
                       "unvoiced_gap", "silence", 1.0,
                       reason="pitch_unvoiced_not_acoustic_silence",
                       acoustic_classification="unknown", silence_confirmed=False,
                       gap_source_start=float(ss), gap_source_end=float(se),
                       quality_definition="Legacy pitch-gap structural score; not acoustic silence or semantic confidence.")

    teacher_runs = _voice_runs(teacher, ts, te)
    if not teacher_runs:
        return [unvoiced_region(ts, te, ss)]
    student_runs = _voice_runs(student, ss, se) if se > ss else []
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    if allow_fallback and len(student_runs) == len(teacher_runs):
        candidates = list(zip(teacher_runs, student_runs))
        # A single run on each side supplies an activity extent, not semantic
        # evidence; multiple runs need compatible relative timing or an omitted
        # sound could shift every subsequent pairing.
        def compatible(pair: tuple[tuple[float, float], tuple[float, float]]) -> bool:
            (a, b), (c, d) = pair
            if d <= c or not _ratio_within_stretch((b - a) / (d - c), max_stretch):
                return False
            if len(candidates) == 1:
                return True
            midpoint = ss + (((a + b) / 2 - ts) / (te - ts)) * (se - ss)
            return c <= midpoint < d

        if all(compatible(pair) for pair in candidates):
            pairs = candidates

    result = []
    teacher_cursor, student_cursor = ts, ss
    for index, (a, b) in enumerate(teacher_runs):
        if a > teacher_cursor:
            result.append(unvoiced_region(teacher_cursor, a, student_cursor))
        if pairs:
            c, d = pairs[index][1]
            result.append(_region(a, b, c, d, "unresolved", "acoustic_fallback", 0.15, False,
                                  reason="untranscribed_activity_no_lyric_correspondence",
                                  anchor_strategy="activity_linear", extreme_stretch=False,
                                  pitch_source="teacher"))
            student_cursor = d
        else:
            result.append(_region(a, b, student_cursor, student_cursor, "unresolved", "mute", 0.0, True,
                                  reason="voiced_teacher_region_without_lyric_correspondence"))
        teacher_cursor = b
    if teacher_cursor < te:
        result.append(unvoiced_region(teacher_cursor, te, student_cursor))
    return result


def _global_map(regions: list[dict], teacher_duration: float, student_duration: float) -> dict:
    pairs = [(0.0, 0.0)]
    for region in regions:
        if region["mode"] in {"synthesize", "acoustic_fallback"}:
            pairs.extend(zip(region["teacher_anchors"], region["student_anchors"]))
    pairs.append((teacher_duration, student_duration))
    pairs.sort(key=lambda pair: pair[0])
    teacher_times, student_times = [], []
    for target, source in pairs:
        target, source = float(target), float(source)
        if teacher_times and target == teacher_times[-1]:
            if source > student_times[-1] and len(teacher_times) > 1:
                # A source insertion is a jump; move its left limit by one ULP
                # so the sparse map stays monotonic without a broad smear.
                left_limit = float(np.nextafter(target, -math.inf))
                if left_limit > teacher_times[-2]:
                    teacher_times[-1] = left_limit
                    teacher_times.append(target)
                    student_times.append(max(student_times[-1], source))
                    continue
            student_times[-1] = max(student_times[-1], source)
            continue
        teacher_times.append(target)
        student_times.append(max(student_times[-1] if student_times else 0.0, source))
    return {"teacher_times": teacher_times, "student_times": student_times,
            "direction": "teacher_to_student", "requires_region_mask": True}


def build_alignment(teacher: dict, student: dict, *, min_quality: float = 0.35,
                    max_stretch: float = 4.0, allow_acoustic_fallback: bool = False) -> dict:
    """Plan teacher-clock editing and report unsupported material explicitly.

    Substitutions are suspected lyric/ASR mismatches; missing words are muted."""
    if not 0 <= min_quality <= 1 or not math.isfinite(max_stretch) or max_stretch < 1:
        raise ValueError("min_quality must be in [0, 1] and max_stretch must be at least 1.")
    td, sd = _duration(teacher), _duration(student)
    tw, teacher_warnings = _units(teacher, td)
    sw, student_warnings = _units(student, sd)
    teacher_voice, student_voice = _voice_extent(teacher, td), _voice_extent(student, sd)
    warnings = teacher_warnings + student_warnings
    matches, active_regions = [], []
    pairs, ambiguities = _correspond(tw, sw, td, sd)
    # The edit script alone can split a one-pass student performance across
    # several repeats of a teacher chorus; snap it onto a single run so the
    # skipped repetitions stay whole instead of fragmenting the timeline.
    pairs, ambiguities = _snap_to_contiguous_window(tw, sw, pairs, ambiguities)
    for ti, si in pairs:
        t, s = (tw[ti] if ti is not None else None), (sw[si] if si is not None else None)
        match = {"teacher_index": t["index"] if t else None,
                 "student_index": s["index"] if s else None,
                 "teacher_unit_index": ti, "student_unit_index": si,
                 "teacher_text": t["text"] if t else None,
                 "student_text": s["text"] if s else None}
        identity = {}
        if any(unit and unit.get("language") == "ja" for unit in (t, s)):
            identity = {"teacher_identity_basis": t.get("identity_basis") if t else None,
                        "student_identity_basis": s.get("identity_basis") if s else None,
                        "phonetic_only": any(unit and unit.get("identity_basis") == "phonetic_mora"
                                             for unit in (t, s))}
            match.update(identity)
        if t and s:
            relation = _relation(t, s)
            quality = math.sqrt(t["quality"] * s["quality"])
            quality *= {"matched": 1.0, "homophone": 0.85, "substitution": 0.48}[relation]
            ambiguity = ambiguities.get((ti, si))
            if ambiguity:
                quality = min(quality, 0.25)
            status = relation if quality >= min_quality else "unresolved"
            if ambiguity:
                status = "unresolved"
            anchors_t, anchors_s, strategy = _local_anchors(t, s)
            ratio = (t["end"] - t["start"]) / (s["end"] - s["start"])
            local_ratios = np.diff(anchors_t) / np.diff(anchors_s)
            lower_limit, upper_limit = (1 / max_stretch) * (1 - _STRETCH_EPSILON), max_stretch * (1 + _STRETCH_EPSILON)
            local_extreme = bool(np.any((local_ratios > upper_limit) | (local_ratios < lower_limit)))
            extreme = not _ratio_within_stretch(ratio, max_stretch)
            unsupported = status == "unresolved" or extreme
            mode = "mute" if unsupported else "synthesize"
            if ambiguity and allow_acoustic_fallback and not extreme:
                mode, unsupported = "acoustic_fallback", False
            reason = "possible_lyric_mismatch_or_asr_error" if relation == "substitution" else relation
            if ambiguity:
                reason = "ambiguous_repeated_lyrics"
            match.update(status=status, relation=relation, quality=float(quality), reason=reason,
                         ambiguous=bool(ambiguity), ambiguity=ambiguity,
                         timing_estimated=t["estimated_timing"] or s["estimated_timing"])
            active_regions.append(_region(t["start"], t["end"], s["start"], s["end"],
                                          status, mode, quality, unsupported,
                                          teacher_anchors=anchors_t, student_anchors=anchors_s,
                                          anchor_strategy=strategy, teacher_index=t["index"],
                                          student_index=s["index"], teacher_text=t["text"],
                                          student_text=s["text"], reason=reason,
                                          ambiguous=bool(ambiguity), ambiguity=ambiguity,
                                          extreme_stretch=extreme, extreme_local_stretch=local_extreme,
                                          minimum_local_stretch=float(local_ratios.min()),
                                          maximum_local_stretch=float(local_ratios.max()),
                                          pitch_source="teacher", **identity))
            if relation == "substitution":
                warnings.append(f"Possible lyric mismatch or ASR error: {t['text']!r} versus {s['text']!r}.")
            if extreme:
                warnings.append(f"Unsupported stretch ratio {ratio:.2f} for teacher word {t['text']!r}.")
            elif local_extreme:
                warnings.append(f"Extreme local phoneme stretch in teacher word {t['text']!r}; inspect audio quality.")
        elif t:
            asr_unknown = not sw and student_voice is not None
            status = "unresolved" if asr_unknown else "missing"
            reason = "student_lyrics_unavailable" if asr_unknown else "no_corresponding_student_lyric_unit"
            match.update(status=status, quality=0.0, reason=reason)
            active_regions.append(_region(t["start"], t["end"], 0.0, 0.0,
                                          status, "mute", 0.0, True,
                                          teacher_index=t["index"], student_index=None,
                                          teacher_text=t["text"], student_text=None,
                                          reason=reason, **identity))
        else:
            asr_unknown = not tw and teacher_voice is not None
            match.update(status="unresolved" if asr_unknown else "extra", quality=0.0,
                         reason="teacher_lyrics_unavailable" if asr_unknown else "discard_unmatched_student_material")
        matches.append(match)
    # If no teacher words are available, silence and failed recognition must be
    # distinguishable. Only observed voicing permits an optional acoustic warp.
    if (not tw or not sw) and allow_acoustic_fallback:
        if teacher_voice and student_voice and teacher_voice[1] > teacher_voice[0] and student_voice[1] > student_voice[0]:
            active_regions = _gap_regions(teacher, student, 0.0, td, 0.0, sd, True, max_stretch)
            unavailable = "teacher and student" if not tw and not sw else "teacher" if not tw else "student"
            warnings.append(f"ASR did not provide {unavailable} lyrics. Activity-only timing has no semantic validation.")
    regions, cursor, source_cursor = [], 0.0, 0.0
    for region in sorted(active_regions, key=lambda item: item["teacher_start"]):
        if region["teacher_start"] > cursor:
            source_gap_end = region["student_start"] if region["mode"] in {"synthesize", "acoustic_fallback"} else source_cursor
            regions.extend(_gap_regions(teacher, student, cursor, region["teacher_start"],
                                        source_cursor, source_gap_end, allow_acoustic_fallback, max_stretch))
        regions.append(region)
        cursor = region["teacher_end"]
        if region["mode"] in {"synthesize", "acoustic_fallback"}:
            source_cursor = region["student_end"]
        elif float(region.get("student_end", 0.0)) > float(region.get("student_start", 0.0)):
            # A matched word that is muted for quality still consumes its student
            # span; the next gap must not reuse that audio as a fallback.
            source_cursor = max(source_cursor, float(region["student_end"]))
    if td > cursor:
        regions.extend(_gap_regions(teacher, student, cursor, td, source_cursor, sd,
                                    allow_acoustic_fallback, max_stretch))
    if any(r.get("reason") == "untranscribed_activity_no_lyric_correspondence" for r in regions):
        warnings.append("Untranscribed voiced gaps have activity-only timing with no semantic validation.")
    if any(r.get("reason") == "voiced_teacher_region_without_lyric_correspondence" for r in regions):
        warnings.append("Untranscribed teacher voicing lacks corresponding student material and is muted.")
    if not tw:
        warnings.append("No timed teacher lyric units were available.")
    if not sw:
        warnings.append("No timed student lyric units were available.")
    if ambiguities:
        warnings.append(f"Ambiguous repeated lyrics affect {len(ambiguities)} matches; progress alone cannot identify the omitted repetition.")
    if any(word["estimated_timing"] for word in tw + sw):
        warnings.append("Multi-unit ASR words use estimated internal boundaries; re-align them for precise editing.")
    statuses = Counter(match["status"] for match in matches)
    phonetic_only_match_count = sum(match.get("phonetic_only", False)
                                    and match.get("relation") == "homophone" for match in matches)
    if phonetic_only_match_count:
        warnings.append("Japanese mora correspondence is phonetic only; matching readings do not establish lexical identity or exclude homophone lyric errors.")
    supported_duration = sum(r["teacher_end"] - r["teacher_start"] for r in regions if r["mode"] == "synthesize")
    lyric_duration = sum(w["end"] - w["start"] for w in tw)
    muted_duration = sum(r["teacher_end"] - r["teacher_start"] for r in regions
                         if r["mode"] == "mute" and r.get("teacher_index") is not None)
    if lyric_duration and muted_duration >= lyric_duration - 1e-8:
        warnings.append("All timed teacher lyric material is muted because correspondence is unsupported; this is not a successful correction.")
    ratios = [r["stretch_ratio"] for r in regions if r["mode"] in {"synthesize", "acoustic_fallback"}]
    metrics = {"teacher_unit_count": len(tw), "student_unit_count": len(sw),
               "status_counts": dict(statuses), "supported_duration": float(supported_duration),
               "teacher_lyric_duration": float(lyric_duration),
               "lyric_coverage": float(supported_duration / lyric_duration) if lyric_duration else 0.0,
               "semantic_match_rate": float(sum(match["status"] in {"matched", "homophone"}
                                                 and not match.get("phonetic_only", False)
                                                 for match in matches) / len(tw)) if tw else 0.0,
               "phonetic_only_match_count": phonetic_only_match_count,
               "unsupported_region_count": sum(r["unsupported"] for r in regions),
               "extreme_stretch_count": sum(bool(r.get("extreme_stretch")) for r in regions),
               "extreme_local_stretch_count": sum(bool(r.get("extreme_local_stretch")) for r in regions),
               "ambiguous_match_count": len(ambiguities),
               "muted_lyric_duration": float(muted_duration),
               "muted_lyric_ratio": float(muted_duration / lyric_duration) if lyric_duration else 0.0,
               "all_lyric_material_muted": bool(lyric_duration and muted_duration >= lyric_duration - 1e-8),
               "minimum_stretch": min(ratios) if ratios else None,
               "maximum_stretch": max(ratios) if ratios else None,
               "acoustic_fallback": any(r["mode"] == "acoustic_fallback" for r in regions)}
    return {"version": 1, "teacher_duration": td, "student_duration": sd,
            "matches": matches, "regions": regions,
            "time_map": _global_map(regions, td, sd), "metrics": metrics,
            "warnings": list(dict.fromkeys(warnings)),
            "policy": {"missing_material": "mute", "substitution": "suspected_lyric_or_asr_mismatch",
                       "min_quality": min_quality, "max_stretch": max_stretch}}


def map_times(plan: dict, times: Any) -> np.ndarray:
    """Map teacher seconds to source seconds; region masks remain mandatory.

    Unsupported regions get a bounded map for diagnostics only, never synthesis."""
    values = np.asarray(times, dtype=np.float64)
    mapping = plan.get("time_map") or {}
    target = np.asarray(mapping.get("teacher_times", []), dtype=float)
    source = np.asarray(mapping.get("student_times", []), dtype=float)
    result = np.interp(values, target, source) if target.size else np.zeros_like(values)
    for region in plan.get("regions", []):
        if region.get("mode") not in {"synthesize", "acoustic_fallback"}:
            continue
        selected = (values >= region["teacher_start"]) & (values < region["teacher_end"])
        local = np.interp(values, region["teacher_anchors"], region["student_anchors"])
        result = np.where(selected, local, result)
    return np.clip(result, 0.0, float(plan.get("student_duration", 0.0)))
