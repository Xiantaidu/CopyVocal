from __future__ import annotations

import argparse
import base64
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from application.artifacts import file_hash, read_json, write_json


BUNDLE_FORMAT = "copyvocal.file"
BUNDLE_VERSION = 1
F0_HOP = 160
F0_SAMPLE_RATE = 16000
F0_STEP = F0_HOP / F0_SAMPLE_RATE
_SAMPLE_RATE = 44100
# CopyVocal's target pitch runs on the vocoder grid (~11.6 ms). Bridge at most
# one missing output frame when resampling it onto the .cvf 10 ms grid.
_TARGET_INTERP_MAX_GAP = 0.030
# One- to three-frame pitch spikes at note seams are below the vocoder's time
# resolution and are not audible in CopyVocal's rendered audio. A short median
# over voiced runs keeps the exported curve and a consumer's render consistent with that audio.
_TARGET_MEDIAN_WINDOW = 5
# A source->output step wider than this means the teacher has material the
# student recording does not cover (intro, instrumental break, a repeated
# chorus). The time grid represents it as a near-vertical step so the gap stays
# silent instead of being interpolated into the student's audio.
_MAX_GAP_JUMP_SECONDS = 0.5


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((float(midi) - 69.0) / 12.0)


def _hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(float(hz), 1e-9) / 440.0)


def _b64_f32(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f4")
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


def _b64_u8(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.uint8)
    return base64.b64encode(array.tobytes(order="C")).decode("ascii")


def _as_float_array(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.size != length:
        result = np.full(length, default, dtype=np.float32)
        count = min(length, array.size)
        if count:
            result[:count] = array[:count]
        return result
    return array.astype(np.float32, copy=False)


def _as_mask(value: Any, length: int, default: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.size != length:
        result = np.full(length, 1 if default else 0, dtype=np.uint8)
        count = min(length, array.size)
        if count:
            result[:count] = array[:count].astype(bool).astype(np.uint8)
        return result
    return array.astype(bool).astype(np.uint8)


def _audio_info(path: Path) -> dict[str, Any]:
    """Read real media metadata; fall back to a zero-filled stub on failure."""
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(path))
        frames = int(info.frames)
        rate = int(info.samplerate)
        return {
            "sampleRate": rate,
            "channels": int(info.channels),
            "sampleCount": frames,
            "durationSeconds": frames / rate if rate else 0.0,
        }
    except Exception:
        pass
    try:
        import wave  # type: ignore

        with wave.open(str(path), "rb") as handle:
            frames = int(handle.getnframes())
            rate = int(handle.getframerate())
            return {
                "sampleRate": rate,
                "channels": int(handle.getnchannels()),
                "sampleCount": frames,
                "durationSeconds": frames / rate if rate else 0.0,
            }
    except Exception:
        return {"sampleRate": _SAMPLE_RATE, "channels": 1, "sampleCount": 0,
                "durationSeconds": 0.0}


def _media_ref(path: Path, bundle_dir: Path, role: str, *, digest: bool) -> dict[str, Any]:
    info = _audio_info(path)
    try:
        relative = Path(path).resolve().relative_to(bundle_dir.resolve())
        rendered_path = relative.as_posix()
    except ValueError:
        rendered_path = str(Path(path).resolve()).replace("\\", "/")
    result: dict[str, Any] = {
        "role": role,
        "path": rendered_path,
        "sampleRate": info["sampleRate"],
        "channels": info["channels"],
        "sampleCount": info["sampleCount"],
        "durationSeconds": info["durationSeconds"],
    }
    if digest:
        result["sha256"] = file_hash(path)
    return result


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def _student_f0_grid(student: dict, frame_count: int) -> np.ndarray:
    pitch = student.get("pitch") or {}
    times = np.asarray(pitch.get("times") or [], dtype=np.float64)
    values = np.asarray(pitch.get("f0_hz") or [], dtype=np.float32)
    result = np.zeros(frame_count, dtype=np.float32)
    if values.size == 0:
        return result
    if times.size == values.size and times.size > 1:
        step = float(times[1] - times[0])
        if abs(step - F0_STEP) <= 1e-4 and times[0] <= 1e-6:
            count = min(frame_count, values.size)
            result[:count] = values[:count]
            return result
        # CopyVocal's RMVPE grid is uniform 10 ms; this branch tolerates a
        # different origin/step rather than silently mis-indexing it.
        for time, value in zip(times, values):
            if value <= 0 or not math.isfinite(float(value)):
                continue
            index = int(round(float(time) / F0_STEP))
            if 0 <= index < frame_count:
                result[index] = max(result[index], float(value))
        return result
    count = min(frame_count, values.size)
    result[:count] = values[:count]
    return result


def _median_smooth_voiced(values: np.ndarray, voiced: np.ndarray,
                          window: int = _TARGET_MEDIAN_WINDOW) -> np.ndarray:
    """Median-smooth voiced runs while keeping unvoiced zero frames intact.
    Removes short seam spikes that CopyVocal's vocoder does not reproduce."""
    result = np.asarray(values, dtype=np.float32).copy()
    if window <= 1:
        return result
    half = window // 2
    total = int(result.size)
    start = 0
    while start < total:
        if not bool(voiced[start]):
            start += 1
            continue
        end = start
        while end < total and bool(voiced[end]):
            end += 1
        run = result[start:end].astype(np.float64)
        for index in range(run.size):
            low = max(0, index - half)
            high = min(run.size, index + half + 1)
            result[start + index] = float(np.median(run[low:high]))
        start = end
    return result


def _target_f0_grid(target: dict[str, np.ndarray] | None, frame_count: int,
                    original: np.ndarray) -> np.ndarray:
    """Resample output-clock target pitch onto the 10 ms source grid.
    Bridges single missing output frames without filling real unvoiced spans."""
    result = np.zeros(frame_count, dtype=np.float32)
    if target is None:
        return result
    times = np.asarray(target.get("times", []), dtype=np.float64)
    f0 = np.asarray(target.get("f0_hz", []), dtype=np.float32)
    if times.size == 0 or f0.size != times.size:
        return result
    source_times = np.asarray(target.get("source_times", times), dtype=np.float64)
    if source_times.size != times.size:
        source_times = times
    rendered = target.get("rendered")
    if rendered is None or np.asarray(rendered).size != times.size:
        rendered = target.get("supported")
    if rendered is None or np.asarray(rendered).size != times.size:
        rendered = np.isfinite(source_times)
    rendered = np.asarray(rendered, dtype=bool)

    valid = rendered & np.isfinite(source_times) & np.isfinite(f0) & (f0 > 0.0)
    if not np.any(valid):
        return result
    f0 = _median_smooth_voiced(f0, valid)
    valid_times = source_times[valid]
    valid_f0 = f0[valid].astype(np.float64)
    order = np.argsort(valid_times, kind="stable")
    valid_times = valid_times[order]
    valid_f0 = valid_f0[order]

    # Time holds map several output frames to one source time; collapse them in
    # the log-frequency domain so averaged values stay musically symmetric.
    unique_times, inverse = np.unique(valid_times, return_inverse=True)
    if unique_times.size != valid_times.size:
        log_sums = np.bincount(inverse, weights=np.log(valid_f0))
        counts = np.bincount(inverse)
        unique_f0 = np.exp(log_sums / np.maximum(counts, 1))
    else:
        unique_f0 = valid_f0

    frame_times = np.arange(frame_count, dtype=np.float64) * F0_STEP
    if unique_times.size == 1:
        near = np.abs(frame_times - unique_times[0]) <= 0.5 * F0_STEP + 1e-9
        result[near] = float(unique_f0[0])
    else:
        right = np.searchsorted(unique_times, frame_times, side="left")
        left = right - 1
        both = (left >= 0) & (right < unique_times.size)
        if np.any(both):
            t0 = unique_times[left[both]]
            t1 = unique_times[right[both]]
            gap = t1 - t0
            weight = np.clip((frame_times[both] - t0) / np.where(gap > 0.0, gap, 1.0), 0.0, 1.0)
            log_value = ((1.0 - weight) * np.log(unique_f0[left[both]])
                         + weight * np.log(unique_f0[right[both]]))
            # An 11.6 ms target grid lands between 10 ms frames; bridging one
            # dropped output frame stays below 30 ms, while real unvoiced spans
            # are far wider and must not be filled with fabricated pitch.
            result[both] = np.where(gap <= _TARGET_INTERP_MAX_GAP,
                                    np.exp(log_value), 0.0)
        # The target grid's frame centres can sit one target step outside the
        # first and last 10 ms frame; extrapolate across that one step so a
        # voiced onset does not lose its first frame, while silence stays empty.
        steps = np.diff(unique_times)
        edge_step = float(np.median(steps)) if steps.size else F0_STEP
        after_last = (left >= 0) & (right >= unique_times.size)
        if np.any(after_last):
            distance = frame_times[after_last] - unique_times[left[after_last]]
            result[after_last] = np.where(distance <= edge_step,
                                          unique_f0[left[after_last]], 0.0)
        before_first = (left < 0) & (right < unique_times.size)
        if np.any(before_first):
            distance = unique_times[right[before_first]] - frame_times[before_first]
            result[before_first] = np.where(distance <= edge_step,
                                            unique_f0[right[before_first]], 0.0)
    # v1 only corrects frames whose original F0 is voiced.
    result[original <= 0.0] = 0.0
    return result


def _map_times(plan: dict, teacher_times: Sequence[float]) -> np.ndarray:
    """Teacher seconds -> student source seconds, mirroring CopyVocal's map_times."""
    values = np.asarray(teacher_times, dtype=np.float64)
    mapping = plan.get("time_map") or {}
    target = np.asarray(mapping.get("teacher_times", []), dtype=np.float64)
    source = np.asarray(mapping.get("student_times", []), dtype=np.float64)
    result = np.interp(values, target, source) if target.size else np.zeros_like(values)
    for region in plan.get("regions", []):
        if region.get("mode") not in {"synthesize", "acoustic_fallback"}:
            continue
        selected = (values >= float(region["teacher_start"])) & (values < float(region["teacher_end"]))
        local = np.interp(values, region["teacher_anchors"], region["student_anchors"])
        result = np.where(selected, local, result)
    return np.clip(result, 0.0, float(plan.get("student_duration", 0.0)))


def _build_notes(teacher: dict, student: dict, plan: dict, target: dict[str, np.ndarray] | None,
                 project_seconds: float, register_offset_cents: float,
                 timing_changed: bool, student_words: Iterable[dict]) -> list[dict]:
    teacher_notes = [note for note in (teacher.get("notes") or []) if _finite(note.get("end")) > _finite(note.get("start"))]
    student_notes = [note for note in (student.get("notes") or []) if _finite(note.get("end")) > _finite(note.get("start"))]
    chosen = teacher_notes or student_notes
    if not chosen:
        return []
    using_teacher = bool(teacher_notes)

    times = np.asarray(target.get("times", []), dtype=np.float64) if target else np.asarray([], dtype=np.float64)
    f0 = np.asarray(target.get("f0_hz", []), dtype=np.float32) if target else np.asarray([], dtype=np.float32)
    source_times = np.asarray(target.get("source_times", times), dtype=np.float64) if target else np.asarray([], dtype=np.float64)
    if target is not None:
        rendered = target.get("rendered")
        if rendered is None or np.asarray(rendered).size != times.size:
            rendered = target.get("supported")
        if rendered is None or np.asarray(rendered).size != times.size:
            rendered = np.isfinite(source_times)
        rendered = np.asarray(rendered, dtype=bool)
    else:
        rendered = np.zeros(0, dtype=bool)

    words = list(student_words)
    result: list[dict] = []
    for note in chosen:
        start = _finite(note.get("start"))
        end = _finite(note.get("end"))
        if end <= start:
            continue
        pitch = 0.0
        source_start = start
        source_end = end
        if using_teacher:
            if timing_changed and times.size:
                mask = (times >= start) & (times <= end) & rendered & np.isfinite(source_times)
                if np.any(mask):
                    source_start = float(np.min(source_times[mask]))
                    source_end = float(np.max(source_times[mask]))
                    candidates = f0[mask & (f0 > 0)]
                    if candidates.size:
                        pitch = float(np.median(candidates))
            if pitch <= 0.0:
                mapped = _map_times(plan, [start, end])
                source_start, source_end = float(mapped[0]), float(mapped[1])
            if pitch <= 0.0:
                pitch = _midi_to_hz(_finite(note.get("midi")) + register_offset_cents / 100.0)
        else:
            source_start = start
            source_end = end
            if times.size:
                mask = (times >= start) & (times <= end) & rendered
                candidates = f0[mask & (f0 > 0)]
                if candidates.size:
                    pitch = float(np.median(candidates))
            if pitch <= 0.0:
                pitch = _midi_to_hz(_finite(note.get("midi")) + register_offset_cents / 100.0)

        source_start = _clamp(source_start, 0.0, project_seconds)
        source_end = _clamp(source_end, 0.0, project_seconds)
        if source_end - source_start < 0.020 or pitch <= 0.0:
            continue
        lyric = ""
        overlap = [word for word in words
                   if _finite(word.get("end")) > source_start and _finite(word.get("start")) < source_end]
        if len(overlap) == 1:
            lyric = str(overlap[0].get("text") or "")
        result.append({
            "start": source_start,
            "end": source_end,
            "pitchHz": pitch,
            "midi": _hz_to_midi(pitch),
            "lyric": lyric,
            "source": "teacher" if using_teacher else "student",
            "confidence": 1.0 if note.get("method") == "GAME" else 0.5,
        })

    result.sort(key=lambda item: (item["start"], item["end"]))
    normalized: list[dict] = []
    for note in result:
        if normalized and note["start"] < normalized[-1]["end"]:
            normalized[-1]["end"] = max(normalized[-1]["start"], note["start"])
        if note["end"] - note["start"] >= 0.020:
            normalized.append(note)
    return normalized


def _build_time_grid(regions: Sequence[dict], duration: float) -> list[dict]:
    """Build the editable source->output time grid at word granularity.

    Region boundaries are word boundaries: each ``synthesize`` region spans one
    word, so its (student_start, teacher_start) and (student_end, teacher_end)
    are the anchors a user would want to drag. The ``teacher_anchors`` /
    ``student_anchors`` arrays are sub-word (phoneme) splits; expanding them puts
    a handle on every syllable boundary (a real 17-word sentence produced 38,
    with gaps down to 30 ms), which is not something anyone edits by hand.

    ``silence`` / ``mute`` regions carry a zero-length student span -- there is
    no source material to map -- so they contribute no anchors.

    Accepts both spellings: the internal plan uses snake_case, while the bundle
    JSON (which the caller builds first) uses camelCase.
    """
    pairs: list[tuple[float, float]] = [(0.0, 0.0)]
    for region in regions:
        student_start = _finite(region.get("student_start", region.get("studentStart")))
        student_end = _finite(region.get("student_end", region.get("studentEnd")))
        teacher_start = _finite(region.get("teacher_start", region.get("teacherStart")))
        teacher_end = _finite(region.get("teacher_end", region.get("teacherEnd")))
        if student_end > student_start:
            pairs.append((student_start, teacher_start))
            pairs.append((student_end, teacher_end))

    # Teacher material with no student counterpart must become a jump, not a
    # stretch. A student performance that starts 22 s into the teacher recording
    # would otherwise interpolate its first 0.16 s across those 22 s. Pin the
    # source position just before the gap and again a hair later at the teacher's
    # entry point: the source axis barely advances while the output axis covers
    # the teacher-only span, so it stays silent instead of stretching audio.
    if len(pairs) > 1:
        first_source, first_output = pairs[1]
        if first_output > first_source + _MAX_GAP_JUMP_SECONDS:
            # Hold the source still at the gap entry, then jump the output axis.
            pairs.insert(1, (first_source - 0.001, first_source))
            pairs[2] = (first_source, first_output)

    cleaned: list[tuple[float, float]] = []
    for source, output in pairs:
        if not math.isfinite(source) or not math.isfinite(output):
            continue
        source = _clamp(source, 0.0, duration)
        output = _clamp(output, 0.0, duration)
        cleaned.append((source, output))
    cleaned.sort(key=lambda pair: (pair[0], pair[1]))

    result: list[tuple[float, float]] = []
    for source, output in cleaned:
        if source <= 1e-6:
            continue
        if source >= duration - 1e-6:
            continue
        if result and source <= result[-1][0] + 1e-6:
            continue
        if result and output <= result[-1][1] + 1e-6:
            continue
        result.append((source, output))

    if not result:
        return [
            {"sourceSeconds": 0.0, "outputSeconds": 0.0, "kind": "clipStart", "confidence": "default"},
            {"sourceSeconds": duration, "outputSeconds": duration, "kind": "clipEnd", "confidence": "default"},
        ]

    handles: list[dict] = [
        {"sourceSeconds": 0.0, "outputSeconds": 0.0, "kind": "clipStart", "confidence": "default"}
    ]
    for index, (source, output) in enumerate(result):
        handles.append({
            "sourceSeconds": source,
            "outputSeconds": output,
            "kind": "onsetVoiced" if index == 0 else "internalOnset",
            "confidence": "high",
        })
    handles.append({
        "sourceSeconds": duration,
        "outputSeconds": duration,
        "kind": "clipEnd",
        "confidence": "default",
    })
    return handles


def _copy_words(words: Sequence[dict]) -> list[dict]:
    result = []
    for word in words:
        result.append({
            "text": str(word.get("text") or ""),
            "start": _finite(word.get("start")),
            "end": _finite(word.get("end")),
            "phonemes": [
                {
                    "text": str(phoneme.get("text") or ""),
                    "start": _finite(phoneme.get("start")),
                    "end": _finite(phoneme.get("end")),
                }
                for phoneme in (word.get("phonemes") or [])
            ],
        })
    return result


def build_bundle(job_dir: str | Path, *, compute_hashes: bool = True) -> tuple[dict, Path]:
    job_dir = Path(job_dir).resolve()
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory does not exist: {job_dir}")
    job = read_json(job_dir / "job.json")
    teacher = read_json(job_dir / "teacher.analysis.json")
    student = read_json(job_dir / "student.analysis.json")
    plan = read_json(job_dir / "alignment.json")
    render = read_json(job_dir / "render.json") if (job_dir / "render.json").is_file() else {}

    target_path = job_dir / "target_pitch.npz"
    target = None
    if target_path.is_file():
        with np.load(target_path, allow_pickle=False) as archive:
            target = {key: archive[key] for key in archive.files}

    timing_changed = bool(target.get("timing_changed", job.get("config", {}).get("change_timing", True))) if target else bool(
        job.get("config", {}).get("change_timing", True) and (job_dir / "student.corrected.wav").is_file())
    experiment = (render.get("experiments") or {}).get("full") or {}
    if timing_changed:
        output_seconds = _finite(experiment.get("duration"))
        if output_seconds <= 0.0:
            output_seconds = _finite(teacher.get("duration")) or _finite(plan.get("teacher_duration"))
    else:
        output_seconds = _finite(student.get("duration"))
    if output_seconds <= 0.0:
        raise ValueError("Cannot determine output duration from render/analysis artifacts.")

    frame_count = int(round(output_seconds / F0_STEP)) + 1
    if frame_count <= 1:
        raise ValueError("Output duration is too short for a bundle.")
    original = _student_f0_grid(student, frame_count)
    target_grid = _target_f0_grid(target, frame_count, original)

    student_audio = _existing(job_dir / "student.reference.wav") or _existing(Path(str(job.get("student", ""))))
    if student_audio is None:
        raise FileNotFoundError("CopyVocal student reference audio is missing.")
    teacher_audio = _existing(job_dir / "teacher.reference.wav") or (
        _existing(Path(str(job.get("teacher", "")))) if job.get("teacher") else None)
    corrected_audio = _existing(job_dir / "student.corrected.wav")

    bundle_dir = job_dir
    media: dict[str, Any] = {
        "student": _media_ref(student_audio, bundle_dir, "student", digest=compute_hashes),
    }
    if teacher_audio is not None:
        media["teacher"] = _media_ref(teacher_audio, bundle_dir, "teacher", digest=compute_hashes)
    if corrected_audio is not None:
        media["corrected"] = _media_ref(corrected_audio, bundle_dir, "corrected", digest=False)

    regions = []
    for region in plan.get("regions", []):
        entry: dict[str, Any] = {
            "teacherStart": _finite(region.get("teacher_start")),
            "teacherEnd": _finite(region.get("teacher_end")),
            "studentStart": _finite(region.get("student_start")),
            "studentEnd": _finite(region.get("student_end")),
            "status": str(region.get("status") or ""),
            "mode": str(region.get("mode") or ""),
            "quality": _finite(region.get("quality")),
            "unsupported": bool(region.get("unsupported")),
            "reason": str(region.get("reason") or ""),
            "teacherAnchors": [_finite(value) for value in (region.get("teacher_anchors") or [])],
            "studentAnchors": [_finite(value) for value in (region.get("student_anchors") or [])],
        }
        if region.get("stretch_ratio") is not None:
            entry["stretchRatio"] = _finite(region.get("stretch_ratio"))
        regions.append(entry)

    register_offset = _finite(target.get("register_offset_cents") if target else 0.0)
    student_words = _copy_words(student.get("words") or [])
    notes = _build_notes(teacher, student, plan, target, output_seconds,
                         register_offset, timing_changed, student_words)
    handles = _build_time_grid(regions, output_seconds) if timing_changed else [
        {"sourceSeconds": 0.0, "outputSeconds": 0.0, "kind": "clipStart", "confidence": "default"},
        {"sourceSeconds": output_seconds, "outputSeconds": output_seconds, "kind": "clipEnd", "confidence": "default"},
    ]

    warnings = list(dict.fromkeys([
        *(plan.get("warnings") or []),
        *(teacher.get("warnings") or []),
        *(student.get("warnings") or []),
    ]))
    unsupported = [region for region in regions if region.get("unsupported")]
    unsupported_seconds = sum(max(0.0, region["teacherEnd"] - region["teacherStart"]) for region in unsupported)
    project_name = job_dir.name or "CopyVocal"

    bundle = {
        "format": BUNDLE_FORMAT,
        "formatVersion": BUNDLE_VERSION,
        "producer": {
            "name": "CopyVocal",
            "version": str(job.get("version", "1")),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "jobStatus": str(job.get("status") or ""),
            "jobId": f"{job.get('input_sha256', {}).get('teacher', '')}:{job.get('input_sha256', {}).get('student', '')}".strip(":"),
            "sourceJobDir": str(job_dir),
        },
        "project": {
            "name": project_name,
            "language": str(job.get("config", {}).get("language", student.get("config", {}).get("language", "zh"))),
            "sampleRate": _SAMPLE_RATE,
        },
        "clocks": {
            "studentSeconds": _finite(student.get("duration")),
            "teacherSeconds": _finite(teacher.get("duration")),
            "outputSeconds": output_seconds,
            "projectSourceSeconds": output_seconds,
            "outputClock": "teacher" if timing_changed else "student",
            "timingChanged": timing_changed,
            "pitchPolicy": str(target.get("pitch_policy", experiment.get("pitch_policy", "none")) if target else "none"),
            "registerOffsetCents": register_offset,
        },
        "media": media,
        "f0": {
            "hopSize": F0_HOP,
            "sampleRate": F0_SAMPLE_RATE,
            "frameStepSeconds": F0_STEP,
            "frameCount": frame_count,
            "original": {
                "encoding": "base64-f32le",
                "values": _b64_f32(original),
                "voiced": _b64_u8(original > 0.0),
            },
            "target": {
                "encoding": "base64-f32le",
                "values": _b64_f32(target_grid),
                "voiced": _b64_u8(target_grid > 0.0),
            },
        },
        "notes": notes,
        "regions": regions,
        "timeGrid": {
            "totalDurationSeconds": output_seconds,
            "handles": handles,
        },
        "words": {
            "student": student_words,
            "teacher": _copy_words(teacher.get("words") or []),
        },
        "markers": [
            {
                "start": region["teacherStart"],
                "end": region["teacherEnd"],
                "clock": "output",
                "kind": "unresolved" if region["unsupported"] else region["mode"] or "region",
                "label": region["reason"] or region["status"],
            }
            for region in regions
            if region["unsupported"] or region["mode"] == "mute"
        ],
        "diagnostics": {
            "warnings": warnings,
            "unresolvedRegionCount": len(unsupported),
            "unsupportedDurationSeconds": unsupported_seconds,
        },
    }
    return bundle, bundle_dir


def export_bundle(job_dir: str | Path, output_path: str | Path | None = None,
                  *, compute_hashes: bool = True) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    bundle, bundle_dir = build_bundle(job_dir, compute_hashes=compute_hashes)
    target = Path(output_path).resolve() if output_path else bundle_dir / "copyvocal.cvf"
    bundle["producer"]["bundlePath"] = str(target)
    write_json(target, bundle)
    return {
        "bundle": str(target),
        "format": BUNDLE_FORMAT,
        "formatVersion": BUNDLE_VERSION,
        "frameCount": bundle["f0"]["frameCount"],
        "noteCount": len(bundle["notes"]),
        "handleCount": len(bundle["timeGrid"]["handles"]),
        "timingChanged": bundle["clocks"]["timingChanged"],
        "outputSeconds": bundle["clocks"]["outputSeconds"],
        "warnings": len(bundle["diagnostics"]["warnings"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a CopyVocal job as a .cvf document.")
    parser.add_argument("job_dir", type=Path, help="CopyVocal output directory (contains job.json)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .cvf path")
    parser.add_argument("--no-hash", action="store_true", help="Skip media SHA-256 hashing")
    args = parser.parse_args(argv)
    summary = export_bundle(args.job_dir, args.output, compute_hashes=not args.no_hash)
    print(f"CopyVocal file: {summary['bundle']}")
    print(f"  notes={summary['noteCount']} handles={summary['handleCount']} "
          f"frames={summary['frameCount']} timingChanged={summary['timingChanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
