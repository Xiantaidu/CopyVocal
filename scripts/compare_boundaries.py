from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


WINDOWS = {
    "pre_25_to_5_ms": (-0.025, -0.005),
    "post_0_to_20_ms": (0.0, 0.020),
    "post_20_to_60_ms": (0.020, 0.060),
    "post_60_to_120_ms": (0.060, 0.120),
}
ZERO_TOLERANCE = 1e-7


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Expected nonempty finite mono audio: {path}")
    return audio, int(rate)


def supported(region: dict) -> bool:
    return region.get("mode") == "synthesize" and not region.get("unsupported", False)


def lexical_barrier(region: dict) -> bool:
    """A rejected explicit teacher word is a hard barrier, including index zero."""
    return not supported(region) and (
        region.get("teacher_index") is not None
        or bool(region.get("teacher_text"))
        or region.get("status") == "missing"
    )


def semantic_signature(plan: dict) -> list[tuple]:
    keys = ("teacher_start", "teacher_end", "student_start", "student_end",
            "mode", "unsupported", "teacher_index", "student_index",
            "teacher_text", "student_text")
    return [tuple(region.get(key) for key in keys) for region in plan["regions"]]


def islands(regions: list[dict], tolerance: float) -> list[dict]:
    result = []
    for index, region in enumerate(regions):
        if not supported(region):
            continue
        if result and region["teacher_start"] <= result[-1]["end"] + tolerance:
            result[-1].update(end=float(region["teacher_end"]), last=index)
        else:
            result.append({"start": float(region["teacher_start"]),
                           "end": float(region["teacher_end"]),
                           "first": index, "last": index})
    return result


def samples(audio: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    first = max(0, min(len(audio), round(start * rate)))
    last = max(first, min(len(audio), round(end * rate)))
    return audio[first:last]


def measure(audio: np.ndarray, rate: int, start: float, end: float) -> dict:
    section = samples(audio, rate, start, end)
    count = len(section)
    energy = float(np.dot(section.astype(np.float64), section.astype(np.float64)))
    return {
        "start_seconds": float(start), "end_seconds": float(max(start, end)),
        "sample_count": count, "duration_seconds": count / rate,
        "rms": math.sqrt(energy / count) if count else None,
        "peak": float(np.max(np.abs(section))) if count else None,
        "sum_squared_amplitude": energy,
    }


def pitch_measure(analysis: dict, start: float, end: float) -> dict:
    pitch = analysis.get("pitch") or {}
    times = np.asarray(pitch.get("times", []), dtype=np.float64)
    voiced = np.asarray(pitch.get("voiced", []), dtype=bool)
    if times.shape != voiced.shape:
        raise ValueError("Pitch times and voiced flags must have identical shapes.")
    selected = np.isfinite(times) & (times >= start) & (times < end)
    count = int(np.count_nonzero(selected))
    active = int(np.count_nonzero(voiced & selected))
    return {"frame_count": count, "voiced_frame_count": active,
            "voiced_frame_ratio": active / count if count else None}


def delta_measure(audio: np.ndarray, rate: int, edge: float, radius: float = 0.010) -> dict:
    section = samples(audio, rate, edge - radius, edge + radius)
    delta = np.abs(np.diff(section.astype(np.float64)))
    return {"radius_seconds": radius, "difference_count": len(delta),
            "maximum_absolute_neighbor_difference": float(delta.max()) if len(delta) else None,
            "p95_absolute_neighbor_difference": float(np.percentile(delta, 95)) if len(delta) else None}


def bounded_windows(audio: np.ndarray, rate: int, edge: float, lower: float,
                    upper: float, analysis: dict | None = None) -> dict:
    result = {}
    for name, (left, right) in WINDOWS.items():
        start = min(upper, max(lower, edge + left))
        end = min(upper, max(start, edge + right))
        row = measure(audio, rate, start, end)
        if analysis is not None:
            row["pitch"] = pitch_measure(analysis, start, end)
        row["requested_offset_seconds"] = [left, right]
        row["bounded"] = start > edge + left + 1e-9 or end < edge + right - 1e-9
        result[name] = row
    return result


def reference_path(directory: Path, name: str, analysis: dict) -> Path:
    path = directory / f"{name}.reference.wav"
    if not path.is_file():
        path = Path(analysis["source"])
    return path.resolve(strict=True)


def aggregate_measures(rows: list[dict]) -> dict:
    count = sum(row["sample_count"] for row in rows)
    energy = sum(row["sum_squared_amplitude"] for row in rows)
    return {"sample_count": count, "sum_squared_amplitude": energy,
            "rms": math.sqrt(energy / count) if count else None,
            "measured_window_count": sum(row["sample_count"] > 0 for row in rows),
            "nonzero_window_count": sum((row["peak"] or 0) > ZERO_TOLERANCE for row in rows)}


def rms_change(baseline: dict, candidate: dict) -> dict:
    before, after = baseline["rms"], candidate["rms"]
    return {
        "baseline": baseline, "candidate": candidate,
        "rms_ratio": after / before if before and after is not None else None,
        "rms_change_db": 20 * math.log10(after / before) if before and after else None,
    }


def compare(baseline_dir: Path, candidate_dir: Path) -> dict:
    plan_path = baseline_dir / "alignment.json"
    plan = read_json(plan_path)
    candidate_plan = candidate_dir / "alignment.json"
    if candidate_plan.is_file() and semantic_signature(plan) != semantic_signature(read_json(candidate_plan)):
        raise ValueError("Baseline and candidate must use the same semantic alignment plan.")
    teacher_path, student_path = (baseline_dir / f"{name}.analysis.json" for name in ("teacher", "student"))
    teacher, student = read_json(teacher_path), read_json(student_path)
    for name, analysis in (("teacher", teacher), ("student", student)):
        candidate_analysis = candidate_dir / f"{name}.analysis.json"
        if candidate_analysis.is_file():
            other = read_json(candidate_analysis)
            for key in ("sample_rate", "sample_count", "words", "pitch"):
                if analysis.get(key) != other.get(key):
                    raise ValueError(f"Baseline and candidate {name} analysis differ: {key}")

    before_path = baseline_dir / "student.corrected.wav"
    after_path = candidate_dir / "student.corrected.wav"
    before, rate = load_audio(before_path)
    after, after_rate = load_audio(after_path)
    teacher_audio_path = reference_path(baseline_dir, "teacher", teacher)
    student_audio_path = reference_path(baseline_dir, "student", student)
    teacher_audio, teacher_rate = load_audio(teacher_audio_path)
    student_audio, student_rate = load_audio(student_audio_path)
    if rate != after_rate or len(before) != len(after):
        raise ValueError("Baseline and candidate output timelines must have equal sample rates and lengths.")
    if rate != int(teacher["sample_rate"]) or len(before) != int(teacher["sample_count"]):
        raise ValueError("Corrected audio must use the teacher analysis timeline.")
    regions = sorted(plan["regions"], key=lambda item: item["teacher_start"])
    support = islands(regions, 1 / rate)
    boundaries, edge_rows, barrier_rows = [], [], []
    for island in support:
        last = regions[island["last"]]
        following = regions[island["last"] + 1:]
        stops = [region for region in following if supported(region) or lexical_barrier(region)]
        stop = stops[0] if stops else None
        upper = float(stop["teacher_start"]) if stop else len(before) / rate
        source_edge = float(last["student_end"])
        source_upper = float(stop["student_start"]) if stop and stop.get("student_start", 0) >= source_edge else len(student_audio) / student_rate
        # A missing word has no usable source extent. Never measure through it
        # and accidentally treat a later student syllable as release material.
        if stop and lexical_barrier(stop) and stop.get("student_start", 0) < source_edge:
            source_upper = source_edge
        target_windows = {
            "baseline": bounded_windows(before, rate, island["end"], island["start"], upper),
            "candidate": bounded_windows(after, rate, island["end"], island["start"], upper),
            "teacher": bounded_windows(teacher_audio, teacher_rate, island["end"], island["start"], upper, teacher),
        }
        source_windows = bounded_windows(student_audio, student_rate, source_edge,
                                          float(regions[island["first"]]["student_start"]),
                                          source_upper, student)
        barriers = [region for region in following if region["teacher_start"] < upper and not supported(region)]
        row = {"teacher_start": island["start"], "teacher_end": island["end"],
               "student_end": source_edge, "ending_teacher_text": last.get("teacher_text"),
               "ending_student_text": last.get("student_text"),
               "target_context_limit_seconds": upper, "source_context_limit_seconds": source_upper,
               "next_stop_kind": "hard_lexical_barrier" if stop and lexical_barrier(stop) else "supported_word" if stop else "audio_end",
               "next_stop_text": stop.get("teacher_text") if stop else None,
               "nonlexical_gap_regions": [{key: region.get(key) for key in ("teacher_start", "teacher_end", "mode", "reason")} for region in barriers],
               "target_windows": target_windows, "student_reference_windows": source_windows}
        boundaries.append(row)
        for kind, edge in (("support_start", island["start"]), ("support_end", island["end"])):
            edge_rows.append({"kind": kind, "teacher_time_seconds": edge,
                              "baseline": delta_measure(before, rate, edge),
                              "candidate": delta_measure(after, rate, edge)})

    for region in regions:
        if not lexical_barrier(region):
            continue
        start, end = float(region["teacher_start"]), float(region["teacher_end"])
        measurements = {}
        for name, audio in (("baseline", before), ("candidate", after)):
            value = measure(audio, rate, start, end)
            section = samples(audio, rate, start, end)
            value.update(nonzero_sample_count=int(np.count_nonzero(np.abs(section) > ZERO_TOLERANCE)),
                         all_samples_zero_within_tolerance=bool(np.all(np.abs(section) <= ZERO_TOLERANCE)))
            measurements[name] = value
        barrier_rows.append({"teacher_start": start, "teacher_end": end,
                             "teacher_text": region.get("teacher_text"), "status": region.get("status"),
                             "mode": region.get("mode"), "reason": region.get("reason"), **measurements})
        for kind, edge in (("hard_lexical_start", start), ("hard_lexical_end", end)):
            edge_rows.append({"kind": kind, "teacher_time_seconds": edge,
                              "baseline": delta_measure(before, rate, edge),
                              "candidate": delta_measure(after, rate, edge)})

    source_jumps = []
    for left, right in zip(regions, regions[1:]):
        if not (supported(left) and supported(right)):
            continue
        if abs(float(left["teacher_end"]) - float(right["teacher_start"])) > 1 / rate:
            continue
        jump = float(right["student_start"]) - float(left["student_end"])
        if abs(jump) <= 1 / student_rate:
            continue
        edge = float(left["teacher_end"])
        row = {"kind": "source_jump", "teacher_time_seconds": edge,
               "student_jump_seconds": jump,
               "left_teacher_text": left.get("teacher_text"), "right_teacher_text": right.get("teacher_text"),
               "baseline": delta_measure(before, rate, edge),
               "candidate": delta_measure(after, rate, edge)}
        edge_rows.append(row)
        source_jumps.append(row)

    interior = {"baseline": [], "candidate": []}
    for island in support:
        start, end = island["start"] + 0.030, island["end"] - 0.030
        if end > start:
            for name, audio in (("baseline", before), ("candidate", after)):
                interior[name].append(measure(audio, rate, start, end))
    window_summary = {}
    for window in WINDOWS:
        window_summary[window] = rms_change(*[
            aggregate_measures([row["target_windows"][name][window] for row in boundaries])
            for name in ("baseline", "candidate")])
    return {
        "version": 1, "cpu_only": True,
        "inputs": {"baseline_directory": str(baseline_dir), "candidate_directory": str(candidate_dir),
                   "alignment": str(plan_path), "teacher_analysis": str(teacher_path), "student_analysis": str(student_path),
                   "baseline_audio": str(before_path), "candidate_audio": str(after_path),
                   "teacher_reference_audio": str(teacher_audio_path), "student_reference_audio": str(student_audio_path)},
        "sample_rate": rate, "sample_count": len(before), "zero_tolerance": ZERO_TOLERANCE,
        "notes": [
            "Windows end at the next supported teacher word, explicit lexical barrier, or audio end.",
            "Voicing is cached analysis evidence; no neural model or pitch extractor is run.",
            "Neighbor-sample differences depend on pitch, phase, and consonants; they do not establish click severity.",
            "RMS and boundary continuity measurements do not prove perceived naturalness or timbre quality.",
        ],
        "aggregate": {"support_island_count": len(support), "hard_lexical_barrier_count": len(barrier_rows),
                      "source_jump_count": len(source_jumps),
                      "boundary_windows": window_summary,
                      "maximum_absolute_neighbor_difference_near_semantic_edges": {
                          name: max((row[name]["maximum_absolute_neighbor_difference"] or 0 for row in edge_rows), default=None)
                          for name in ("baseline", "candidate")},
                      "supported_interiors": {"edge_trim_seconds": 0.030,
                          **rms_change(aggregate_measures(interior["baseline"]), aggregate_measures(interior["candidate"]))},
                      "lexical_barrier_zero_checks_pass": {name: all(row[name]["all_samples_zero_within_tolerance"] for row in barrier_rows)
                                                           for name in ("baseline", "candidate")}},
        "boundaries": boundaries, "semantic_edge_differences": edge_rows,
        "source_jump_joins": source_jumps, "hard_lexical_barriers": barrier_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exported vocal boundaries on CPU without judging perceived naturalness.")
    parser.add_argument("--baseline", type=Path, required=True, help="Directory containing the original corrected WAV and cached analysis.")
    parser.add_argument("--candidate", type=Path, required=True, help="Directory containing the candidate corrected WAV.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON report.")
    args = parser.parse_args()
    report = compare(args.baseline.resolve(strict=True), args.candidate.resolve(strict=True))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Boundary comparison saved: {output}", flush=True)


if __name__ == "__main__":
    main()
