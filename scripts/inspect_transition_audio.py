from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


ZERO_TOLERANCE = 1e-7


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def semantic_hash(plan: dict) -> str:
    keys = ("teacher_start", "teacher_end", "student_start", "student_end",
            "teacher_index", "student_index", "mode", "unsupported",
            "teacher_anchors", "student_anchors")
    return canonical_hash({"regions": [{key: region.get(key) for key in keys} for region in plan["regions"]]})


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, rate = sf.read(path, dtype="float32")
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Expected finite nonempty mono audio: {path}")
    return audio, int(rate)


def nonlexical_gap(region: dict) -> bool:
    if region.get("teacher_index") is not None or region.get("teacher_text"):
        return False
    return region.get("mode") in {"silence", "mute"} and region.get("status") != "missing"


class AudioMetrics:
    def __init__(self, audio: np.ndarray, rate: int):
        self.audio, self.rate = audio, rate
        self.energy = np.r_[0.0, np.cumsum(audio.astype(np.float64) ** 2)]
        count_type = np.uint32 if len(audio) < np.iinfo(np.uint32).max else np.uint64
        self.nonzero = np.concatenate((np.zeros(1, dtype=count_type),
                                       np.cumsum(np.abs(audio) > ZERO_TOLERANCE, dtype=count_type)))
        crossings = np.r_[False, np.signbit(audio[1:]) != np.signbit(audio[:-1])]
        self.crossings = np.concatenate((np.zeros(1, dtype=count_type), np.cumsum(crossings, dtype=count_type)))

    def windows(self, centers: np.ndarray, duration: float = 0.020) -> dict[str, np.ndarray]:
        centers = np.asarray(centers, dtype=np.float64)
        indices = np.rint(np.nan_to_num(centers) * self.rate).astype(np.int64)
        radius = max(1, round(duration * self.rate / 2))
        first = np.clip(indices - radius, 0, len(self.audio))
        last = np.clip(indices + radius, 0, len(self.audio))
        count = last - first
        valid = np.isfinite(centers) & (count > 0) & (centers >= 0) & (centers <= len(self.audio) / self.rate)
        rms = np.sqrt(np.maximum(0, self.energy[last] - self.energy[first]) / np.maximum(1, count))
        allzero = self.nonzero[last] == self.nonzero[first]
        crossing_count = self.crossings[last] - self.crossings[np.minimum(first + 1, last)]
        zcr = crossing_count / np.maximum(1, count - 1)
        return {"valid": valid, "rms": rms, "allzero": allzero, "zcr": zcr,
                "sample_count": count}


def shared_alignment(baseline: Path, candidate: Path, explicit: Path | None) -> tuple[dict, dict]:
    paths = [baseline / "alignment.json", candidate / "alignment.json"]
    if explicit is None and any(not path.is_file() for path in paths):
        raise ValueError("Both directories need alignment.json, or pass --alignment for an explicitly shared plan.")
    selected = explicit.resolve(strict=True) if explicit else paths[0]
    plan = read_json(selected)
    digest = canonical_hash(plan)
    mapping_digest = semantic_hash(plan)
    evidence = []
    for path in paths:
        if path.is_file():
            other = read_json(path)
            if semantic_hash(other) != mapping_digest:
                raise ValueError(f"Alignment differs from the shared semantic plan: {path}")
            evidence.append({"path": str(path), "sha256": sha256(path),
                             "canonical_sha256": canonical_hash(other), "semantic_mapping_sha256": mapping_digest})
    return plan, {"shared_path": str(selected), "canonical_sha256": digest,
                  "semantic_mapping_sha256": mapping_digest,
                  "explicit_shared_plan": explicit is not None, "directory_plan_files": evidence}


def common_file(baseline: Path, candidate: Path, filename: str) -> Path:
    paths = [directory / filename for directory in (candidate, baseline)]
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find shared {filename} in either directory.")


def target_map(path: Path, centers: np.ndarray) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Source mapping evidence is required: {path}")
    with np.load(path) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
            raise ValueError(f"Target times must be nonempty and increasing: {path}")
        right = np.searchsorted(times, centers).clip(0, len(times) - 1)
        left = np.maximum(0, right - 1)
        index = np.where(np.abs(times[left] - centers) <= np.abs(times[right] - centers), left, right)
        mapped = {"frame_time": times[index]}
        for key in ("source_times", "f0_hz", "source_voiced", "supported", "rendered", "acoustic_context"):
            if key in archive:
                value = np.asarray(archive[key])
                if value.shape != times.shape:
                    raise ValueError(f"Target array has an invalid shape: {key}")
                mapped[key] = value[index]
        if "source_times" not in mapped:
            raise ValueError(f"Source times are missing: {path}")
    return mapped, {"path": str(path), "sha256": sha256(path)}


def render_metadata(directory: Path) -> dict:
    path = directory / "render.json"
    if not path.is_file():
        return {}
    report = read_json(path)
    return report.get("experiments", {}).get("full", report)


def audit_source_map(path: Path, metadata: dict) -> dict:
    with np.load(path) as archive:
        times = np.asarray(archive["times"], dtype=np.float64)
        source = np.asarray(archive["source_times"], dtype=np.float64)
        rendered = np.asarray(archive["rendered"], dtype=bool) if "rendered" in archive else np.isfinite(source)
        refined = np.asarray(archive["time_map_refined"], dtype=bool) if "time_map_refined" in archive else np.zeros(len(times), dtype=bool)
    finite = np.isfinite(source)
    pairs = finite[:-1] & finite[1:] & rendered[:-1] & rendered[1:]
    delta = np.diff(source)
    backwards, frozen = pairs & (delta < -1e-9), pairs & (np.abs(delta) <= 1e-9)
    held = np.zeros(len(times), dtype=bool)
    for context in metadata.get("acoustic_contexts", []):
        if context.get("source_strategy") == "hold":
            held |= (times >= context["teacher_start"]) & (times < context["teacher_end"])
    expected_held = held[:-1] & held[1:]

    def examples(mask):
        return [{"teacher_start": float(times[index]), "teacher_end": float(times[index + 1]),
                 "source_start": finite_number(source[index]), "source_end": finite_number(source[index + 1]),
                 "source_step_seconds": finite_number(delta[index]),
                 "inside_held_context": bool(expected_held[index])}
                for index in np.flatnonzero(mask)[:100]]

    windows = []
    for transition in metadata.get("acoustic_transitions", []):
        selected = (times >= transition["teacher_start"]) & (times < transition["teacher_end"])
        indices = np.flatnonzero(selected)
        values = source[selected]
        steps = np.diff(values)
        # Include both outer joins so a good internal DTW cannot hide a backwards seam.
        joins = np.zeros(len(delta), dtype=bool)
        if len(indices):
            joins[max(0, indices[0] - 1):min(len(delta), indices[-1] + 1)] = True
        windows.append({"teacher_start": transition["teacher_start"], "teacher_end": transition["teacher_end"],
                        "source_strategy": transition.get("source_strategy"), "frame_count": len(values),
                        "all_finite": bool(np.isfinite(values).all()),
                        "strictly_increasing": bool(np.isfinite(values).all() and np.all(steps > 1e-9)),
                        "minimum_source_step_seconds": finite_number(steps.min()) if len(steps) else None,
                        "backwards_join_count": int(np.count_nonzero(backwards & joins)),
                        "frozen_join_count": int(np.count_nonzero(frozen & joins))})
    return {"rendered_nonfinite_source_frame_count": int(np.count_nonzero(rendered & ~finite)),
            "global_finite_adjacent_pair_count": int(np.count_nonzero(pairs)),
            "global_backwards_pair_count": int(np.count_nonzero(backwards)),
            "global_frozen_pair_count": int(np.count_nonzero(frozen)),
            "expected_held_frozen_pair_count": int(np.count_nonzero(frozen & expected_held)),
            "unexpected_frozen_pair_count": int(np.count_nonzero(frozen & ~expected_held)),
            "refined_frame_count": int(np.count_nonzero(refined)),
            "refined_windows": windows,
            "all_refined_windows_strictly_increasing": all(window["strictly_increasing"] for window in windows),
            "backwards_examples": examples(backwards), "frozen_examples": examples(frozen)}


def finite_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def metric_row(metrics: dict, index: int) -> dict:
    if not metrics["valid"][index]:
        return {"rms": None, "zero_crossing_ratio": None, "allzero": None}
    return {"rms": float(metrics["rms"][index]),
            "zero_crossing_ratio": float(metrics["zcr"][index]),
            "allzero": bool(metrics["allzero"][index])}


def mask_summary(selected: np.ndarray, audible: np.ndarray, output: dict,
                 teacher: dict, floor: float, ratio: float, hop: float) -> dict:
    selected = selected & teacher["valid"] & output["valid"]
    active = selected & audible
    low = active & (output["rms"] < np.maximum(floor, teacher["rms"] * ratio))
    zero = active & output["allzero"]
    count = int(np.count_nonzero(active))
    return {"nonlexical_gap_frame_count": int(np.count_nonzero(selected)),
            "teacher_audible_frame_count": count,
            "teacher_audible_output_allzero_frame_count": int(np.count_nonzero(zero)),
            "teacher_audible_output_lowenergy_frame_count": int(np.count_nonzero(low)),
            "audible_frames_lost_to_allzero_ratio": float(np.count_nonzero(zero) / count) if count else None,
            "audible_frames_lowenergy_ratio": float(np.count_nonzero(low) / count) if count else None,
            "audible_frame_grid_duration_seconds": count * hop,
            "allzero_frame_grid_duration_seconds": int(np.count_nonzero(zero)) * hop}


def run(args) -> dict:
    baseline, candidate = args.baseline.resolve(strict=True), args.candidate.resolve(strict=True)
    plan, plan_evidence = shared_alignment(baseline, candidate, args.alignment)
    paths = {"teacher": common_file(baseline, candidate, "teacher.reference.wav"),
             "student": common_file(baseline, candidate, "student.reference.wav"),
             "baseline": baseline / "student.corrected.wav", "candidate": candidate / "student.corrected.wav"}
    audio, rates = {}, {}
    for name, path in paths.items():
        audio[name], rates[name] = load_audio(path)
    rate = rates["teacher"]
    if any(rates[name] != rate or len(audio[name]) != len(audio["teacher"])
           for name in ("baseline", "candidate")):
        raise ValueError("Teacher, baseline, and candidate must have equal sample rates and lengths.")
    if rates["student"] != rate:
        raise ValueError("Student reference audio must use the teacher sample rate.")
    duration = len(audio["teacher"]) / rate
    if not (0 <= args.start < args.end <= duration):
        raise ValueError("Excerpt must have 0 <= start < end <= teacher duration.")
    if not (0 <= args.table_start < args.table_end <= duration):
        raise ValueError("Table must have 0 <= table-start < table-end <= teacher duration.")
    hop = 0.010
    centers = (np.arange(math.ceil(duration / hop)) + 0.5) * hop
    centers = centers[centers < duration]
    metrics = {name: AudioMetrics(wave, rates[name]) for name, wave in audio.items()}
    measured = {name: metrics[name].windows(centers) for name in ("teacher", "baseline", "candidate")}
    threshold = args.audible_threshold
    if args.adaptive_threshold:
        threshold = max(threshold, 3 * float(np.percentile(measured["teacher"]["rms"], 10)))
    audible = measured["teacher"]["rms"] > threshold
    gap_mask = np.zeros(len(centers), dtype=bool)
    regions = sorted(plan["regions"], key=lambda row: row["teacher_start"])
    gaps = []
    for index, region in enumerate(regions):
        if not nonlexical_gap(region):
            continue
        selected = (centers >= float(region["teacher_start"])) & (centers < float(region["teacher_end"]))
        gap_mask |= selected
        gaps.append({"region_index": index, **{key: region.get(key) for key in
                     ("teacher_start", "teacher_end", "student_start", "student_end", "mode", "reason")},
                     "baseline": mask_summary(selected, audible, measured["baseline"], measured["teacher"],
                                              args.low_energy_floor, args.low_energy_ratio, hop),
                     "candidate": mask_summary(selected, audible, measured["candidate"], measured["teacher"],
                                               args.low_energy_floor, args.low_energy_ratio, hop)})
    # An overlapping explicit lyric or supported word always wins over a coarse gap label.
    for region in regions:
        if not nonlexical_gap(region):
            gap_mask &= ~((centers >= region["teacher_start"]) & (centers < region["teacher_end"]))

    render_reports = {name: render_metadata(directory) for name, directory in
                      (("baseline", baseline), ("candidate", candidate))}
    safe_transition_mask = np.zeros(len(centers), dtype=bool)
    safe_transition_gaps = []
    for transition in render_reports["candidate"].get("acoustic_transitions", []):
        start, end = transition.get("gap_teacher_start"), transition.get("gap_teacher_end")
        if start is None or end is None or end <= start:
            continue
        selected = gap_mask & (centers >= start) & (centers < end)
        safe_transition_mask |= selected
        safe_transition_gaps.append({"teacher_start": start, "teacher_end": end,
                                     "source_strategy": transition.get("source_strategy"),
                                     **{name: mask_summary(selected, audible, measured[name], measured["teacher"],
                                                           args.low_energy_floor, args.low_energy_ratio, hop)
                                        for name in ("baseline", "candidate")}})
    lexical_barriers = []
    for region in regions:
        if region.get("mode") not in {"mute", "silence"} or nonlexical_gap(region):
            continue
        start, end = round(region["teacher_start"] * rate), round(region["teacher_end"] * rate)
        row = {key: region.get(key) for key in ("teacher_start", "teacher_end", "teacher_text", "mode", "reason")}
        for name in ("baseline", "candidate"):
            section = audio[name][max(0, start):min(len(audio[name]), end)]
            row[name] = {"sample_count": len(section),
                         "nonzero_sample_count": int(np.count_nonzero(np.abs(section) > ZERO_TOLERANCE)),
                         "peak": float(np.max(np.abs(section))) if len(section) else 0.0,
                         "allzero": bool(np.all(np.abs(section) <= ZERO_TOLERANCE))}
        lexical_barriers.append(row)

    table_centers = np.arange(args.table_start + hop / 2, args.table_end, hop)
    maps, map_evidence = {}, {}
    for name, directory in (("baseline", baseline), ("candidate", candidate)):
        maps[name], map_evidence[name] = target_map(directory / "target_pitch.npz", table_centers)
    table_metrics = {name: metrics[name].windows(table_centers, duration=hop)
                     for name in ("teacher", "baseline", "candidate")}
    source_metrics = {name: metrics["student"].windows(maps[name]["source_times"], duration=hop)
                      for name in ("baseline", "candidate")}
    table = []
    for index, center in enumerate(table_centers):
        active_regions = [region for region in regions if region["teacher_start"] <= center < region["teacher_end"]]
        row = {"start_seconds": float(center - hop / 2), "end_seconds": float(center + hop / 2),
               "teacher": metric_row(table_metrics["teacher"], index),
               "semantic_regions": [{key: region.get(key) for key in ("mode", "teacher_text", "reason")}
                                    for region in active_regions]}
        for name in ("baseline", "candidate"):
            mapped = maps[name]
            row[name] = metric_row(table_metrics[name], index)
            row[name]["mapped_student_reference"] = metric_row(source_metrics[name], index)
            row[name]["target_frame_time_seconds"] = float(mapped["frame_time"][index])
            for key in ("source_times", "f0_hz", "source_voiced", "supported", "rendered", "acoustic_context"):
                if key in mapped:
                    value = mapped[key][index]
                    row[name][key] = bool(value) if np.issubdtype(mapped[key].dtype, np.bool_) else finite_number(value)
        table.append(row)

    excerpt_directory = args.excerpt_dir.resolve() if args.excerpt_dir else args.output.resolve().parent / "transition_excerpts"
    excerpt_directory.mkdir(parents=True, exist_ok=True)
    first, last = round(args.start * rate), round(args.end * rate)
    excerpts = {}
    for name in ("teacher", "baseline", "candidate"):
        path = excerpt_directory / f"{name}_{args.start:.3f}-{args.end:.3f}.wav"
        sf.write(path, audio[name][first:last], rate, subtype="PCM_24")
        excerpts[name] = {"path": str(path), "sha256": sha256(path), "sample_count": last - first,
                          "start_seconds": first / rate, "end_seconds": last / rate}

    aggregate = {name: mask_summary(gap_mask, audible, measured[name], measured["teacher"],
                                    args.low_energy_floor, args.low_energy_ratio, hop)
                 for name in ("baseline", "candidate")}
    safe_aggregate = {name: mask_summary(safe_transition_mask, audible, measured[name], measured["teacher"],
                                         args.low_energy_floor, args.low_energy_ratio, hop)
                      for name in ("baseline", "candidate")}
    return {
        "version": 1, "cpu_only": True,
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "shared_alignment": plan_evidence, "target_maps": map_evidence,
        "measurement_policy": {"frame_hop_seconds": hop, "audibility_rms_window_seconds": 0.020,
                               "table_rms_window_seconds": hop, "teacher_audible_rms_threshold": threshold,
                               "adaptive_threshold": args.adaptive_threshold,
                               "lowenergy_absolute_floor": args.low_energy_floor,
                               "lowenergy_teacher_relative_ratio": args.low_energy_ratio,
                               "allzero_tolerance": ZERO_TOLERANCE},
        "aggregate": aggregate, "nonlexical_gaps": gaps,
        "safe_acoustic_transition_gap_aggregate": safe_aggregate,
        "safe_acoustic_transition_gaps": safe_transition_gaps,
        "source_map_audit": {name: audit_source_map(directory / "target_pitch.npz", render_reports[name])
                             for name, directory in (("baseline", baseline), ("candidate", candidate))},
        "lexical_barrier_zero_checks": {"regions": lexical_barriers,
            "all_pass": {name: all(row[name]["allzero"] for row in lexical_barriers) for name in ("baseline", "candidate")}},
        "transition_table_10ms": table, "excerpts": excerpts,
        "notes": [
            "Audibility means RMS exceeding a declared threshold; it is not a perceptual loudness judgment.",
            "Low energy compares two singers' levels and does not establish intelligibility or audio quality.",
            "Whole-track statistics use 20 ms windows on a 10 ms grid; durations describe the grid and not exact silence spans.",
            "The safe transition subset intersects candidate acoustic transition gap endpoints with nonlexical gap frames; refined word interiors are excluded.",
            "Mapped student RMS/ZCR uses a 10 ms source window centered on the nearest saved target frame source time.",
            "The excerpts are separate reference/listening artifacts; this script never writes teacher audio into corrected output.",
            "Planner recommendation: distinguish unvoiced acoustic activity from silence using waveform energy and spectral evidence; F0 absence alone is insufficient.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect nonlexical vocal transitions and export CPU-only listening excerpts.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, help="Explicit shared alignment when an archived run has no plan copy.")
    parser.add_argument("--start", type=float, default=46.8, help="Teacher-clock excerpt start in seconds.")
    parser.add_argument("--end", type=float, default=48.2, help="Teacher-clock excerpt end in seconds.")
    parser.add_argument("--table-start", type=float, default=47.3)
    parser.add_argument("--table-end", type=float, default=47.7)
    parser.add_argument("--excerpt-dir", type=Path)
    parser.add_argument("--audible-threshold", type=float, default=0.003)
    parser.add_argument("--adaptive-threshold", action="store_true",
                        help="Raise the audible threshold to at least 3 times the teacher RMS 10th percentile.")
    parser.add_argument("--low-energy-ratio", type=float, default=0.1)
    parser.add_argument("--low-energy-floor", type=float, default=0.0003)
    args = parser.parse_args()
    for name in ("audible_threshold", "low_energy_ratio", "low_energy_floor"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    report = run(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Transition inspection saved: {output}", flush=True)
    print(f"Listening excerpts saved: {Path(next(iter(report['excerpts'].values()))['path']).parent}", flush=True)


if __name__ == "__main__":
    main()
