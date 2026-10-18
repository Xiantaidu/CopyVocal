# Usage: .\python\python.exe scripts\pitch_probe.py outputs\short
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SETTINGS = (("off", False), ("teacher", False), ("teacher", True), ("adaptive", True))


def main() -> int:
    output = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else output / "pitch_probe"
    destination.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    from application.artifacts import read_json
    from inference.correction import render_student
    from inference.slicer.copyvocal_slicer import downmix_safe
    from inference.vocoder import PcNsfVocoder

    teacher, student = (read_json(output / f"{name}.analysis.json") for name in ("teacher", "student"))
    plan = read_json(output / "alignment.json")
    sources = {}
    for name, analysis in (("teacher", teacher), ("student", student)):
        waveform, sr = sf.read(analysis["source"], dtype="float32", always_2d=True)
        waveform, _ = downmix_safe(waveform)
        target_sr = int(analysis["sample_rate"])
        if sr != target_sr:
            divisor = math.gcd(sr, target_sr)
            waveform = resample_poly(waveform, target_sr // divisor, sr // divisor).astype(np.float32)
        sources[name] = waveform
    vocoder = PcNsfVocoder(ROOT / "models" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
                           device="cuda" if _cuda() else "cpu")
    rows = []
    for pitch_mode, map_teacher in SETTINGS:
        target = destination / f"{pitch_mode}{'_contour' if map_teacher else ''}.npz"
        wave, diagnostics = render_student(
            sources["student"], teacher, student, plan, vocoder, mode="full",
            change_timing=True, pitch_mode=pitch_mode, map_teacher_pitch=map_teacher,
            target_pitch_path=target, teacher_audio=sources["teacher"],
        )
        register = diagnostics["register_pitch"]
        with np.load(target) as saved:
            voiced = np.asarray(saved["voiced"], dtype=bool)
            levels = np.asarray(saved["f0_hz"], dtype=np.float64)[voiced]
        rows.append({
            "pitch_mode": pitch_mode, "map_teacher_pitch": map_teacher,
            "policy": diagnostics["pitch_policy"],
            "placement": register["placement"],
            "offset_cents": round(float(register["offset_cents"]), 2),
            "raw_offset_cents": (round(float(register["raw_offset_cents"]), 2)
                                 if register["raw_offset_cents"] is not None else None),
            "comparable_regions": register["comparable_regions"],
            "contour_shift_cents": round(float(register["contour_shift_cents"]), 2),
            "note_shift_measured": register["note_shift_measured_count"],
            "note_shift_applied": register["note_shift_applied_count"],
            "target_median_hz": round(float(np.median(levels)), 2) if len(levels) else None,
            "voiced_frames": int(voiced.sum()),
            "sample_count": int(diagnostics["sample_count"]),
            "peak": round(float(np.max(np.abs(wave))), 4),
            "warnings": [item for item in diagnostics["warnings"] if "pitch" in item.lower()],
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (destination / "pitch_probe.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSummary written: {destination / 'pitch_probe.json'}")
    return 0


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    raise SystemExit(main())
