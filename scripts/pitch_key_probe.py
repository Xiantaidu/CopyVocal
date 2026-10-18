r"""Diagnose why adaptive placement can sound out of tune.

Measures, on an existing output directory's analyses and alignment, three
separate things that could each explain an out-of-tune result:

1. How far each word is moved. Adaptive placement moves a word by
   ``offset - difference``, so measurement spread around the register median
   becomes per-word detuning of words that were already correct.
2. Whether each side is on a stable scale, and which key that is. Key detection
   is only meaningful if the singing really sits on a scale.
3. Whether the aims themselves land on the student's scale. A target that is
   chromatically transposed off the student's key is wrong even when the output
   matches it exactly, which is a defect this project's own accuracy metric
   cannot see.

    .\python\python.exe -X utf8 scripts\pitch_key_probe.py outputs\long_hard
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCALES = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10)}
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
SEMITONE = 100.0


def midi(f0):
    return 69.0 + 12.0 * math.log2(f0 / 440.0)


def scale_distance(value, tonic, scale):
    """Distance in cents from a MIDI value to the nearest tone of a key."""
    offsets = [abs((value - tonic - degree)) % 12 for degree in scale]
    return min(min(item, 12 - item) for item in offsets) * SEMITONE


def key_score(values, tonic, scale):
    """Mean distance from a MIDI value to the nearest tone of a key, in cents."""
    if not len(values):
        return float("inf")
    return float(sum(scale_distance(value, tonic, scale) for value in values) / len(values))


def detect_key(values):
    """Best (tonic, scale) by mean distance to the nearest scale tone."""
    best = None
    for tonic in range(12):
        for name, scale in SCALES.items():
            score = key_score(values, tonic, scale)
            if best is None or score < best[0]:
                best = (score, tonic, name)
    return {"tonic": best[1], "tonic_name": PITCH_NAMES[best[1]], "scale": best[2],
            "mean_cents": round(best[0], 2)}


def levels(pitch, regions, side):
    """Median MIDI level per matched region, on one side's own interval."""
    import numpy as np
    times = np.asarray(pitch["times"], dtype=float)
    f0 = np.asarray(pitch["f0_hz"], dtype=float)
    voiced = np.asarray(pitch.get("voiced", f0 > 0), dtype=bool) & (f0 > 0)
    out = []
    for region in regions:
        start, stop = float(region[f"{side}_start"]), float(region[f"{side}_end"])
        selected = voiced & (times >= start) & (times < stop)
        out.append(float(np.median(f0[selected])) if np.any(selected) else None)
    return out


def main() -> int:
    import numpy as np

    output = Path(sys.argv[1]).resolve()
    teacher = json.loads((output / "teacher.analysis.json").read_text(encoding="utf-8"))
    student = json.loads((output / "student.analysis.json").read_text(encoding="utf-8"))
    plan = json.loads((output / "alignment.json").read_text(encoding="utf-8"))
    regions = [region for region in plan["regions"]
               if region.get("mode") == "synthesize" and "student_start" in region
               and "student_end" in region]
    print(f"{output.name}: {len(regions)} matched regions with sung pitch on both sides")

    teacher_levels = levels(teacher["pitch"], regions, "teacher")
    student_levels = levels(student["pitch"], regions, "student")
    pairs = [(midi(t), midi(s)) for t, s in zip(teacher_levels, student_levels)
             if t and s]
    if not pairs:
        print("No comparable regions; nothing to diagnose.")
        return 1
    teacher_midi = np.array([item[0] for item in pairs])
    student_midi = np.array([item[1] for item in pairs])
    differences = (student_midi - teacher_midi) * SEMITONE

    print("\n1. How far each word is moved")
    median = float(np.median(differences))
    offset = round(median / SEMITONE) * SEMITONE
    shifts = offset - differences
    print(f"   difference: median {median:+.1f}c  p5 {np.percentile(differences, 5):+.1f}c"
          f"  p95 {np.percentile(differences, 95):+.1f}c"
          f"  spread {(np.percentile(differences, 95) - np.percentile(differences, 5)):.1f}c")
    print(f"   quantized offset {offset:+.1f}c;  that leaves a {abs(median - offset):.1f}c global bias")
    for limit in (10.0, 25.0, 50.0):
        moved = np.abs(shifts) > limit
        print(f"   words moved by more than {limit:4.0f}c: {int(moved.sum()):4d} / {len(shifts)}"
              f"  ({100 * moved.mean():5.1f}%)  largest {np.max(np.abs(shifts)):7.1f}c")

    print("\n2. Is each side on a stable scale?")
    teacher_key = detect_key(teacher_midi)
    student_key = detect_key(student_midi)
    for label, key in (("teacher", teacher_key), ("student", student_key)):
        print(f"   {label}: {key['tonic_name']} {key['scale']:5s}"
              f"  mean distance to its scale {key['mean_cents']:5.2f}c")
    print(f"   same tonic and scale: {teacher_key['tonic'] == student_key['tonic'] and teacher_key['scale'] == student_key['scale']}")

    print("\n3. Do the aims land inside the student's own key?")
    tonic, scale = student_key["tonic"], SCALES[student_key["scale"]]
    for label, values in (("student's own singing", student_midi),
                          ("teacher chromatically transposed", teacher_midi + offset / SEMITONE),
                          ("teacher as sung (untouched)", teacher_midi)):
        deviations = np.array([scale_distance(value, tonic, scale) for value in values])
        on_scale = deviations < 25.0
        print(f"   {label:32s} on the student's scale: {100 * on_scale.mean():5.1f}%"
              f"   median distance {np.median(deviations):5.2f}c"
              f"   worst {deviations.max():6.1f}c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
