from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from application.artifacts import write_json
from application.opentune_bundle import build_bundle, export_bundle


SAMPLE_RATE = 44100


def _wav(path: Path, seconds: float) -> None:
    import soundfile as sf

    sf.write(path, np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32), SAMPLE_RATE)


def _analysis(duration: float, *, notes=None, words=None, f0_value=200.0) -> dict:
    frames = int(round(duration / 0.01)) + 1
    times = [round(index * 0.01, 6) for index in range(frames)]
    f0 = [0.0] * frames
    for index in range(10, min(frames, 31)):
        f0[index] = f0_value
    return {
        "source": "synthetic.wav",
        "sample_rate": SAMPLE_RATE,
        "sample_count": int(round(duration * SAMPLE_RATE)),
        "duration": duration,
        "words": words or [],
        "pitch": {"times": times, "f0_hz": f0, "voiced": [value > 0 for value in f0]},
        "notes": notes or [],
        "warnings": [],
        "config": {"language": "en"},
    }


def make_job(root: Path, *, timing_changed: bool = True, student_seconds: float = 1.0,
             target_step: float = 0.01, target_spike: tuple[int, float] | None = None) -> Path:
    job = root / "job"
    job.mkdir()
    _wav(job / "student.reference.wav", student_seconds)
    _wav(job / "teacher.reference.wav", 1.0)
    write_json(job / "job.json", {
        "version": 1,
        "status": "completed",
        "teacher": str(job / "teacher.reference.wav"),
        "student": str(job / "student.reference.wav"),
        "output": str(job),
        "input_sha256": {"teacher": "t", "student": "s"},
        "config": {"language": "en", "change_timing": timing_changed},
    })
    teacher = _analysis(1.0, notes=[{"start": 0.1, "end": 0.3, "midi": 60.0, "lyric": "", "method": "GAME"}])
    student = _analysis(student_seconds, notes=[{"start": 0.1, "end": 0.3, "midi": 59.0, "lyric": "", "method": "GAME"}])
    write_json(job / "teacher.analysis.json", teacher)
    write_json(job / "student.analysis.json", student)
    write_json(job / "alignment.json", {
        "version": 1,
        "teacher_duration": 1.0,
        "student_duration": student_seconds,
        "matches": [],
        "regions": [
            {"teacher_start": 0.0, "teacher_end": 0.1, "student_start": 0.0, "student_end": 0.1,
             "status": "matched", "mode": "silence", "quality": 1.0, "unsupported": False},
            {"teacher_start": 0.1, "teacher_end": 0.3, "student_start": 0.1, "student_end": 0.3,
             "status": "matched", "mode": "synthesize", "quality": 1.0, "unsupported": False,
             "teacher_anchors": [0.1, 0.2, 0.3], "student_anchors": [0.1, 0.2, 0.3]},
            {"teacher_start": 0.3, "teacher_end": 1.0, "student_start": 0.3, "student_end": 1.0,
             "status": "matched", "mode": "silence", "quality": 1.0, "unsupported": False},
        ],
        "time_map": {"teacher_times": [0.0, 1.0], "student_times": [0.0, student_seconds]},
        "metrics": {},
        "warnings": [],
        "policy": {},
    })
    output_seconds = 1.0 if timing_changed else student_seconds
    frame_count = int(round(output_seconds / target_step))
    times = np.arange(frame_count, dtype=np.float64) * target_step + target_step * 0.5
    source_times = times.copy()
    rendered = np.ones(frame_count, dtype=bool)
    f0 = np.full(frame_count, 220.0, dtype=np.float32)
    f0[times < 0.1] = 0.0
    if target_spike is not None:
        spike_index, spike_value = target_spike
        f0[spike_index] = spike_value
    np.savez_compressed(
        job / "target_pitch.npz",
        times=times,
        f0_hz=f0,
        voiced=f0 > 0,
        source_times=source_times,
        supported=rendered,
        rendered=rendered,
        acoustic_context=np.zeros(frame_count, dtype=bool),
        time_map_refined=np.zeros(frame_count, dtype=bool),
        acoustic_energy_gain=np.ones(frame_count, dtype=np.float32),
        pitch_policy=np.array("teacher_contour_student_register"),
        register_offset_cents=np.float64(0.0),
        register_offset_available=np.bool_(True),
        note_shift_applied_count=np.int64(0),
        timing_changed=np.bool_(timing_changed),
    )
    write_json(job / "render.json", {"experiments": {"full": {
        "sample_rate": SAMPLE_RATE,
        "sample_count": int(round(output_seconds * SAMPLE_RATE)),
        "duration": output_seconds,
        "timing_changed": timing_changed,
        "pitch_policy": "teacher_contour_student_register",
    }}})
    return job


def decode(array: dict, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(array), dtype=dtype)


class OpenTuneBundleTests(unittest.TestCase):
    def test_bundle_contract_and_time_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            job = make_job(Path(directory), timing_changed=True, student_seconds=1.2)
            bundle, _ = build_bundle(job)
            self.assertEqual(bundle["format"], "opentune.copyvocal-bundle")
            self.assertEqual(bundle["formatVersion"], 1)
            self.assertAlmostEqual(bundle["clocks"]["projectSourceSeconds"],
                                   bundle["clocks"]["outputSeconds"])
            frame_count = bundle["f0"]["frameCount"]
            self.assertEqual(frame_count, 101)
            original = decode(bundle["f0"]["original"]["values"], "<f4")
            target = decode(bundle["f0"]["target"]["values"], "<f4")
            self.assertEqual(len(original), frame_count)
            self.assertEqual(len(target), frame_count)
            self.assertEqual(len(decode(bundle["f0"]["original"]["voiced"], np.uint8)), frame_count)
            # Target is zero wherever the student has no F0: OpenTune can only
            # correct voiced source frames.
            self.assertTrue(np.all(target[original <= 0] == 0))
            self.assertGreater(float(target.max()), 0.0)

            handles = bundle["timeGrid"]["handles"]
            self.assertGreater(len(handles), 2)
            sources = [handle["sourceSeconds"] for handle in handles]
            outputs = [handle["outputSeconds"] for handle in handles]
            self.assertEqual(handles[0]["kind"], "clipStart")
            self.assertEqual(handles[-1]["kind"], "clipEnd")
            self.assertEqual(handles[0]["confidence"], "default")
            self.assertEqual(handles[-1]["confidence"], "default")
            self.assertTrue(all(a < b for a, b in zip(sources, sources[1:])))
            self.assertTrue(all(a < b for a, b in zip(outputs, outputs[1:])))
            self.assertAlmostEqual(sources[-1], bundle["clocks"]["projectSourceSeconds"])
            self.assertAlmostEqual(outputs[-1], bundle["clocks"]["projectSourceSeconds"])

            self.assertEqual(len(bundle["notes"]), 1)
            note = bundle["notes"][0]
            self.assertGreaterEqual(note["start"], 0.0)
            self.assertLessEqual(note["end"], bundle["clocks"]["projectSourceSeconds"])
            self.assertGreater(note["pitchHz"], 0.0)

    def test_export_writes_file(self):
        with tempfile.TemporaryDirectory() as directory:
            job = make_job(Path(directory), timing_changed=True)
            summary = export_bundle(job)
            self.assertTrue(Path(summary["bundle"]).is_file())
            written = json.loads(Path(summary["bundle"]).read_text(encoding="utf-8"))
            self.assertEqual(written["producer"]["name"], "CopyVocal")
            self.assertEqual(written["f0"]["hopSize"], 160)

    def test_student_clock_bundle_uses_identity_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            job = make_job(Path(directory), timing_changed=False)
            bundle, _ = build_bundle(job)
            self.assertFalse(bundle["clocks"]["timingChanged"])
            self.assertEqual(bundle["clocks"]["outputClock"], "student")
            self.assertEqual(len(bundle["timeGrid"]["handles"]), 2)
            self.assertEqual(bundle["timeGrid"]["handles"][0]["kind"], "clipStart")
            self.assertEqual(bundle["timeGrid"]["handles"][1]["kind"], "clipEnd")

    def test_vocoder_grid_target_is_resampled_without_holes(self):
        # CopyVocal's target pitch uses the vocoder hop (~11.6 ms), so a naive
        # nearest-frame scatter left periodic zero frames on OpenTune's 10 ms
        # grid.  Interpolation must fill every voiced source frame.
        with tempfile.TemporaryDirectory() as directory:
            job = make_job(Path(directory), timing_changed=True, target_step=0.0116)
            bundle, _ = build_bundle(job)
            frame_count = bundle["f0"]["frameCount"]
            original = decode(bundle["f0"]["original"]["values"], "<f4")
            target = decode(bundle["f0"]["target"]["values"], "<f4")
            self.assertEqual(len(target), frame_count)
            voiced = original > 0
            self.assertTrue(np.any(voiced))
            self.assertTrue(np.all(target[voiced] > 0))
            self.assertTrue(np.all(target[original <= 0] == 0))

    def test_single_frame_target_spike_is_smoothed(self):
        # One-frame pitch spikes at note seams are not reproduced by CopyVocal's
        # vocoder; the exported curve must not show them as vibrato.
        with tempfile.TemporaryDirectory() as directory:
            job = make_job(Path(directory), timing_changed=True, target_step=0.0116,
                           target_spike=(15, 110.0))
            bundle, _ = build_bundle(job)
            target = decode(bundle["f0"]["target"]["values"], "<f4")
            window = target[int(0.15 / 0.01):int(0.22 / 0.01)]
            self.assertTrue(np.all(window > 180.0))


if __name__ == "__main__":
    unittest.main()
