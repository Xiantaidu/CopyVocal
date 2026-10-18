"""Remeasure exported audio in a process isolated from the neural vocoder."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--teacher-analysis", type=Path, required=True)
    parser.add_argument("--student-analysis", type=Path,
                        help="Required when the output keeps the student's timeline.")
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--target-pitch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("dml", "cpu"), default="dml")
    parser.add_argument("--clock", choices=("teacher", "student"),
                        help="Timeline the render used; defaults to what target_pitch.npz declares.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from application.artifacts import read_json, write_json
    from inference.verification import verify_output
    student = read_json(args.student_analysis) if args.student_analysis else None
    report = verify_output(args.audio, read_json(args.teacher_analysis), read_json(args.alignment),
                           args.target_pitch, root=ROOT, device=args.device,
                           student_analysis=student, clock=args.clock)
    if "measured_pitch" in report:
        pitch_path = args.output.with_name(args.output.stem + ".measured_pitch.json")
        write_json(pitch_path, report.pop("measured_pitch"))
        report["measured_pitch_path"] = str(pitch_path.resolve())
    write_json(args.output, report)
    print(f"Verification saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
