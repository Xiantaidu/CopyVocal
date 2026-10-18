"""Remeasure rendered word boundaries without teacher-forced output lyrics."""
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device", choices=("dml", "cpu"), default="dml")
    parser.add_argument("--language")
    parser.add_argument("--minimum-quality", type=float, default=0.6)
    parser.add_argument("--tolerance-ms", type=float, default=50.0)
    parser.add_argument("--max-chunk-seconds", type=float)
    parser.add_argument("--clock", choices=("teacher", "student"), default="teacher",
                        help="The timeline the render used; teacher boundaries only describe a teacher-clock output.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from application.artifacts import read_json, write_json
    from inference.timing_verification import verify_timing
    report = verify_timing(args.audio, read_json(args.teacher_analysis),
        args.work_dir or args.output.parent / "timing_verification", root=ROOT,
        device=args.device, language=args.language, minimum_quality=args.minimum_quality,
        tolerance_seconds=args.tolerance_ms / 1000, max_chunk_seconds=args.max_chunk_seconds,
        clock=args.clock)
    write_json(args.output, report)
    print(f"Timing verification saved: {args.output}", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
