from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both analysis tracks in an isolated process.")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("dml", "cpu"), default="dml")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--max-chunk-seconds", type=float, default=18.0)
    parser.add_argument("--context-seconds", type=float, default=0.75)
    parser.add_argument("--no-game", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from application.artifacts import write_json
    from inference.analysis import analyze_pair
    teacher, student = analyze_pair(
        args.teacher, args.student, args.output, ROOT, device=args.device,
        language=args.language, max_chunk_seconds=args.max_chunk_seconds,
        context_seconds=args.context_seconds,
        include_game=not args.no_game,
    )
    write_json(args.output / "teacher.analysis.json", teacher)
    write_json(args.output / "student.analysis.json", student)
    print("Analysis artifacts saved.", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
