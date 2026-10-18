"""Align a student's vocal pitch and timing to a teacher recording."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from application.languages import SUPPORTED_LANGUAGES


def main() -> int:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=ROOT / "settings" / "default.json")
    config_args, _ = config_parser.parse_known_args()
    config = json.loads(config_args.config.read_text(encoding="utf-8"))
    if config.get("sample_rate", 44100) != 44100:
        config_parser.error("The bundled vocoder requires sample_rate=44100.")
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--teacher", type=Path, default=ROOT / "Teacher.wav")
    parser.add_argument("--student", type=Path, default=ROOT / "Student.wav")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "short")
    parser.add_argument("--onnx-device", choices=("dml", "cpu"), default=config.get("onnx_device", "dml"))
    parser.add_argument("--vocoder-device", choices=("auto", "cuda", "cpu"), default=config.get("vocoder_device", "auto"))
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default=config.get("language", "zh"))
    parser.add_argument("--change-timing", action=argparse.BooleanOptionalAction,
                        default=config.get("change_timing", True),
                        help="Warp the student's timeline onto the teacher's. Disable to correct pitch only.")
    parser.add_argument("--pitch-mode", choices=("off", "teacher", "adaptive"),
                        default=config.get("pitch_mode", "off"),
                        help="off leaves the pitch exactly as sung. teacher moves the result into the teacher's "
                             "register at the teacher's own level. adaptive puts each word on the teacher's note "
                             "level while the melody as a whole stays in the student's own register.")
    parser.add_argument("--map-teacher-pitch", action=argparse.BooleanOptionalAction,
                        default=config.get("map_teacher_pitch", False),
                        help="Use the teacher's continuous pitch, so slides and vibrato are the teacher's. In "
                             "teacher mode this chooses the teacher's contour over the student's. In adaptive "
                             "mode the teacher's contour already carries the note levels, so this transposes it "
                             "into the student's register instead of placing words individually.")
    parser.add_argument("--max-chunk-seconds", type=float, default=config.get("max_chunk_seconds", 18.0))
    parser.add_argument("--context-seconds", type=float, default=config.get("context_seconds", 0.75))
    parser.add_argument("--minimum-alignment-quality", type=float, default=config.get("minimum_alignment_quality", 0.35))
    parser.add_argument("--max-stretch", type=float, default=config.get("maximum_stretch", 4.0))
    parser.add_argument("--experiments", action="store_true", help="Also render identity, pitch-only and time-only A/B files.")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true", help="Reuse matching analysis files from the output directory.")
    parser.add_argument("--no-game", action="store_true", help="Skip optional note extraction.")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--verify-timing", action="store_true", help="Remeasure output words with independent ASR/TiFA; adds model inference.")
    parser.add_argument("--allow-acoustic-fallback", action="store_true", default=config.get("allow_acoustic_fallback", False), help="Allow explicitly low-confidence activity-only timing.")
    parser.add_argument("--log-floor", type=float, default=config.get("vocoder_log_floor", 1e-5))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from application.copyvocal import run_job
    try:
        result = run_job(args.teacher, args.student, args.output, onnx_device=args.onnx_device,
                         vocoder_device=args.vocoder_device, language=args.language,
                         experiments=args.experiments, analyze_only=args.analyze_only,
                         skip_analysis=args.skip_analysis, include_game=not args.no_game,
                         verify=not args.no_verify, max_chunk_seconds=args.max_chunk_seconds,
                         allow_acoustic_fallback=args.allow_acoustic_fallback, log_floor=args.log_floor,
                         context_seconds=args.context_seconds, min_quality=args.minimum_alignment_quality,
                         max_stretch=args.max_stretch, verify_timing=args.verify_timing,
                         change_timing=args.change_timing, pitch_mode=args.pitch_mode,
                         map_teacher_pitch=args.map_teacher_pitch)
    except KeyboardInterrupt:
        logging.error("Processing cancelled.")
        return 130
    except Exception:
        logging.exception("Processing failed.")
        return 1
    print(f"Job status: {result['status']}")
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
