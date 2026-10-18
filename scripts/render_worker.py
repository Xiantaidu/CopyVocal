"""Render controlled experiments in an isolated CUDA process."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--experiments", action="store_true")
    parser.add_argument("--log-floor", type=float, default=1e-5)
    parser.add_argument("--allow-acoustic-fallback", action="store_true")
    # run_job and the desktop interface always pass one variant of each axis, so
    # these defaults only apply to a direct worker invocation.
    parser.add_argument("--change-timing", action=argparse.BooleanOptionalAction, default=True,
                        help="Warp the student's timeline onto the teacher's.")
    parser.add_argument("--pitch-mode", choices=("off", "teacher", "adaptive"), default="off",
                        help="off leaves the pitch as sung; teacher moves the result into the teacher's "
                             "register; adaptive puts each word on the teacher's note level while the "
                             "melody stays in the student's own register.")
    parser.add_argument("--map-teacher-pitch", action=argparse.BooleanOptionalAction, default=False,
                        help="Use the teacher's continuous pitch; without it the student's own pitch movement "
                             "is kept and each matched word is only moved onto its target note.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from application.artifacts import file_hash, read_json, write_json
    from inference.correction import render_student
    from inference.slicer.copyvocal_slicer import downmix_safe
    from inference.vocoder import PcNsfVocoder

    torch.set_num_threads(min(8, torch.get_num_threads()))
    torch.manual_seed(20261015)
    teacher = read_json(args.output / "teacher.analysis.json")
    student = read_json(args.output / "student.analysis.json")
    plan = read_json(args.output / "alignment.json")
    sources = {}
    for name, analysis in (("teacher", teacher), ("student", student)):
        if file_hash(Path(analysis["source"])) != analysis.get("input_sha256"):
            raise ValueError(f"{name} input changed after analysis; rerun analysis.")
        waveform, sr = sf.read(analysis["source"], dtype="float32", always_2d=True)
        waveform, _ = downmix_safe(waveform)
        target_sr = int(analysis["sample_rate"])
        if sr != target_sr:
            divisor = math.gcd(sr, target_sr)
            waveform = resample_poly(waveform, target_sr // divisor, sr // divisor).astype(np.float32)
        if len(waveform) != int(analysis["sample_count"]):
            raise ValueError("Audio changed after analysis; rerun analysis.")
        sources[name] = waveform
        sf.write(args.output / f"{name}.reference.wav", waveform, target_sr, subtype="PCM_24")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    logging.info("Loading PC-NSF-HiFiGAN on %s.", device)
    started = time.perf_counter()
    vocoder = PcNsfVocoder(ROOT / "models" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
                           device=device, log_floor=args.log_floor)
    results = {"device": device, "torch_version": torch.__version__, "log_floor": args.log_floor,
               "model_load": vocoder.load_diagnostics,
               "loaded_parameter_count": vocoder.loaded_parameter_count, "experiments": {}}
    modes = ("identity", "pitch_only", "time_only", "full") if args.experiments else ("full",)
    for mode in modes:
        logging.info("Rendering %s.", mode)
        beginning = time.perf_counter()
        pitch_path = args.output / ("target_pitch.npz" if mode == "full" else f"{mode}.target_pitch.npz")
        audio, diagnostics = render_student(
            sources["student"], teacher, student, plan, vocoder, mode=mode,
            change_timing=args.change_timing, pitch_mode=args.pitch_mode,
            map_teacher_pitch=args.map_teacher_pitch,
            allow_acoustic_fallback=args.allow_acoustic_fallback, target_pitch_path=pitch_path,
            teacher_audio=sources["teacher"],
        )
        peak = float(np.max(np.abs(audio)))
        gain = min(1.0, 0.98 / peak) if peak > 0 else 1.0
        audio *= gain
        filename = "student.corrected.wav" if mode == "full" else f"student.{mode}.wav"
        sf.write(args.output / filename, audio, vocoder.sample_rate, subtype="PCM_24")
        diagnostics.update(filename=filename, output_gain=gain,
                           elapsed_seconds=time.perf_counter() - beginning)
        results["experiments"][mode] = diagnostics
        write_json(args.output / "render.json", results)
    results["elapsed_seconds"] = time.perf_counter() - started
    if device == "cuda":
        results["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    write_json(args.output / "render.json", results)
    logging.info("Render artifacts saved.")


if __name__ == "__main__":
    main()
