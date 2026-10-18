from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import librosa
import numpy as np
import soundfile as sf
import torch

from inference.correction import sample_pitch
from inference.vocoder import PcNsfVocoder


def measure_pitch(audio, sr, target_times, target_f0):
    """Use independent YIN with wide bounds to reveal residual/octave errors."""
    measured = librosa.yin(audio, sr=sr, fmin=50, fmax=1600, frame_length=2048,
                           hop_length=256, trough_threshold=0.1, center=False)
    times = (np.arange(len(measured)) * 256 + 1024) / sr
    reference, voiced = sample_pitch({"times": target_times, "f0_hz": target_f0, "voiced": np.asarray(target_f0) > 0}, times)
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=256, center=False)[0]
    valid = voiced & (rms > max(0.005, np.max(rms) * 0.1)) & (times > 0.04) & (times < len(audio) / sr - 0.04)
    if not valid.any():
        return {"valid_frame_count": 0}
    error = 1200 * np.log2(measured[valid] / reference[valid])
    return {"valid_frame_count": int(valid.sum()), "median_absolute_error_cents": float(np.median(np.abs(error))),
            "median_signed_error_cents": float(np.median(error)),
            "p95_absolute_error_cents": float(np.percentile(np.abs(error), 95)),
            "gross_error_ratio_over_100_cents": float(np.mean(np.abs(error) > 100)),
            "measured_median_hz": float(np.median(measured[valid])),
            "target_median_hz": float(np.median(reference[valid]))}


def select_vowel(analysis):
    pitch = analysis["pitch"]
    best = None
    vowels = {"a", "o", "e", "i", "u", "v", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "ong", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng", "ve", "van", "vn"}
    for word in analysis["words"]:
        for phone in word.get("phonemes", []):
            if phone["text"].lower() not in vowels or phone["end"] - phone["start"] < 0.3:
                continue
            start, end = phone["start"] + 0.025, phone["end"] - 0.025
            center = (start + end) / 2
            start, end = max(start, center - 0.18), min(end, center + 0.18)
            times = np.linspace(start, end, 48)
            f0, voiced = sample_pitch(pitch, times)
            if np.mean(voiced) < 0.95:
                continue
            cents = 1200 * np.log2(f0[voiced])
            score = float(np.median(np.abs(cents - np.median(cents))))
            if best is None or score < best[0]:
                best = (score, start, end, phone["text"])
    if best is None:
        raise RuntimeError("No suitable voiced vowel segment was found for pitch stress checks.")
    return {"start": best[1], "end": best[2], "phoneme": best[3], "pitch_mad_cents": best[0]}


def main():
    parser = argparse.ArgumentParser(description="Reproduce short PC-NSF reconstruction and pitch-control stress checks.")
    parser.add_argument("--source", type=Path, default=Path("Student.wav"))
    parser.add_argument("--analysis", type=Path, default=Path("outputs/short/student.analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vocoder_smoke"))
    parser.add_argument("--model", type=Path, default=Path("models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    audio, sr = sf.read(args.source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 44100:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        sr = 44100
    if len(audio) != analysis["sample_count"]:
        raise ValueError("Analysis and source audio lengths differ.")
    vocoder = PcNsfVocoder(args.model, device=args.device)
    report = {"source": str(args.source.resolve()), "torch_version": torch.__version__,
              "device": str(vocoder.device), "sample_rate": sr,
              "loaded_state_tensor_count": vocoder.loaded_parameter_count,
              "strict_load": vocoder.load_diagnostics, "identity": {}, "stress": {},
              "listening_assessment": "Not performed. Objective diagnostics do not certify listening quality."}
    identity_waves = {}
    for floor in (1e-5, 1e-9):
        vocoder.log_floor = floor
        mel = vocoder.mel(audio)
        times = vocoder.frame_times(mel.shape[1])
        f0, _ = sample_pitch(analysis["pitch"], times)
        waveform = vocoder.synthesize(mel, f0, sample_count=len(audio))
        label = "1e-5" if floor == 1e-5 else "1e-9"
        sf.write(args.output / ("student.identity.wav" if floor == 1e-5 else "student.identity.floor_1e-9.wav"), waveform, sr, subtype="FLOAT")
        identity_waves[label] = waveform
        measured = measure_pitch(waveform, sr, times, f0)
        report["identity"][label] = {"finite": bool(np.isfinite(waveform).all()), "sample_count": len(waveform),
                                    "expected_sample_count": len(audio), "peak": float(np.max(np.abs(waveform))),
                                    "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))), "yin": measured}
        print(f"Completed identity floor={label}", flush=True)
    difference = identity_waves["1e-5"] - identity_waves["1e-9"]
    report["floor_comparison"] = {"difference_rms": float(np.sqrt(np.mean(difference.astype(np.float64) ** 2))),
                                  "difference_peak": float(np.max(np.abs(difference))),
                                  "normalized_difference_rms": float(np.sqrt(np.mean(difference.astype(np.float64) ** 2)) / max(1e-12, np.sqrt(np.mean(identity_waves["1e-5"].astype(np.float64) ** 2))))}
    vowel = select_vowel(analysis)
    report["stress_segment"] = vowel
    first, last = round(vowel["start"] * sr), round(vowel["end"] * sr)
    segment = audio[first:last]
    sf.write(args.output / "vowel.original.wav", segment, sr, subtype="FLOAT")
    vocoder.log_floor = 1e-5
    mel = vocoder.mel(segment)
    times = vocoder.frame_times(mel.shape[1])
    original_f0, voiced = sample_pitch(analysis["pitch"], times + first / sr)
    for shift in (-12, -5, -2, 0, 2, 5, 12):
        target = original_f0 * 2 ** (shift / 12)
        waveform = vocoder.synthesize(mel, target, sample_count=len(segment))
        sf.write(args.output / f"vowel.shift_{shift:+d}.wav", waveform, sr, subtype="FLOAT")
        report["stress"][f"{shift:+d}"] = {"finite": bool(np.isfinite(waveform).all()), "sample_count": len(waveform),
                                            "peak": float(np.max(np.abs(waveform))),
                                            "yin": measure_pitch(waveform, sr, times, target)}
        print(f"Completed vowel shift={shift:+d}", flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    (args.output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
