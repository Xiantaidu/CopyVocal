from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np


class PcNsfVocoder:
    """A 44.1 kHz, 128-bin, natural-log mel PC-NSF-HiFiGAN adapter.

    Audio is mono float32; mel arrays are [128, frames] with frame centers (index + 0.5) * hop / rate."""

    def __init__(self, model_dir: str | Path, device: str = "cuda", log_floor: float = 1e-5):
        import torch
        from .generator import Generator

        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available; use device='cpu'.")
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8-sig"))
        self.sample_rate = int(config["sampling_rate"])
        self.hop_size = int(config["hop_size"])
        self.n_fft = int(config["n_fft"])
        self.win_size = int(config["win_size"])
        self.n_mels = int(config["num_mels"])
        self.fmin, self.fmax = float(config["fmin"]), float(config["fmax"])
        self.log_floor = float(log_floor)
        if (self.sample_rate, self.hop_size, self.n_fft, self.win_size, self.n_mels) != (44100, 512, 2048, 2048, 128):
            raise ValueError("This adapter requires the 44.1 kHz / hop 512 / 128-bin PC-NSF model.")
        if int(np.prod(config["upsample_rates"])) != self.hop_size:
            raise ValueError("Vocoder upsample rates do not match hop_size.")
        if not np.isfinite(self.log_floor) or self.log_floor <= 0:
            raise ValueError("log_floor must be positive and finite.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*", category=FutureWarning)
            self.model = Generator(SimpleNamespace(**config))
        checkpoint = torch.load(self.model_dir / "model.ckpt", map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
            raise ValueError("Checkpoint must contain a generator state dictionary.")
        # Loading strictly is essential: legacy weight_g/weight_v must not be ignored.
        result = self.model.load_state_dict(checkpoint["generator"], strict=True)
        self.loaded_parameter_count = len(checkpoint["generator"])
        self.load_diagnostics = {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}
        self.model.eval()
        self.model.remove_weight_norm()
        self.model.to(self.device)
        self._mel_basis = None
        self._window = None

    def frame_times(self, frame_count: int) -> np.ndarray:
        return (np.arange(frame_count, dtype=np.float64) + 0.5) * self.hop_size / self.sample_rate

    def mel(self, audio: np.ndarray) -> np.ndarray:
        """Extract Slaney-normalized mel amplitudes followed by natural log.

        Follows openvpi/SingingVocoders wav2mel.py: reflect pad, Hann, center=False."""
        import torch
        from librosa.filters import mel as mel_filter

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Audio must be a nonempty finite mono array.")
        frame_count = math.ceil(len(audio) / self.hop_size)
        pad = (self.win_size - self.hop_size) // 2
        tail = frame_count * self.hop_size - len(audio)
        # Extend only the final reflection so complete training-contract frames
        # stay identical even when the last audio hop is incomplete.
        padded = np.pad(audio, (pad, pad + tail), mode="reflect" if len(audio) > 1 else "edge")
        waveform = torch.from_numpy(padded).unsqueeze(0)
        if self._mel_basis is None:
            self._mel_basis = torch.from_numpy(mel_filter(
                sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels,
                fmin=self.fmin, fmax=self.fmax, htk=False, norm="slaney",
            )).float()
            self._window = torch.hann_window(self.win_size)
        # Bound peak STFT storage even for hour-long inputs; each block shares
        # the already reflected waveform and therefore has identical edges.
        blocks = []
        with torch.inference_mode():
            for first in range(0, frame_count, 2048):
                last = min(frame_count, first + 2048)
                section = waveform[:, first * self.hop_size:(last - 1) * self.hop_size + self.n_fft]
                spectrum = torch.stft(
                    section, self.n_fft, hop_length=self.hop_size,
                    win_length=self.win_size, window=self._window, center=False,
                    normalized=False, onesided=True, return_complex=True,
                ).abs()
                blocks.append(torch.log(torch.clamp(self._mel_basis @ spectrum[0], min=self.log_floor)).numpy())
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    def _phase_prefix(self, f0: np.ndarray) -> np.ndarray:
        if not self.model.mini_nsf:
            return np.zeros(len(f0) + 1, dtype=np.float64)
        delta = np.diff(f0.astype(np.float64), append=float(f0[-1]))
        increments = f0.astype(np.float64) * self.model.upp / self.model.source_sr
        increments += 0.5 * delta * (self.model.upp - 1) / self.model.source_sr
        return np.concatenate(([0.0], np.cumsum(increments))) % 1.0

    def synthesize(
        self, mel: np.ndarray, f0_hz: np.ndarray, sample_count: int | None = None,
        *, chunk_frames: int = 768, context_frames: int = 32, overlap_frames: int = 16,
    ) -> np.ndarray:
        import torch

        mel = np.asarray(mel, dtype=np.float32)
        f0 = np.asarray(f0_hz, dtype=np.float32).reshape(-1)
        if mel.ndim != 2 or mel.shape[0] != self.n_mels or mel.shape[1] != len(f0):
            raise ValueError("Expected mel [128, T] and F0 [T] with the same T.")
        if not len(f0) or not np.isfinite(mel).all() or not np.isfinite(f0).all() or np.any(f0 < 0):
            raise ValueError("Mel and F0 must be nonempty, finite, and F0 nonnegative.")
        if chunk_frames < 2 or context_frames < 1 or not 0 <= overlap_frames < chunk_frames:
            raise ValueError("Invalid chunk/context/overlap frame counts.")
        sample_count = len(f0) * self.hop_size if sample_count is None else int(sample_count)
        if not 0 <= sample_count <= len(f0) * self.hop_size:
            raise ValueError("sample_count must fit the supplied mel frame count.")
        output = np.zeros(len(f0) * self.hop_size, dtype=np.float32)
        weights = np.zeros_like(output)
        phases = self._phase_prefix(f0)
        stride = chunk_frames - overlap_frames
        with torch.inference_mode():
            for first in range(0, len(f0), stride):
                last = min(first + chunk_frames, len(f0))
                left, right = max(0, first - context_frames), min(len(f0), last + context_frames)
                mel_tensor = torch.from_numpy(np.ascontiguousarray(mel[:, left:right])).unsqueeze(0).to(self.device)
                f0_tensor = torch.from_numpy(np.ascontiguousarray(f0[left:right])).unsqueeze(0).to(self.device)
                waveform = self.model(mel_tensor, f0_tensor, phase=float(phases[left]))[0, 0]
                start = (first - left) * self.hop_size
                count = (last - first) * self.hop_size
                section = waveform[start:start + count].float().cpu().numpy()
                if len(section) != count:
                    raise RuntimeError("Generator output does not match the model hop_size.")
                weight = np.ones(count, dtype=np.float32)
                fade = min(overlap_frames * self.hop_size, count)
                if first > 0 and fade:
                    weight[:fade] = np.sin(np.linspace(0, np.pi / 2, fade, dtype=np.float64)) ** 2
                if last < len(f0) and fade:
                    weight[-fade:] = np.cos(np.linspace(0, np.pi / 2, fade, dtype=np.float64)) ** 2
                a, b = first * self.hop_size, last * self.hop_size
                output[a:b] += section * weight
                weights[a:b] += weight
                if last == len(f0):
                    break
        output /= np.maximum(weights, 1e-8)
        if not np.isfinite(output).all():
            raise RuntimeError("Vocoder produced nonfinite output.")
        return output[:sample_count]

