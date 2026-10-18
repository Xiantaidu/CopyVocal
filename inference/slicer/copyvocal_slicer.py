"""Deterministic, bounded audio chunks with context and disjoint ownership."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_sample: int
    end_sample: int
    core_start_sample: int
    core_end_sample: int
    sample_rate: int

    @property
    def start(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end(self) -> float:
        return self.end_sample / self.sample_rate

    @property
    def core_start(self) -> float:
        return self.core_start_sample / self.sample_rate

    @property
    def core_end(self) -> float:
        return self.core_end_sample / self.sample_rate

    def to_dict(self) -> dict:
        return {**asdict(self), "start": self.start, "end": self.end,
                "core_start": self.core_start, "core_end": self.core_end}


def downmix_safe(audio: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Preserve the strongest channel when a mean would cancel vocal energy."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim not in (1, 2):
        raise ValueError("Audio must have shape [samples] or [samples, channels].")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains non-finite samples.")
    if audio.ndim == 1:
        return np.ascontiguousarray(audio), []
    if audio.shape[1] == 0:
        raise ValueError("Audio has no channels.")
    energy = np.mean(np.square(audio.astype(np.float64)), axis=0) if len(audio) else np.zeros(audio.shape[1])
    loudest = int(np.argmax(energy))
    mono = audio.mean(axis=1)
    mean_energy = float(np.mean(np.square(mono.astype(np.float64)))) if len(mono) else 0.0
    if energy[loudest] > 1e-12 and mean_energy < 0.0625 * energy[loudest]:
        return np.ascontiguousarray(audio[:, loudest]), [
            f"Channel averaging caused phase cancellation; selected channel {loudest}."
        ]
    return np.ascontiguousarray(mono), []


def slice_audio(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    max_seconds: float = 18.0,
    min_seconds: float = 4.0,
    context_seconds: float = 0.75,
    minimum_silence: float = 0.12,
) -> list[AudioChunk]:
    """Cover every sample once in cores; use silence for cuts when available.

    Context is included in the hard duration bound. Core midpoint ownership
    removes duplicate words and notes after inference on overlapping chunks.
    """
    if sample_rate <= 0 or max_seconds <= 0 or min_seconds < 0 or context_seconds < 0:
        raise ValueError("Invalid audio slicing parameters.")
    if max_seconds <= 2 * context_seconds or min_seconds > max_seconds - 2 * context_seconds:
        raise ValueError("Chunk duration must leave room for context and a core.")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("slice_audio requires a mono waveform.")
    count = len(waveform)
    if not count:
        return []
    context = int(round(context_seconds * sample_rate))
    maximum = max(1, int(np.floor(max_seconds * sample_rate)))
    core_max = maximum - 2 * context
    core_min = max(1, int(round(min_seconds * sample_rate)))
    hop = max(1, int(round(sample_rate * 0.01)))
    frame = max(hop, int(round(sample_rate * 0.02)))
    squared = np.square(waveform.astype(np.float64))
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, count, hop)
    ends = np.minimum(starts + frame, count)
    rms = np.sqrt((cumulative[ends] - cumulative[starts]) / np.maximum(1, ends - starts))
    peak_db = float(20 * np.log10(max(float(np.max(rms)), 1e-12)))
    threshold_db = max(-60.0, min(-38.0, peak_db - 28.0))
    silent = rms <= 10 ** (threshold_db / 20)
    changes = np.diff(np.concatenate(([False], silent, [False])).astype(np.int8))
    quiet = []
    for begin, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        left = int(starts[begin])
        right = min(count, int(starts[min(end, len(starts) - 1)]))
        if (right - left) / sample_rate >= minimum_silence:
            quiet.append((left, right))

    cuts = [0]
    while count - cuts[-1] > core_max:
        lower = cuts[-1] + core_min
        upper = cuts[-1] + core_max
        candidates = [min(upper, (left + right) // 2) for left, right in quiet
                      if right >= lower and left <= upper and min(upper, (left + right) // 2) >= lower]
        cut = max(candidates) if candidates else upper
        if count - cut < core_min and count - cuts[-1] <= 2 * core_max:
            cut = min(upper, cuts[-1] + (count - cuts[-1]) // 2)
        cuts.append(cut)
    cuts.append(count)
    return [AudioChunk(index, max(0, begin - context), min(count, end + context), begin, end, sample_rate)
            for index, (begin, end) in enumerate(zip(cuts, cuts[1:]))]
