from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
from typing import Callable, Iterable
import uuid

import numpy as np
import soundfile as sf

from inference.slicer.copyvocal_slicer import AudioChunk, downmix_safe, slice_audio

logger = logging.getLogger(__name__)
ANALYSIS_VERSION = "copyvocal-analysis-1.1-ja"
SAMPLE_RATE = 44100
SILENCE_TOKENS = {"SP", "AP", "EP", "sil", "br", "pau", "GS"}
ASR_PROMPT = (
    "Transcribe only the lyrics actually sung in this audio. Preserve repeated "
    "words and the original language. Do not complete missing lyrics, add "
    "punctuation, translate, or invent text during silence."
)


def atomic_json(path: str | Path, payload: dict | list) -> None:
    """Commit complete JSON files atomically, without NaN or Infinity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_analysis(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"source", "sample_rate", "sample_count", "duration", "words", "pitch", "notes"}
    if not isinstance(data, dict) or not required.issubset(data):
        raise ValueError(f"Invalid analysis JSON: {path}")
    return data


def _cached(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _check_cancel(checker) -> None:
    if checker and checker():
        raise InterruptedError("Analysis cancelled.")


class _FingerprintIndex:
    def __init__(self, directory: Path):
        self.path = directory / "fingerprints.json"
        self.entries = _cached(self.path) or {}

    def file(self, path: Path) -> str:
        path = path.resolve(strict=True)
        stat = path.stat()
        identity = f"{path}|{stat.st_size}|{stat.st_mtime_ns}|{stat.st_ctime_ns}"
        if identity not in self.entries:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            self.entries[identity] = digest.hexdigest()
        return self.entries[identity]

    def tree(self, path: Path) -> str:
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")
        files = sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        if not files:
            raise FileNotFoundError(f"Model directory is empty: {path}")
        return _key([(p.relative_to(path).as_posix(), self.file(p)) for p in files])

    def save(self) -> None:
        atomic_json(self.path, self.entries)


def _key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    from scipy.signal import resample_poly
    divisor = math.gcd(source_rate, target_rate)
    return np.ascontiguousarray(resample_poly(waveform, target_rate // divisor, source_rate // divisor), dtype=np.float32)


def _text(result) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text") or result.get("transcript") or "").strip()
    return str(getattr(result, "text", "") or "").strip()


def _japanese_resources(root: Path) -> dict[str, Path]:
    models = root / "models"
    paths = {"japanese_asr": models / "romajiASR", "japanese_g2p": models / "kashi-g2p-onnx"}
    required = [paths["japanese_asr"] / "model.onnx", paths["japanese_asr"] / "phoneme_vocab.json",
                paths["japanese_g2p"] / "model.onnx", paths["japanese_g2p"] / "resources" / "lexicon.txz",
                models / "tifa-1.0-onnx" / "dictionaries" / "japanese_dict_full.txt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Japanese analysis requires a project-local resource: {path}")
    for directory in paths.values():
        for path in [directory, *directory.rglob("*")]:
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Japanese model resources must be physical project-local copies: {path}")
            if path.is_file() and path.stat().st_nlink > 1:
                raise ValueError(f"Japanese model resources must not be hard links: {path}")
    return paths


def _japanese_moras(result) -> tuple[list[str], list[str]]:
    from inference.API.lfa_api import _normalize_asr_phoneme_token, _phoneme_tokens_to_romaji_moras
    raw = result.get("phonemes") if isinstance(result, dict) else None
    if not isinstance(raw, list):
        raise RuntimeError("Japanese romaji ASR did not return its phoneme token stream.")
    tokens = [normalized for token in raw if (normalized := _normalize_asr_phoneme_token(str(token)))]
    moras = [mora for mora in _phoneme_tokens_to_romaji_moras(tokens) if mora not in SILENCE_TOKENS]
    return tokens, moras


def _japanese_semantic_readings(text: str, pipeline) -> list[dict]:
    from inference.TiFA.aligner import _merge_mora_phonemes
    readings = []
    for index, word in enumerate(pipeline.convert(text, languages=["ja"])):
        candidates, seen = [], set()
        for reading in word.readings:
            for path in reading.paths:
                candidate = [{"kana": group.script, "phonemes": list(group.phonemes),
                              "romaji": _merge_mora_phonemes(list(group.phonemes))} for group in path]
                identity = _key(candidate)
                if identity not in seen:
                    candidates.append(candidate)
                    seen.add(identity)
        readings.append({"source_word_index": index, "surface": word.text, "candidates": candidates})
    return readings


def _run_japanese_pronunciation(tracks, model_path, g2p_path, alignment_path, device, cancel_checker, progress):
    from inference.romaji_asr.runtime import RomajiASROnnxModel
    model = None
    try:
        for track in tracks:
            for chunk in track.active:
                _check_cancel(cancel_checker)
                destination = track.wav(chunk).with_suffix(".asr.json")
                transcript = _cached(destination)
                if transcript is None or "mora_tokens" in transcript:
                    continue
                if model is None:
                    model = RomajiASROnnxModel.from_model_path(model_path, device=device)
                progress(f"Recognizing Japanese phonemes in {track.label} chunk {chunk.index + 1}/{len(track.chunks)}.")
                results = model.transcribe([str(track.wav(chunk))], batch_size=1)
                if len(results) != 1 or results[0] is None:
                    raise RuntimeError(f"Japanese romaji ASR failed for {track.label} chunk {chunk.index}.")
                tokens, moras = _japanese_moras(results[0])
                transcript.update({"alignment_text": " ".join(moras), "romaji_tokens": tokens,
                                   "mora_tokens": moras, "asr_engine": "qwen+romaji"})
                atomic_json(destination, transcript)
    finally:
        if model is not None:
            del model
        gc.collect()

    # Release the direct ASR session before the contextual G2P session starts.
    from inference.ja_onnx_g2p import get_ja_onnx_runtime
    from inference.TiFA.g2p_pipeline import build_g2p_pipeline
    pipeline = None
    for track in tracks:
        for chunk in track.active:
            _check_cancel(cancel_checker)
            destination = track.wav(chunk).with_suffix(".asr.json")
            transcript = _cached(destination)
            if transcript is None or "mora_tokens" not in transcript:
                continue
            if "semantic_readings" not in transcript:
                if transcript["text"]:
                    if pipeline is None:
                        if get_ja_onnx_runtime(g2p_path) is None:
                            raise RuntimeError(f"Japanese contextual G2P could not load its project-local bundle: {g2p_path}")
                        pipeline = build_g2p_pipeline(alignment_path, japanese_g2p_engine="kashi-g2p-onnx",
                                                      kashi_g2p_model_dir=g2p_path)
                    transcript["semantic_readings"] = _japanese_semantic_readings(transcript["text"], pipeline)
                else:
                    transcript["semantic_readings"] = []
                transcript["semantic_reading_basis"] = "qwen_text_g2p_unaligned"
                atomic_json(destination, transcript)
            track.wav(chunk).with_suffix(".txt").write_text(transcript["alignment_text"], encoding="utf-8")


def _quality(word: dict, waveform: np.ndarray, sample_rate: int) -> tuple[float, list[str]]:
    issues = []
    duration = word["end"] - word["start"]
    if duration < 0.035:
        issues.append("Word is shorter than 35 ms.")
    if duration > 5.0:
        issues.append("Word exceeds 5 seconds; check for missing or repeated lyrics.")
    begin = max(0, int(word["start"] * sample_rate))
    end = min(len(waveform), int(math.ceil(word["end"] * sample_rate)))
    energy = float(np.mean(np.square(waveform[begin:end].astype(np.float64)))) if end > begin else 0.0
    if energy < 1e-8:
        issues.append("Aligned word falls in near silence.")
    phones = word["phonemes"]
    if not phones:
        issues.append("Word has no valid aligned phonemes.")
    if any(phone["end"] - phone["start"] < 0.015 for phone in phones):
        issues.append("Word contains a phoneme shorter than 15 ms.")
    return max(0.0, 1.0 - 0.3 * len(issues)), issues


def _deduplicate_words(words: list[dict], warnings: list[str]) -> list[dict]:
    words = sorted(words, key=lambda word: (word["start"], word["end"]))
    output = []
    for word in words:
        if output:
            previous = output[-1]
            intersection = min(previous["end"], word["end"]) - max(previous["start"], word["start"])
            shorter = min(previous["end"] - previous["start"], word["end"] - word["start"])
            if word["text"] == previous["text"] and intersection > 0.45 * shorter:
                if word["quality"] > previous["quality"]:
                    output[-1] = word
                continue
            if intersection > 0.015:
                previous["quality"] = min(previous["quality"], 0.25)
                word["quality"] = min(word["quality"], 0.25)
                warnings.append(f"Chunk boundary word overlap at {word['start']:.3f}s requires review.")
        output.append(word)
    return output


def _write_textgrid(path: Path, analysis: dict) -> None:
    def quote(text):
        return str(text).replace('"', '""').replace("\n", " ")
    duration = analysis["duration"]
    tiers = [("words", analysis["words"]), ("phones", [phone for word in analysis["words"] for phone in word["phonemes"]])]
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0", f"xmax = {duration:.9f}",
             "tiers? <exists>", "size = 2", "item []:"]
    for index, (name, spans) in enumerate(tiers, 1):
        intervals = []
        cursor = 0.0
        for span in spans:
            start, end = max(0.0, float(span["start"])), min(duration, float(span["end"]))
            if end <= start or start < cursor - 1e-7:
                continue
            if start > cursor + 1e-7:
                intervals.append((cursor, start, ""))
            intervals.append((start, end, span["text"]))
            cursor = end
        if cursor < duration:
            intervals.append((cursor, duration, ""))
        lines.extend([f"    item [{index}]:", '        class = "IntervalTier"', f'        name = "{name}"',
                      "        xmin = 0", f"        xmax = {duration:.9f}", f"        intervals: size = {len(intervals)}"])
        for number, (start, end, text) in enumerate(intervals, 1):
            lines.extend([f"        intervals [{number}]:", f"            xmin = {start:.9f}",
                          f"            xmax = {end:.9f}", f'            text = "{quote(text)}"'])
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _Track:
    def __init__(self, label: str, source: Path, cache: Path, cache_key: str, config: dict):
        self.label, self.source = label, source
        audio, rate = sf.read(str(source), dtype="float32", always_2d=True)
        audio, self.warnings = downmix_safe(audio)
        if not len(audio):
            raise ValueError(f"Audio is empty: {source}")
        self.audio = _resample(audio, rate, SAMPLE_RATE)
        self.chunks = slice_audio(self.audio, SAMPLE_RATE, max_seconds=config["max_chunk_seconds"],
                                  min_seconds=config["min_chunk_seconds"], context_seconds=config["context_seconds"])
        self.cache = cache / cache_key
        self.cache.mkdir(parents=True, exist_ok=True)
        self.chunk_dir = self.cache / "chunks"
        self.chunk_dir.mkdir(exist_ok=True)
        self.active = []
        for chunk in self.chunks:
            wave = self.audio[chunk.start_sample:chunk.end_sample]
            energy = float(np.mean(np.square(wave.astype(np.float64))))
            if energy > 1e-10:
                self.active.append(chunk)
                path = self.chunk_dir / str(chunk.index) / f"chunk_{chunk.index}.wav"
                path.parent.mkdir(exist_ok=True)
                if not path.exists() or sf.info(str(path)).subtype != "PCM_16":
                    # Qwen's wave/pydub input path does not accept IEEE float WAV.
                    sf.write(str(path), wave, SAMPLE_RATE, subtype="PCM_16")
        if not self.active:
            self.warnings.append("Audio is near silence; lyric analysis was not attempted.")

    def wav(self, chunk: AudioChunk) -> Path:
        return self.chunk_dir / str(chunk.index) / f"chunk_{chunk.index}.wav"


def _run_asr(tracks: list[_Track], model: Path, device: str, language: str, cancel_checker, progress):
    from inference.API.asr_api import AsrSubprocessSession, QWEN_ASR_LANGUAGE_NAMES
    with AsrSubprocessSession(startup_timeout_sec=300, batch_timeout_sec=300) as session:
        for track in tracks:
            for chunk in track.active:
                _check_cancel(cancel_checker)
                destination = track.wav(chunk).with_suffix(".asr.json")
                cached = _cached(destination)
                if cached is None:
                    progress(f"Transcribing {track.label} chunk {chunk.index + 1}/{len(track.chunks)}.")
                    results = session.transcribe([str(track.wav(chunk))], model_path=str(model), device=device,
                        asr_lang=QWEN_ASR_LANGUAGE_NAMES.get(language, "Chinese"), asr_prompt=ASR_PROMPT,
                        asr_batch_size=1, cancel_checker=cancel_checker)
                    if len(results) != 1 or results[0] is None:
                        track.warnings.append(f"ASR failed for chunk {chunk.index}; retry analysis before correction.")
                        continue
                    cached = {"text": _text(results[0])}
                    atomic_json(destination, cached)
                track.wav(chunk).with_suffix(".txt").write_text(cached["text"], encoding="utf-8")


def _run_alignment(tracks: list[_Track], model_path: Path, device: str, language: str, cancel_checker, progress,
                   japanese_g2p_path: Path | None = None):
    from inference.TiFA.aligner import run_tifa_fa
    from inference.TiFA.runtime import load_tifa_model
    model = None
    try:
        for track in tracks:
            for chunk in track.active:
                _check_cancel(cancel_checker)
                destination = track.wav(chunk).with_suffix(".aligned.json")
                if _cached(destination) is not None:
                    continue
                transcript = _cached(track.wav(chunk).with_suffix(".asr.json"))
                if transcript is None:
                    continue
                alignment_text = transcript.get("alignment_text", transcript["text"])
                if not alignment_text:
                    atomic_json(destination, {"words": [], "warnings": [f"ASR returned no lyrics for chunk {chunk.index}."]})
                    continue
                if language == "ja" and "mora_tokens" not in transcript:
                    raise RuntimeError("Japanese alignment requires independent romaji ASR evidence.")
                track.wav(chunk).with_suffix(".txt").write_text(alignment_text, encoding="utf-8")
                if model is None:
                    model = load_tifa_model(model_path, device=device, use_cache=False)
                progress(f"Aligning {track.label} chunk {chunk.index + 1}/{len(track.chunks)}.")
                japanese_options = ({"japanese_g2p_engine": "kashi-g2p-onnx",
                                     "kashi_g2p_model_dir": japanese_g2p_path} if language == "ja" else {})
                predictions, display = run_tifa_fa(model, track.wav(chunk).parent, language=language,
                    cancel_checker=cancel_checker, preserve_gaps=True, raw_spans=True, **japanese_options)
                prediction = predictions.get(f"chunk_{chunk.index}")
                if prediction is None:
                    track.warnings.append(f"TiFA failed for chunk {chunk.index}; retry analysis before correction.")
                    continue
                raw_words = prediction[2]
                warnings = list(getattr(raw_words, "raw_alignment_issues", []))
                words = []
                mora_display = iter(display.get(f"chunk_{chunk.index}", []))
                for word in raw_words:
                    if word.text in SILENCE_TOKENS:
                        continue
                    display_pair = next(mora_display, (str(word.text), "")) if language == "ja" else None
                    start, end = word.start + chunk.start, word.end + chunk.start
                    midpoint = (start + end) / 2
                    if not chunk.core_start <= midpoint < chunk.core_end:
                        continue
                    value = {"text": str(word.text), "start": start, "end": end,
                             "phonemes": [{"text": str(phone.text), "start": phone.start + chunk.start,
                                           "end": phone.end + chunk.start} for phone in word.phonemes]}
                    if language == "ja":
                        from inference.API.lfa_api import _direct_moras_to_display_tokens
                        kana = str(display_pair[1])
                        if not kana or kana.isascii():
                            kana = _direct_moras_to_display_tokens("ja", "kana", [str(display_pair[0])])[0]
                        value.update({"language": "ja", "unit_kind": "mora", "identity_basis": "phonetic_mora",
                                      "romaji": str(display_pair[0]), "kana": kana,
                                      "chunk_index": chunk.index})
                    value["quality"], value["issues"] = _quality(value, track.audio, SAMPLE_RATE)
                    if warnings:
                        value["quality"] = min(value["quality"], 0.5)
                    words.append(value)
                if not words:
                    warnings.append(f"Chunk {chunk.index} has lyrics but no valid core word spans.")
                atomic_json(destination, {"words": words, "warnings": warnings})
    finally:
        if model is not None:
            model.sessions.clear()
            del model
        gc.collect()


def pitch_from_result(result, duration: float) -> dict:
    """Restore original UV after the upstream API's interpolated MIDI output."""
    midi = np.asarray(result.midi_pitch, dtype=np.float64).reshape(-1)
    if result.voiced_mask is None:
        raise RuntimeError("RMVPE did not provide an independent voiced mask.")
    voiced = np.asarray(result.voiced_mask, dtype=bool).reshape(-1)
    if len(voiced) != len(midi):
        raise RuntimeError("RMVPE returned different MIDI and voicing lengths.")
    times = np.arange(len(midi), dtype=np.float64) * result.time_step_seconds
    valid = times < duration
    times, midi, voiced = times[valid], midi[valid], voiced[valid]
    voiced &= np.isfinite(midi)
    f0 = np.zeros(len(midi), dtype=np.float64)
    f0[voiced] = 440.0 * np.power(2.0, (midi[voiced] - 69.0) / 12.0)
    return {"times": times.tolist(), "f0_hz": f0.tolist(), "voiced": voiced.tolist(),
            "time_step_seconds": float(result.time_step_seconds), "method": "RMVPE", "uv_interpolated": False}


def _run_pitch(tracks: list[_Track], model_path: Path, device: str, cancel_checker, progress, threshold: float):
    from inference.API.rmvpe_api import RmvpeTranscriber
    model = None
    try:
        for track in tracks:
            _check_cancel(cancel_checker)
            destination = track.pitch_cache
            if _cached(destination) is not None:
                continue
            if model is None:
                model = RmvpeTranscriber(model_path, device=device, batch_size=1, threshold=threshold)
            progress(f"Extracting continuous pitch from {track.label}.")
            result = model.infer(track.audio, SAMPLE_RATE, cancel_checker=cancel_checker)
            pitch = pitch_from_result(result, len(track.audio) / SAMPLE_RATE)
            atomic_json(destination, pitch)
    finally:
        if model is not None:
            model.shutdown()
            del model
        gc.collect()


def _run_notes(tracks: list[_Track], model_path: Path, device: str, language: str, cancel_checker, progress):
    from inference.API.game_api import extract_pitches_only, load_game_model
    model = None
    try:
        for track in tracks:
            for chunk in track.active:
                _check_cancel(cancel_checker)
                destination = track.wav(chunk).with_suffix(".notes.json")
                if _cached(destination) is not None:
                    continue
                if model is None:
                    model = load_game_model(str(model_path), device=device, use_cache=False)
                progress(f"Detecting notes in {track.label} chunk {chunk.index + 1}/{len(track.chunks)}.")
                waveform = track.audio[chunk.start_sample:chunk.end_sample]
                waveform = _resample(waveform, SAMPLE_RATE, model.samplerate)
                notes = extract_pitches_only([{"waveform": waveform, "offset": chunk.start}], model.samplerate,
                    model, ts=[0.1, 0.25, 0.5, 0.75, 1.0], seg_threshold=0.2, seg_radius=0.02,
                    est_threshold=0.2, batch_size=1, cancel_checker=cancel_checker, language=language)
                values = []
                for note in notes:
                    if not all(np.isfinite([note.onset, note.offset, note.pitch])):
                        continue
                    midpoint = (note.onset + note.offset) / 2
                    if chunk.core_start <= midpoint < chunk.core_end and note.offset > note.onset:
                        values.append({"start": max(0.0, float(note.onset)),
                                       "end": min(len(track.audio) / SAMPLE_RATE, float(note.offset)),
                                       "midi": float(note.pitch), "lyric": str(note.lyric), "method": "GAME"})
                atomic_json(destination, values)
    finally:
        if model is not None:
            model.shutdown()
            del model
        gc.collect()


def analyze_pair(
    teacher_path: str | Path,
    student_path: str | Path,
    output_dir: str | Path,
    root: str | Path,
    device: str = "dml",
    language: str = "zh",
    *,
    max_chunk_seconds: float = 18.0,
    min_chunk_seconds: float = 4.0,
    context_seconds: float = 0.75,
    include_game: bool = True,
    rmvpe_threshold: float = 0.03,
    stages: Iterable[str] | None = None,
    cache_dir: str | Path | None = None,
    cancel_checker: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, dict]:
    """Analyze both tracks independently, never copying teacher lyrics.

    ``stages`` selects lyrics/pitch/notes; Windows callers need a ``__main__`` guard."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    language = str(language or "zh").strip().lower()
    if language not in {"zh", "ja", "en", "yue"}:
        raise ValueError("Analysis language must be zh, ja, en, or yue.")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).resolve() if cache_dir else root / "cache" / "analysis"
    cache.mkdir(parents=True, exist_ok=True)
    selected = set(stages if stages is not None else ("lyrics", "pitch", "notes"))
    if not selected.issubset({"lyrics", "pitch", "notes"}):
        raise ValueError("Analysis stages must be lyrics, pitch, or notes.")
    if not include_game:
        selected.discard("notes")
    progress = progress or logger.info
    config = {"version": ANALYSIS_VERSION, "device": device, "language": language,
              "max_chunk_seconds": max_chunk_seconds, "min_chunk_seconds": min_chunk_seconds,
              "context_seconds": context_seconds, "include_game": include_game,
              "rmvpe_threshold": rmvpe_threshold, "asr_prompt": ASR_PROMPT}
    models = {"asr": root / "models" / "Qwen3-ASR-1.7B-dml", "alignment": root / "models" / "tifa-1.0-onnx",
              "pitch": root / "models" / "RMVPE", "notes": root / "models" / "GAME-1.0.3-medium-onnx"}
    if language == "ja":
        models.update(_japanese_resources(root))
        config.update({"japanese_asr_engine": "qwen+romaji", "japanese_g2p_engine": "kashi-g2p-onnx",
                       "japanese_identity_basis": "phonetic_mora"})
    index = _FingerprintIndex(cache)
    _check_cancel(cancel_checker)
    progress("Checking analysis input, model, and configuration fingerprints.")
    fingerprints = {name: index.tree(path) for name, path in models.items() if name != "notes" or include_game}
    code_files = [Path(__file__)] + [root / "inference" / relative for relative in (
        "slicer/copyvocal_slicer.py", "TiFA/aligner.py", "TiFA/runtime.py", "TiFA/decoding.py",
        "TiFA/g2p_pipeline.py", "API/asr_api.py", "API/rmvpe_api.py", "API/game_api.py", "device_utils.py")]
    if language == "ja":
        code_files.extend(root / "inference" / relative for relative in (
            "API/lfa_api.py", "romaji_asr/runtime.py", "romaji_asr/common.py", "ja_onnx_g2p.py",
            "TiFA/japanese_onnx.py", "TiFA/japanese_lexicon.py", "TiFA/g2p/converters/japanese.py",
            "kashi_g2p_ja/g2p_onnx_runtime.py", "HubertFA/tools/g2p.py", "LyricFA/tools/JaG2p.py"))
    fingerprints["code"] = _key([index.file(path) for path in code_files])
    pitch_code = _key({
        "functions": [inspect.getsource(function) for function in
                      (_run_pitch, pitch_from_result, _resample, downmix_safe, _Track.__init__)],
        "runtime_files": [index.file(root / "inference" / relative) for relative in
                          ("API/rmvpe_api.py", "device_utils.py")],
        "numpy": np.__version__, "soundfile": sf.__version__,
    })
    tracks = []
    for label, source in (("teacher", teacher_path), ("student", student_path)):
        source = Path(source).resolve(strict=True)
        input_hash = index.file(source)
        cache_key = _key({"input": input_hash, "models": fingerprints, "config": config})
        track = _Track(label, source, cache, cache_key, config)
        track.input_sha256 = input_hash
        pitch_key = _key({"input": input_hash, "model": fingerprints["pitch"], "code": pitch_code,
                          "device": device, "threshold": rmvpe_threshold, "sample_rate": SAMPLE_RATE})
        track.pitch_cache = cache / "pitch" / pitch_key / "pitch.json"
        tracks.append(track)
    index.save()
    if "lyrics" in selected:
        _run_asr(tracks, models["asr"], device, language, cancel_checker, progress)
        if language == "ja":
            _run_japanese_pronunciation(tracks, models["japanese_asr"], models["japanese_g2p"],
                                       models["alignment"], device, cancel_checker, progress)
        _run_alignment(tracks, models["alignment"], device, language, cancel_checker, progress,
                       japanese_g2p_path=models.get("japanese_g2p"))
    if "pitch" in selected:
        _run_pitch(tracks, models["pitch"], device, cancel_checker, progress, rmvpe_threshold)
    if "notes" in selected:
        _run_notes(tracks, models["notes"], device, language, cancel_checker, progress)
    outputs = []
    for track in tracks:
        words, chunks, notes = [], [], []
        lyrics_complete = True
        notes_complete = include_game
        for chunk in track.chunks:
            transcript = _cached(track.wav(chunk).with_suffix(".asr.json")) if chunk in track.active else {"text": ""}
            aligned = _cached(track.wav(chunk).with_suffix(".aligned.json")) if chunk in track.active else {"words": [], "warnings": []}
            chunk_notes = _cached(track.wav(chunk).with_suffix(".notes.json")) if chunk in track.active else []
            lyrics_complete &= aligned is not None
            notes_complete &= chunk_notes is not None
            if aligned is not None:
                words.extend(aligned["words"])
                track.warnings.extend(aligned["warnings"])
            if chunk_notes is not None:
                notes.extend(chunk_notes)
            chunk_value = {**chunk.to_dict(), "text": transcript["text"] if transcript else "",
                           "alignment_complete": aligned is not None}
            if language == "ja" and transcript:
                chunk_value.update({key: transcript[key] for key in
                    ("alignment_text", "romaji_tokens", "mora_tokens", "asr_engine", "semantic_readings",
                     "semantic_reading_basis") if key in transcript})
            chunks.append(chunk_value)
        words = _deduplicate_words(words, track.warnings)
        pitch = _cached(track.pitch_cache)
        completed = []
        if lyrics_complete:
            completed.append("lyrics")
        if pitch is not None:
            completed.append("pitch")
        if notes_complete:
            completed.append("notes")
        if not words:
            track.warnings.append("No reliable lyric spans are available; lyric correction must not be reported as successful.")
        missing = {"lyrics", "pitch"} - set(completed)
        if missing:
            track.warnings.append("Incomplete analysis stages: " + ", ".join(sorted(missing)) + ".")
        analysis = {"source": str(track.source), "sample_rate": SAMPLE_RATE, "sample_count": len(track.audio),
                    "duration": len(track.audio) / SAMPLE_RATE, "words": words, "chunks": chunks,
                    "pitch": pitch or {"times": [], "f0_hz": [], "voiced": []},
                    "notes": sorted(notes, key=lambda note: note["start"]), "warnings": list(dict.fromkeys(track.warnings)),
                    "completed_stages": completed, "cache_key": track.cache.name, "config": config,
                    "input_sha256": track.input_sha256,
                    "provenance": {"input_sha256": track.input_sha256, "models": fingerprints,
                                   "config_sha256": _key(config), "analysis_version": ANALYSIS_VERSION},
                    "quality_definition": "Structural and energy checks; not a calibrated model confidence.",
                    "model_fingerprints": fingerprints}
        atomic_json(output_dir / f"{track.label}.analysis.json", analysis)
        _write_textgrid(output_dir / f"{track.label}.TextGrid", analysis)
        outputs.append(analysis)
    return outputs[0], outputs[1]
