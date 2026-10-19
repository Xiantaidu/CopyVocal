from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from application.artifacts import ROOT
from application.languages import SUPPORTED_LANGUAGES
from gui.i18n import tr

SETTINGS_PATH = ROOT / "settings" / "default.json"
PREFS_PATH = ROOT / "settings" / "gui_prefs.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "gui"
CONFIG_FILENAME = "gui_settings.json"
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma")
ONNX_DEVICES = ("dml", "cpu")
PITCH_MODES = ("off", "teacher", "adaptive")
VOCODER_DEVICES = ("auto", "cuda", "cpu")
LANGUAGES = SUPPORTED_LANGUAGES
REQUIRED_ANALYSIS_FILES = ("teacher.analysis.json", "student.analysis.json")


def _number(value: float) -> str:
    return f"{float(value):g}"


def _choice(value, allowed: tuple, fallback: str) -> str:
    return value if value in allowed else fallback


def suggested_output(student: str | Path) -> Path:
    stem = Path(student).stem if str(student).strip() else "job"
    return ROOT / "outputs" / f"{stem}.corrected"


@dataclass
class JobRequest:
    """One correction run, in the command-line pipeline's own vocabulary."""

    teacher: str = ""
    student: str = ""
    output: str = ""
    config: str = ""
    onnx_device: str = "dml"
    vocoder_device: str = "auto"
    language: str = "zh"
    max_chunk_seconds: float = 18.0
    context_seconds: float = 0.75
    minimum_alignment_quality: float = 0.35
    maximum_stretch: float = 4.0
    log_floor: float = 1e-5
    include_game: bool = True
    allow_acoustic_fallback: bool = False
    change_timing: bool = True
    pitch_mode: str = "off"
    map_teacher_pitch: bool = False
    experiments: bool = False
    verify: bool = True
    verify_timing: bool = False
    analyze_only: bool = False
    skip_analysis: bool = False

    @classmethod
    def from_defaults(cls) -> "JobRequest":
        """Start from settings/default.json so both entry points agree."""
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        request = cls()
        request.output = str(DEFAULT_OUTPUT)
        request.onnx_device = _choice(settings.get("onnx_device"), ONNX_DEVICES, request.onnx_device)
        request.vocoder_device = _choice(settings.get("vocoder_device"), VOCODER_DEVICES, request.vocoder_device)
        request.language = _choice(settings.get("language"), LANGUAGES, request.language)
        for name, key in (("max_chunk_seconds", "max_chunk_seconds"), ("context_seconds", "context_seconds"),
                          ("minimum_alignment_quality", "minimum_alignment_quality"),
                          ("maximum_stretch", "maximum_stretch"), ("log_floor", "vocoder_log_floor")):
            try:
                setattr(request, name, float(settings[key]))
            except (KeyError, TypeError, ValueError):
                continue
        if isinstance(settings.get("allow_acoustic_fallback"), bool):
            request.allow_acoustic_fallback = settings["allow_acoustic_fallback"]
        if isinstance(settings.get("change_timing"), bool):
            request.change_timing = settings["change_timing"]
        if isinstance(settings.get("map_teacher_pitch"), bool):
            request.map_teacher_pitch = settings["map_teacher_pitch"]
        request.pitch_mode = _choice(settings.get("pitch_mode"), PITCH_MODES, request.pitch_mode)
        return request

    @property
    def teacher_path(self) -> Path:
        return Path(self.teacher)

    @property
    def student_path(self) -> Path:
        return Path(self.student)

    @property
    def output_path(self) -> Path:
        return Path(self.output)

    def config_path(self) -> Path:
        return self.output_path / CONFIG_FILENAME

    def validate(self) -> list[str]:
        """Return human-readable problems that must be fixed before a run."""
        problems: list[str] = []
        for label, value in ((tr("label_teacher_short"), self.teacher),
                             (tr("label_student_short"), self.student)):
            text = value.strip()
            if not text:
                problems.append(tr("validate_choose_recording").format(label=label))
            elif not Path(text).is_file():
                problems.append(tr("validate_file_not_found").format(label=label, path=text))
        if not problems:
            if self.teacher_path.resolve() == self.student_path.resolve():
                problems.append(tr("validate_same_file"))
        if not self.output.strip():
            problems.append(tr("validate_choose_output"))
        elif self.output_path.exists() and not self.output_path.is_dir():
            problems.append(tr("validate_output_is_file").format(path=self.output))
        if self.analyze_only and self.skip_analysis:
            problems.append(tr("validate_conflict"))
        if self.skip_analysis:
            missing = [name for name in REQUIRED_ANALYSIS_FILES if not (self.output_path / name).is_file()]
            if missing:
                problems.append(tr("validate_missing_analysis").format(
                    files="、".join(missing) + f"（位于 {self.output}）"))
        for name, value, low, high in ((tr("advanced_max_chunk"), self.max_chunk_seconds, 0.0, 600.0),
                                       (tr("advanced_context"), self.context_seconds, 0.0, 10.0),
                                       (tr("advanced_quality"), self.minimum_alignment_quality, 0.0, 1.0),
                                       (tr("advanced_stretch"), self.maximum_stretch, 1.0, 100.0),
                                       (tr("advanced_floor"), self.log_floor, 0.0, 1.0)):
            if not low < float(value) <= high:
                problems.append(tr("validate_range").format(name=name, low=low, high=high))
        return problems

    def effective_config(self) -> dict:
        return {"sample_rate": 44100, "onnx_device": self.onnx_device,
                "vocoder_device": self.vocoder_device, "language": self.language,
                "max_chunk_seconds": self.max_chunk_seconds, "context_seconds": self.context_seconds,
                "minimum_alignment_quality": self.minimum_alignment_quality,
                "maximum_stretch": self.maximum_stretch,
                "allow_acoustic_fallback": self.allow_acoustic_fallback,
                "change_timing": self.change_timing,
                "pitch_mode": self.pitch_mode,
                "map_teacher_pitch": self.map_teacher_pitch,
                "vocoder_log_floor": self.log_floor}

    def write_config(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.effective_config(), indent=2) + "\n", encoding="utf-8")

    def argv(self) -> list[str]:
        """Command-line arguments for scripts/copyvocal_cli.py."""
        arguments = ["--config", str(self.config_path()),
                     "--teacher", self.teacher.strip(), "--student", self.student.strip(),
                     "--output", self.output,
                     "--onnx-device", self.onnx_device, "--vocoder-device", self.vocoder_device,
                     "--language", self.language,
                     "--max-chunk-seconds", _number(self.max_chunk_seconds),
                     "--context-seconds", _number(self.context_seconds),
                     "--minimum-alignment-quality", _number(self.minimum_alignment_quality),
                     "--max-stretch", _number(self.maximum_stretch),
                     "--log-floor", _number(self.log_floor)]
        if not self.include_game:
            arguments.append("--no-game")
        if self.allow_acoustic_fallback:
            arguments.append("--allow-acoustic-fallback")
        # Pitch and timing switches are always stated explicitly, so unchecking an
        # option cannot silently leave the settings file's value in force.
        arguments.append("--change-timing" if self.change_timing else "--no-change-timing")
        arguments.extend(["--pitch-mode", self.pitch_mode])
        arguments.append("--map-teacher-pitch" if self.map_teacher_pitch else "--no-map-teacher-pitch")
        if self.experiments:
            arguments.append("--experiments")
        if not self.verify:
            arguments.append("--no-verify")
        if self.verify_timing:
            arguments.append("--verify-timing")
        if self.analyze_only:
            arguments.append("--analyze-only")
        if self.skip_analysis:
            arguments.append("--skip-analysis")
        return arguments

    def to_prefs(self) -> dict:
        values = asdict(self)
        values.pop("config", None)
        return values

    @classmethod
    def from_prefs(cls, values: dict, fallback: "JobRequest") -> "JobRequest":
        """Restore a request, ignoring unknown keys and unusable values."""
        request = cls(**{field.name: getattr(fallback, field.name) for field in fields(cls)})
        for field in fields(cls):
            if field.name not in values or field.name == "config":
                continue
            value = values[field.name]
            if isinstance(getattr(request, field.name), bool):
                if isinstance(value, bool):
                    setattr(request, field.name, value)
            elif isinstance(getattr(request, field.name), (int, float)) and not isinstance(
                    getattr(request, field.name), bool):
                try:
                    setattr(request, field.name, float(value))
                except (TypeError, ValueError):
                    continue
            elif isinstance(value, str):
                setattr(request, field.name, value)
        for name, allowed in (("onnx_device", ONNX_DEVICES), ("vocoder_device", VOCODER_DEVICES),
                              ("language", LANGUAGES), ("pitch_mode", PITCH_MODES)):
            setattr(request, name, _choice(getattr(request, name), allowed, getattr(fallback, name)))
        # A preferences file from the build that had a 'change pitch' checkbox and no
        # pitch switch still says what its author wanted: apply the equivalent mode
        # rather than silently dropping the request back to "leave the pitch alone".
        if "pitch_mode" not in values and isinstance(values.get("change_pitch"), bool):
            request.pitch_mode = "teacher" if values["change_pitch"] else "off"
        if not request.output:
            request.output = fallback.output
        return request
