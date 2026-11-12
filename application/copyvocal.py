from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

from application.artifacts import ROOT, code_fingerprint, file_hash, read_json, write_json
from application.languages import SUPPORTED_LANGUAGES


# Each worker runs in its own process so a GPU backend's contexts are released
# before the next backend starts.
def run_worker(script: str, arguments: list[str], output: Path) -> None:
    command = [sys.executable, "-X", "utf8", "-u", str(ROOT / "scripts" / script), *arguments]
    environment = os.environ.copy()
    environment["V2M_FFMPEG"] = str(ROOT / "ffmpeg.exe")
    environment["PATH"] = str(ROOT) + os.pathsep + environment.get("PATH", "")
    logging.info("Starting %s.", script)
    with (output / "processing.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except KeyboardInterrupt:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
            else:
                process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"{script} failed with exit code {code}. See {output / 'processing.log'}.")


def run_job(teacher: Path, student: Path, output: Path, *, onnx_device: str = "dml",
            vocoder_device: str = "auto", language: str = "zh", experiments: bool = False,
            analyze_only: bool = False, skip_analysis: bool = False, include_game: bool = True,
            verify: bool = True, max_chunk_seconds: float = 18.0,
            allow_acoustic_fallback: bool = False, log_floor: float = 1e-5,
            context_seconds: float = 0.75, min_quality: float = 0.35,
            max_stretch: float = 4.0, verify_timing: bool = False,
            change_timing: bool = True, pitch_mode: str = "off",
            map_teacher_pitch: bool = False, export_copyvocal_file: bool = False) -> dict:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Choose from {', '.join(SUPPORTED_LANGUAGES)}.")
    teacher, student, output = teacher.resolve(), student.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    job = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
           "teacher": str(teacher), "student": str(student), "output": str(output),
           "input_sha256": {},
           "config": {"onnx_device": onnx_device, "vocoder_device": vocoder_device, "language": language,
                      "max_chunk_seconds": max_chunk_seconds, "context_seconds": context_seconds,
                      "minimum_alignment_quality": min_quality, "maximum_stretch": max_stretch,
                      "include_game": include_game,
                      "allow_acoustic_fallback": allow_acoustic_fallback, "log_floor": log_floor,
                      "experiments": experiments, "verify": verify, "verify_timing": verify_timing,
                      "change_timing": change_timing, "pitch_mode": pitch_mode,
                      "map_teacher_pitch": map_teacher_pitch,
                      "export_copyvocal_file": export_copyvocal_file,
                      "skip_analysis": skip_analysis},
           "stages": {}}
    write_json(output / "job.json", job)
    try:
        from application.report import write_status_report
        write_status_report(output, job)
        teacher, student = teacher.resolve(strict=True), student.resolve(strict=True)
        job["input_sha256"] = {"teacher": file_hash(teacher), "student": file_hash(student)}
        job["code_sha256"] = code_fingerprint()
        job["resource_manifest_sha256"] = file_hash(ROOT / "settings" / "resource_manifest.json")
        write_json(output / "job.json", job)
        if not skip_analysis:
            arguments = ["--teacher", str(teacher), "--student", str(student), "--output", str(output),
                         "--device", onnx_device, "--language", language,
                         "--max-chunk-seconds", str(max_chunk_seconds),
                         "--context-seconds", str(context_seconds)]
            if not include_game:
                arguments.append("--no-game")
            run_worker("analysis_worker.py", arguments, output)
        ta, sa = (read_json(output / f"{label}.analysis.json") for label in ("teacher", "student"))
        for label, analysis in (("teacher", ta), ("student", sa)):
            if analysis.get("input_sha256") != job["input_sha256"][label]:
                raise ValueError(f"{label} analysis does not match the current input file.")
            if analysis.get("config", {}).get("language") != language:
                raise ValueError(f"{label} analysis language does not match {language}; rerun without --skip-analysis.")
            if language == "ja" and analysis.get("config", {}).get("japanese_asr_engine") != "qwen+romaji":
                raise ValueError(f"{label} analysis lacks the Japanese pronunciation pipeline; rerun without --skip-analysis.")
            current_source = str(teacher if label == "teacher" else student)
            if analysis.get("source") != current_source:
                analysis.setdefault("original_analysis_source", analysis.get("source"))
                analysis["source"] = current_source
                write_json(output / f"{label}.analysis.json", analysis)
        job["stages"]["analysis"] = {"teacher": ta.get("completed_stages", []),
                                      "student": sa.get("completed_stages", [])}
        if analyze_only:
            job["status"] = "analysis_complete"
            return job
        if not ta["pitch"]["times"] or not sa["pitch"]["times"]:
            raise ValueError("Pitch analysis is missing; correction cannot run.")
        from inference.alignment import build_alignment
        plan = build_alignment(ta, sa, min_quality=min_quality, max_stretch=max_stretch,
                               allow_acoustic_fallback=allow_acoustic_fallback)
        write_json(output / "alignment.json", plan)
        job["stages"]["alignment"] = plan["metrics"]
        if not any(region.get("mode") == "synthesize" or
                   (allow_acoustic_fallback and region.get("mode") == "acoustic_fallback")
                   for region in plan["regions"]):
            raise ValueError("No supported vocal correspondences; inspect analysis and alignment artifacts.")
        # The render's timeline is decided by this flag, and everything downstream
        # that measures the output has to know it.
        output_clock = "teacher" if change_timing else "student"
        arguments = ["--output", str(output), "--device", vocoder_device, "--log-floor", str(log_floor)]
        if experiments:
            arguments.append("--experiments")
        if allow_acoustic_fallback:
            arguments.append("--allow-acoustic-fallback")
        # State all three correction axes explicitly, so a worker default can never
        # silently override the job's configuration.
        arguments.append("--change-timing" if change_timing else "--no-change-timing")
        arguments.extend(["--pitch-mode", pitch_mode])
        arguments.append("--map-teacher-pitch" if map_teacher_pitch else "--no-map-teacher-pitch")
        run_worker("render_worker.py", arguments, output)
        job["stages"]["render"] = read_json(output / "render.json")
        if verify:
            run_worker("verify_worker.py", ["--audio", str(output / "student.corrected.wav"),
                       "--teacher-analysis", str(output / "teacher.analysis.json"),
                       "--student-analysis", str(output / "student.analysis.json"),
                       "--alignment", str(output / "alignment.json"),
                       "--target-pitch", str(output / "target_pitch.npz"),
                       "--output", str(output / "verification.json"), "--device", onnx_device,
                       "--clock", output_clock], output)
            job["stages"]["verification"] = read_json(output / "verification.json")
            if not job["stages"]["verification"].get("structural_checks_passed", False):
                raise RuntimeError("Rendered waveform failed structural verification; inspect verification.json.")
        if verify_timing:
            run_worker("timing_verify_worker.py", ["--audio", str(output / "student.corrected.wav"),
                       "--teacher-analysis", str(output / "teacher.analysis.json"),
                       "--output", str(output / "timing.verification.json"), "--device", onnx_device,
                       "--language", language, "--max-chunk-seconds", str(max_chunk_seconds),
                       "--clock", output_clock], output)
            timing = read_json(output / "timing.verification.json")
            job["stages"]["timing_verification"] = timing
            if verify:
                verification = job["stages"]["verification"]
                previous = verification.get("word_timing_validation", {}).get("reason")
                verification["limitations"] = [item for item in verification.get("limitations", []) if item != previous]
                measured = bool(timing["measurement"]["output_word_boundaries_remeasured"])
                validation = {
                    "measured": measured,
                    "report_path": str(output / "timing.verification.json"),
                    "coverage": timing.get("coverage"),
                    "reliable_boundary_errors": timing.get("reliable_boundary_errors"),
                    "method": timing["measurement"]["method"],
                }
                if not measured:
                    validation["reason"] = timing["measurement"].get("reason")
                    for item in timing.get("limitations", []):
                        if item not in verification["limitations"]:
                            verification["limitations"].append(item)
                verification["word_timing_validation"] = validation
                write_json(output / "verification.json", verification)
        metrics = plan["metrics"]
        unresolved = (metrics["unsupported_region_count"] or metrics.get("acoustic_fallback") or
                      metrics.get("ambiguous_match_count") or metrics.get("extreme_local_stretch_count") or
                      metrics.get("phonetic_only_match_count") or
                      metrics.get("status_counts", {}).get("substitution", 0))
        if verify_timing:
            timing_coverage = job["stages"]["timing_verification"].get("coverage")
            if timing_coverage is None:
                # The output keeps the student's timeline, so no boundary comparison
                # was possible; that is reported, not silently counted as success.
                unresolved = True
            else:
                unresolved = unresolved or timing_coverage["within_tolerance_ratio_of_all_teacher_units"] != 1.0
        job["status"] = "completed_with_unresolved" if unresolved else "completed"
        from application.report import write_report
        write_report(output, ta, sa, plan, job)
        if export_copyvocal_file:
            # The bundle is an editing hand-off, not a second render: it reuses
            # the artifacts already written above.
            from application.copyvocal_file import export_bundle
            try:
                job["stages"]["copyvocal_file"] = export_bundle(output)
            except Exception as error:
                job["stages"]["copyvocal_file"] = {"error": str(error)}
                logging.exception("CopyVocal file export failed.")
        logging.info("Saved corrected vocal: %s", output / "student.corrected.wav")
        return job
    except BaseException as error:
        job["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
        job["error"] = str(error)
        raise
    finally:
        job["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "job.json", job)
        if job["status"] not in {"completed", "completed_with_unresolved"}:
            from application.report import write_status_report
            write_status_report(output, job)
