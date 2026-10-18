"""Audit physical resources and the isolated portable runtime without loading models."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
IMPORTS = ("numpy", "scipy", "scipy.signal", "librosa", "librosa.filters", "soundfile",
           "soxr", "onnxruntime", "tokenizers", "torch", "yaml", "textgrid", "opencc", "pyopenjtalk")
RESOURCE_ROOTS = ("models", "python", "inference/qwen3asr_dml/bin", "ffmpeg.exe")


def local(path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(ROOT)
    except (OSError, ValueError):
        return False


def physical_inventory():
    records, links, errors = [], [], []
    directory_count = 0
    stack = [ROOT / item for item in RESOURCE_ROOTS]
    while stack:
        path = stack.pop()
        try:
            info = path.stat(follow_symlinks=False)
            relative = path.relative_to(ROOT).as_posix()
            if path.is_symlink() or getattr(info, "st_file_attributes", 0) & REPARSE:
                links.append({"path": relative, "kind": "symbolic_link_or_reparse_point"})
                continue
            if stat.S_ISDIR(info.st_mode):
                directory_count += 1
                with os.scandir(path) as children:
                    stack.extend(Path(child.path) for child in children)
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    links.append({"path": relative, "kind": "hard_link", "link_count": info.st_nlink})
                records.append({"path": relative, "bytes": info.st_size,
                                "modified_ns": info.st_mtime_ns, "link_count": info.st_nlink})
        except OSError as error:
            errors.append({"path": str(path), "error": str(error)})
    return sorted(records, key=lambda item: item["path"]), directory_count, links, errors


def hash_record(record):
    record = record.copy()
    path = ROOT / record["path"]
    # Interpreter caches can be written by concurrent worker processes. Every
    # cache is still checked for links; immutable runtime sources are hashed.
    if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
        record["hash_omitted"] = "regenerable_interpreter_bytecode"
        return record
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        info = path.stat()
        if info.st_size != record["bytes"] or info.st_mtime_ns != record["modified_ns"]:
            record["hash_error"] = "file_changed_during_audit"
        else:
            record["sha256"] = digest.hexdigest()
    except OSError as error:
        record["hash_error"] = str(error)
    return record


def native_modules():
    if os.name != "nt":
        return []
    from ctypes import wintypes
    psapi, kernel = ctypes.WinDLL("psapi", use_last_error=True), ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    process = kernel.GetCurrentProcess()
    handles = (wintypes.HMODULE * 4096)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModules(process, handles, ctypes.sizeof(handles), ctypes.byref(needed)):
        raise OSError(ctypes.get_last_error(), "EnumProcessModules failed")
    paths = []
    for handle in handles[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        buffer = ctypes.create_unicode_buffer(32768)
        if psapi.GetModuleFileNameExW(process, handle, buffer, len(buffer)):
            paths.append(buffer.value)
    return sorted(set(paths), key=str.lower)


def runtime_audit():
    imports, errors = [], []
    for name in IMPORTS:
        try:
            module = importlib.import_module(name)
            origin = getattr(module, "__file__", None)
            imports.append({"name": name, "origin": origin,
                            "version": str(getattr(module, "__version__", "")),
                            "local": origin is not None and local(origin)})
        except Exception as error:
            errors.append({"name": name, "error": str(error)})
    loaded_origins = sorted({str(module.__file__) for module in list(sys.modules.values())
                             if getattr(module, "__file__", None) and not str(module.__file__).startswith("<")})
    external_python = [path for path in loaded_origins if not local(path)]
    native = []
    for path in native_modules():
        if local(path):
            category = "project"
        elif Path(path).resolve().is_relative_to(Path(os.environ.get("SystemRoot", "C:/Windows")).resolve()):
            category = "windows_system_or_driver"
        else:
            category = "external_native_dependency"
        native.append({"path": path, "category": category})
    return {"executable": sys.executable, "python_version": sys.version,
            "isolated": bool(sys.flags.isolated), "ignore_environment": bool(sys.flags.ignore_environment),
            "site_initialization_disabled": bool(sys.flags.no_site),
            "sys_path": list(sys.path), "external_sys_path": [path for path in sys.path if not local(path)],
            "imports": imports, "import_errors": errors,
            "loaded_python_origin_count": len(loaded_origins), "external_python_origins": external_python,
            "native_modules": native,
            "external_native_dependencies": [item for item in native if item["category"] == "external_native_dependency"],
            "gpu_models_loaded": False,
            "scope": "Import origins and current-process DLLs; model inference and GPU initialization are not executed."}


def source_path_audit():
    findings = []
    upstream_pattern = re.compile(r"(?:D:[/\\]+Vocal2Midi|Z:[/\\])", re.I)
    allowed = {"scripts/bootstrap.py", "scripts/bootstrap_torch.py", "scripts/audit_local_resources.py"}
    for directory in ("application", "inference", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative in allowed:
                continue
            for line_number, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
                if upstream_pattern.search(line):
                    findings.append({"path": relative, "line": line_number, "text": line.strip()})
    return {"external_upstream_literals_in_runtime_source": findings,
            "excluded_provenance_tools": sorted(allowed),
            "scope": "Windows source-drive literals in executable source. Provenance manifests and bootstrap defaults are not runtime dependencies."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "settings/local_resource_audit.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    if Path(sys.executable).resolve() != (ROOT / "python/python.exe").resolve():
        raise RuntimeError("Run this audit with the project's portable python/python.exe.")
    runtime = runtime_audit()
    print(f"Imported {len(runtime['imports'])} local dependency modules without loading inference models.", flush=True)
    records, directories, links, errors = physical_inventory()
    print(f"Checking {len(records)} physical files in {directories} directories; hashing immutable resources.", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        hashed = []
        for index, record in enumerate(executor.map(hash_record, records), 1):
            hashed.append(record)
            if index % 10000 == 0:
                print(f"Hashed/checked {index}/{len(records)} files.", flush=True)
    index = {record["path"]: record for record in hashed}
    manifest_path = ROOT / "settings/resource_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_errors = []
    for expected in manifest["files"]:
        observed = index.get(expected["path"])
        if observed is None or observed.get("sha256") != expected["sha256"] or observed["bytes"] != expected["bytes"]:
            manifest_errors.append(expected["path"])
    sources = source_path_audit()
    hash_errors = [record for record in hashed if record.get("hash_error")]
    violations = {
        "links": links, "filesystem_errors": errors, "hash_errors": hash_errors,
        "original_manifest_mismatches": manifest_errors,
        "external_sys_path": runtime["external_sys_path"],
        "external_python_origins": runtime["external_python_origins"],
        "external_native_dependencies": runtime["external_native_dependencies"],
        "import_errors": runtime["import_errors"],
        "nonlocal_imports": [item for item in runtime["imports"] if not item["local"]],
        "upstream_runtime_source_literals": sources["external_upstream_literals_in_runtime_source"],
    }
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True).stdout
    lock = ROOT / "settings/requirements-lock.txt"
    lock.write_text("# Snapshot of the physically bundled Windows CPython 3.12 runtime.\n"
                    "# Generated by python/python.exe -m pip freeze --all.\n"
                    "# This preserves versions; it is not an offline wheel archive or a minimal dependency specification.\n"
                    "# PyTorch uses its cu129 build. Do not replace onnxruntime-directml with another ONNX Runtime distribution.\n"
                    + freeze.replace("\r\n", "\n"), encoding="utf-8")
    ffmpeg = subprocess.run([str(ROOT / "ffmpeg.exe"), "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[:3]
    audit = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
             "root": str(ROOT), "passed": not any(violations.values()),
             "runtime": runtime, "source_path_audit": sources, "violations": violations,
             "original_manifest_resource_count": len(manifest["files"]),
             "inventory_summary": {"file_count": len(hashed), "directory_count": directories,
                                   "total_bytes": sum(item["bytes"] for item in hashed),
                                   "sha256_file_count": sum("sha256" in item for item in hashed),
                                   "bytecode_hash_omission_count": sum("hash_omitted" in item for item in hashed)},
             "requirements_lock": str(lock.relative_to(ROOT)), "ffmpeg_build": ffmpeg,
             "elapsed_seconds": time.perf_counter() - started, "files": hashed,
             "limits": ["Windows system DLLs and hardware drivers are operating-system prerequisites, not bundled files.",
                        "Import and source-path checks are complemented by separate end-to-end model runs; no upstream drive is renamed or disconnected by this audit.",
                        "Python bytecode files are link-checked but not hashed because worker processes may regenerate them.",
                        "Provenance source paths and bootstrap defaults document copying and are not import/search paths."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("passed", "inventory_summary", "elapsed_seconds")}, indent=2), flush=True)
    if not audit["passed"]:
        print(json.dumps(violations, indent=2, ensure_ascii=False), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
