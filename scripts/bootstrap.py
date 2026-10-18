"""Make physical, verified copies of the upstream runtime and model bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = (
    "Qwen3-ASR-1.7B-dml", "GAME-1.0.3-medium-onnx", "tifa-1.0-onnx", "RMVPE",
    "kashi-g2p-onnx", "romajiASR",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_bundle(source: Path, target: Path) -> None:
    print(f"Copying {source.name} -> {target}", flush=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def verify_file(path: Path) -> None:
    info = path.stat(follow_symlinks=False)
    if path.is_symlink() or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise RuntimeError(f"Reparse points are forbidden: {path}")
    if path.is_file() and info.st_nlink != 1:
        raise RuntimeError(f"Hard links are forbidden: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path("D:/Vocal2Midi"))
    parser.add_argument("--vocoder", type=Path,
                        default=Path("Z:/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    for name in ("application", "inference", "models", "python", "scripts", "settings",
                 "tests", "docs", "outputs", "cache", "licenses"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        copy_bundle(args.upstream / "inference", ROOT / "inference")
        for name in ("__init__.py", "config.py", "exceptions.py"):
            copy_bundle(args.upstream / "application" / name, ROOT / "application" / name)
        for name in MODEL_NAMES:
            copy_bundle(args.upstream / "models" / name, ROOT / "models" / name)
        copy_bundle(args.vocoder, ROOT / "models" / args.vocoder.name)
        copy_bundle(args.upstream / "python", ROOT / "python")
        copy_bundle(args.upstream / "ffmpeg.exe", ROOT / "ffmpeg.exe")
        for name in ("LICENSE", "NOTICE", "ACKNOWLEDGEMENTS.md"):
            copy_bundle(args.upstream / name, ROOT / "licenses" / f"Vocal2Midi-{name}")
        print("Physical copy completed.", flush=True)
    records = []
    bundles = [ROOT / "models", ROOT / "inference" / "qwen3asr_dml" / "bin"]
    for bundle in bundles:
        for path in sorted(bundle.rglob("*")):
            verify_file(path)
            if path.is_file():
                records.append({"path": path.relative_to(ROOT).as_posix(),
                                "bytes": path.stat().st_size, "sha256": sha256(path)})
    runtime_count = 0
    for path in (ROOT / "python").rglob("*"):
        verify_file(path)
        runtime_count += path.is_file()
    manifest = {"version": 1, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "copy_method": "shutil.copy2 (physical files)",
                "sources": {"upstream": str(args.upstream), "vocoder": str(args.vocoder)},
                "files": records, "runtime_file_count": runtime_count,
                "verification": {"reparse_points": 0, "hard_links": 0}}
    target = ROOT / "settings" / "resource_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(records)} hashed resources and {runtime_count} runtime files "
          f"in {time.time() - started:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
