"""Verify or physically copy the self-contained Japanese inference resources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
JAPANESE_BUNDLES = ("models/kashi-g2p-onnx", "models/romajiASR")
REQUIRED_FILES = (
    "models/kashi-g2p-onnx/model.onnx",
    "models/kashi-g2p-onnx/resources/lexicon.txz",
    "models/romajiASR/model.onnx",
    "models/romajiASR/model.meta.json",
    "models/romajiASR/phoneme_vocab.json",
    "models/tifa-1.0-onnx/dictionaries/ja_lexicon.txz",
    "models/tifa-1.0-onnx/dictionaries/japanese_dict_full.txt",
)
SHARED_FILES = (
    "models/Qwen3-ASR-1.7B-dml/qwen3_asr_encoder_backend.fp16.onnx",
    "models/Qwen3-ASR-1.7B-dml/qwen3_asr_encoder_frontend.fp16.onnx",
    "models/Qwen3-ASR-1.7B-dml/qwen3_asr_llm.f16.gguf",
)
RUNTIME_DIRECTORY = "python/Lib/site-packages/pyopenjtalk"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_physical(path: Path, root: Path) -> None:
    """Reject links in a resource and every parent inside its project root."""
    root = root.absolute()
    path = path.absolute()
    if not path.is_relative_to(root):
        raise ValueError(f"Resource escapes project root: {path}")
    for item in (path, *path.parents):
        info = item.stat(follow_symlinks=False)
        if item.is_symlink() or getattr(info, "st_file_attributes", 0) & REPARSE:
            raise ValueError(f"Linked resource is forbidden: {item}")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise ValueError(f"Hard-linked resource is forbidden: {item}")
        if item == root:
            break


def audit_file(relative: str, root: Path, source: Path) -> dict:
    target, original = root / relative, source / relative
    record = {"path": relative}
    try:
        assert_physical(target, root)
        assert_physical(original, source)
        record.update(bytes=target.stat().st_size, sha256=sha256(target),
                      source_sha256=sha256(original), link_count=target.stat().st_nlink)
        record["passed"] = record["sha256"] == record["source_sha256"]
        if not record["passed"]:
            record["error"] = "Source and local SHA256 differ."
    except (OSError, ValueError) as error:
        record.update(passed=False, error=str(error))
    return record


def runtime_files(source: Path) -> list[str]:
    runtime = source / RUNTIME_DIRECTORY
    return [item.relative_to(source).as_posix() for item in sorted(runtime.rglob("*"))
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"]


def copy_japanese_bundles(root: Path, source: Path) -> None:
    """Copy bytes; never create symlinks, hardlinks, or external search paths."""
    for relative in JAPANESE_BUNDLES:
        source_bundle = source / relative
        assert_physical(source_bundle, source)
        for original in sorted(source_bundle.rglob("*")):
            assert_physical(original, source)
            target = root / original.relative_to(source)
            if original.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                assert_physical(target, root)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                assert_physical(target.parent, root)
                if target.exists():
                    assert_physical(target, root)
                shutil.copy2(original, target)
                assert_physical(target, root)


def cpu_smoke(root: Path) -> dict:
    """Load Japanese models on CPU only and exercise local G2P resources."""
    import importlib
    import os

    sys.path.insert(0, str(root))
    import pyopenjtalk
    from inference.kashi_g2p_ja.g2p_onnx_runtime import G2POnnxRuntime
    from inference.romaji_asr.runtime import RomajiASROnnxModel

    origins = {}
    for name in ("pyopenjtalk", "inference.kashi_g2p_ja.g2p_onnx_runtime",
                 "inference.romaji_asr.runtime", "onnxruntime", "numpy"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(root.resolve()):
            raise RuntimeError(f"Nonlocal import: {name}: {origin}")
        origins[name] = str(origin)
    dictionary = Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR))
    assert_physical(dictionary, root)
    text = "君と歌う空"
    frontend = pyopenjtalk.run_frontend(text)
    g2p = G2POnnxRuntime(model_dir=root / "models/kashi-g2p-onnx", prefer_dml=False)
    prediction = g2p.predict(text)
    romaji = RomajiASROnnxModel.from_model_path(root / "models/romajiASR", provider="cpu")
    providers = {"kashi_g2p": g2p.session.get_providers(),
                 "romaji_asr": romaji.session.get_providers()}
    if any(value != ["CPUExecutionProvider"] for value in providers.values()):
        raise RuntimeError(f"Unexpected non-CPU providers: {providers}")
    if not prediction.get("reading") or not frontend:
        raise RuntimeError("Japanese frontend returned no reading.")
    return {"passed": True, "origins": origins, "open_jtalk_dictionary": str(dictionary),
            "providers": providers, "text": text, "kashi_reading": prediction["reading"],
            "open_jtalk_reading": "".join(item["pron"] for item in frontend),
            "romaji_vocabulary_size": len(romaji.id2token)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Upstream project used for byte-for-byte comparison.")
    parser.add_argument("--output", type=Path, default=ROOT / "settings/japanese_resource_audit.json")
    parser.add_argument("--copy", action="store_true", help="Physically copy missing Japanese model bundles first.")
    parser.add_argument("--skip-shared-qwen", action="store_true", help="Skip hashing the already shared Qwen bundle.")
    parser.add_argument("--cpu-smoke", action="store_true", help="Exercise Japanese model loading on CPU only.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "settings/resource_manifest.json").read_text(encoding="utf-8"))
    source = (args.source or Path(manifest["sources"]["upstream"])).resolve()
    if source == ROOT.resolve():
        raise ValueError("The upstream source must differ from this project.")
    if args.copy:
        copy_japanese_bundles(ROOT, source)
    paths = sorted(set(REQUIRED_FILES) | set(runtime_files(source)) | {
        f"{RUNTIME_DIRECTORY}/__init__.py",
        f"{RUNTIME_DIRECTORY}/open_jtalk_dic_utf_8-1.11/sys.dic",
    })
    if not args.skip_shared_qwen:
        paths.extend(SHARED_FILES)
    records = [audit_file(relative, ROOT, source) for relative in paths]
    report = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "source": str(source), "root": str(ROOT), "files": records,
              "file_count": len(records), "passed": all(record["passed"] for record in records),
              "shared_qwen_included": not args.skip_shared_qwen,
              "production_path": "romaji ASR -> TiFA mora alignment; Qwen -> kashi G2P alternative",
              "optional_runtime": "pyopenjtalk fallback with project-local Open JTalk dictionary; MeCab and UniDic are not used by kashi G2P."}
    if args.cpu_smoke:
        try:
            report["cpu_smoke"] = cpu_smoke(ROOT)
        except Exception as error:
            report.update(passed=False, cpu_smoke={"passed": False, "error": str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "file_count": len(records),
                      "total_bytes": sum(record.get("bytes", 0) for record in records),
                      "failures": [record for record in records if not record["passed"]],
                      "cpu_smoke": report.get("cpu_smoke"), "report": str(args.output)},
                     indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
