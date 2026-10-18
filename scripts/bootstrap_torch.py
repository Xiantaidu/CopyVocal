"""Copy a compatible local PyTorch distribution into the isolated runtime."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-packages", type=Path, required=True)
    args = parser.parse_args()
    source = args.site_packages.resolve(strict=True)
    target = ROOT / "python" / "Lib" / "site-packages"
    packages = {distribution.metadata["Name"].lower().replace("_", "-"): distribution
                for distribution in importlib.metadata.distributions(path=[str(source)])}
    names = ("torch", "filelock", "typing-extensions", "sympy", "mpmath", "networkx",
             "jinja2", "markupsafe", "fsspec")
    manifest = {"source": str(source), "copy_method": "physical copy", "packages": []}
    for name in names:
        distribution = packages[name]
        count = 0
        size = 0
        print(f"Copying {name} {distribution.version}", flush=True)
        for item in distribution.files or []:
            original = Path(distribution.locate_file(item)).resolve()
            if not original.is_relative_to(source) or not original.is_file():
                continue
            relative = original.relative_to(source)
            if "__pycache__" in relative.parts or original.suffix == ".pyc":
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not (destination.is_file() and destination.stat().st_size == original.stat().st_size
                    and destination.stat().st_mtime_ns == original.stat().st_mtime_ns):
                shutil.copy2(original, destination)
            if destination.is_symlink() or destination.stat().st_nlink != 1:
                raise RuntimeError(f"Physical copy verification failed: {destination}")
            size += destination.stat().st_size
            count += 1
        manifest["packages"].append({"name": name, "version": distribution.version,
                                     "files": count, "bytes": size})
    (ROOT / "settings" / "synthesis_runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print("Local synthesis runtime copied.", flush=True)


if __name__ == "__main__":
    main()
