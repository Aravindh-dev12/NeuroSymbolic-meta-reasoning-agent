"""Stage or push this project as a Kaggle script kernel."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "data" / "kaggle_kernel"


def copytree(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(src, dst, ignore=ignore)


def stage_kernel() -> Path:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    copytree(REPO_ROOT / "neurosymbolic_agent", BUILD_DIR / "neurosymbolic_agent")
    shutil.copy2(REPO_ROOT / "kaggle" / "run_agent.py", BUILD_DIR / "run_agent.py")
    shutil.copy2(REPO_ROOT / "README.md", BUILD_DIR / "README.md")
    shutil.copy2(REPO_ROOT / "pyproject.toml", BUILD_DIR / "pyproject.toml")

    metadata = json.loads((REPO_ROOT / "kaggle" / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = os.getenv("KAGGLE_KERNEL_ID", metadata["id"])
    (BUILD_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return BUILD_DIR


def push_kernel(path: Path) -> int:
    return subprocess.run(["kaggle", "kernels", "push", "-p", str(path)], check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage or push the Kaggle kernel package.")
    parser.add_argument("--stage", action="store_true", help="Only stage the Kaggle package")
    parser.add_argument("--push", action="store_true", help="Stage and push with the Kaggle CLI")
    args = parser.parse_args()

    path = stage_kernel()
    print(f"Staged Kaggle kernel at: {path}")

    if args.push:
        return push_kernel(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
