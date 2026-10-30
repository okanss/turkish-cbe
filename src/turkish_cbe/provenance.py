"""Fingerprints for planning artefacts; deliberately excludes model results."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from . import __version__
from .config import loads_config


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    """Hash the explicit source/document surface, never .env or generated data."""
    root = root.resolve()
    candidates = list(root.glob("*.md"))
    candidates += [root / name for name in
                   ("pyproject.toml", "requirements-build.lock", ".gitignore")]
    for directory, suffixes in (("src", {".py"}), ("tests", {".py"}),
                                ("configs", {".toml"}), ("data", {".md"})):
        base = root / directory
        if base.is_symlink():
            raise ValueError(f"source symlink is not allowed: {directory}")
        if base.exists():
            candidates += [p for p in base.rglob("*") if p.suffix in suffixes]
    hashes = {}
    for path in sorted(candidates):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"source symlink is not allowed: {path.name}")
        if path.is_file():
            hashes[path.relative_to(root).as_posix()] = sha256_bytes(path.read_bytes())
    if not hashes:
        raise ValueError("no project sources found")
    return hashes


def _git_state(root: Path) -> dict:
    """Only claim a revision when root is the actual checkout root."""
    try:
        def git(*args: str) -> str:
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
            return {"commit": None, "dirty": None, "status": "unversioned_project"}
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain")), "status": "git"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "status": "unversioned_project"}


def build_manifest(config_path: Path, project_root: Path) -> dict:
    config_path = Path(config_path)
    project_root = Path(project_root).resolve()
    config_bytes = config_path.read_bytes()
    config = loads_config(config_bytes)
    resolved = config.to_dict()
    hashes = source_hashes(project_root)
    packages = sorted({(d.metadata.get("Name", "unknown"), d.version)
                       for d in importlib.metadata.distributions()})
    return {
        "manifest_schema_version": 1,
        "artifact_type": "planning_manifest",
        "status": "planning_only",
        "contains_model_results": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "config_file": config_path.name,
        "config_sha256": sha256_bytes(config_bytes),
        "resolved_config": resolved,
        "resolved_config_sha256": sha256_bytes(canonical_json(resolved).encode("utf-8")),
        "source_files_sha256": hashes,
        "source_tree_sha256": sha256_bytes(canonical_json(hashes).encode("utf-8")),
        "git": _git_state(project_root),
        "environment": {
            "python": sys.version, "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "installed_distributions": [{"name": n, "version": v} for n, v in packages],
        },
        "dataset": None,
        "template_bundle": None,
        "budget": config.budget(),
        "execution_ready": False,
        "pending_gates": list(config.pending_gates),
        "limitations": ["Planning command does not bind a generated dataset; use dataset_manifest.json.",
                        "Runner, response metrics and model snapshots remain unavailable; linguistic review is pending.",
                        "A planning manifest is not an experiment record."],
    }


def write_manifest(manifest: dict, output: Path) -> None:
    """Exclusive creation prevents accidental destruction of an earlier record."""
    output = Path(output)
    payload = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
