"""Resolve and verify pinned private datasets without downloading during inference."""
from __future__ import annotations

import hashlib
from . import json_codec as json
import os
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "DATA_ASSETS.json"


def load_manifest(path=None):
    manifest = json.loads(Path(path or MANIFEST_PATH).read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("repo_type") != "dataset"
            or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", ""))
            or not re.fullmatch(r"[\w.-]+/[\w.-]+", manifest.get("repo_id", ""))):
        raise ValueError("Invalid pinned data manifest")
    known = set()
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in item["path"]
                or not item["path"].startswith("data/") or item["path"] in known
                or type(item["bytes"]) is not int or item["bytes"] < 0
                or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])):
            raise ValueError("Invalid data asset entry")
        known.add(item["path"])
    for paths in manifest["groups"].values():
        if not paths or any(path not in known for path in paths):
            raise ValueError("Invalid data asset group")
    return manifest


def cache_directory(manifest=None, cache_root=None):
    manifest = manifest or load_manifest()
    base = Path(cache_root or os.environ.get("PII_DATA_CACHE", ROOT / ".cache/private-data"))
    return base.expanduser().resolve() / manifest["revision"]


def _target(directory, relative):
    target = (directory / relative).resolve()
    if not target.is_relative_to(directory.resolve()):
        raise ValueError("Data asset path escapes cache directory")
    return target


def data_path(relative, *, cache_root=None):
    """Return a known asset's cache path; never access the network."""
    manifest = load_manifest()
    if relative not in {item["path"] for item in manifest["files"]}:
        raise ValueError(f"Unknown private data asset: {relative}")
    return _target(cache_directory(manifest, cache_root), relative)


def _matches(path, item):
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]


def require_data_path(relative):
    manifest = load_manifest()
    item = next((item for item in manifest["files"] if item["path"] == relative), None)
    if item is None:
        raise ValueError(f"Unknown private data asset: {relative}")
    path = _target(cache_directory(manifest), relative)
    if not _matches(path, item):
        raise FileNotFoundError(
            f"Private data asset is missing or fails SHA256 verification: {relative}. "
            "Run python -m scripts.download_data_assets --group all first."
        )
    return path


def ensure_data(groups=("benchmark", "runtime"), *, check_only=False,
                cache_root=None, manifest_path=None):
    """Download only missing/invalid selected assets, then verify every SHA256."""
    manifest = load_manifest(manifest_path)
    groups = tuple(groups)
    unknown = set(groups) - set(manifest["groups"]) - {"all"}
    if unknown or not groups:
        raise ValueError(f"Unknown or empty data groups: {sorted(unknown)}")
    selected = ({item["path"] for item in manifest["files"]} if "all" in groups else
                {path for group in groups for path in manifest["groups"][group]})
    items = [item for item in manifest["files"] if item["path"] in selected]
    directory = cache_directory(manifest, cache_root)
    invalid = [item for item in items if not _matches(_target(directory, item["path"]), item)]
    if invalid and not check_only:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=manifest["repo_id"], repo_type="dataset", revision=manifest["revision"],
            local_dir=str(directory), allow_patterns=[item["path"] for item in invalid],
            token=True, force_download=True,
        )
    invalid = [item["path"] for item in items if not _matches(_target(directory, item["path"]), item)]
    if invalid:
        raise FileNotFoundError("Missing or corrupt private data assets: " + ", ".join(invalid))
    return {item["path"]: _target(directory, item["path"]) for item in items}
