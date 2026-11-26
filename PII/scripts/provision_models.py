#!/usr/bin/env python3
"""Provision reviewed runtime models into local, deployable directories.

This command is intentionally separate from service startup: inference workers
must never download or replace models while handling production traffic.
"""
from __future__ import annotations

import argparse
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_GLINER_REPOSITORY = "fastino/gliner2-privacy-filter-PII-multi"
DEFAULT_GLINER_OUTPUT = Path("models/gliner2-primary")
DEFAULT_SPACY_MODEL = "en_core_web_sm"


def _empty_or_missing(path: Path) -> bool:
    return not path.exists() or not any(path.iterdir())


def provision_gliner(
    repository: str,
    output: Path,
    revision: str | None,
    force: bool,
    offline_snapshot: Path | None = None,
) -> None:
    if output.exists() and not _empty_or_missing(output):
        if not force:
            raise RuntimeError(f"{output} already exists; use --force to replace it.")
        shutil.rmtree(output)
    if offline_snapshot is not None:
        snapshot = offline_snapshot.expanduser().resolve()
        required_files = ("config.json", "model.safetensors", "tokenizer.json")
        if not snapshot.is_dir() or any(not (snapshot / name).is_file() for name in required_files):
            raise RuntimeError(
                f"Offline snapshot {snapshot} is incomplete; expected {', '.join(required_files)}."
            )
        # Follow cache symlinks so the deployable artifact is self-contained.
        if output.exists():
            output.rmdir()  # Existing non-empty directories were handled above.
        shutil.copytree(snapshot, output, symlinks=False)
    else:
        output.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repository,
            revision=revision,
            local_dir=str(output),
            local_dir_use_symlinks=False,
        )
    manifest = {
        "repository": repository,
        "requested_revision": revision or "default-branch",
        "provisioned_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Stage 9 approved GLiNER primary/fallback checkpoint",
    }
    (output / "provisioning_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"GLiNER model provisioned at {output}")


def provision_spacy(model_name: str) -> None:
    import spacy.util

    if spacy.util.is_package(model_name):
        print(f"spaCy model already installed: {model_name}")
        return
    import subprocess

    subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    print(f"spaCy model installed: {model_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision required Stage 9 model assets before deployment.")
    parser.add_argument("--gliner-repository", default=DEFAULT_GLINER_REPOSITORY)
    parser.add_argument("--gliner-revision", default=None, help="Required in production: immutable Hugging Face commit SHA.")
    parser.add_argument("--gliner-output", type=Path, default=DEFAULT_GLINER_OUTPUT)
    parser.add_argument(
        "--offline-snapshot",
        type=Path,
        help="Verified local Hugging Face snapshot to copy without any Hub request.",
    )
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL)
    parser.add_argument("--skip-spacy", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace an existing local GLiNER artifact directory.")
    args = parser.parse_args()

    if not args.gliner_revision:
        print("WARNING: no immutable --gliner-revision supplied; record the resolved revision before production approval.", file=sys.stderr)
    provision_gliner(
        args.gliner_repository,
        args.gliner_output,
        args.gliner_revision,
        args.force,
        args.offline_snapshot,
    )
    if not args.skip_spacy:
        provision_spacy(args.spacy_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
