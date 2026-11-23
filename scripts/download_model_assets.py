"""Restore pinned private model assets without replacing inference source files."""
from __future__ import annotations

import argparse
import hashlib
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
from pathlib import Path


def verify(root: Path, files: list[dict], checksums: bool) -> None:
    errors = []
    for item in files:
        target = root / item['path']
        if not target.is_file() or target.stat().st_size != item['bytes']:
            errors.append(item['path'] + ': missing or incorrect size')
        elif checksums:
            with target.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != item['sha256']:
                errors.append(item['path'] + ': checksum mismatch')
    if errors:
        raise SystemExit('Model verification failed:\n' + '\n'.join(errors))
    print(f"Verified {len(files)} model assets ({'SHA256' if checksums else 'file sizes'}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check-only', action='store_true', help='Check existing files without network access.')
    parser.add_argument('--verify-sha256', action='store_true', help='Also read and hash every model asset.')
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest = json.loads((root / 'MODEL_ASSETS.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        target = (root / item['path']).resolve()
        if not target.is_relative_to(root):
            raise SystemExit('Manifest path escapes project root.')
    if not args.check_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit('Install huggingface_hub, then authenticate with hf auth login.')
        snapshot_download(
            repo_id=manifest['repo_id'], revision=manifest['revision'],
            local_dir=str(root), allow_patterns=[item['path'] for item in manifest['files']],
            token=True,
        )
    verify(root, manifest['files'], args.verify_sha256)


if __name__ == '__main__':
    main()
