"""Regression coverage for online and offline model provisioning inputs."""
from __future__ import annotations

from pathlib import Path
from pii_pipeline import json_codec as json
import tempfile
import unittest

from scripts.provision_models import provision_gliner


class ProvisionModelsTests(unittest.TestCase):
    def test_copies_verified_offline_snapshot_and_records_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            for name in ("config.json", "model.safetensors", "tokenizer.json"):
                (snapshot / name).write_text(name, encoding="utf-8")
            output = root / "model"

            provision_gliner("example/model", output, "immutable-sha", False, snapshot)

            self.assertEqual((output / "model.safetensors").read_text(encoding="utf-8"), "model.safetensors")
            manifest = json.loads((output / "provisioning_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["requested_revision"], "immutable-sha")

    def test_rejects_incomplete_offline_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "incomplete"
            snapshot.mkdir()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                provision_gliner("example/model", root / "model", "immutable-sha", False, snapshot)
