"""Private assets must be pinned, verified, contained, and reusable offline."""
from __future__ import annotations

import hashlib
from pii_pipeline import json_codec as json
import subprocess
import sys
from pathlib import Path

import pytest

from pii_pipeline import data_assets


def test_asset_and_configuration_utilities_do_not_initialize_neural_libraries():
    script = (
        "import sys; import pii_pipeline; "
        "from pii_pipeline import PipelineConfig, load_pipeline_config; "
        "from pii_pipeline.data_assets import load_manifest; "
        "assert isinstance(load_pipeline_config(), PipelineConfig); "
        "assert load_manifest()['repo_type'] == 'dataset'; "
        "assert not {'torch', 'presidio_analyzer', 'transformers', 'onnxruntime'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def private_assets(tmp_path):
    contents = {"data/gold/benchmark.csv": b"row_index,context\n1,Synthetic example\n",
                "data/gazetteers/names.txt": b"Synthetic\n"}
    manifest = {"schema_version": 1, "repo_id": "test/private-data", "repo_type": "dataset",
                "revision": "a" * 40,
                "files": [{"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                          for path, raw in contents.items()],
                "groups": {"benchmark": ["data/gold/benchmark.csv"],
                           "runtime": ["data/gazetteers/names.txt"]}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest, contents, tmp_path / "cache"


def test_download_is_pinned_authenticated_and_only_selected_assets(private_assets, monkeypatch):
    path, manifest, contents, cache = private_assets
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        for relative in kwargs["allow_patterns"]:
            target = data_assets.Path(kwargs["local_dir"]) / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[relative])

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    result = data_assets.ensure_data(("benchmark",), cache_root=cache, manifest_path=path)
    assert set(result) == {"data/gold/benchmark.csv"}
    assert len(calls) == 1
    assert calls[0]["revision"] == manifest["revision"]
    assert calls[0]["repo_type"] == "dataset" and calls[0]["token"] is True
    assert calls[0]["allow_patterns"] == ["data/gold/benchmark.csv"]
    # Verified cached assets must work without a second network request.
    assert data_assets.ensure_data(("benchmark",), cache_root=cache, manifest_path=path) == result
    assert len(calls) == 1


def test_check_only_never_downloads_missing_or_corrupt_data(private_assets, monkeypatch):
    path, manifest, contents, cache = private_assets
    def unexpected(**kwargs):
        pytest.fail("Offline verification accessed the network")
    monkeypatch.setattr("huggingface_hub.snapshot_download", unexpected)
    with pytest.raises(FileNotFoundError):
        data_assets.ensure_data(("benchmark",), check_only=True, cache_root=cache, manifest_path=path)
    target = cache / manifest["revision"] / "data/gold/benchmark.csv"
    target.parent.mkdir(parents=True)
    raw = contents["data/gold/benchmark.csv"]
    target.write_bytes(b"X" * len(raw))
    with pytest.raises(FileNotFoundError, match="corrupt"):
        data_assets.ensure_data(("benchmark",), check_only=True, cache_root=cache, manifest_path=path)


def test_bad_download_is_rejected(private_assets, monkeypatch):
    path, manifest, contents, cache = private_assets
    def corrupt_download(**kwargs):
        target = data_assets.Path(kwargs["local_dir"]) / "data/gold/benchmark.csv"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"corrupt download")
    monkeypatch.setattr("huggingface_hub.snapshot_download", corrupt_download)
    with pytest.raises(FileNotFoundError, match="corrupt"):
        data_assets.ensure_data(("benchmark",), cache_root=cache, manifest_path=path)


@pytest.mark.parametrize("relative", ["../secret", "data/../../secret", "/data/secret", "data\\secret"])
def test_manifest_cannot_escape_cache(private_assets, relative):
    path, manifest, _, _ = private_assets
    manifest["files"][0]["path"] = relative
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        data_assets.load_manifest(path)


def test_mutable_revision_is_rejected(private_assets):
    path, manifest, _, _ = private_assets
    manifest["revision"] = "main"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        data_assets.load_manifest(path)


def test_missing_runtime_resource_fails_with_setup_guidance(private_assets, monkeypatch):
    path, _, _, cache = private_assets
    monkeypatch.setattr(data_assets, "MANIFEST_PATH", path)
    monkeypatch.setenv("PII_DATA_CACHE", str(cache))
    with pytest.raises(FileNotFoundError, match="download_data_assets"):
        data_assets.require_data_path("data/gazetteers/names.txt")


def test_failed_inference_does_not_score_or_write_metrics(tmp_path, monkeypatch):
    from scripts import run_benchmark
    output = tmp_path / "failed-run"
    monkeypatch.setattr(run_benchmark, "ensure_data", lambda: {"data/gold/unified_benchmark.csv": tmp_path / "benchmark.csv"})
    monkeypatch.setattr("sys.argv", ["run_benchmark", "--output-dir", str(output)])
    def fail(*args, **kwargs):
        assert kwargs["check"] is True
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(run_benchmark.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        run_benchmark.main()
    assert not (output / "metrics.json").exists()


@pytest.mark.parametrize("flag,group,count", [("--real-only", "real_development_pending", 500),
                                              ("--synthetic-only", "synthetic_regression", 28)])
def test_benchmark_run_selects_the_same_group_for_inference_and_scoring(tmp_path, monkeypatch, flag, group, count):
    from scripts import run_benchmark, evaluate_unified_benchmark
    from unittest.mock import Mock
    benchmark = tmp_path / "benchmark.csv"
    output = tmp_path / "run"
    monkeypatch.setattr(run_benchmark, "ensure_data", lambda: {"data/gold/unified_benchmark.csv": benchmark})
    inference = Mock()
    scoring = Mock(return_value={"groups": {group: {"rows_scored": count}}})
    monkeypatch.setattr(run_benchmark.subprocess, "run", inference)
    monkeypatch.setattr(evaluate_unified_benchmark, "evaluate", scoring)
    monkeypatch.setattr("sys.argv", ["run_benchmark", flag, "--device", "cpu", "--indic-device", "cpu", "--output-dir", str(output)])
    run_benchmark.main()
    command = inference.call_args.args[0]
    assert command[command.index("--input") + 1] == str(benchmark)
    assert command[command.index("--benchmark-group") + 1] == group
    assert command[command.index("--indic-device") + 1] == "cpu"
    assert "--max-rows" not in command
    assert scoring.call_args.kwargs["selected_groups"] == [group]
    assert (output / "data_provenance.json").is_file()
