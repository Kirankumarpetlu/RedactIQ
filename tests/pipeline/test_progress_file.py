from pii_pipeline import json_codec as json
from pathlib import Path

from pii_pipeline.run_csv_through_stage9 import _write_pipeline_progress


def test_progress_replace_retries_transient_windows_lock(tmp_path, monkeypatch):
    original = Path.replace
    calls = 0

    def intermittently_locked(self, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("destination briefly opened by metrics reader")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", intermittently_locked)
    monkeypatch.setattr("pii_pipeline.run_csv_through_stage9.time.sleep", lambda _: None)
    _write_pipeline_progress(
        tmp_path, status="running", total_rows=500, processed_rows=1,
        latencies_ms=[10.0], total_start=0.0, device="cpu", hardware=None,
        gliner_threshold=.5, min_entity_score=0.0, entity_count=1,
        input_format="csv",
    )
    assert calls == 2
    assert json.loads((tmp_path / "pipeline_progress.json").read_text())["processed_rows"] == 1
