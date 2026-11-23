import csv

from pii_pipeline.stage_timing_report import stage_timing_rows, write_stage_timing_reports


def test_stage_report_keeps_every_nested_call_and_does_not_merge_repeats(tmp_path):
    profiles = [
        {"row_index": 1, "pipeline": {"stages": [
            {"stage_name": "run", "depth": 0, "duration_ms": 10, "status": "ok"},
            {"stage_name": "recognizer", "depth": 1, "duration_ms": 2, "status": "ok"},
            {"stage_name": "recognizer", "depth": 1, "duration_ms": 3, "status": "ok"},
        ]}},
        {"row_index": 2, "pipeline": {"stages": [
            {"stage_name": "run", "depth": 0, "duration_ms": 20, "status": "ok"},
            {"stage_name": "recognizer", "depth": 1, "duration_ms": 4, "status": "ok"},
        ]}},
    ]
    calls, summaries = stage_timing_rows(profiles)
    assert [(item["row_index"], item["call_index"]) for item in calls
            if item["stage_name"] == "recognizer"] == [(1, 1), (1, 2), (2, 1)]
    recognizer = next(item for item in summaries if item["stage_name"] == "recognizer")
    assert (recognizer["calls"], recognizer["rows"], recognizer["total_ms"]) == (3, 2, 9)

    call_path, summary_path = write_stage_timing_reports(profiles, tmp_path)
    with call_path.open(encoding="utf-8", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 5
    with summary_path.open(encoding="utf-8", newline="") as stream:
        assert {row["stage_name"] for row in csv.DictReader(stream)} == {"run", "recognizer"}
