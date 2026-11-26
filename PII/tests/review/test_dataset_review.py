"""Human-review provenance, pending-sheet repair and prediction alignment."""
from __future__ import annotations

import csv
from pii_pipeline import json_codec as json

import pytest

from scripts.align_review_predictions import align
from scripts.evaluate_pipeline import PipelineEvaluator
from scripts.prepare_precision_dataset import has_human_review_signoff, validate_reviewed
from scripts.repair_real_labelling_sheet import prepare_pending


def reference(reviewer):
    return dict(row_index="1", context="No private details.", verified_semantic_tags="[]",
                review_status="reviewed", annotation_scope="exhaustive", reviewer=reviewer)


@pytest.mark.parametrize("reviewer", ["ChatGPT", "chatgpt", "Codex", "GPT-5", "AI", "Claude"])
def test_ai_reviewer_does_not_supply_human_signoff(tmp_path, reviewer):
    row = reference(reviewer)
    assert not has_human_review_signoff(row)
    with pytest.raises(ValueError, match="human exhaustive review"):
        validate_reviewed([row])
    path = tmp_path / "reference.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    with pytest.raises(ValueError, match="exhaustively reviewed"):
        PipelineEvaluator().load_ground_truth(str(path))


def test_named_reviewer_signoff_and_pending_draft_are_distinct():
    row = reference("Kiran")
    validate_reviewed([row])
    assert has_human_review_signoff(row)
    row["review_status"] = "pending"
    assert not has_human_review_signoff(row)


def copied_row():
    tags = [{"entity_type": "PERSON", "text": "Maya", "start": 5, "end": 9}]
    return {"row_index": "real-001", "context": "Name Maya", "predicted_tags": json.dumps(tags),
            "verified_semantic_tags": json.dumps(tags)}


def test_repair_preserves_suggestions_without_creating_gold():
    legacy = [{"row_id": "1", "context": "Name Maya", "pii_dictionary": '{"PERSON":["Maya"]}'}]
    prepared, summary = prepare_pending([copied_row()], legacy)
    row = prepared[0]
    assert row["verified_semantic_tags"] == ""
    assert json.loads(row["predicted_tags"])[0]["text"] == "Maya"
    assert json.loads(row["legacy_reference_suggestions"])[0]["text"] == "Maya"
    assert row["review_status"] == "pending" and summary["legacy_matching_rows"] == 1
    with pytest.raises(ValueError, match="human exhaustive review"):
        validate_reviewed(prepared)


def test_repair_refuses_to_erase_independent_or_reviewed_annotations():
    independent = copied_row()
    independent["verified_semantic_tags"] = "[]"
    with pytest.raises(ValueError, match="differ"):
        prepare_pending([independent], [])
    reviewed = copied_row()
    reviewed["review_status"] = "reviewed"
    with pytest.raises(ValueError, match="reviewed"):
        prepare_pending([reviewed], [])


def test_align_review_predictions_matches_text_and_keeps_review_ids():
    reference = [{"row_index": "real-001", "context": "Name Maya"}]
    predictions = [{"row_index": "1", "context": "Name Maya", "pipeline_entities": "[]"}]
    assert align(reference, predictions) == [{"row_index": "real-001", "context": "Name Maya", "pipeline_entities": "[]"}]
    with pytest.raises(ValueError, match="No matching prediction"):
        align(reference, [{**predictions[0], "context": "Different text"}])
