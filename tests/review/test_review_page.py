"""Annotation pages must preserve original text and require human sign-off."""
from __future__ import annotations

from pii_pipeline import json_codec as json

import pytest

from scripts.build_ground_truth_review_page import render_page, review_payload


def draft_row():
    return {"row_index": "benchmark-example", "context": "Hi 😀 Mira",
            "chosen_reference_tags": json.dumps([
                {"entity_type": "PERSON", "text": "Mira", "start": 5, "end": 9}]),
            "review_issues": '["Pending review"]', "review_status": "pending"}


def test_general_export_preserves_unicode_and_non_numeric_ids_without_signoff():
    rows = [draft_row(), {"row_index": "another-id", "context": "A second transcript."}]
    payload = review_payload(rows)
    assert len(payload) == 2
    assert payload[0]["row_index"] == "benchmark-example"
    assert payload[0]["context"] == rows[0]["context"]
    assert payload[0]["suggestions"][0]["text"] == "Mira"
    assert payload[0]["uncertainties"] == ["Pending review"]
    assert all("review_status" not in row and "verified_semantic_tags" not in row for row in payload)


@pytest.mark.parametrize("corruption", ["fingerprint", "offsets", "duplicate-id", "empty"])
def test_invalid_review_sources_fail_before_export(corruption):
    rows = [draft_row()]
    if corruption == "fingerprint":
        rows[0]["source_fingerprint"] = "0" * 64
    elif corruption == "offsets":
        rows[0]["chosen_reference_tags"] = '[{"entity_type":"PERSON","text":"Mira","start":6,"end":10}]'
    elif corruption == "duplicate-id":
        rows.append(draft_row())
    else:
        rows = []
    with pytest.raises(ValueError):
        review_payload(rows)


def test_source_text_cannot_break_out_of_embedded_json():
    context = '</script><script>alert("example")</script>&'
    payload = review_payload([{"row_index": "example", "context": context}])
    template = '<script type="application/json">__TRANSCRIPT_DATA__</script>'
    html = render_page(payload, template)
    assert context not in html
    assert html.count('</script>') == 1
    serialized = html.removeprefix('<script type="application/json">').removesuffix('</script>')
    assert json.loads(serialized)[0]["context"] == context
