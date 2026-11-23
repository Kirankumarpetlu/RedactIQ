import json

import httpx
import pytest
from presidio_analyzer import RecognizerResult

from pii_pipeline.decider_stage import CRITERIA, MODEL, LocalLightningDecider, OpenRouterDecider, gap_spans
from pii_pipeline.orchestrator import RoutedResults
from pii_pipeline.policy.entity_policy import resolve_entities
from pii_pipeline.recognizers.indic_identifiers import build_indic_identifier_recognizers


def _candidate(text, value, label, recognizer):
    start = text.index(value)
    return RecognizerResult(label, start, start + len(value), .82,
                            recognition_metadata={"recognizer_name": recognizer})


@pytest.fixture
def hosted_env(monkeypatch):
    monkeypatch.setenv("PII_ALLOW_HOSTED_DECIDER", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-test-key")


def test_openrouter_candidates_preserve_offsets_rules_and_cache(hosted_env):
    text = "Nurse saw Mira Sample. Email: sample@example.invalid."
    requests = []

    def reply(request):
        payload = json.loads(request.content)
        assert payload["model"] == MODEL
        assert list(payload["questions"]) == ["span"]
        assert payload["questions"]["span"]["type"] == "choice"
        assert "⟦Mira Sample⟧" in payload["state"]
        requests.append(payload)
        return httpx.Response(200, json={"model": MODEL, "answers": {"span": {
            "type": "choice", "choice": "PERSON",
            "probabilities": {label: float(label == "PERSON") for label in CRITERIA},
        }}})

    decider = OpenRouterDecider([text], transport=httpx.MockTransport(reply))
    email = _candidate(text, "sample@example.invalid", "EMAIL_ADDRESS", "EmailRecognizer")
    try:
        candidates, counts, trace = decider.analyze(text, RoutedResults([email], []))
        assert len(requests) == counts["decider_calls"] == 1
        assert any(item.get("reason") == "rule_validated" for item in trace)
        assert {c["entity_type"]: text[c["start"]:c["end"]] for c in candidates} == {
            "PERSON": "Mira Sample", "EMAIL_ADDRESS": "sample@example.invalid",
        }
        repeated, counts, _ = decider.analyze(text, RoutedResults([email], []))
        assert repeated == candidates
        assert counts["decider_cache_hits"] == 1 and counts["decider_calls"] == 0
        with pytest.raises(ValueError, match="allowlisted"):
            decider.analyze("Outside the generated corpus.", RoutedResults([], []))
        assert len(requests) == 1
    finally:
        decider.close()


def test_openrouter_failure_preserves_presidio_candidate(hosted_env):
    text = "Mira visited the clinic."
    candidate = _candidate(text, "Mira", "PERSON", "SpacyRecognizer")

    def reply(request):
        return httpx.Response(429)

    decider = OpenRouterDecider([text], transport=httpx.MockTransport(reply))
    try:
        kept, counts, trace = decider.analyze(text, RoutedResults([candidate], []))
        assert len(kept) == 1 and kept[0]["score"] == .82
        assert counts["decider_calls"] == counts["decider_failed"] == 1
        assert trace[0]["failure"] == "http_429"
        with pytest.raises(RuntimeError, match="http_429"):
            decider.warm()
    finally:
        decider.close()


def test_openrouter_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("PII_ALLOW_HOSTED_DECIDER", raising=False)
    with pytest.raises(ValueError, match="PII_ALLOW_HOSTED_DECIDER"):
        OpenRouterDecider(())


@pytest.mark.parametrize(
    "text, value, label, recognizer, expected",
    [
        ("The infant is 7 months old. Follow-up tomorrow.",
         "7", "AGE", "AgeRecognizer", ("7", "AGE")),
        ("Contact e-mail address is patient@sample.invalid. Thanks.",
         "patient@sample.invalid", "ACCOUNT_ID", "UPIHandleRecognizer",
         ("patient@sample.invalid", "EMAIL")),
        ("UPI ID: patient@sample.invalid. Payment received.",
         "patient@sample.invalid", "ACCOUNT_ID", "UPIHandleRecognizer",
         ("patient@sample.invalid", "ACCOUNT_ID")),
        ("Street address = 408 Maple Road; follow-up tomorrow.",
         "Maple Road", "LOCATION", "SpacyRecognizer", ("408 Maple Road", "LOCATION")),
    ],
)
def test_openrouter_normalizes_bounded_fields_before_model(
    hosted_env, text, value, label, recognizer, expected,
):
    def unexpected_request(request):
        pytest.fail("A validated field or its fragments reached OpenRouter")

    decider = OpenRouterDecider([text], transport=httpx.MockTransport(unexpected_request))
    try:
        result = _candidate(text, value, label, recognizer)
        candidates, counts, _ = decider.analyze(text, RoutedResults([result], []))
        selected, _ = resolve_entities(text, candidates)
        assert counts["decider_calls"] == 0
        assert [(entity["text"], entity["canonical_entity_type"]) for entity in selected] == [expected]
    finally:
        decider.close()


def test_indic_gap_names_require_a_bounded_cue():
    assert list(gap_spans("दर्ज नाम। मरीज को सिरदर्द है। రోగి పేరు. తలనొప్పి తగ్గింది.")) == []
    text = "इनको कहते हैं अनया वर्मा।"
    assert [(text[start:end], hint) for start, end, hint in gap_spans(text)] == [("अनया वर्मा", "PERSON")]


def test_bare_numbers_never_reach_the_decider():
    assert list(gap_spans("BP 120/82. Dose 25 mg. 3 days. మోతాదు ౨౫. मात्रा २५।")) == []


def test_telugu_gap_reuses_only_whole_rule_backed_values():
    text = "పేరు అనయ. అనయకు మందు ఇచ్చారు. అనయ వచ్చింది."
    start = text.index("అనయ")
    anchor = {"entity_type": "PERSON", "start": start, "end": start + len("అనయ")}
    assert [(a, text[a:b], hint) for a, b, hint in gap_spans(text, [anchor])] == [
        (start, "అనయ", "PERSON"), (text.rindex("అనయ"), "అనయ", "PERSON"),
    ]


def test_local_lightning_requires_evidence_for_a_numeric_label(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PII_ALLOW_HOSTED_DECIDER", raising=False)
    text = "Mira visited the clinic. Sam Example followed. Dose 25 mg."
    requests = []

    def reply(request):
        assert str(request.url) == "http://127.0.0.1:8080/v1/systemone"
        assert "Authorization" not in request.headers
        payload = json.loads(request.content)
        assert list(payload["questions"]) == ["span"]
        state = payload["state"]
        assert any(f"⟦{name}⟧" in state for name in ("Mira", "Sam Example"))
        requests.append(state)
        criteria = payload["questions"]["span"]["criteria"]
        if "⟦Sam Example⟧" in state:
            assert criteria.keys() == {"PERSON", "not_pii"}
            answer = "not_pii"
        else:
            assert criteria == CRITERIA
            answer = "AGE"
        return httpx.Response(200, json={"model": LocalLightningDecider.model, "answers": {"span": {
            "type": "choice", "choice": answer,
            "probabilities": {label: float(label == answer) for label in criteria},
        }}})

    decider = LocalLightningDecider(8080, transport=httpx.MockTransport(reply))
    try:
        guess = _candidate(text, "Mira", "PERSON", "SpacyRecognizer")
        kept, counts, trace = decider.analyze(text, RoutedResults([guess], []))
        assert [(text[c["start"]:c["end"]], c["entity_type"]) for c in kept] == [("Mira", "PERSON")]
        assert len(requests) == counts["decider_calls"] == 2
        assert {entry["reason"] for entry in trace} == {"not_pii", "relabel_requires_rule_evidence"}
    finally:
        decider.close()


@pytest.mark.parametrize("recognizer, text, expected", [
    ("IndicNameContextRecognizer", "రోగి పూర్తి పేరు ఏమిటి? అనయ వర్మ. జ్వరం తగ్గింది.", "అనయ వర్మ"),
    ("TeluguNameCueRecognizer", "నన్ను అనయ వర్మ అని పిలుస్తారు.", "అనయ వర్మ"),
    ("SpokenPlaceRecognizer", "మీరు ఎక్కడ నివసిస్తున్నారు? చింతల పేట. ఫాలో అప్ తరువాత.", "చింతల పేట"),
    ("SpokenPlaceRecognizer", "రోగి చంద్రపేటలో నివసిస్తున్నారు.", "చంద్రపేట"),
    ("AgeRecognizer", "The infant is 7 months old. Follow-up tomorrow.", "7"),
    ("LabelledMrnRecognizer", "Patient ID: 555-011-0234. Follow-up tomorrow.", "555-011-0234"),
    ("LabelledMrnRecognizer", "Record number is DEMO-4242. Temp 98.", "DEMO-4242"),
    ("LabelledMrnRecognizer", "రోగి గుర్తింపు సంఖ్య: ౧౨౩౪౫౬౭౮౯. మోతాదు 25.", "౧౨౩౪౫౬౭౮౯"),
])
def test_explicit_field_recognizers_emit_exact_values(recognizer, text, expected):
    rule = next(rule for rule in build_indic_identifier_recognizers() if rule.name == recognizer)
    values = {text[result.start:result.end] for result in rule.analyze(text, [])}
    assert expected in values


def test_age_rule_keeps_symptom_duration_visible():
    age = next(rule for rule in build_indic_identifier_recognizers() if rule.name == "AgeRecognizer")
    assert age.analyze("The cough started 7 months ago. Dose 25 mg.", ["AGE"]) == []


def test_local_rule_age_does_not_spend_a_confirmation_request():
    text = "Age: 42. The cough started 7 months ago."
    age = next(rule for rule in build_indic_identifier_recognizers() if rule.name == "AgeRecognizer")
    decider = LocalLightningDecider(8080, transport=httpx.MockTransport(lambda _: pytest.fail("rule ages skip H2O")))
    try:
        candidates, counts, _ = decider.analyze(text, RoutedResults([], age.analyze(text, ["AGE"])))
        selected, _ = resolve_entities(text, candidates)
        assert [(entity["text"], entity["canonical_entity_type"]) for entity in selected] == [("42", "AGE")]
        assert counts["decider_calls"] == 0
    finally:
        decider.close()


def test_labelled_medical_id_wins_over_a_phone_guess():
    text = "Patient ID: 555-011-0234. Temp 98."
    rule = next(rule for rule in build_indic_identifier_recognizers() if rule.name == "LabelledMrnRecognizer")
    phone = _candidate(text, "555-011-0234", "PHONE_NUMBER", "PhoneRecognizer")
    decider = LocalLightningDecider(8080, transport=httpx.MockTransport(lambda _: pytest.fail("validated fields skip H2O")))
    try:
        kept, counts, _ = decider.analyze(text, RoutedResults([phone], rule.analyze(text, [])))
        selected, _ = resolve_entities(text, kept)
        assert counts["decider_calls"] == 0
        assert [(entity["text"], entity["canonical_entity_type"]) for entity in selected] == [
            ("555-011-0234", "MEDICAL_RECORD_NUMBER"),
        ]
    finally:
        decider.close()
