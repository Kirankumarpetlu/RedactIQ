"""Small synthetic checks for the Hindi identity rule boundaries."""
from pii_pipeline.recognizers.hindi_identity import HindiNameRecognizer, HindiOrganizationRecognizer


def _surfaces(recognizer, text):
    return [text[result.start:result.end] for result in recognizer.analyze(text, [])]


def test_hindi_name_rules_keep_titles_suffixes_and_postpositions_outside():
    text = "मेरी माता का नाम श्री नंदिता जोशी जी है।"
    assert _surfaces(HindiNameRecognizer(), text) == ["नंदिता जोशी"]


def test_hindi_cues_do_not_make_predicates_into_names():
    recognizer = HindiNameRecognizer()
    for text in ("मैं इंतज़ार कर रही हूँ।", "मैं बोल रहा हूँ।",
                 "डॉक्टर हो जाइए।", "डॉक्टर I am having vomiting"):
        assert _surfaces(recognizer, text) == []
    assert _surfaces(recognizer, "डॉक्टर राम।") == ["राम"]


def test_hindi_organization_requires_personal_relation_and_named_surface():
    recognizer = HindiOrganizationRecognizer()
    assert _surfaces(recognizer, "मुझे नवोदय क्लिनिक में भर्ती किया गया।") == ["नवोदय क्लिनिक"]
    assert _surfaces(recognizer, "क्लिनिक में भर्ती किया गया।") == []
    assert _surfaces(recognizer, "कंपनी के नियम बदले।") == []


def test_hindi_rules_are_noop_on_english_and_other_indic_script():
    for text in ("My name is Nadia Holt.", "నా పేరు దీపక్ రావు."):
        assert _surfaces(HindiNameRecognizer(), text) == []
        assert _surfaces(HindiOrganizationRecognizer(), text) == []
