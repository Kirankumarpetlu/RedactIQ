"""Generate synthetic Hindi evaluation cases without reading repository data.

Fixed seeds are committed. Gold offsets are constructed from labelled segments,
never searched by surface, so repeated words remain unambiguous.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from pii_pipeline import json_codec as json
from .seeds import COMMON, TRICKY

SEEDS = {"A": 22072026, "B": 22082026, "C": 22092026, "heldout": 22102026}


@dataclass(frozen=True)
class Span:
    type: str
    start: int
    end: int
    surface: str


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    script: str
    text: str
    gold: tuple[Span, ...]


def assemble(case_id: str, split: str, script: str, segments: list[tuple[str, str]]) -> Case:
    text = ""
    gold = []
    for label, surface in segments:
        start = len(text)
        text += surface
        if label:
            gold.append(Span(label, start, len(text), surface))
    assert all(text[s.start:s.end] == s.surface for s in gold)
    return Case(case_id, split, script, text, tuple(gold))


def _perturb(base: list[tuple[str, str]], variant: int) -> list[tuple[str, str]]:
    seg = list(base)
    if variant & 1:
        seg = [(label, value.translate(str.maketrans("०१२३४५६७८९", "0123456789"))) for label, value in seg]
    if variant & 2:
        seg = [(label, value.replace("।", "").replace(";", ",")) for label, value in seg]
    if variant & 4:
        seg = [(label, value.replace("  ", " ").replace(" हूँ", " हूं").replace(" हूँ", " हूं")) for label, value in seg]
    if variant & 8:
        seg = [("", "मरीज़: ")] + seg
    if variant & 16:
        seg = seg + [("", " नर्स ने विवरण दर्ज किया।")]
    return seg


NEGATIVE_FRAMES = [
    "मैं अभी ठीक हूँ।", "कृपया बाद में बात करें।", "आज काम जल्दी समाप्त हुआ।",
    "कमल का फूल पानी में है।", "शांति बनाए रखें।", "सागर बहुत विशाल है।",
    "आशा बनी हुई है।", "राजा की कहानी पढ़ी।", "आकाश में बादल हैं।",
    "दवा की मात्रा ५० मिलीग्राम है।", "रक्तचाप १२०/८० दर्ज है।", "नब्ज़ ७२ है।",
    "कमरा १२ साफ है।", "वार्ड ४ में जगह है।", "पृष्ठ ४५ देखें।",
    "पाँच साल से उपचार चल रहा है।", "तीन महीने की गर्भावस्था दर्ज है।",
    "अस्पताल आज बंद है।", "कंपनी के नियम बदले।", "घर पर आराम करें।",
]


def generate() -> list[Case]:
    cases = []
    for group, seeds in (("A", COMMON), ("B", TRICKY)):
        rng = random.Random(SEEDS[group])
        for index, base in enumerate(seeds):
            cases.append(assemble(f"{group}-seed-{index:02d}", group, "Devanagari", base))
            order = list(range(32))
            rng.shuffle(order)
            for variant in order:
                cases.append(assemble(f"{group}-{index:02d}-{variant:02d}", group, "Devanagari", _perturb(base, variant)))
    rng = random.Random(SEEDS["C"])
    for index in range(220):
        frame = NEGATIVE_FRAMES[index % len(NEGATIVE_FRAMES)]
        prefix = "चिकित्सक: " if rng.randrange(2) else ""
        suffix = " अभी जवाब दें।" if rng.randrange(2) else ""
        cases.append(assemble(f"C-{index:03d}", "C", "Devanagari", [("", prefix + frame + suffix)]))
    return cases


def dump(path: Path) -> None:
    rows = []
    for case in generate():
        rows.append({"id": case.id, "split": case.split, "script": case.script,
                     "text": case.text, "gold": [vars(span) for span in case.gold]})
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    dump(parser.parse_args().output)
