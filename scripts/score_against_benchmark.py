"""Compatibility scorer for externally supplied literal-string reference exports.

Current unified benchmark evaluations use scripts.run_benchmark or
scripts.evaluate_unified_benchmark. This helper retains the old pii_dictionary
format for review-sheet repair and compatibility checks. It rejects reviewed
offset annotations so they cannot silently be scored against legacy labels.
"""
from __future__ import annotations

import argparse
import csv
try:
    from pii_pipeline import json_codec as json
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _ProjectRootPath
    sys.path.insert(0, str(_ProjectRootPath(__file__).resolve().parents[1]))
    from pii_pipeline import json_codec as json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

csv.field_size_limit(10 ** 9)

# Benchmark type -> the pipeline labels that count as the same thing. Used only
# for the type-accuracy figure; a span still counts as found under any label,
# because a misplaced placeholder is recoverable and an exposed identifier is not.
TYPE_ALIASES = {
    "PERSON": {"PERSON", "person", "first_name", "last_name", "full_name", "middle_name"},
    "PATIENT_ID": {"MEDICAL_ID", "MEDICAL_RECORD_NUMBER", "national_id_number", "account_id",
                   "sensitive_account_id", "GOVERNMENT_ID", "HEALTH_PLAN_BENEFICIARY_NUM",
                   "US_BANK_NUMBER", "US_PASSPORT", "US_SSN", "US_DRIVER_LICENSE", "account_number"},
    "LOCATION": {"LOCATION", "ADDRESS", "address", "city", "state_or_region", "street_address",
                 "country", "postal_code"},
    "DATE": {"DATE_TIME", "sensitive_date", "date_of_birth", "document_date", "expiration_date",
             "transaction_date"},
    "AGE": {"AGE"},
}


# Reverse of TYPE_ALIASES: a pipeline label -> the benchmark class it claims.
PIPE_TO_CLASS: Dict[str, str] = {}
for _cls, _labels in TYPE_ALIASES.items():
    for _label in _labels:
        PIPE_TO_CLASS[_label] = _cls

_SCRIPT_HINTS = (
    ("DEVANAGARI", "Hindi"), ("TELUGU", "Telugu"), ("KANNADA", "Kannada"),
    ("TAMIL", "Tamil"), ("BENGALI", "Bengali"), ("MALAYALAM", "Malayalam"),
)


def detect_language(text: str) -> str:
    """Dominant Indic script in a row, else Hinglish.

    These transcripts are code-mixed, so this reports the script that carries
    most of the letters rather than pretending a row has a single language.
    """
    import unicodedata
    counts: Dict[str, int] = {}
    for ch in text:
        if not ch.isalpha() or ch.isascii():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for prefix, language in _SCRIPT_HINTS:
            if name.startswith(prefix):
                counts[language] = counts.get(language, 0) + 1
                break
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "Hinglish"


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def normalise(text: str) -> str:
    """Fold the whitespace variants an export introduces.

    The benchmark came back with U+00A0 non-breaking spaces where the run has
    ordinary ones. Same character count, so offsets still line up 1:1 — but an
    exact-string join silently drops the row, which cost 13 of 49 rows on the
    first scoring pass.
    """
    return text.replace("\u00a0", " ").replace("\u200b", "")


def spans_of(text: str, needle: str) -> List[Tuple[int, int]]:
    """Every occurrence of `needle` in `text`."""
    out, start = [], text.find(needle)
    while start != -1:
        out.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    return out


def covered(span: Tuple[int, int], chars: Set[int]) -> float:
    a, b = span
    if b <= a:
        return 0.0
    return len(set(range(a, b)) & chars) / (b - a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="fraction of a true span that must be redacted to count as found")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    bench = list(csv.DictReader(open(args.benchmark, encoding="utf-8-sig")))
    if not bench or not {"context", "pii_dictionary"}.issubset(bench[0]):
        raise ValueError("Legacy scoring requires context and pii_dictionary; use evaluate_pipeline.py for offset review sheets")
    if "review_status" in bench[0]:
        raise ValueError("Review sheets must be reviewed and scored with evaluate_pipeline.py using verified_semantic_tags; legacy pii_dictionary is not the verified reference")
    run = list(csv.DictReader(open(args.run, encoding="utf-8-sig")))
    by_ctx = {normalise(r["context"]): r for r in run}

    tp = fn = 0                      # true spans found / missed
    pred_total = pred_hit = 0        # predicted spans, and those touching a true span
    type_right = type_wrong = 0
    missed: List[Tuple[str, str, str]] = []
    false_pos: Counter = Counter()
    fp_examples: Dict[str, List[str]] = defaultdict(list)
    per_type = defaultdict(lambda: {"total": 0, "found": 0})
    ent_tp: Counter = Counter()
    ent_fp: Counter = Counter()
    ent_fn: Counter = Counter()
    lang_tp: Counter = Counter()
    lang_fp: Counter = Counter()
    lang_fn: Counter = Counter()
    lang_rows: Counter = Counter()
    rows_matched = rows_missing = 0
    clean_rows = clean_rows_touched = 0
    clean_row_fp = 0

    for b in bench:
        r = by_ctx.get(normalise(b["context"]))
        if r is None:
            rows_missing += 1
            continue
        rows_matched += 1
        text = normalise(r["context"])
        language = detect_language(text)
        lang_rows[language] += 1
        truth = json.loads(b["pii_dictionary"] or "{}")
        preds = json.loads(r.get("pipeline_entities") or "[]")

        redacted_chars: Set[int] = set()
        for e in preds:
            redacted_chars.update(range(e["start"], e["end"]))

        true_spans: List[Tuple[Tuple[int, int], str]] = []
        for ttype, values in truth.items():
            for value in values:
                for sp in spans_of(text, normalise(value)) or []:
                    true_spans.append((sp, ttype))

        if not true_spans:
            clean_rows += 1
            if preds:
                clean_rows_touched += 1
                clean_row_fp += len(preds)

        true_char: Set[int] = set()
        for sp, _ in true_spans:
            true_char.update(range(*sp))

        for sp, ttype in true_spans:
            per_type[ttype]["total"] += 1
            if covered(sp, redacted_chars) >= args.min_overlap:
                tp += 1
                per_type[ttype]["found"] += 1
                hit_labels = {e["entity_type"] for e in preds
                              if e["start"] < sp[1] and e["end"] > sp[0]}
                if hit_labels & TYPE_ALIASES.get(ttype, set()):
                    type_right += 1
                else:
                    type_wrong += 1
                ent_tp[ttype] += 1
                lang_tp[language] += 1
            else:
                fn += 1
                ent_fn[ttype] += 1
                lang_fn[language] += 1
                missed.append((b["row_id"], ttype, text[sp[0]:sp[1]]))

        for e in preds:
            pred_total += 1
            if covered((e["start"], e["end"]), true_char) > 0:
                pred_hit += 1
            else:
                ent_fp[PIPE_TO_CLASS.get(e["entity_type"], "OTHER")] += 1
                lang_fp[language] += 1
                surface = (e.get("text") or text[e["start"]:e["end"]]).strip()
                false_pos[e["entity_type"]] += 1
                if len(fp_examples[e["entity_type"]]) < 6:
                    fp_examples[e["entity_type"]].append(surface)

    precision = pred_hit / pred_total if pred_total else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"benchmark : {args.benchmark}")
    print(f"run       : {args.run}")
    print(f"rows      : matched {rows_matched}, not found in run {rows_missing}")
    print(f"criterion : a true span counts as found when >= {args.min_overlap:.0%} of it is redacted")
    print()
    print("=" * 62)
    print(f"  PRECISION   {precision*100:6.1f}%    ({pred_hit} of {pred_total} detections hit real PII)")
    print(f"  RECALL      {recall*100:6.1f}%    ({tp} of {tp+fn} true PII spans redacted)")
    print(f"  F1          {f1*100:6.1f}%")
    print("=" * 62)
    print()
    print(f"placeholder type correct on found spans: {type_right}/{type_right+type_wrong}"
          f" ({100*type_right/(type_right+type_wrong):.1f}%)" if type_right + type_wrong else "")
    print()

    print("RECALL BY TYPE")
    for t in sorted(per_type, key=lambda k: -per_type[k]["total"]):
        v = per_type[t]
        print(f"   {t:12} {v['found']:>3}/{v['total']:<3}  {100*v['found']/v['total']:5.1f}%")
    print()

    if missed:
        print(f"PII LEFT EXPOSED ({len(missed)})")
        for rid, t, s in missed:
            print(f"   row {rid:>3}  {t:12} {s!r}")
        print()

    print("ENTITY-LEVEL PRECISION / RECALL / F1")
    print(f"   {'type':12}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>9}{'recall':>9}{'F1':>9}")
    ent_rows = {}
    for t in sorted(set(ent_tp) | set(ent_fn) | set(ent_fp), key=lambda k: -(ent_tp[k] + ent_fn[k])):
        p_, r_, f_ = prf(ent_tp[t], ent_fp[t], ent_fn[t])
        ent_rows[t] = {"tp": ent_tp[t], "fp": ent_fp[t], "fn": ent_fn[t],
                       "precision_pct": round(p_ * 100, 1), "recall_pct": round(r_ * 100, 1),
                       "f1_pct": round(f_ * 100, 1)}
        print(f"   {t:12}{ent_tp[t]:>5}{ent_fp[t]:>5}{ent_fn[t]:>5}"
              f"{p_*100:>8.1f}%{r_*100:>8.1f}%{f_*100:>8.1f}%")
    print()

    print("LANGUAGE-LEVEL PRECISION / RECALL / F1")
    print(f"   {'language':12}{'rows':>6}{'TP':>5}{'FP':>5}{'FN':>5}{'prec':>9}{'recall':>9}{'F1':>9}")
    lang_rows_out = {}
    for lg in sorted(lang_rows, key=lambda k: -(lang_tp[k] + lang_fn[k])):
        p_, r_, f_ = prf(lang_tp[lg], lang_fp[lg], lang_fn[lg])
        lang_rows_out[lg] = {"rows": lang_rows[lg], "tp": lang_tp[lg], "fp": lang_fp[lg],
                             "fn": lang_fn[lg], "precision_pct": round(p_ * 100, 1),
                             "recall_pct": round(r_ * 100, 1), "f1_pct": round(f_ * 100, 1)}
        print(f"   {lg:12}{lang_rows[lg]:>6}{lang_tp[lg]:>5}{lang_fp[lg]:>5}{lang_fn[lg]:>5}"
              f"{p_*100:>8.1f}%{r_*100:>8.1f}%{f_*100:>8.1f}%")
    print()

    print(f"FALSE POSITIVES ({pred_total - pred_hit} of {pred_total} detections)")
    for t, n in false_pos.most_common():
        print(f"   {t:24} {n:>4}   e.g. {fp_examples[t][:4]}")
    print()
    print(f"rows the benchmark marks as containing NO PII: {clean_rows}")
    print(f"   of those, rows the pipeline redacted something in: {clean_rows_touched}"
          f"  ({clean_row_fp} spans, all false positives)")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({
            "benchmark": args.benchmark, "run": args.run,
            "reference_status": "legacy_unreviewed",
            "rows_matched": rows_matched, "rows_missing": rows_missing,
            "recall_min_span_coverage": args.min_overlap,
            "precision_pct": round(precision * 100, 2),
            "recall_pct": round(recall * 100, 2),
            "f1_pct": round(f1 * 100, 2),
            "true_spans": tp + fn, "found": tp, "missed": fn,
            "detections": pred_total, "detections_on_pii": pred_hit,
            "type_correct": type_right, "type_wrong": type_wrong,
            "per_type": {k: dict(v) for k, v in per_type.items()},
            "exposed": [{"row": r, "type": t, "text": s} for r, t, s in missed],
            "false_positive_types": dict(false_pos),
            "entity_level": ent_rows,
            "language_level": lang_rows_out,
            "clean_rows": clean_rows, "clean_rows_touched": clean_rows_touched,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport written: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
