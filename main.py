"""Run the synthetic benchmark or process a file through local H2O Lightning."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def synthetic_rows():
    """Generate original text and exact annotations from fictional templates."""
    names = (
        ("Mira Example", "मीरा उदाहरण", "మీరా ఉదాహరణ"),
        ("Ravi Sample", "रवि नमूना", "రవి నమూనా"),
        ("Anika Fable", "अनिका कथा", "అనిక కథ"),
        ("Dev Fiction", "देव कल्पना", "దేవ్ కల్పన"),
        ("Leela Story", "लीला कहानी", "లీల కథనం"),
        ("Arun Testcase", "अरुण परीक्षण", "అరుణ్ పరీక్ష"),
        ("Nila Imaginary", "नीला कल्पित", "నీల కల్పితం"),
        ("Kiran Mock", "किरण अभ्यास", "కిరణ్ అభ్యాసం"),
        ("Tara Placeholder", "तारा उदाहरण", "తార ఉదాహరణ"),
        ("Ishan Sandbox", "ईशान नमूना", "ఈశాన్ నమూనా"),
    )
    row_index = 0
    for index, (english, hindi, telugu) in enumerate(names, start=1):
        age = f"{34 + index} years old"
        place = f"{index} Imaginary Lane"
        email = f"example-{index}@example.invalid"
        medical_id = f"DEMO-{4100 + index}"
        for context, values in (
            (f"The nurse noted {english} arrived for review. Headache improved.", [(english, "PERSON")]),
            (f"The patient's name is {english}. They report fatigue.", [(english, "PERSON")]),
            (f"दर्ज नाम: {hindi}। मरीज को सिरदर्द है।", [(hindi, "PERSON")]),
            (f"రోగి పేరు {telugu}. తలనొప్పి తగ్గింది.", [(telugu, "PERSON")]),
            (f"The fictional patient is {age}. Started oral fluids.", [(age.split()[0], "AGE")]),
            (f"Patient address: {place}. Continue routine observation.", [(place, "LOCATION")]),
            (f"Contact email: {email}. Clinical review is complete.", [(email, "EMAIL")]),
            (f"Patient medical ID: {medical_id}. Blood pressure was normal.", [(medical_id, "MEDICAL_ID")]),
            (f"Fictional observation {index}: cough improved after rest. Continue oral fluids.", []),
            (f"काल्पनिक अवलोकन {index}। हल्का सिरदर्द और थकान है। తలనొప్పి తగ్గింది.", []),
        ):
            row_index += 1
            annotations = []
            for value, label in values:
                start = context.index(value)
                annotations.append({
                    "entity_type": label, "start": start, "end": start + len(value), "text": value,
                })
            yield {"row_index": row_index, "context": context,
                   "verified_semantic_tags": json.dumps(annotations, ensure_ascii=False)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h2o-port", type=int, help="Loopback port of the H2O Lightning llama.cpp server")
    parser.add_argument("--input", type=lambda path: Path(path).resolve(), help="Input file for the local server")
    parser.add_argument("--limit-rows", type=int, default=100)
    parser.add_argument("--save-diagnostics", action="store_true", help="Write per-row candidate diagnostics")
    args = parser.parse_args()
    if args.input is not None and args.h2o_port is None:
        parser.error("--input requires --h2o-port so file contents stay local")
    if args.limit_rows < 1:
        parser.error("--limit-rows must be positive")

    root = Path(__file__).resolve().parent
    os.chdir(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from pii_pipeline.decider_stage import LocalLightningDecider, OpenRouterDecider
    from pii_pipeline.run_csv_through_stage9 import run_pipeline_on_file
    from scripts.score_pending_groundtruth import score

    rows = list(synthetic_rows())[:args.limit_rows] if args.input is None else []
    try:
        decider = (LocalLightningDecider(args.h2o_port) if args.h2o_port is not None else
                   OpenRouterDecider(row["context"] for row in rows))
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    backend = "h2o_local" if args.h2o_port is not None else "decider_synthetic"
    run_dir = root / "output" / f"first_{args.limit_rows}_{backend}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    predictions = run_dir / "predictions.csv"
    reference = run_dir / "synthetic_reference.csv"
    try:
        run_dir.mkdir(parents=True)
        if args.input is None:
            with reference.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        summary = run_pipeline_on_file(
            input_file=str(args.input or reference),
            output_csv=str(predictions),
            config_path="config/pipeline_config.yaml",
            device="cpu",
            max_rows=args.limit_rows,
            save_diagnostics=args.save_diagnostics or args.input is None,
            decider=decider,
        )
        if args.input is not None:
            return 0
        metrics = score(reference, predictions)
        metrics["reference_status"] = "synthetic_generated_smoke_test"
        metrics["decider"] = summary["decider"]
        metrics["total_time_seconds"] = summary["total_time_seconds"]
        benchmark = run_dir / f"benchmark_first_{len(rows)}.json"
        benchmark.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Benchmark report    : {benchmark}")
    finally:
        decider.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
