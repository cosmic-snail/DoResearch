import argparse
import json
from pathlib import Path
from typing import Any

from rulex.evaluation.span_metrics import evaluate_span_predictions
from rulex.schema.ruleframe import validate_ruleframe


def run_null_baseline(
    input_path: Path,
    predictions_path: Path,
    results_path: Path,
) -> dict[str, float | int]:
    records = _read_jsonl(input_path)
    prediction_rows = [
        {"doc_id": record["doc_id"], "rule_type": record["rule_type"], "predicted_spans": []}
        for record in records
    ]
    predictions = {
        (row["doc_id"], row["rule_type"]): row["predicted_spans"] for row in prediction_rows
    }
    metrics = evaluate_span_predictions(records, predictions)
    _write_jsonl(predictions_path, prediction_rows)
    _write_json(results_path, metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an always-no-answer baseline.")
    parser.add_argument("input", type=Path, help="Path to RuleFrame JSONL")
    parser.add_argument("predictions", type=Path, help="Path for predictions JSONL output")
    parser.add_argument("results", type=Path, help="Path for metrics JSON output")
    args = parser.parse_args(argv)

    metrics = run_null_baseline(args.input, args.predictions, args.results)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(validate_ruleframe(json.loads(line)))
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
