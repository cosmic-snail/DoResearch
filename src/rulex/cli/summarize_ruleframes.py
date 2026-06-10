import argparse
import json
from pathlib import Path
from typing import Any

from rulex.schema.ruleframe import validate_ruleframe


def summarize_file(input_path: Path, output_path: Path) -> dict[str, float | int]:
    records = _read_jsonl(input_path)
    summary = summarize_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    documents = {record["doc_id"] for record in records}
    rule_types = {record["rule_type"] for record in records}
    answerable = [record for record in records if record["metadata"]["has_answer"]]
    no_answer = [record for record in records if not record["metadata"]["has_answer"]]
    gold_span_counts = [len(record["gold_spans"]) for record in records]
    gold_span_lengths = [
        len(span["text"])
        for record in records
        for span in record["gold_spans"]
    ]

    return {
        "records": len(records),
        "documents": len(documents),
        "rule_types": len(rule_types),
        "answerable_records": len(answerable),
        "no_answer_records": len(no_answer),
        "total_gold_spans": sum(gold_span_counts),
        "max_gold_spans_per_record": max(gold_span_counts) if gold_span_counts else 0,
        "avg_gold_span_chars": _mean(gold_span_lengths),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize RuleFrame JSONL records.")
    parser.add_argument("input", type=Path, help="Path to RuleFrame JSONL")
    parser.add_argument("output", type=Path, help="Path for summary JSON output")
    args = parser.parse_args(argv)

    summary = summarize_file(args.input, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(validate_ruleframe(json.loads(line)))
    return records


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
