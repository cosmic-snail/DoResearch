import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROFILE_COLUMNS = [
    "rule_type",
    "records",
    "answerable_records",
    "no_answer_records",
    "answer_rate",
    "total_gold_spans",
    "avg_gold_spans_per_answerable",
    "max_gold_spans_per_record",
    "avg_answer_chars",
    "avg_context_chars",
]


def profile_rule_types(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["rule_type"])].append(record)

    rows = [_profile_group(rule_type, group) for rule_type, group in grouped.items()]
    return sorted(rows, key=lambda row: row["rule_type"])


def write_rule_type_profile_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in PROFILE_COLUMNS})


def _profile_group(rule_type: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    answerable = [record for record in records if record["metadata"]["has_answer"]]
    gold_span_counts = [len(record["gold_spans"]) for record in records]
    answer_lengths = [
        len(span["text"])
        for record in records
        for span in record["gold_spans"]
    ]
    context_lengths = [len(record["source_text"]) for record in records]

    return {
        "rule_type": rule_type,
        "records": len(records),
        "answerable_records": len(answerable),
        "no_answer_records": len(records) - len(answerable),
        "answer_rate": len(answerable) / len(records) if records else 0.0,
        "total_gold_spans": sum(gold_span_counts),
        "avg_gold_spans_per_answerable": (
            sum(len(record["gold_spans"]) for record in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        "max_gold_spans_per_record": max(gold_span_counts) if gold_span_counts else 0,
        "avg_answer_chars": _mean(answer_lengths),
        "avg_context_chars": _mean(context_lengths),
    }


def _mean(values: Sequence[int]) -> float:
    return sum(values) / len(values) if values else 0.0
