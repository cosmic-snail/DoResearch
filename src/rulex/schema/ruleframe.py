from collections.abc import Mapping
from typing import Any

REQUIRED_FIELDS = {
    "doc_id",
    "domain",
    "rule_type",
    "query",
    "source_text",
    "gold_spans",
    "predicted_spans",
    "frame",
    "metadata",
}

FRAME_FIELDS = ("condition", "action", "value", "exception", "evidence")


def empty_frame() -> dict[str, list[Any]]:
    return {field: [] for field in FRAME_FIELDS}


def validate_ruleframe(record: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"RuleFrame missing required fields: {sorted(missing)}")

    frame = record["frame"]
    if not isinstance(frame, Mapping):
        raise ValueError("RuleFrame frame must be a mapping")
    missing_frame_fields = set(FRAME_FIELDS) - frame.keys()
    if missing_frame_fields:
        raise ValueError(f"RuleFrame frame missing fields: {sorted(missing_frame_fields)}")

    source_text = record["source_text"]
    for span_group in ("gold_spans", "predicted_spans"):
        for span in record[span_group]:
            _validate_span(source_text, span)

    return record


def _validate_span(source_text: str, span: Mapping[str, Any]) -> None:
    missing = {"text", "start", "end"} - span.keys()
    if missing:
        raise ValueError(f"Span missing required fields: {sorted(missing)}")

    start = span["start"]
    end = span["end"]
    text = span["text"]
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("Span start and end must be integers")
    if start < 0 or end < start or end > len(source_text):
        raise ValueError("Span offsets are outside source_text bounds")
    if source_text[start:end] != text:
        raise ValueError("Span text does not match source_text at offsets")
