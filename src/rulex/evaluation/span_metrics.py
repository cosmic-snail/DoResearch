import re
from collections.abc import Mapping, Sequence
from typing import Any

_TOKEN_RE = re.compile(r"\b\w+\b")
_NUMERIC_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|percent|dollars?|\$\s*\d+(?:\.\d+)?|days?|years?|months?|weeks?|hours?)\b",
    re.IGNORECASE,
)
# also match standalone "$1,234" style
_DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def token_iou(gold_text: str, predicted_text: str) -> float:
    gold_tokens = set(_tokens(gold_text))
    predicted_tokens = set(_tokens(predicted_text))
    if not gold_tokens and not predicted_tokens:
        return 1.0
    if not gold_tokens or not predicted_tokens:
        return 0.0
    return len(gold_tokens & predicted_tokens) / len(gold_tokens | predicted_tokens)


def token_f1(gold_text: str, predicted_text: str) -> float:
    gold_tokens = _tokens(gold_text)
    predicted_tokens = _tokens(predicted_text)
    if not gold_tokens and not predicted_tokens:
        return 1.0
    if not gold_tokens or not predicted_tokens:
        return 0.0

    gold_counts = _counts(gold_tokens)
    predicted_counts = _counts(predicted_tokens)
    overlap = sum(min(gold_counts[token], predicted_counts.get(token, 0)) for token in gold_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def boundary_f1(
    gold_spans: Sequence[Mapping[str, Any]],
    predicted_spans: Sequence[Mapping[str, Any]],
) -> float:
    if not gold_spans or not predicted_spans:
        return 0.0
    return max(_span_boundary_f1(gold, predicted) for gold in gold_spans for predicted in predicted_spans)


# ── Patch 1: Multi-Span Recall ──────────────────────────────────────────────

def multi_span_recall(
    gold_spans: Sequence[Mapping[str, Any]],
    predicted_spans: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
) -> float:
    """Fraction of *gold_spans* matched by at least one prediction with IoU >= threshold.

    Unlike ``_best_token_f1`` which only looks at the single best overlap,
    this metric penalises systems that find only 1 of N required spans.
    """
    if not gold_spans:
        return 1.0
    if not predicted_spans:
        return 0.0
    hits = 0
    for gold in gold_spans:
        for pred in predicted_spans:
            if token_iou(str(gold.get("text", "")), str(pred.get("text", ""))) >= iou_threshold:
                hits += 1
                break
    return hits / len(gold_spans)


# ── Patch 2: Numeric Match ──────────────────────────────────────────────────

def numeric_match(
    gold_spans: Sequence[Mapping[str, Any]],
    predicted_spans: Sequence[Mapping[str, Any]],
) -> float:
    """Fraction of gold *(value, unit)* pairs that appear verbatim in predictions.

    Each "(value, unit)" pair must match exactly (same digits + same normalised unit).
    Returns 1.0 when gold contains no numeric content (trivially correct).
    """
    gold_pairs = _extract_numeric_pairs(
        " ".join(str(s.get("text", "")) for s in gold_spans)
    )
    if not gold_pairs:
        return 1.0
    pred_pairs = _extract_numeric_pairs(
        " ".join(str(s.get("text", "")) for s in predicted_spans)
    )
    hits = len(gold_pairs & pred_pairs)
    return hits / len(gold_pairs)


def _extract_numeric_pairs(text: str) -> set[tuple[str, str]]:
    """Extract normalised (value, unit) pairs from *text*."""
    pairs: set[tuple[str, str]] = set()

    # value+unit pairs: "30 days", "0.5%", "5%", etc.
    for m in _NUMERIC_UNIT_RE.finditer(text):
        value = m.group(1)
        unit = m.group(2).lower().strip()
        if unit in ("percent",):
            unit = "%"
        if unit.startswith("dollar") or unit.startswith("$"):
            value = m.group(1).replace(",", "")
            unit = "$"
        pairs.add((value, unit))

    # standalone dollar amounts: "$1,234"
    for m in _DOLLAR_RE.finditer(text):
        value = m.group(1).replace(",", "")
        pairs.add((value, "$"))

    return pairs


def evaluate_span_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> dict[str, float | int]:
    answerable_records = [record for record in records if record.get("metadata", {}).get("has_answer")]
    no_answer_records = [record for record in records if not record.get("metadata", {}).get("has_answer")]

    exact_scores: list[float] = []
    f1_scores: list[float] = []
    boundary_scores: list[float] = []
    msr_scores: list[float] = []
    nm_scores: list[float] = []
    for record in answerable_records:
        predicted_spans = predictions.get((record["doc_id"], record["rule_type"]), [])
        exact_scores.append(_best_exact(record["gold_spans"], predicted_spans))
        f1_scores.append(_best_token_f1(record["gold_spans"], predicted_spans))
        boundary_scores.append(boundary_f1(record["gold_spans"], predicted_spans))
        msr_scores.append(multi_span_recall(record["gold_spans"], predicted_spans))
        nm_scores.append(numeric_match(record["gold_spans"], predicted_spans))

    no_answer_hits = 0
    for record in no_answer_records:
        predicted_spans = predictions.get((record["doc_id"], record["rule_type"]), [])
        if not predicted_spans:
            no_answer_hits += 1

    return {
        "records": len(records),
        "answerable_records": len(answerable_records),
        "no_answer_records": len(no_answer_records),
        "exact_match": _mean(exact_scores),
        "token_f1": _mean(f1_scores),
        "boundary_f1": _mean(boundary_scores),
        "multi_span_recall": _mean(msr_scores),
        "numeric_match": _mean(nm_scores),
        "average_precision": _average_precision(records, predictions),
        "no_answer_accuracy": no_answer_hits / len(no_answer_records) if no_answer_records else 0.0,
    }


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _counts(tokens: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _best_exact(
    gold_spans: Sequence[Mapping[str, Any]],
    predicted_spans: Sequence[Mapping[str, Any]],
) -> float:
    for gold in gold_spans:
        for predicted in predicted_spans:
            if gold.get("start") == predicted.get("start") and gold.get("end") == predicted.get("end"):
                return 1.0
            if str(gold.get("text", "")).strip() == str(predicted.get("text", "")).strip():
                return 1.0
    return 0.0


def _best_token_f1(
    gold_spans: Sequence[Mapping[str, Any]],
    predicted_spans: Sequence[Mapping[str, Any]],
) -> float:
    if not gold_spans or not predicted_spans:
        return 0.0
    return max(
        token_f1(str(gold.get("text", "")), str(predicted.get("text", "")))
        for gold in gold_spans
        for predicted in predicted_spans
    )


def _span_boundary_f1(gold: Mapping[str, Any], predicted: Mapping[str, Any]) -> float:
    gold_boundaries = {("start", gold.get("start")), ("end", gold.get("end"))}
    predicted_boundaries = {("start", predicted.get("start")), ("end", predicted.get("end"))}
    overlap = len(gold_boundaries & predicted_boundaries)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_boundaries)
    recall = overlap / len(gold_boundaries)
    return 2 * precision * recall / (precision + recall)


def _average_precision(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> float:
    total_positive_records = sum(1 for record in records if record.get("metadata", {}).get("has_answer"))
    if total_positive_records == 0:
        return 0.0

    candidates: list[tuple[float, bool]] = []
    for record in records:
        predicted_spans = predictions.get((record["doc_id"], record["rule_type"]), [])
        if not predicted_spans:
            continue

        score = max(float(predicted.get("score", 0.0)) for predicted in predicted_spans)
        hit = bool(record.get("metadata", {}).get("has_answer")) and (
            _best_token_iou(record.get("gold_spans", []), predicted_spans) >= 0.5
        )
        candidates.append((score, hit))

    if not candidates:
        return 0.0

    true_positives = 0
    precision_sum = 0.0
    for rank, (_, hit) in enumerate(sorted(candidates, key=lambda item: item[0], reverse=True), start=1):
        if hit:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / total_positive_records


def _best_token_iou(
    gold_spans: Sequence[Mapping[str, Any]],
    predicted_spans: Sequence[Mapping[str, Any]],
) -> float:
    if not gold_spans or not predicted_spans:
        return 0.0
    return max(
        token_iou(str(gold.get("text", "")), str(predicted.get("text", "")))
        for gold in gold_spans
        for predicted in predicted_spans
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
