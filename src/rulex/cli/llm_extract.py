"""CLI: run LLM-based span extraction on a RuleFrame JSONL dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from rulex.evaluation.span_metrics import evaluate_span_predictions
from rulex.models.llm_extractor import (
    LLMClient,
    _load_prompt_strategies,
    _pick_few_shot_examples,
)
from rulex.schema.ruleframe import validate_ruleframe

logger = logging.getLogger(__name__)


def run_llm_extraction(args: argparse.Namespace) -> dict[str, Any]:
    """Main extraction loop. Returns aggregate metrics dict."""
    strategies = _load_prompt_strategies()
    strategy_name = args.prompt_strategy
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown prompt strategy {strategy_name!r}. Available: {list(strategies)}"
        )
    strategy = strategies[strategy_name]

    client = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    records = _read_records(args.input, limit=args.max_records)
    train_records: list[dict[str, Any]] = []
    few_shot_n = 0
    if strategy_name in ("few_shot_3",):
        few_shot_n = 3
    elif strategy_name in ("few_shot_5",):
        few_shot_n = 5

    if few_shot_n and args.train:
        train_records = _read_records(args.train)
        logger.info("Loaded %d training records for few-shot example pool", len(train_records))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    results_path = args.output_dir / "results.json"
    log_path = args.output_dir / "extraction.log"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )

    prediction_rows: list[dict[str, Any]] = []
    parse_errors = 0
    total_spans = 0
    total_no_answer = 0
    t0 = time.monotonic()

    for i, record in enumerate(records):
        doc_id = record["doc_id"]
        rule_type = record["rule_type"]
        query = record["query"]
        context = record["source_text"]

        if args.max_context_chars and len(context) > args.max_context_chars:
            context = context[: args.max_context_chars]

        examples: list[dict[str, Any]] | None = None
        if few_shot_n and train_records:
            examples = _pick_few_shot_examples(
                train_records, rule_type, n=few_shot_n, max_context_chars=3000
            )
            # ensure no example comes from the same doc
            examples = [ex for ex in examples if ex.get("query") != query]

        try:
            result = client.extract_spans(
                context=context, query=query, strategy=strategy, examples=examples
            )
        except Exception as exc:
            logger.error("LLM call failed for doc=%s type=%s: %s", doc_id, rule_type, exc)
            result = {
                "answerable": False,
                "spans": [],
                "raw_response": "",
                "parse_error": str(exc),
            }

        if result["parse_error"]:
            parse_errors += 1
        if not result["answerable"]:
            total_no_answer += 1
        total_spans += len(result["spans"])

        predicted_spans = []
        for span in result["spans"]:
            predicted_spans.append(
                {
                    "text": span["text"],
                    "start": span["start"],
                    "end": span["end"],
                    "score": span.get("score", 0.9),
                }
            )

        prediction_rows.append(
            {
                "doc_id": doc_id,
                "rule_type": rule_type,
                "predicted_spans": predicted_spans,
                "_raw_response": result.get("raw_response", ""),
                "_parse_error": result.get("parse_error"),
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d] %.1f rec/min | parse_err=%d | spans=%d | no_answer=%d",
                i + 1,
                len(records),
                rate * 60,
                parse_errors,
                total_spans,
                total_no_answer,
            )

    elapsed = time.monotonic() - t0
    logger.info("Extraction complete: %d records in %.1f min", len(records), elapsed / 60)

    # ── write predictions ──────────────────────────────────────────────────
    _write_jsonl(predictions_path, prediction_rows)

    # ── evaluate ───────────────────────────────────────────────────────────
    predictions_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (row["doc_id"], row["rule_type"])
        predictions_map[key] = row["predicted_spans"]

    metrics = evaluate_span_predictions(records, predictions_map)
    metrics["parse_error_rate"] = parse_errors / len(records) if records else 0.0
    metrics["total_extracted_spans"] = total_spans
    metrics["llm_no_answer_rate"] = total_no_answer / len(records) if records else 0.0
    metrics["extraction_time_min"] = round(elapsed / 60, 2)
    metrics["model"] = args.model
    metrics["prompt_strategy"] = strategy_name

    _write_json(results_path, metrics)
    logger.info("Results written to %s", results_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    return metrics


# ── helpers ─────────────────────────────────────────────────────────────────

def _read_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw = path.read_text(encoding="utf-8")
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        records.append(validate_ruleframe(json.loads(stripped)))
        if limit and len(records) >= limit:
            break
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


# ── main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM-based span extraction on RuleFrame JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="RuleFrame JSONL to extract from")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--train", type=Path, help="Training RuleFrame JSONL (for few-shot examples)")
    parser.add_argument("--prompt-strategy", default="zero_shot",
                        choices=["zero_shot", "few_shot_3", "few_shot_5", "chain_of_thought"])
    parser.add_argument("--model", default="deepseek-chat",
                        help="Model name (e.g. deepseek-chat, gpt-4o)")
    parser.add_argument("--api-key", help="API key (or set DEEPSEEK_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--base-url", default="https://api.deepseek.com",
                        help="API base URL (default: DeepSeek)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-records", type=int, help="Limit records (for smoke testing)")
    parser.add_argument("--max-context-chars", type=int,
                        help="Truncate context to N chars (default: full)")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(argv)
    try:
        run_llm_extraction(args)
    except Exception:
        logger.exception("Fatal error during LLM extraction")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
