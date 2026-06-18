"""CLI: LLM-as-Judge — structured quality evaluation of span predictions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from rulex.models.llm_extractor import LLMClient
from rulex.schema.ruleframe import validate_ruleframe

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """\
You are an expert evaluator of clause extraction systems for legal and \
regulatory documents. Your task is to compare a PREDICTED extraction \
against a GOLD STANDARD extraction and rate its quality on structured \
dimensions.

RATING SCALES:

1. completeness (1-5): Were ALL relevant text fragments found?
   5 = every gold span has a matching prediction (IoU >= 0.5)
   3 = main span(s) found but secondary spans partially missed
   1 = completely wrong, empty, or irrelevant

2. precision (1-5): Is the predicted text free of irrelevant content?
   5 = spans contain ONLY the relevant clause, no preamble, no noise
   3 = relevant clause found but with noticeable extra text
   1 = mostly noise or wrong content

3. boundary_exactness (1-5): How accurate are the start/end cut points?
   5 = exact character-level match on all spans
   3 = rough sentence-level match but start/end drift by a few words
   1 = completely different location or severely truncated

4. rule_usability (yes / no): Could this extraction serve as the basis
   for encoding an executable rule with condition/obligation/value/exception
   fields? Think: does it capture WHAT must be done, by WHOM, under WHAT
   conditions, with WHAT thresholds?

5. critical_flaw (string or null): Identify the single most severe issue.
   Examples: "missed duration threshold (7 days)", "included unrelated
   preamble text", "false positive on clause type not present in contract",
   "truncated before the numerical value". If no flaw, return null.

Return ONLY valid JSON. No markdown, no explanation outside the JSON block.\
"""

JUDGE_USER_TEMPLATE = """\
CONTRACT EXCERPT (context surrounding the relevant clause):
---
{context_excerpt}
---

GOLD STANDARD EXTRACTION:
{gold_formatted}

PREDICTED EXTRACTION (by {model_label}):
{pred_formatted}

Evaluate the prediction. Return JSON:
{{
  "completeness": <1-5>,
  "precision": <1-5>,
  "boundary_exactness": <1-5>,
  "rule_usability": <true/false>,
  "critical_flaw": <"description" or null>
}}\
"""


def _context_window(source_text: str, spans: list[dict[str, Any]], margin: int = 400) -> str:
    """Return a text window covering *spans* with *margin* chars on each side."""
    if not spans:
        return source_text[: margin * 2]
    starts = [s.get("start", 0) for s in spans if isinstance(s.get("start"), int) and s["start"] >= 0]
    ends = [s.get("end", 0) for s in spans if isinstance(s.get("end"), int) and s["end"] >= 0]
    if not starts or not ends:
        return source_text[: margin * 2]
    win_start = max(0, min(starts) - margin)
    win_end = min(len(source_text), max(ends) + margin)
    return source_text[win_start:win_end]


def _format_spans(spans: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, s in enumerate(spans, 1):
        parts.append(f"  [{i}] \"{s.get('text', '')}\"")
    return "\n".join(parts) if parts else "  (none)"


def run_judge_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    client = LLMClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=0.0,
        max_tokens=1024,
        timeout=args.timeout,
    )

    records = _read_records(args.input, limit=args.max_records)
    preds = _load_predictions(args.predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "judge.log"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stderr)],
    )

    model_label = args.model_label or args.model

    verdicts: list[dict[str, Any]] = []
    ok = 0
    parse_err = 0
    api_err = 0
    t0 = time.monotonic()

    # only judge answerable records (no-answer cases are trivial for span eval)
    answerable = [r for r in records if r.get("metadata", {}).get("has_answer")]

    for i, record in enumerate(answerable):
        key = (record["doc_id"], record["rule_type"])
        predicted_spans = preds.get(key, [])

        gold_text = _format_spans(record["gold_spans"])
        pred_text = _format_spans(predicted_spans)
        ctx = _context_window(record["source_text"], record["gold_spans"] + predicted_spans)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": JUDGE_SYSTEM.strip()},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    context_excerpt=ctx,
                    gold_formatted=gold_text,
                    pred_formatted=pred_text,
                    model_label=model_label,
                ),
            },
        ]

        try:
            raw = client.chat(messages)
        except Exception as exc:
            logger.error("API failed for %s/%s: %s", record["doc_id"][:40], record["rule_type"], exc)
            api_err += 1
            verdicts.append(
                {
                    "doc_id": record["doc_id"],
                    "rule_type": record["rule_type"],
                    "completeness": 0,
                    "precision": 0,
                    "boundary_exactness": 0,
                    "rule_usability": False,
                    "critical_flaw": f"API_ERROR: {exc}",
                    "_raw": "",
                }
            )
            continue

        # Parse JSON from response
        try:
            import re as _re

            json_match = _re.search(r"\{[\s\S]*\}", raw)
            parsed = json.loads(json_match.group(0) if json_match else raw)
        except Exception:
            logger.warning("JSON parse failed for %s/%s", record["doc_id"][:40], record["rule_type"])
            parse_err += 1
            parsed = {
                "completeness": 0,
                "precision": 0,
                "boundary_exactness": 0,
                "rule_usability": False,
                "critical_flaw": "PARSE_ERROR",
            }

        verdict = {
            "doc_id": record["doc_id"],
            "rule_type": record["rule_type"],
            **{k: parsed.get(k) for k in ("completeness", "precision", "boundary_exactness", "rule_usability", "critical_flaw")},
            "_raw": raw[:300],
        }
        verdicts.append(verdict)
        ok += 1

        if (i + 1) % 5 == 0:
            elapsed = time.monotonic() - t0
            logger.info(
                "[%d/%d] %.1f rec/min | ok=%d parse_err=%d api_err=%d",
                i + 1,
                len(answerable),
                (i + 1) / elapsed * 60 if elapsed > 0 else 0,
                ok,
                parse_err,
                api_err,
            )

    elapsed = time.monotonic() - t0

    # Aggregate
    avg_comp = sum(v.get("completeness", 0) for v in verdicts) / len(verdicts) if verdicts else 0.0
    avg_prec = sum(v.get("precision", 0) for v in verdicts) / len(verdicts) if verdicts else 0.0
    avg_bound = sum(v.get("boundary_exactness", 0) for v in verdicts) / len(verdicts) if verdicts else 0.0
    usable = sum(1 for v in verdicts if v.get("rule_usability")) / len(verdicts) if verdicts else 0.0

    summary = {
        "model": args.model,
        "model_label": model_label,
        "records_judged": len(verdicts),
        "avg_completeness": round(avg_comp, 2),
        "avg_precision": round(avg_prec, 2),
        "avg_boundary_exactness": round(avg_bound, 2),
        "rule_usability_rate": round(usable, 3),
        "parse_error_rate": round(parse_err / len(answerable), 4) if answerable else 0.0,
        "api_error_rate": round(api_err / len(answerable), 4) if answerable else 0.0,
        "elapsed_min": round(elapsed / 60, 2),
    }

    # Write outputs
    (args.output_dir / "judge_verdicts.jsonl").write_text(
        "\n".join(json.dumps(v, ensure_ascii=True) for v in verdicts),
        encoding="utf-8",
    )
    (args.output_dir / "judge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Judge complete: %d verdicts in %.1f min", len(verdicts), elapsed / 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


# ── helpers ─────────────────────────────────────────────────────────────────

def _read_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        records.append(validate_ruleframe(json.loads(line)))
        if limit and len(records) >= limit:
            break
    return records


def _load_predictions(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    preds: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["doc_id"], r["rule_type"])
        preds[key] = r.get("predicted_spans", [])
    return preds


# ── main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation of span predictions.")
    parser.add_argument("--input", type=Path, required=True, help="RuleFrame JSONL (ground truth)")
    parser.add_argument("--predictions", type=Path, required=True, help="Predictions JSONL to judge")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--model", default="deepseek-chat", help="LLM model for judging")
    parser.add_argument("--model-label", help="Label for the model being judged (default: --model)")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-records", type=int, help="Limit records (default: all answerable)")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(argv)
    try:
        run_judge_evaluation(args)
    except Exception:
        logger.exception("Fatal error in judge evaluation")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
