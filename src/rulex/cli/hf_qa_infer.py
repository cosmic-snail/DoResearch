"""Run a fine-tuned HuggingFace extractive QA model on RuleFrame JSONL and output predictions."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run HF extractive QA inference on RuleFrame JSONL"
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--input", type=Path, required=True, help="Test RuleFrame JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--max-records", type=int, help="Limit records (for testing)")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    logger.info("device=%s", device)

    logger.info("Loading model from %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(str(args.model))
    model.to(device)
    model.eval()
    logger.info("Model loaded. Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # Read test records
    records = _read_jsonl(args.input, limit=args.max_records)
    logger.info("Records to process: %d", len(records))

    # Group by document
    from collections import defaultdict
    doc_groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        doc_groups[r["doc_id"]].append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    log_path = args.output_dir / "inference.log"

    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    if args.verbose:
        logger.addHandler(logging.StreamHandler(sys.stderr))

    start_time = time.time()
    total = len(records)
    processed = 0
    all_preds: list[dict] = []

    # Truncate long contexts for speed (most relevant info is early in the document)
    MAX_CONTEXT_CHARS = 25000

    with torch.no_grad():
        for doc_id, doc_records in doc_groups.items():
            context = doc_records[0]["source_text"]
            if len(context) > MAX_CONTEXT_CHARS:
                context = context[:MAX_CONTEXT_CHARS]
                logger.info("Truncated context for %s from %d to %d chars",
                            doc_id[:50], len(doc_records[0]["source_text"]), MAX_CONTEXT_CHARS)

            for record in doc_records:
                query = record["query"]
                spans = _predict_spans(
                    model=model,
                    tokenizer=tokenizer,
                    query=query,
                    context=context,
                    max_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    device=device,
                )
                all_preds.append({
                    "doc_id": record["doc_id"],
                    "rule_type": record["rule_type"],
                    "predicted_spans": spans,
                })
                processed += 1

                if processed % 410 == 0 or processed == 1:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed * 60 if elapsed > 0 else 0
                    remaining = (total - processed) / rate if rate > 0 else 0
                    logger.info(
                        "[%d/%d] %.1f rec/min | remaining ~%.1f min",
                        processed, total, rate, remaining,
                    )

    # Write all predictions at once
    with open(predictions_path, "w", encoding="utf-8") as out_f:
        for pred in all_preds:
            out_f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    logger.info("DONE. %d records in %.1f min (%.1f rec/min)", total, elapsed / 60, total / elapsed * 60)

    summary = {
        "model_path": str(args.model),
        "records": total,
        "elapsed_min": round(elapsed / 60, 2),
        "device": device_str,
    }
    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def _predict_spans(
    model: Any,
    tokenizer: Any,
    query: str,
    context: str,
    max_length: int,
    doc_stride: int,
    device: Any,
) -> list[dict[str, Any]]:
    import torch
    import numpy as np

    enc = tokenizer(
        query,
        context,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
    )

    n_features = len(enc["input_ids"])

    # Track best span across all windows
    global_best_score = float("-inf")
    global_best_cls_score = float("-inf")
    global_best_char_start = 0
    global_best_char_end = 0

    SUB_BATCH_SIZE = 32
    for batch_start in range(0, n_features, SUB_BATCH_SIZE):
        batch_end = min(batch_start + SUB_BATCH_SIZE, n_features)

        batch_input_ids = torch.tensor(
            enc["input_ids"][batch_start:batch_end], device=device
        )
        batch_attention_mask = torch.tensor(
            enc["attention_mask"][batch_start:batch_end], device=device
        )

        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

        s_logits = outputs.start_logits.cpu().numpy()
        e_logits = outputs.end_logits.cpu().numpy()

        for local_idx in range(batch_end - batch_start):
            f_idx = batch_start + local_idx
            offsets = enc["offset_mapping"][f_idx]
            L = min(len(s_logits[local_idx]), 512)

            sl = s_logits[local_idx][:L]
            el = e_logits[local_idx][:L]

            # CLS score for no-answer
            cls_score = float(sl[0] + el[0])
            if cls_score > global_best_cls_score:
                global_best_cls_score = cls_score

            # Find best non-CLS span: for each end position, find best start
            best_score = -1e9
            best_s = 0
            best_e = 0

            for e in range(1, L):
                if offsets[e] == (0, 0):
                    continue
                s_start = max(1, e - 29)
                s_end = e
                if s_start >= s_end:
                    continue
                best_s_in_window = s_start + int(np.argmax(sl[s_start:s_end]))
                if offsets[best_s_in_window] == (0, 0) or offsets[best_s_in_window][0] >= offsets[e][1]:
                    # Scan for valid start
                    for s in range(s_start, s_end):
                        if offsets[s] != (0, 0) and offsets[s][0] < offsets[e][1]:
                            score = float(sl[s] + el[e])
                            if score > best_score:
                                best_score = score
                                best_s = s
                                best_e = e
                else:
                    score = float(sl[best_s_in_window] + el[e])
                    if score > best_score:
                        best_score = score
                        best_s = best_s_in_window
                        best_e = e

            if best_score > global_best_score:
                global_best_score = best_score
                global_best_char_start = offsets[best_s][0]
                global_best_char_end = offsets[best_e][1]

    # No-answer decision: ALWAYS return best span for CUAD evaluation
    # (CLS threshold is tuned for SQuAD, not CUAD)

    if global_best_score <= -1e8:
        return []

    span_text = context[global_best_char_start:global_best_char_end]
    return [{
        "start": global_best_char_start,
        "end": global_best_char_end,
        "text": span_text,
    }]


def _merge_overlapping_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(spans) <= 1:
        return spans
    spans = sorted(spans, key=lambda s: s["start"])
    merged = []
    current = spans[0]
    for next_span in spans[1:]:
        inter_start = max(current["start"], next_span["start"])
        inter_end = min(current["end"], next_span["end"])
        inter = max(0, inter_end - inter_start)
        union = (current["end"] - current["start"]) + (next_span["end"] - next_span["start"]) - inter
        iou = inter / union if union > 0 else 0
        if iou > 0.5:
            current["start"] = min(current["start"], next_span["start"])
            current["end"] = max(current["end"], next_span["end"])
        else:
            merged.append(current)
            current = next_span
    merged.append(current)
    return merged


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


if __name__ == "__main__":
    raise SystemExit(main())
