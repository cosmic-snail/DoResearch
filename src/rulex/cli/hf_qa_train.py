import argparse
import json
from pathlib import Path
from typing import Any

from rulex.models.qa_feature_prep import answer_token_positions
from rulex.schema.ruleframe import validate_ruleframe


class RuleFrameQADataset:
    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]


def train_hf_qa_baseline(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
    from transformers.data.data_collator import default_data_collator

    train_records = _read_jsonl(args.train, limit=args.max_train_records)
    eval_records = _read_jsonl(args.eval, limit=args.max_eval_records) if args.eval else []

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    train_features = build_qa_features(
        train_records,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        doc_stride=args.doc_stride,
    )
    eval_features = build_qa_features(
        eval_records,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        doc_stride=args.doc_stride,
    ) if eval_records else []

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        use_cpu=args.device == "cpu",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RuleFrameQADataset(train_features),
        eval_dataset=RuleFrameQADataset(eval_features) if eval_features else None,
        data_collator=default_data_collator,
    )
    train_output = trainer.train()
    eval_metrics = trainer.evaluate() if eval_features else {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model_name": args.model_name,
        "device": args.device,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_features": len(train_features),
        "eval_features": len(eval_features),
        "max_seq_length": args.max_seq_length,
        "doc_stride": args.doc_stride,
        "max_steps": args.max_steps,
        "train_loss": float(train_output.training_loss),
        "eval_metrics": {key: float(value) for key, value in eval_metrics.items()},
        "torch_version": torch.__version__,
    }
    (args.output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def build_qa_features(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    doc_stride: int,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for record in records:
        tokenized = tokenizer(
            record["query"],
            record["source_text"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for feature_index in range(len(tokenized["input_ids"])):
            offsets = tokenized["offset_mapping"][feature_index]
            sequence_ids = tokenized.sequence_ids(feature_index)
            cls_index = tokenized["input_ids"][feature_index].index(tokenizer.cls_token_id)
            start_position, end_position = _positions_for_record(
                record,
                offsets=offsets,
                sequence_ids=sequence_ids,
                cls_index=cls_index,
            )
            features.append(
                {
                    "input_ids": tokenized["input_ids"][feature_index],
                    "attention_mask": tokenized["attention_mask"][feature_index],
                    "start_positions": start_position,
                    "end_positions": end_position,
                }
            )
    return features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a HuggingFace extractive QA baseline.")
    parser.add_argument("--train", type=Path, required=True, help="Training RuleFrame JSONL")
    parser.add_argument("--eval", type=Path, help="Evaluation RuleFrame JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-eval-records", type=int)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    metrics = train_hf_qa_baseline(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def _positions_for_record(
    record: dict[str, Any],
    offsets: list[tuple[int, int]],
    sequence_ids: list[int | None],
    cls_index: int,
) -> tuple[int, int]:
    if not record["gold_spans"]:
        return cls_index, cls_index
    first_span = record["gold_spans"][0]
    return answer_token_positions(
        offsets=offsets,
        sequence_ids=sequence_ids,
        answer_start=first_span["start"],
        answer_end=first_span["end"],
        cls_index=cls_index,
    )


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(validate_ruleframe(json.loads(line)))
            if limit is not None and len(records) >= limit:
                break
    return records


if __name__ == "__main__":
    raise SystemExit(main())
