from collections.abc import Iterator, Mapping
from typing import Any

from rulex.schema.ruleframe import empty_frame, validate_ruleframe


def convert_cuad_to_ruleframes(cuad: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Convert CUAD's SQuAD-style JSON into RuleFrame records."""
    for document in cuad.get("data", []):
        doc_id = document.get("title", "")
        for paragraph in document.get("paragraphs", []):
            source_text = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                record = {
                    "doc_id": doc_id,
                    "domain": "legal",
                    "rule_type": _infer_rule_type(qa),
                    "query": qa.get("question", ""),
                    "source_text": source_text,
                    "gold_spans": [_answer_to_span(answer) for answer in answers],
                    "predicted_spans": [],
                    "frame": empty_frame(),
                    "metadata": {
                        "has_answer": bool(answers),
                        "answer_count": len(answers),
                        "source_dataset": "CUAD",
                    },
                }
                yield validate_ruleframe(record)


def _infer_rule_type(qa: Mapping[str, Any]) -> str:
    qa_id = str(qa.get("id", ""))
    if "__" in qa_id:
        return qa_id.rsplit("__", maxsplit=1)[-1]
    return qa.get("question", qa_id)


def _answer_to_span(answer: Mapping[str, Any]) -> dict[str, Any]:
    text = answer["text"]
    start = int(answer["answer_start"])
    return {
        "text": text,
        "start": start,
        "end": start + len(text),
    }
