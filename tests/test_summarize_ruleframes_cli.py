import json

from rulex.cli.summarize_ruleframes import summarize_file


def test_summarize_file_reports_core_dataset_counts(tmp_path):
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "summary.json"
    records = [
        {
            "doc_id": "doc-1",
            "domain": "legal",
            "rule_type": "Insurance",
            "query": "Does the agreement require insurance?",
            "source_text": "The supplier shall maintain insurance.",
            "gold_spans": [{"text": "maintain insurance", "start": 19, "end": 37}],
            "predicted_spans": [],
            "frame": {"condition": [], "action": [], "value": [], "exception": [], "evidence": []},
            "metadata": {"has_answer": True, "answer_count": 1, "source_dataset": "CUAD"},
        },
        {
            "doc_id": "doc-1",
            "domain": "legal",
            "rule_type": "Non-Compete",
            "query": "Does the agreement include a non-compete?",
            "source_text": "The supplier shall maintain insurance.",
            "gold_spans": [],
            "predicted_spans": [],
            "frame": {"condition": [], "action": [], "value": [], "exception": [], "evidence": []},
            "metadata": {"has_answer": False, "answer_count": 0, "source_dataset": "CUAD"},
        },
        {
            "doc_id": "doc-2",
            "domain": "legal",
            "rule_type": "Insurance",
            "query": "Does the agreement require insurance?",
            "source_text": "The buyer must carry insurance and cyber insurance.",
            "gold_spans": [
                {"text": "carry insurance", "start": 15, "end": 30},
                {"text": "cyber insurance", "start": 35, "end": 50},
            ],
            "predicted_spans": [],
            "frame": {"condition": [], "action": [], "value": [], "exception": [], "evidence": []},
            "metadata": {"has_answer": True, "answer_count": 2, "source_dataset": "CUAD"},
        },
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    summary = summarize_file(input_path, output_path)

    assert summary["records"] == 3
    assert summary["documents"] == 2
    assert summary["rule_types"] == 2
    assert summary["answerable_records"] == 2
    assert summary["no_answer_records"] == 1
    assert summary["total_gold_spans"] == 3
    assert summary["max_gold_spans_per_record"] == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary
