import json

from rulex.cli.smoke_eval import run_oracle_smoke_eval


def test_run_oracle_smoke_eval_writes_predictions_and_results(tmp_path):
    input_path = tmp_path / "records.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    results_path = tmp_path / "results.json"
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
            "doc_id": "doc-2",
            "domain": "legal",
            "rule_type": "Non-Compete",
            "query": "Does the agreement include a non-compete?",
            "source_text": "No relevant clause.",
            "gold_spans": [],
            "predicted_spans": [],
            "frame": {"condition": [], "action": [], "value": [], "exception": [], "evidence": []},
            "metadata": {"has_answer": False, "answer_count": 0, "source_dataset": "CUAD"},
        },
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    metrics = run_oracle_smoke_eval(input_path, predictions_path, results_path)

    prediction_rows = [
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert metrics["records"] == 2
    assert metrics["token_f1"] == 1.0
    assert json.loads(results_path.read_text(encoding="utf-8"))["exact_match"] == 1.0
    assert prediction_rows[0]["predicted_spans"][0]["text"] == "maintain insurance"
    assert prediction_rows[1]["predicted_spans"] == []


def test_run_oracle_smoke_eval_reads_jsonl_with_unicode_line_separator(tmp_path):
    input_path = tmp_path / "records.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    results_path = tmp_path / "results.json"
    source_text = "Clause heading\u2028The supplier shall maintain insurance."
    answer_text = "maintain insurance"
    record = {
        "doc_id": "doc-1",
        "domain": "legal",
        "rule_type": "Insurance",
        "query": "Does the agreement require insurance?",
        "source_text": source_text,
        "gold_spans": [
            {
                "text": answer_text,
                "start": source_text.index(answer_text),
                "end": source_text.index(answer_text) + len(answer_text),
            }
        ],
        "predicted_spans": [],
        "frame": {"condition": [], "action": [], "value": [], "exception": [], "evidence": []},
        "metadata": {"has_answer": True, "answer_count": 1, "source_dataset": "CUAD"},
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics = run_oracle_smoke_eval(input_path, predictions_path, results_path)

    assert metrics["records"] == 1
    assert metrics["exact_match"] == 1.0
