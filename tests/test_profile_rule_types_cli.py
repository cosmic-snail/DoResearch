import csv
import json

from rulex.cli.profile_rule_types import profile_file


def test_profile_file_reads_ruleframes_and_writes_csv(tmp_path):
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "profile.csv"
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
            "rule_type": "Insurance",
            "query": "Does the agreement require insurance?",
            "source_text": "No relevant clause.",
            "gold_spans": [],
            "predicted_spans": [],
            "frame": {"condition": [], "action": [], "value": [], "exception": [], "evidence": []},
            "metadata": {"has_answer": False, "answer_count": 0, "source_dataset": "CUAD"},
        },
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    rows = profile_file(input_path, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert rows[0]["rule_type"] == "Insurance"
    assert written[0]["records"] == "2"
    assert written[0]["answer_rate"] == "0.5"
