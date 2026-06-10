import csv

from rulex.analysis.rule_type_profile import profile_rule_types, write_rule_type_profile_csv


def test_profile_rule_types_computes_per_type_difficulty_features(tmp_path):
    records = [
        {
            "doc_id": "doc-1",
            "rule_type": "Insurance",
            "source_text": "a" * 100,
            "gold_spans": [{"text": "maintain insurance", "start": 10, "end": 28}],
            "metadata": {"has_answer": True},
        },
        {
            "doc_id": "doc-2",
            "rule_type": "Insurance",
            "source_text": "b" * 80,
            "gold_spans": [],
            "metadata": {"has_answer": False},
        },
        {
            "doc_id": "doc-3",
            "rule_type": "Non-Compete",
            "source_text": "c" * 120,
            "gold_spans": [
                {"text": "non-compete", "start": 5, "end": 16},
                {"text": "restricted business", "start": 30, "end": 49},
            ],
            "metadata": {"has_answer": True},
        },
    ]

    rows = profile_rule_types(records)

    assert rows[0]["rule_type"] == "Insurance"
    assert rows[0]["records"] == 2
    assert rows[0]["answerable_records"] == 1
    assert rows[0]["no_answer_records"] == 1
    assert rows[0]["answer_rate"] == 0.5
    assert rows[0]["avg_gold_spans_per_answerable"] == 1.0
    assert rows[1]["rule_type"] == "Non-Compete"
    assert rows[1]["total_gold_spans"] == 2
    assert rows[1]["max_gold_spans_per_record"] == 2


def test_write_rule_type_profile_csv_writes_stable_columns(tmp_path):
    output_path = tmp_path / "profile.csv"
    rows = [
        {
            "rule_type": "Insurance",
            "records": 2,
            "answerable_records": 1,
            "no_answer_records": 1,
            "answer_rate": 0.5,
            "total_gold_spans": 1,
            "avg_gold_spans_per_answerable": 1.0,
            "max_gold_spans_per_record": 1,
            "avg_answer_chars": 18.0,
            "avg_context_chars": 90.0,
        }
    ]

    write_rule_type_profile_csv(output_path, rows)

    with output_path.open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert written[0]["rule_type"] == "Insurance"
    assert written[0]["answer_rate"] == "0.5"
