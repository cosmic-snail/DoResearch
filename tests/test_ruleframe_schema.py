import pytest

from rulex.schema.ruleframe import validate_ruleframe


def test_validate_ruleframe_accepts_minimal_valid_record():
    record = {
        "doc_id": "contract-1",
        "domain": "legal",
        "rule_type": "Insurance",
        "query": "Does the agreement require insurance?",
        "source_text": "The supplier shall maintain insurance with limits of $1M.",
        "gold_spans": [
            {
                "text": "maintain insurance with limits of $1M",
                "start": 19,
                "end": 56,
            }
        ],
        "predicted_spans": [],
        "frame": {
            "condition": [],
            "action": [],
            "value": [],
            "exception": [],
            "evidence": [],
        },
        "metadata": {
            "has_answer": True,
            "answer_count": 1,
            "source_dataset": "CUAD",
        },
    }

    assert validate_ruleframe(record) == record


def test_validate_ruleframe_rejects_span_offsets_that_do_not_match_text():
    record = {
        "doc_id": "contract-1",
        "domain": "legal",
        "rule_type": "Insurance",
        "query": "Does the agreement require insurance?",
        "source_text": "The supplier shall maintain insurance with limits of $1M.",
        "gold_spans": [
            {
                "text": "insurance",
                "start": 0,
                "end": 9,
            }
        ],
        "predicted_spans": [],
        "frame": {
            "condition": [],
            "action": [],
            "value": [],
            "exception": [],
            "evidence": [],
        },
        "metadata": {
            "has_answer": True,
            "answer_count": 1,
            "source_dataset": "CUAD",
        },
    }

    with pytest.raises(ValueError, match="does not match source_text"):
        validate_ruleframe(record)
