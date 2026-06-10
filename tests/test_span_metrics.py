from rulex.evaluation.span_metrics import boundary_f1, evaluate_span_predictions, token_iou


def test_token_iou_uses_lowercase_word_tokens():
    assert token_iou("Maintain insurance, with limits.", "maintain limits") == 0.5


def test_boundary_f1_gives_partial_credit_for_one_correct_boundary():
    gold_spans = [{"text": "maintain insurance", "start": 10, "end": 28}]
    predicted_spans = [{"text": "maintain insurance today", "start": 10, "end": 34}]

    assert boundary_f1(gold_spans, predicted_spans) == 0.5


def test_evaluate_span_predictions_scores_answer_and_no_answer_records():
    records = [
        {
            "doc_id": "doc-1",
            "rule_type": "Insurance",
            "gold_spans": [{"text": "maintain insurance", "start": 10, "end": 28}],
            "metadata": {"has_answer": True},
        },
        {
            "doc_id": "doc-2",
            "rule_type": "Non-Compete",
            "gold_spans": [],
            "metadata": {"has_answer": False},
        },
    ]
    predictions = {
        ("doc-1", "Insurance"): [{"text": "maintain insurance", "start": 10, "end": 28, "score": 0.91}],
        ("doc-2", "Non-Compete"): [],
    }

    metrics = evaluate_span_predictions(records, predictions)

    assert metrics["records"] == 2
    assert metrics["answerable_records"] == 1
    assert metrics["no_answer_records"] == 1
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["boundary_f1"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["no_answer_accuracy"] == 1.0


def test_evaluate_span_predictions_penalizes_high_confidence_false_positives_in_ap():
    records = [
        {
            "doc_id": "doc-1",
            "rule_type": "Insurance",
            "gold_spans": [{"text": "maintain insurance", "start": 10, "end": 28}],
            "metadata": {"has_answer": True},
        },
        {
            "doc_id": "doc-2",
            "rule_type": "Non-Compete",
            "gold_spans": [],
            "metadata": {"has_answer": False},
        },
    ]
    predictions = {
        ("doc-1", "Insurance"): [{"text": "maintain insurance", "start": 10, "end": 28, "score": 0.7}],
        ("doc-2", "Non-Compete"): [{"text": "non-compete", "start": 0, "end": 11, "score": 0.9}],
    }

    metrics = evaluate_span_predictions(records, predictions)

    assert metrics["average_precision"] == 0.5
    assert metrics["no_answer_accuracy"] == 0.0
