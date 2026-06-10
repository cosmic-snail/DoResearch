from rulex.data_adapters.cuad_adapter import convert_cuad_to_ruleframes


def test_convert_cuad_to_ruleframes_preserves_answer_and_no_answer_cases():
    context = (
        "This Supply Agreement is made today. "
        "The supplier shall maintain insurance with limits of $1M."
    )
    answer_text = "maintain insurance with limits of $1M"
    answer_start = context.index(answer_text)
    cuad = {
        "data": [
            {
                "title": "Supply_Agreement",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "id": "Supply_Agreement__Insurance",
                                "question": "Does the agreement require insurance?",
                                "answers": [
                                    {
                                        "text": answer_text,
                                        "answer_start": answer_start,
                                    }
                                ],
                            },
                            {
                                "id": "Supply_Agreement__Non-Compete",
                                "question": "Does the agreement include a non-compete?",
                                "answers": [],
                            },
                        ],
                    }
                ],
            }
        ]
    }

    records = list(convert_cuad_to_ruleframes(cuad))

    assert len(records) == 2
    assert records[0]["doc_id"] == "Supply_Agreement"
    assert records[0]["rule_type"] == "Insurance"
    assert records[0]["metadata"]["has_answer"] is True
    assert records[0]["metadata"]["answer_count"] == 1
    assert records[0]["gold_spans"][0] == {
        "text": answer_text,
        "start": answer_start,
        "end": answer_start + len(answer_text),
    }
    assert records[1]["rule_type"] == "Non-Compete"
    assert records[1]["metadata"]["has_answer"] is False
    assert records[1]["metadata"]["answer_count"] == 0
    assert records[1]["gold_spans"] == []
