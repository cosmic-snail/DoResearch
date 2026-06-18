import json

from rulex.cli.cuad_to_ruleframe import convert_file


def test_convert_file_writes_ruleframe_jsonl(tmp_path):
    context = "The supplier shall maintain insurance."
    answer_text = "maintain insurance"
    input_path = tmp_path / "cuad.json"
    output_path = tmp_path / "cuad.jsonl"
    input_path.write_text(
        json.dumps(
            {
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
                                                "answer_start": context.index(answer_text),
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    written = convert_file(input_path, output_path)

    assert written == 1
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["doc_id"] == "Supply_Agreement"
    assert rows[0]["rule_type"] == "Insurance"
