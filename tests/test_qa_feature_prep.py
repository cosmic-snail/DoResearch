from rulex.models.qa_feature_prep import answer_token_positions, best_text_span


def test_answer_token_positions_maps_char_offsets_to_context_tokens():
    offsets = [(0, 0), (0, 4), (5, 13), (14, 23), (24, 28), (29, 37), (0, 0)]
    sequence_ids = [None, 0, 1, 1, 1, 1, None]

    start, end = answer_token_positions(
        offsets=offsets,
        sequence_ids=sequence_ids,
        answer_start=5,
        answer_end=23,
        cls_index=0,
    )

    assert (start, end) == (2, 3)


def test_answer_token_positions_uses_cls_when_answer_not_inside_window():
    offsets = [(0, 0), (0, 4), (50, 60), (61, 70), (0, 0)]
    sequence_ids = [None, 0, 1, 1, None]

    start, end = answer_token_positions(
        offsets=offsets,
        sequence_ids=sequence_ids,
        answer_start=5,
        answer_end=23,
        cls_index=0,
    )

    assert (start, end) == (0, 0)


def test_best_text_span_selects_highest_scoring_valid_context_span():
    context = "The supplier shall maintain insurance."
    offsets = [(0, 0), (0, 3), (4, 12), (13, 18), (19, 27), (28, 37), (0, 0)]
    sequence_ids = [None, 1, 1, 1, 1, 1, None]
    start_logits = [0.0, 0.1, 0.2, 0.1, 5.0, 0.1, 0.0]
    end_logits = [0.0, 0.1, 0.2, 0.1, 0.1, 5.0, 0.0]

    span = best_text_span(
        context=context,
        offsets=offsets,
        sequence_ids=sequence_ids,
        start_logits=start_logits,
        end_logits=end_logits,
        max_answer_tokens=4,
    )

    assert span == {
        "text": "maintain insurance",
        "start": 19,
        "end": 37,
        "score": 10.0,
    }
