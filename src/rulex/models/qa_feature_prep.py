from collections.abc import Sequence
from typing import Any


def answer_token_positions(
    offsets: Sequence[tuple[int, int]],
    sequence_ids: Sequence[int | None],
    answer_start: int,
    answer_end: int,
    cls_index: int,
) -> tuple[int, int]:
    context_indices = [idx for idx, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
    if not context_indices:
        return cls_index, cls_index

    context_start = context_indices[0]
    context_end = context_indices[-1]
    if offsets[context_start][0] > answer_start or offsets[context_end][1] < answer_end:
        return cls_index, cls_index

    token_start = context_start
    while token_start <= context_end and offsets[token_start][0] <= answer_start:
        token_start += 1
    token_start -= 1

    token_end = context_end
    while token_end >= context_start and offsets[token_end][1] >= answer_end:
        token_end -= 1
    token_end += 1

    return token_start, token_end


def best_text_span(
    context: str,
    offsets: Sequence[tuple[int, int]],
    sequence_ids: Sequence[int | None],
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    max_answer_tokens: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    context_indices = [idx for idx, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
    for start_index in context_indices:
        for end_index in context_indices:
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_tokens:
                continue
            start_char, _ = offsets[start_index]
            _, end_char = offsets[end_index]
            if end_char <= start_char:
                continue
            score = float(start_logits[start_index]) + float(end_logits[end_index])
            if best is None or score > best["score"]:
                best = {
                    "text": context[start_char:end_char],
                    "start": start_char,
                    "end": end_char,
                    "score": score,
                }
    return best or {"text": "", "start": 0, "end": 0, "score": 0.0}
