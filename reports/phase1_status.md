# Phase 1 Status

## Completed

- Converted official CUAD training split from `train_separate_questions.json` to RuleFrame JSONL.
- Converted official CUAD test split from `test.json` to RuleFrame JSONL.
- Generated train/test corpus summaries.
- Generated 41-clause-type difficulty profiles for train and test.
- Ran an always-no-answer null baseline on the test split.

## Key Counts

| Split | Records | Documents | Rule Types | Answerable | No-answer | Gold Spans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train separate questions | 22,450 | 408 | 41 | 11,180 | 11,270 | 11,180 |
| Test | 4,182 | 102 | 41 | 1,244 | 2,938 | 2,643 |

## Null Baseline on Test

| Metric | Value |
| --- | ---: |
| Exact match | 0.0 |
| Token F1 | 0.0 |
| Boundary F1 | 0.0 |
| Average precision | 0.0 |
| No-answer accuracy | 1.0 |

## Notes

- `train_separate_questions.json` splits multi-answer training questions into separate single-answer records. The adapter strips suffixes such as `_0` from rule types so `Document Name_0` maps back to `Document Name`.
- `test.json` keeps multi-answer questions grouped; this is the right split for multi-span evaluation.
- Full RoBERTa reproduction is not started yet because this environment has `transformers` but not `torch`, `datasets`, or `evaluate`.
