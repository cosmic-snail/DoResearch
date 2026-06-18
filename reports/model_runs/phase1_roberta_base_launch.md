# Phase 1 RoBERTa-base Full Baseline Launch

## Status

Started in a persistent `tmux` session on 2026-06-10.

## Session

- tmux session: `cuad_roberta_base`
- log file: `outputs/phase1_roberta_base/full_train.log`
- exit-code file: `outputs/phase1_roberta_base/full_train.exit`
- output directory: `outputs/phase1_roberta_base`

## Command

```bash
PYTHONPATH=src TOKENIZERS_PARALLELISM=false /Users/simon/miniforge3/bin/python3 -u -m rulex.cli.hf_qa_train \
  --train data/processed/ruleframe/cuad_train.jsonl \
  --eval data/processed/ruleframe/cuad_test.jsonl \
  --output-dir outputs/phase1_roberta_base \
  --model-name roberta-base \
  --max-seq-length 512 \
  --doc-stride 128 \
  --batch-size 16 \
  --learning-rate 3e-5 \
  --epochs 2 \
  --device auto
```

## Monitoring

```bash
tmux attach -t cuad_roberta_base
tail -f outputs/phase1_roberta_base/full_train.log
cat outputs/phase1_roberta_base/full_train.exit
```

## Observed Speed

A foreground trial reached training and reported `137,970` total steps. Early-step speed on the local MPS setup was about `2.5s/step`, implying a multi-day run. The full run was therefore relaunched under `tmux` rather than kept attached to the Codex tool session.
