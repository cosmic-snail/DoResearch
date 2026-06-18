# Phase 1 RoBERTa-base Full Baseline Paused

## Pause Status

Paused on 2026-06-11 after user request.

## Last Observed Progress

- Total planned steps: `137,970`
- Last observed step: about `37,526`
- Progress: about `27%`
- Last observed epoch: about `0.544`
- Runtime shown by tqdm: about `24:26:25`
- Recent per-step speed: about `2.3-3.4s/it`

## Last Log Snippet

```text
27%|██▋       | 37526/137970 [24:26:25<94:35:39,  3.39s/it]^C
```

## Files

- Log: `outputs/phase1_roberta_base/full_train.log`
- Output directory: `outputs/phase1_roberta_base`

## Notes

The run was interrupted with `Ctrl-C` via tmux. No final `train_metrics.json` was produced because training did not complete. The current training script uses `save_strategy="no"`, so this run should be treated as an interrupted baseline attempt rather than a resumable checkpoint.
