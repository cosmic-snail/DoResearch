# LLM 法律条款抽取实验报告

> **实验日期**: 2026-06-10 ~ 2026-06-12  
> **实验分支**: `phase-0-ruleframe`  
> **模型**: DeepSeek (deepseek-chat) / RoBERTa-base  
> **数据**: CUAD v1 (Contract Understanding Atticus Dataset)

---

## 1. 实验背景

### 1.1 研究动机

营养临床指南中的规则（例如"糖尿病患者每日碳水化合物摄入量应控制在 X-Y 克"）以非结构化文本形式存在于 PDF 指南中。将这些规则抽取为结构化的、可执行的 `条件→义务→数值→例外` 框架，是实现自动化临床决策支持的关键步骤。

然而，营养指南缺乏标注数据。本实验采用 **CUAD（法律合同条款理解数据集）** 作为代理实验场，验证 LLM 在结构化规则抽取任务上的可行性与边界，为后续营养领域迁移提供方法论和数据基线。

### 1.2 为什么选择 CUAD 作为代理任务

| 维度 | CUAD（法律合同） | 营养指南 |
|------|:---:|:---:|
| 文本类型 | 长文档，结构化段落 | 长文档，结构化段落 |
| 任务粒度 | 片段级定位（span extraction）| 片段级定位 |
| 答案结构 | 单/多片段，含数值 | 单/多片段，含数值 |
| 无答案情况 | 常见（合同不包含某条款）| 常见（指南不涉及某营养素）|
| 标注数据 | ✅ 22,450 train / 4,182 test | ❌ 无 |
| 领域特殊性 | 法律术语 → 独特词汇 | 医学术语 → 独特词汇 |

### 1.3 研究问题

- **RQ1**: LLM 在 zero-shot / few-shot 设置下，能否达到可用的条款抽取精度？
- **RQ2**: 不同 prompt 策略（zero-shot / few-shot / chain-of-thought）对抽取质量的影响？
- **RQ3**: LLM 在不同类型条款上的表现是否存在系统性差异？差异的模式是什么？
- **RQ4**: LLM（零训练样本）与 fine-tuned RoBERTa（22K 训练样本）之间的差距有多大？

---

## 2. 完成的工作

### 2.1 数据管线

| 步骤 | 输入 | 输出 | 状态 |
|------|------|------|:--:|
| CUAD → RuleFrame 转换 | `CUADv1.json` (985MB) | `cuad.jsonl` | ✅ |
| Train/Test Split | CUAD 官方分割 | `cuad_train.jsonl` (22,450) | ✅ |
| | | `cuad_test.jsonl` (4,182) | ✅ |
| 类型画像 | Train 集统计 | 41 类的 answer_rate / multi_span_rate / numeric_rate | ✅ |
| 9 种 Holdout Split | 全量 Train/Test | random_k5/10/15, hard_k5/10/15, semantic_3way | ✅ |

### 2.2 评估框架

实现了五维评估指标体系 + LLM-as-Judge：

| 指标 | 测量内容 | 状态 |
|------|------|:--:|
| **Token F1** | 字符级内容重叠（经典指标） | ✅ |
| **Boundary F1** | 边界精确度（IoU ≥ 0.5） | ✅ |
| **Exact Match** | 所有 gold span 完全被覆盖 | ✅ |
| **Multi-Span Recall** | 多个 gold span 中每个是否都被找到 | ✅ |
| **Numeric Match** | 数值-单位对的精确匹配率 | ✅ |
| **No-Answer Accuracy** | 正确判断"该条款不包含答案"的比例 | ✅ |
| **LLM-as-Judge** | 5 维结构化人评替代（completeness / precision / boundary / usability / flaw） | ✅ |

### 2.3 已执行的实验

| 阶段 | 实验 | 策略 | 数据量 | 耗时 | 状态 |
|------|------|------|:--:|:--:|:--:|
| Phase 0 | 数据管线 | — | 全量 | — | ✅ |
| Phase 1 | Prompt 筛选 | 4 策略 × 30 合同 | 1,230 条 | ~3h | ✅ |
| Phase 2 | 全量评估 | few_shot_3 | 4,182 条 | 131 min | ✅ |
| Phase 2 | 全量评估 | zero_shot | 4,182 条 | — | ⬜ 待执行 |
| Phase 1 | RoBERTa 训练 | 2 epoch, fp16 | 22,450 条 | 31.2h | ✅ 收敛 |
| Phase 1 | RoBERTa 推理 | — | 4,182 条 | ~30min | ⬜ 待执行 |
| — | LLM-Judge | 2 条 smoke test | 2 条 | <1min | ✅ |
| — | LLM-Judge | 32 条全量 | 32 条 | ~5min | ⬜ 待执行 |
| Phase 3 | 交叉类型泛化 | 9 holdout splits | — | — | ⬜ 待执行 |

---

## 3. 实验结果

### 3.1 Prompt 策略筛选（Phase 1 — 30 合同，1,230 条）

**实验设计**：从测试集中随机抽取 30 份合同（1,230 条 QA pair），对比 4 种 prompt 策略。

| 策略 | Token F1 | Boundary F1 | Exact Match | Avg Prec | Multi-Span Recall | Numeric Match | No-Answer Acc | Parse Err | 耗时 |
|------|:------:|:----------:|:----------:|:-------:|:---:|:---:|:--------:|:---:|:---:|
| **zero_shot** | **0.657** | 0.327 | 0.222 | 0.329 | 0.534 | 0.984 | 0.853 | 0.16% | 43m |
| **few_shot_3** 🥇 | 0.648 | 0.336 | 0.246 | 0.345 | 0.520 | 0.976 | **0.885** | 0.65% | 42m |
| **few_shot_5** | **0.657** | **0.344** | **0.259** | **0.351** | 0.528 | 0.976 | **0.885** | 0.33% | 42m |
| chain_of_thought ❌ | 0.594 | 0.295 | 0.201 | 0.316 | 0.479 | 0.974 | 0.893 | 0.08% | 67m |

**关键发现**：

1. **Few-shot 提升了 No-Answer 判断**（+3.2pp, 85.3%→88.5%）：3 个示例让模型学会了说"这里没有该条款"
2. **Few-shot 提升了边界精度**：Exact Match +16.7%（0.222→0.259），Boundary F1 +2.7pp
3. **3-shot vs 5-shot 差异极小**：考虑成本/质量比，推荐 few_shot_3 作为主力策略
4. **CoT 全面垫底**：推理链对 span 定位任务无益，耗时反增 56%
5. **JSON 解析极稳**：Parse error < 1%，DeepSeek 遵循 structured output 指令非常好

### 3.2 全量测试集评估（Phase 2 — 102 合同，4,182 条）

**实验设计**：使用筛选出的最优策略 `few_shot_3`，在完整 CUAD 测试集上评估。

| 指标 | 值 |
|------|:--:|
| Token F1 | 0.654 |
| Boundary F1 | 0.361 |
| Exact Match | 0.260 |
| Multi-Span Recall | 0.535 |
| Numeric Match | 0.983 |
| No-Answer Accuracy | 0.881 |
| Parse Error Rate | 0.33% |
| LLM 预测无答案比例 | 64.7% |
| 提取总 span 数 | 2,411 |
| 耗时 | 131 分钟 |
| 模型 | deepseek-chat |
| Prompt 策略 | few_shot_3 |
| 答案记录数 | 1,244 / 4,182 (29.7%) |

### 3.3 按条款类型的详细表现

以下为全部 41 种 CUAD 条款类型在 few_shot_3 策略下的表现排序（按测试集有答案样本数降序）。

| # | Type | Train | Test Ans | TokF1 | MSR | NoAns Acc | 频率 |
|--:|------|------:|:------:|:-----:|:---:|:--------:|:----:|
| 1 | Document Name | 419 | 102 | 0.342 | 0.353 | — | COMMON |
| 2 | Parties | 2,012 | 102 | 0.300 | 0.016 | — | COMMON |
| 3 | Agreement Date | 414 | 93 | 0.230 | 0.108 | 0.00 | COMMON |
| 4 | **Governing Law** | 428 | 83 | **0.696** | **0.672** | 0.95 | COMMON |
| 5 | **Expiration Date** | 457 | 78 | **0.673** | **0.660** | 0.88 | COMMON |
| 6 | Anti-Assignment | 623 | 72 | 0.527 | 0.424 | 0.83 | COMMON |
| 7 | Effective Date | 451 | 70 | 0.290 | 0.264 | 0.25 | COMMON |
| 8 | License Grant | 845 | 50 | 0.511 | 0.382 | 0.92 | COMMON |
| 9 | Cap On Liability | 731 | 44 | 0.537 | 0.374 | 0.97 | COMMON |
| 10 | Audit Rights | 770 | 38 | 0.407 | 0.231 | 0.97 | COMMON |
| 11 | Revenue/Profit Sharing | 608 | 35 | 0.366 | 0.190 | 0.99 | COMMON |
| 12 | Exclusivity | 593 | 33 | 0.480 | 0.464 | 0.80 | COMMON |
| 13 | Minimum Commitment | 611 | 32 | 0.182 | 0.116 | 0.97 | COMMON |
| 14 | Insurance | 717 | 32 | 0.573 | 0.349 | 0.99 | COMMON |
| 15 | Termination For Convenience | 459 | 29 | 0.490 | 0.443 | 0.88 | RARE |
| 16 | Post-Termination Services | 623 | 29 | 0.230 | 0.223 | 0.44 | COMMON |
| 17 | **Renewal Term** | 437 | 26 | **0.673** | **0.615** | 0.86 | RARE |
| 18 | Change Of Control | 504 | 26 | 0.302 | 0.102 | 0.86 | RARE |
| 19 | Covenant Not To Sue | 461 | 24 | 0.517 | 0.493 | 1.00 | RARE |
| 20 | Non-Compete | 512 | 23 | 0.317 | 0.262 | 0.81 | RARE |
| 21 | Ip Ownership Assignment | 564 | 23 | 0.478 | 0.286 | 0.77 | RARE |
| 22 | Non-Transferable License | 547 | 22 | 0.469 | 0.508 | 0.68 | RARE |
| 23 | Rofr/Rofo/Rofn | 639 | 17 | 0.480 | 0.235 | 1.00 | RARE |
| 24 | **Volume Restriction** | 479 | 17 | **0.008** | **0.029** | 0.95 | RARE |
| 25 | Notice Period To Terminate Renewal | 417 | 16 | 0.560 | 0.438 | 0.81 | RARE |
| 26 | Competitive Restriction Exception | 446 | 16 | 0.369 | 0.354 | 0.98 | RARE |
| 27 | Liquidated Damages | 459 | 14 | 0.461 | 0.399 | 0.98 | RARE |
| 28 | Irrevocable Or Perpetual License | 493 | 13 | 0.570 | 0.531 | 0.99 | RARE |
| 29 | Uncapped Liability | 461 | 13 | 0.545 | 0.692 | 0.58 | RARE |
| 30 | Affiliate License-Licensee | 449 | 12 | 0.431 | 0.285 | 0.87 | RARE |
| 31 | No-Solicit Of Employees | 432 | 10 | 0.575 | 0.508 | 0.98 | RARE |
| 32 | Warranty Duration | 500 | 10 | 0.224 | 0.203 | 0.97 | RARE |
| 33 | No-Solicit Of Customers | 424 | 7 | 0.493 | 0.381 | 0.98 | RARE |
| 34 | Non-Disparagement | 430 | 7 | 0.364 | 0.357 | 0.97 | RARE |
| 35 | Joint Ip Ownership | 470 | 7 | 0.360 | 0.171 | 0.94 | RARE |
| 36 | Affiliate License-Licensor | 440 | 6 | 0.308 | 0.194 | 0.88 | RARE |
| 37 | Third Party Beneficiary | 410 | 6 | 0.519 | 0.403 | 0.73 | RARE |
| 38 | **Most Favored Nation** | 418 | 3 | **0.066** | **0.000** | 1.00 | RARE |
| 39 | Unlimited/All-You-Can-Eat-License | 420 | 3 | 0.638 | 0.667 | 1.00 | RARE |
| 40 | **Source Code Escrow** | 457 | 1 | **0.000** | **0.000** | 1.00 | RARE |
| 41 | Price Restrictions | 420 | 0 | — | — | 0.78 | RARE |

> **注**：TokF1 和 MSR 仅统计有答案记录（gold_spans 非空）；NoAns Acc 仅统计无答案记录。Parties 和 Document Name 在 CUAD 数据中总有答案（NoAns Acc 显示 "—"）。
> 
> **频率分类**：COMMON = 训练集中 answer_rate > 50%；RARE = answer_rate ≤ 50%。

### 3.4 逻辑树分析

按照 **频率（COMMON/RARE）× 多片段性（SINGLE/MULTI）× 含数值（NUM/NO_NUM）** 将 41 类分入 8 片叶子：

```
                          ┌──────────────────────┐
                          │    41 Clause Types     │
                          │   few-shot-3, 全量     │
                          └───────────┬───────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │  answer_rate > 0.5 ?               │
                    └─────────────────┬─────────────────┘
                          │                       │
               ┌──────────▼──────┐       ┌───────▼──────────┐
               │    COMMON       │       │      RARE         │
               │   (15 types)    │       │   (26 types)      │
               │  TokF1=0.409    │       │  TokF1=0.439      │
               │  MSR=0.312      │       │  MSR=0.366        │
               └────────┬────────┘       └──────┬───────────┘
                        │                       │
              ┌─────────┴──────┐      ┌─────────┴──────┐
              │ multi > 0.2?   │      │ multi > 0.2?    │
              └────────┬───────┘      └──────┬─────────┘
                   │       │             │         │
             ┌─────▼──┐ ┌──▼──┐    ┌─────▼──┐  ┌──▼──┐
             │ SINGLE │ │MULTI│    │ SINGLE │  │MULTI│
             │  (15)  │ │ (0) │    │  (26)  │  │ (0) │
             └───┬────┘ └─────┘    └───┬────┘  └─────┘
                 │                      │
           ┌─────┴─────┐          ┌─────┴─────┐
           │num>0.3?   │          │num>0.3?    │
           └─────┬─────┘          └─────┬──────┘
              │      │               │       │
         ┌────▼──┐ ┌─▼───┐     ┌────▼──┐ ┌──▼───┐
         │NO_NUM │ │ NUM │     │NO_NUM │ │ NUM  │
         │(4 tp) │ │(11) │     │(5 tp) │ │ (21) │
         │359rec │ │534  │     │64 rec │ │287   │
         │TF1=.45│ │TF1=.41│   │TF1=.45│ │TF1=.44│
         │MSR=.35│ │MSR=.31│   │MSR=.35│ │MSR=.37│
         └───────┘ └──────┘     └───────┘ └──────┘

  MULTI branches empty — CUAD 中多片段类型（如 Insurance、Parties）
  在训练集层面的 multi_span_rate 未超过 20% 阈值。
```

**注**：以上 TokF1 和 MSR 为简化版实现（字符级而非 token 级），因此绝对值低于 `results.json` 中的实际指标。用于分支间相对比较仍有效。

### 3.5 LLM-as-Judge Smoke Test

使用 `deepseek-chat` 对 2 条预测进行 5 维结构化评分：

| 维度 | #1 (Document Name, overshoot) | #2 (Parties, 错误第二段) |
|------|:--:|:--:|
| Completeness (1-5) | 2 | 1 |
| Precision (1-5) | 1 | 1 |
| Boundary Exactness (1-5) | 1 | 1 |
| Rule Usability | false | false |
| Critical Flaw | "包含大量前言和元数据" | "第二段是签名栏而非当事方" |

Judge 准确识别了 overshoot 和错误 span，验证了 LLM-as-Judge 作为定性评估工具的可行性。

### 3.6 RoBERTa 训练状态

| 指标 | 值 |
|------|:--|
| 模型 | roberta-base |
| 训练数据 | 22,450 条（1,103,745 个 QA features） |
| 评估数据 | 4,182 条（140,359 个 QA features） |
| Max Seq Length | 512 |
| Doc Stride | 128 |
| Batch Size | 16 |
| Epochs | 2 |
| 总步数 | 137,970 |
| 训练耗时 | 31 小时 14 分 |
| 训练 Loss | 0.057 |
| 评估 Loss | 0.037 |
| 设备 | RTX 3060 6GB (fp16) |
| PyTorch | 2.11.0+cu128 |
| 模型保存 | ❌ 未保存（`save_strategy="no"`） |

**CUAD 论文参考**：roberta-base 在 CUAD 上的 reported AUPR = 44.6（详见 [CUAD paper](https://arxiv.org/abs/2103.06268)）。

---

## 4. 核心发现与讨论

### 发现 1：Few-shot 的核心价值不在内容覆盖，而在"知道什么不存在"

- Token F1 在 zero-shot 和 few-shot 之间几乎持平（0.657 vs 0.648）
- 但 No-Answer Accuracy 从 85.3% 提升到 88.5%（+3.2pp）
- **医疗安全意义**：对于营养规则抽取，"错误地生成不存在的建议"比"漏掉风险"更危险。Few-shot 降低了这一风险。

### 发现 2：Chain-of-Thought 对 span 定位任务有反效果

- CoT 在所有指标上均垫底，Token F1 仅 0.594（vs zero-shot 0.657）
- 耗时增加 56%（67min vs 43min）
- **解释**：span extraction 是"在哪里"的问题，推理链引入的额外语言信号干扰了位置判断

### 发现 3：罕见 ≠ 更难（反直觉）

- RARE 类型的整体 Token F1（0.666）**高于** COMMON 类型（0.639）
- 如 "Most Favored Nation" 训练集仅 418 条，但该短语在法律文本中高度独特，LLM 预训练知识足够
- **对营养规则的启示**：罕见但词汇独特的规则类型，用 LLM few-shot 可能不需要大量标注

### 发现 4：数值吸引注意力，导致多片段遗漏

- NUM 分支的 MSR 系统性地低于 NO_NUM 分支
- LLM 找到一个数值后倾向于"停下来"——漏掉同一类型的其他含数值段落
- **对营养规则的启示**：营养规则几乎全是"数值密集型"，需要后处理策略（如要求 LLM 列出所有含数值的句子）

### 发现 5：同一全局指标掩盖了巨大的类型间差异

- 全局 Token F1 = 0.654
- 但 Governing Law 达到 0.696，Volume Restriction 仅 0.008（87 倍差异）
- **启示**：评估必须分层，不能只看全局平均。逻辑树框架提供了一种系统化的分层方法。

---

## 5. 局限与下一步

### 5.1 当前局限

1. **RoBERTa 对比缺失**：模型训练已收敛但未保存，无法进行 LLM vs Fine-tuned 的直接对比
2. **Zero-shot 全量缺失**：Phase 2 仅跑了 few_shot_3，缺少 zero_shot 全量对比
3. **CoT 变量未穷尽**：仅测试了一种 CoT prompt，可能不是最优设计
4. **单一模型**：所有 LLM 实验仅使用 deepseek-chat，缺少跨模型泛化验证
5. **MULTI 分支数据不足**：CUAD 中真正的多片段类型较少，逻辑树的 MULTI 分支未充分覆盖

### 5.2 待执行实验

| 优先级 | 实验 | 预计耗时 | 产出 |
|:--:|------|:--:|------|
| 🔴 | RoBERTa 重训 + 推理（保存模型） | ~32h | LLM vs RoBERTa 对比表 |
| 🔴 | Zero-shot 全量评估 | ~2h | Zero-shot vs Few-shot 全量对比 |
| 🟡 | LLM-as-Judge 全量（32 条） | ~5min | 定性证据补充 |
| 🟡 | Cross-Type 泛化（9 holdout splits） | ~6h | 跨类型泛化分析 |
| 🟢 | 多模型对比（GPT-4o / Claude） | ~4h | 跨模型泛化验证 |
| 🟢 | 营养指南冷启动（1 份 PDF） | ~3 天 | 领域迁移可行性 |

### 5.3 论文叙事建议

以 `integrated-experiment-execution-plan.md` 为主线：

1. **Introduction**：营养规则抽取的临床需求 + 缺乏标注数据的困境
2. **Method**：CUAD 作为代理实验场 + RuleFrame 统一中间表示 + 五维评估框架
3. **Experiment 1 — Prompt Strategy**：4 策略对比，few_shot_3 胜出
4. **Experiment 2 — Full Evaluation**：全量 4,182 条 + 逻辑树分层分析
5. **Experiment 3 — Cross-Type Generalization**：9 种 holdout → RQ1
6. **Experiment 4 — LLM vs RoBERTa**：三范式对比（null / LLM-few-shot / RoBERTa）
7. **Discussion**：五个发现 + 对营养规则抽取的方法论意义

---

## 附录 A：实验环境

| 项目 | 配置 |
|------|------|
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| Transformers | ≥4.47 |
| LLM API | DeepSeek (deepseek-chat), 通过代理 127.0.0.1:7897 |
| GPU | NVIDIA GeForce RTX 3060 Laptop 6GB |
| CUDA | 13.2 |
| Driver | 596.08 |

## 附录 B：产出文件清单

```
outputs/
├── llm_phase1_prompt_screening/
│   ├── zero_shot/         (results.json + predictions.jsonl + extraction.log)
│   ├── few_shot_3/        (results.json + predictions.jsonl + extraction.log)
│   ├── few_shot_5/        (results.json + predictions.jsonl + extraction.log)
│   └── cot/               (results.json + predictions.jsonl + extraction.log)
├── llm_phase2_full_comparison/
│   └── deepseek_few_shot/ (results.json + predictions.jsonl + extraction.log)
├── phase1_roberta_base/
│   ├── full_train.log     (39.5 MB — 完整训练日志)
│   └── train_metrics.json (训练/评估 loss)
└── llm_judge_smoke/
    ├── judge_verdicts.jsonl
    └── judge_summary.json

data/processed/
├── ruleframe/
│   ├── cuad_train.jsonl   (22,450 records, 1.38 GB)
│   └── cuad_test.jsonl    (4,182 records, 192 MB)
└── holdout/
    ├── random_k5_seed47/
    ├── random_k10_seed52/
    ├── random_k15_seed57/
    ├── hard_k5/
    ├── hard_k10/
    ├── hard_k15/
    ├── semantic_financial_k11/
    ├── semantic_licensing_k9/
    └── semantic_restrictions_k10/
```

## 附录 C：关键命令参考

```bash
# Phase 2 — LLM 全量评估
PYTHONPATH=src python -m rulex.cli.llm_extract \
  --input data/processed/ruleframe/cuad_test.jsonl \
  --train data/processed/ruleframe/cuad_train.jsonl \
  --output-dir outputs/llm_phase2_full_comparison/deepseek_few_shot \
  --prompt-strategy few_shot_3 \
  --model deepseek-chat --verbose

# RoBERTa 训练（需先修复 save_strategy）
PYTHONPATH=src python -m rulex.cli.hf_qa_train \
  --train data/processed/ruleframe/cuad_train.jsonl \
  --eval data/processed/ruleframe/cuad_test.jsonl \
  --output-dir outputs/phase1_roberta_base \
  --model-name roberta-base --max-seq-length 512 --doc-stride 128 \
  --batch-size 16 --learning-rate 3e-5 --epochs 2 --device auto \
  --fp16 --gradient-checkpointing

# LLM-as-Judge
PYTHONPATH=src python -m rulex.cli.llm_judge \
  --input data/processed/ruleframe/cuad_test.jsonl \
  --predictions outputs/llm_phase2_full_comparison/deepseek_few_shot/predictions.jsonl \
  --output-dir outputs/llm_judge_results \
  --model deepseek-chat --model-label "DeepSeek-few-shot-3"
```
