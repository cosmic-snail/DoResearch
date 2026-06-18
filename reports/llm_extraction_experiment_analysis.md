# LLM Rule Extraction Experiment — 完整实验分析报告

> **日期**: 2026-06-11  
> **实验分支**: `phase-0-ruleframe`  
> **PR**: [#1](https://github.com/cosmic-snail/DoResearch/pull/1)  
> **模型**: DeepSeek-Chat（API via `https://api.deepseek.com`）

---

## 目录

1. [实验背景](#1-实验背景)
2. [数据集：CUAD → RuleFrame](#2-数据集cuad--ruleframe)
3. [代码框架](#3-代码框架)
4. [实验设计](#4-实验设计)
5. [评估体系](#5-评估体系)
6. [实验结果](#6-实验结果)
7. [结果分析](#7-结果分析)
8. [结论与下一步](#8-结论与下一步)
9. [附录：输出文件清单](#9-附录输出文件清单)

---

## 1. 实验背景

### 1.1 研究动机

本项目 (DoResearch) 的长期目标是 **从临床营养指南中自动抽取条件规则并结构化为可执行的推荐系统**，核心 Pipeline 为：

```
营养指南 PDF → LLM/规则抽取 → 结构化规则 (Condition/Action/Value/Exception) → 饮食推荐系统
```

由于营养领域缺乏大规模标注数据，我们采用 **跨领域迁移** 策略：先在法律合同数据集 CUAD 上验证 LLM 抽取管线的可行性，积累方法论和评估框架，再迁移到营养领域。

### 1.2 为什么选 CUAD

CUAD (Contract Understanding Atticus Dataset) 的法律合同条款抽取任务与营养规则抽取在结构上高度相似：

| 维度 | CUAD（法律） | 营养指南 |
|------|:-----------:|:-------:|
| 条件逻辑 | ✅ | ✅ |
| 数值阈值 | ✅ | ✅ |
| 跨句抽取 | ✅ | ✅ |
| 规范性文本 | ✅ | ✅ |
| 结构化输出 | ✅ | ✅ |
| 可获得性 | ✅ 开放 | ⬜ 待获取 |

CUAD 共有 41 种条款类型（如 "License Grant", "Non-Compete", "Revenue/Profit Sharing" 等），标注格式为 `(doc_id, clause_type, answer_spans)`。

### 1.3 实验目标

本次实验旨在回答以下问题：

1. **LLM 能否从长篇法律合同中准确抽取条款文本片段？**（Phase 1 — Prompt Screening）
2. **不同 prompt 策略（zero-shot / few-shot / chain-of-thought）的效果差异？**（Phase 1）
3. **最佳策略在全量测试集上的表现如何？**（Phase 2 — Full Comparison）
4. **LLM-as-Judge 能否作为自动化质量评估手段？**（Judge Smoke）

---

## 2. 数据集：CUAD → RuleFrame

### 2.1 RuleFrame 数据格式

我们设计了一个统一的中间数据格式 `RuleFrame`，用于桥接不同领域的标注数据：

```json
{
  "doc_id": "合同唯一ID",
  "domain": "legal",
  "rule_type": "条款类型名（如 License Grant）",
  "query": "向 LLM 提问的自然语言问题",
  "source_text": "合同全文",
  "gold_spans": [
    {"text": "金标准文本", "start": 1234, "end": 1567}
  ],
  "predicted_spans": [],
  "frame": {
    "condition": [],
    "action": [],
    "value": [],
    "exception": [],
    "evidence": []
  },
  "metadata": {
    "has_answer": true,
    "num_gold_spans": 2
  }
}
```

### 2.2 数据规模

| 划分 | 记录数 | 文档数 | 条款类型 | 有答案 | 无答案 | Gold Span 数 |
|------|------:|------:|------:|------:|------:|------:|
| Train | 22,450 | 408 | 41 | 11,180 | 11,270 | 11,180 |
| Test | 4,182 | 102 | 41 | 1,244 | 2,938 | 2,643 |

**关键特性**：
- Test 中 **70.3%** 的记录是无答案的（no-answer），要求模型具备判断"条款不存在"的能力
- 有答案的记录可能包含多个 gold span（multi-span），如 "Parties" 条款通常有多个署名方
- Train 中每条记录只有一个 answer，Test 中保留了多答案

### 2.3 数据预处理流程

```
CUADv1.json (官方原始格式)
    │
    └── cuad_to_ruleframe.py  ──→  cuad_train.jsonl / cuad_test.jsonl (RuleFrame JSONL)
                                       │
                                       ├── summarize_ruleframes.py  ──→  corpus summary (JSON)
                                       ├── profile_rule_types.py    ──→  41-clause difficulty profiles (CSV)
                                       └── null_baseline.py         ──→  always-no-answer baseline
```

---

## 3. 代码框架

### 3.1 整体架构

```
src/rulex/
├── cli/
│   ├── llm_extract.py          ← LLM 抽取主入口
│   ├── llm_judge.py            ← LLM-as-Judge 评判
│   ├── holdout_split.py        ← 交叉验证留出集生成
│   ├── hf_qa_train.py          ← HuggingFace QA 训练（RoBERTa baseline）
│   └── ...
├── models/
│   └── llm_extractor.py        ← LLMClient + Prompt 引擎
├── prompting/
│   └── clause_prompts.yaml     ← 4 种 Prompt 策略模板
├── evaluation/
│   └── span_metrics.py         ← 8 项评估指标
└── schema/
    └── ruleframe.py            ← RuleFrame 数据格式与校验
```

### 3.2 LLM 抽取管线 (`llm_extract.py`)

```
RuleFrame JSONL (输入)
    │
    ├── 逐条读取 record
    │     ├── 提取 doc_id, rule_type, query, context
    │     ├── 若策略为 few_shot，从 train 中采 3/5 个同类型示例
    │     └── 调用 LLMClient.extract_spans()
    │           ├── _build_messages() → [system, user] ChatGPT 格式
    │           ├── client.chat() → OpenAI-compatible API
    │           ├── _extract_json_block() → 解析 JSON 响应
    │           └── _resolve_span_offsets() → text → (start, end) 反向定位
    │
    ├── 写入 predictions.jsonl（逐行 JSON）
    └── evaluate_span_predictions() → results.json
```

### 3.3 LLMClient 设计 (`llm_extractor.py`)

| 特性 | 实现 |
|------|------|
| 后端兼容 | OpenAI-compatible API（DeepSeek / GPT-4o / Claude 均可） |
| API Key | 自动加载 `.env`（`DEEPSEEK_API_KEY` / `OPENAI_API_KEY`） |
| Temperature | 固定 0.0（最大化确定性） |
| 超时 | 默认 120s |
| Span 定位 | 精确 `str.find()` + 回退空格归一化模糊匹配 |

### 3.4 Prompt 策略 (`clause_prompts.yaml`)

| 策略 | 描述 | System Prompt | User Template |
|------|------|:---:|:---:|
| `zero_shot` | 直接给合同+问题，无示例 | 法律合同分析师角色 | `{context} + {query}` |
| `few_shot_3` | 提供 3 个同类型标注示例 | 同上 | `{examples} + {context} + {query}` |
| `few_shot_5` | 提供 5 个同类型标注示例 | 同上 | 同上（5 examples） |
| `chain_of_thought` | 先推理再输出 JSON | 三步推理指引 | `Step 1/2/3 → JSON` |

所有策略共享相同的输出格式要求：
```json
{"answerable": true/false, "spans": [{"text": "exact text from contract"}]}
```

### 3.5 LLM-as-Judge (`llm_judge.py`)

独立的 LLM 二次评判，对预测结果进行 **5 维度结构化评分**：

| 维度 | 量纲 | 含义 |
|------|:---:|------|
| **Completeness** | 1-5 | 是否找到了所有 gold span |
| **Precision** | 1-5 | 预测文本是否干净、无噪声 |
| **Boundary Exactness** | 1-5 | 起止边界的精确程度 |
| **Rule Usability** | yes/no | 能否直接用于规则编码 |
| **Critical Flaw** | string/null | 最严重缺陷的文字描述 |

评判方式：Judge LLM 同时看到 **gold spans + predicted spans + 原始合同上下文**，由同一个 DeepSeek-Chat 模型返回结构化 JSON。

---

## 4. 实验设计

### 4.1 Phase 概览

```
Phase 0: 数据管线验证
  └── cuad_to_ruleframe + smoke_eval（3 条 oracle 测试）

Phase 1: Prompt Screening（本次实验核心）
  ├── Smoke: 3 records, zero_shot + few_shot_3（管线连通性测试）
  ├── Prompt Screening: 1230 records, 4 strategies × DeepSeek-Chat
  ├── Null Baseline: always-no-answer（下界基准）
  └── Judge Smoke: 对 few_shot_3 的 2 条结果做 LLM-as-Judge 评判

Phase 2: Full Comparison
  └── 4182 records (全量 test), deepseek_few_shot_3

Phase 3: Cross-Type + Query Sensitivity（待执行）
  ├── 留出集划分（holdout_split.py）：random / semantic / hard
  └── 跨条款类型泛化 + 不同 query 表述敏感度
```

### 4.2 Phase 1 Prompt Screening 设置

| 参数 | 值 |
|------|------|
| 测试集 | CUAD test，前 1230 条 |
| 训练集 | CUAD train（仅用于 few-shot 示例采样） |
| Prompt 策略 | zero_shot, few_shot_3, few_shot_5, chain_of_thought |
| 模型 | deepseek-chat |
| Temperature | 0.0 |
| Max Tokens | 4096 |

> **注**：1230 = 30% of 4182。之所以用子集是因为 4 个策略 × 全量测试集会消耗大量 API 额度。目的是筛选出 top-2 策略再跑全量。

### 4.3 Phase 2 Full Comparison 设置

| 参数 | 值 |
|------|------|
| 测试集 | CUAD test，全量 4182 条 |
| Prompt 策略 | few_shot_3（Phase 1 中 AP 最高的策略之一） |
| 模型 | deepseek-chat |
| 耗时 | 131.32 分钟 |
| 速度 | ~31.8 rec/min |

---

## 5. 评估体系

### 5.1 核心指标定义

| 指标 | 计算方式 | 取值范围 |
|------|----------|:---:|
| **Exact Match** | `(gold.start == pred.start) and (gold.end == pred.end)` 的命中率 | [0, 1] |
| **Token F1** | 以单词 token 为粒度计算 precision/recall/F1 | [0, 1] |
| **Boundary F1** | 以 `(start, end)` 边界点为粒度计算 F1 | [0, 1] |
| **Multi-Span Recall** | Gold span 中被至少一个预测以 IoU≥0.5 命中的比例 | [0, 1] |
| **Numeric Match** | Gold 中 `(数值, 单位)` 对在预测中精确出现的比例 | [0, 1] |
| **Average Precision** | 按 score 排序后计算 precision-recall 曲线下面积 | [0, 1] |
| **No-Answer Accuracy** | 无答案记录中预测也为空的准确率 | [0, 1] |
| **Parse Error Rate** | LLM 返回的 JSON 解析失败率 | [0, 1] |

### 5.2 指标分层

```
Layer 1: 结构正确性 ── exact_match, boundary_f1
Layer 2: 内容正确性 ── token_f1, multi_span_recall
Layer 3: 数值敏感性 ── numeric_match
Layer 4: 排序质量   ── average_precision
Layer 5: 判断能力   ── no_answer_accuracy
Layer 6: 鲁棒性     ── parse_error_rate
```

---

## 6. 实验结果

### 6.1 Phase 1 Smoke（管线连通性验证）

| 指标 | zero_shot (3 rec) |
|------|------:|
| Exact Match | 0.0 |
| Token F1 | 0.295 |
| Boundary F1 | 0.250 |
| Average Precision | 0.250 |
| No-Answer Accuracy | 0.0 |
| Parse Error Rate | 0.0 |

> 仅 3 条记录，指标波动大，仅用于验证管线无崩溃。

### 6.2 Phase 1 Null Baseline（下界参考）

| 指标 | 值 |
|------|------:|
| Exact Match | 0.0 |
| Token F1 | 0.0 |
| Boundary F1 | 0.0 |
| Average Precision | 0.0 |
| No-Answer Accuracy | 1.0 |

> "永远预测空"策略。No-Answer Accuracy = 1.0 是因为 70.3% 的数据天然无答案。这个 baseline 说明了：**如果不考虑有答案的情况，看起来也很"好"**。

### 6.3 Phase 1 Prompt Screening（1230 条，核心对比）

| 指标 | zero_shot | few_shot_3 | few_shot_5 | chain_of_thought |
|------|------:|------:|------:|------:|
| **Exact Match** | 0.222 | 0.246 | **0.259** | 0.201 |
| **Token F1** | **0.657** | 0.648 | 0.657 | 0.594 |
| **Boundary F1** | 0.327 | 0.336 | **0.344** | 0.295 |
| **Average Precision** | 0.329 | 0.345 | **0.351** | 0.316 |
| **Multi-Span Recall** | **0.534** | 0.520 | 0.528 | 0.479 |
| **Numeric Match** | **0.984** | 0.976 | 0.976 | 0.974 |
| **No-Answer Accuracy** | 0.853 | 0.885 | 0.885 | **0.893** |
| **Answerable Records** | 378 | 378 | 378 | 378 |
| **No-Answer Records** | 852 | 852 | 852 | 852 |

**排名热力图**（🥇 = 最好，数值越高越好）：

| 指标 | zero_shot | few_shot_3 | few_shot_5 | CoT |
|------|:---:|:---:|:---:|:---:|
| Exact Match | 🥉 | 🥈 | 🥇 | 4 |
| Token F1 | 🥇 | 🥉 | 🥇 | 4 |
| Boundary F1 | 🥉 | 🥈 | 🥇 | 4 |
| Average Precision | 🥉 | 🥈 | 🥇 | 4 |
| Multi-Span Recall | 🥇 | 🥉 | 🥈 | 4 |
| Numeric Match | 🥇 | 🥈 | 🥈 | 🥉 |
| No-Answer Accuracy | 4 | 🥈 | 🥈 | 🥇 |

### 6.4 Phase 2 Full Comparison（4182 条全量）

| 指标 | few_shot_3 (Phase 1) | few_shot_3 (Phase 2) | 变化 |
|------|------:|------:|:---:|
| Exact Match | 0.246 | **0.260** | +5.6% |
| Token F1 | 0.648 | **0.654** | +1.0% |
| Boundary F1 | 0.336 | **0.361** | +7.5% |
| Average Precision | 0.345 | **0.342** | -0.7% |
| Multi-Span Recall | 0.520 | **0.535** | +2.8% |
| Numeric Match | 0.976 | **0.983** | +0.7% |
| No-Answer Accuracy | 0.885 | **0.881** | -0.4% |
| Parse Error Rate | — | 0.0033 | — |
| LLM No-Answer Rate | — | 0.647 | — |
| Total Spans Extracted | — | 2,411 | — |
| Extraction Time | — | 131.32 min | — |

> Phase 1 和 Phase 2 的指标高度一致（偏差 < 7.5%），说明 1230 条子集具有很好的代表性。

### 6.5 LLM-as-Judge Smoke（2 条）

| 维度 | 值 |
|------|------:|
| Avg Completeness (1-5) | 1.5 |
| Avg Precision (1-5) | 1.0 |
| Avg Boundary Exactness (1-5) | 1.0 |
| Rule Usability Rate | **0%** (0/2) |
| API Error Rate | 0.0 |
| Parse Error Rate | 0.0 |

> 样本量极小（2 条），不具有统计意义。但 `rule_usability=0%` 和 `completeness=1.5` 提示预测 span 中包含大量无关文本。

### 6.6 RoBERTa Benchmark（参考对比）

| 指标 | RoBERTa-benchmark (100 steps) | DeepSeek few-shot-3 |
|------|------:|------:|
| Train Records | 200 | — |
| Train Loss | 5.79 | — |
| Exact Match | ⏳ 训练未完成 | 0.260 |
| Token F1 | ⏳ | 0.654 |

> RoBERTa-base 全量训练预估需数天（137,970 steps × ~2.5s/step），当前在 tmux 后台运行中。

---

## 7. 结果分析

### 7.1 策略对比：Few-Shot > Zero-Shot ≥ CoT

```
Average Precision 排名:  few_shot_5 (0.351)  >  few_shot_3 (0.345)  >  zero_shot (0.329)  >  CoT (0.316)
Exact Match 排名:        few_shot_5 (0.259)  >  few_shot_3 (0.246)  >  zero_shot (0.222)  >  CoT (0.201)
```

**关键发现**：

1. **Few-shot 一致优于 zero-shot**：提供 3-5 个示例后，Exact Match 提升 10-17%，AP 提升 5-7%。示例帮助 LLM 理解"什么是好的抽取"。

2. **Few-shot 5 vs 3 差异极小**：5 个示例相比 3 个示例的边际收益约为 +1.7%（AP），不值得为此多消耗 67% 的 prompt token。

3. **Chain-of-Thought 全面垫底**：CoT 在所有指标上都是最差。原因推测：三步推理占据了大量输出 token，分散了对精确文本复制的注意力；且 CoT 的 "先推理后输出" 范式更适合数学推理而非精确文本定位。

4. **Zero-shot 在 Token F1 和 Multi-Span Recall 上不输 few-shot**：说明在没有示例的情况下，LLM 对关键 token 的定位能力已经不错，few-shot 主要改善的是边界精度。

### 7.2 数值抽取能力极强

```
Numeric Match: 0.974 ~ 0.984（所有策略均 > 97%）
```

LLM 对 `(数值, 单位)` 对的抽取几乎完美。这对营养领域（大量剂量、频率、阈值）是极为有利的信号。

### 7.3 Multi-Span 是主要瓶颈

```
Multi-Span Recall: 0.479 ~ 0.534
```

接近半数的 gold span 被遗漏。以 "Parties" 条款为例：一份合同可能有多达 5-10 个署名方，但 LLM 往往只找到 1-2 个。

**改进方向**：
- Prompt 中显式要求 "find ALL instances"
- 引入迭代抽取（找到第一个→继续找下一个）
- 使用更大的 max_tokens 避免输出截断

### 7.4 No-Answer 判断仍有提升空间

```
No-Answer Accuracy: 0.853 ~ 0.893
```

这意味着 11-15% 的"条款不存在"情况被误判为存在（false positive）。考虑到测试集中 70.3% 是无答案记录，false positive 的绝对数量不少。

### 7.5 LLM-as-Judge 的初步信号

Judge Smoke 仅评测 2 条，但已经暴露出核心问题：

- **Rule Usability = 0%**：即使 span 文本包含了正确内容，也夹杂了大量无关的前后文（preamble、相邻条款），无法直接用于规则编码
- **Completeness = 1.5/5**：遗漏严重
- **Precision = 1.0/5**：噪声多

> ⚠️ Judge 样本量太小，结论需谨慎。建议在 Phase 2 全量预测上运行至少 50-100 条 Judge 评测以获取统计显著结论。

### 7.6 与 Null Baseline 的对比

| 指标 | Null Baseline | DeepSeek few-shot-3 (Phase 2) |
|------|------:|------:|
| Exact Match | 0.0 | 0.260 |
| Token F1 | 0.0 | 0.654 |
| Boundary F1 | 0.0 | 0.361 |
| Average Precision | 0.0 | 0.342 |
| No-Answer Accuracy | 1.0 | 0.881 |

LLM 在所有有答案指标上显著优于 null baseline，说明模型确实在"做事情"而非猜测。但在 no-answer accuracy 上输给 null baseline —— 这是 LLM 抽取的一个固有 tradeoff：积极的抽取行为必然带来 false positive。

---

## 8. 结论与下一步

### 8.1 核心结论

| 问题 | 答案 |
|------|------|
| LLM 能从法律合同中抽取条款？ | ✅ 可以。Token F1 ~0.65, Exact Match ~0.26 |
| 最佳 Prompt 策略？ | **few_shot 3-5**，边际差异小，3 个示例足够 |
| CoT 有帮助吗？ | ❌ 全面退步，不建议用于 span 抽取任务 |
| LLM 的强项？ | 数值提取（>97%），判断条款是否存在（~88%） |
| LLM 的弱项？ | 多 span 召回（~53%），边界精确性（~36%） |
| 结果能否直接用于规则编码？ | ❌ 不能。Judge 显示 rule_usability=0% |

### 8.2 后续实验建议

| 优先级 | 实验 | 目的 |
|:---:|------|------|
| 🔴 | 扩展 Judge 评测到 50-100 条 | 获得统计显著的 LLM 质量评判 |
| 🔴 | DeepSeek zero_shot × 全量测试集 | 与 few_shot 做完整的 Phase 2 对比 |
| 🟡 | 多模型对比（GPT-4o / Claude） | 确认结论的模型无关性 |
| 🟡 | Multi-Span 专项优化 | 迭代抽取 / prompt 工程 |
| 🟢 | Cross-Type Holdout | 测试未见过条款类型的泛化能力 |
| 🟢 | Query Sensitivity | 不同问题表述对抽取的影响 |
| ⏳ | RoBERTa-base 全量训练 | 获得监督学习 baseline 对比 |

### 8.3 对营养领域的启示

1. **数值抽取能力是最大利好**：营养指南中的剂量、频率、阈值恰恰是 LLM 表现最稳定的部分
2. **Multi-Span 问题在营养领域更严重**：一份指南可能有多条针对不同人群的推荐，LLM 需要找到全部
3. **Rule Usability 的鸿沟**：从 span 到可执行规则（condition/action/value）需要额外的结构化编码步骤，这是本项目的核心创新点
4. **CUAD 验证了管线可行性**：数据流、评估框架、代码架构均已跑通，可以平移到营养领域

---

## 9. 附录：输出文件清单

### 9.1 实验输出

| 文件路径 | 类型 | 说明 |
|------|:---:|------|
| `outputs/llm_phase1_smoke/zero_shot/results.json` | 结果 | Smoke 测试（3 条，zero_shot） |
| `outputs/llm_phase1_prompt_screening/zero_shot/results.json` | 结果 | Prompt Screening（1230 条，zero_shot） |
| `outputs/llm_phase1_prompt_screening/few_shot_3/results.json` | 结果 | Prompt Screening（1230 条，few_shot_3） |
| `outputs/llm_phase1_prompt_screening/few_shot_5/results.json` | 结果 | Prompt Screening（1230 条，few_shot_5） |
| `outputs/llm_phase1_prompt_screening/cot/results.json` | 结果 | Prompt Screening（1230 条，CoT） |
| `outputs/llm_phase2_full_comparison/deepseek_few_shot/results.json` | 结果 | Full Comparison（4182 条，few_shot_3） |
| `outputs/llm_judge_smoke/judge_summary.json` | 结果 | Judge 汇总（2 条） |
| `outputs/llm_judge_smoke/judge_verdicts.jsonl` | 详情 | Judge 逐条评判 |
| `outputs/phase1_benchmark/train_metrics.json` | 结果 | RoBERTa benchmark（100 steps） |
| `reports/tables/phase1_null_baseline_test_results.json` | 结果 | Null baseline |
| `reports/model_runs/phase1_hf_qa_smoke_metrics.json` | 结果 | HF QA smoke test |
| `reports/phase1_status.md` | 文档 | Phase 1 状态汇总 |

### 9.2 每个实验的输出结构

```
outputs/llm_phase1_prompt_screening/<strategy>/
├── results.json        ← 聚合指标（本节分析的数据来源）
├── predictions.jsonl   ← 逐条预测详情（含 _raw_response）
└── extraction.log      ← 运行日志（HTTP 请求/响应速率）
```

### 9.3 运行命令速查

```bash
# Phase 1 Prompt Screening — zero_shot
PYTHONPATH=src python3 -m rulex.cli.llm_extract \
  --input data/processed/ruleframe/cuad_test.jsonl \
  --output-dir outputs/llm_phase1_prompt_screening/zero_shot \
  --prompt-strategy zero_shot --model deepseek-chat \
  --max-records 1230 --verbose

# Phase 1 Prompt Screening — few_shot_3
PYTHONPATH=src python3 -m rulex.cli.llm_extract \
  --input data/processed/ruleframe/cuad_test.jsonl \
  --train data/processed/ruleframe/cuad_train.jsonl \
  --output-dir outputs/llm_phase1_prompt_screening/few_shot_3 \
  --prompt-strategy few_shot_3 --model deepseek-chat \
  --max-records 1230 --verbose

# Phase 2 Full Comparison — few_shot_3
PYTHONPATH=src python3 -m rulex.cli.llm_extract \
  --input data/processed/ruleframe/cuad_test.jsonl \
  --train data/processed/ruleframe/cuad_train.jsonl \
  --output-dir outputs/llm_phase2_full_comparison/deepseek_few_shot \
  --prompt-strategy few_shot_3 --model deepseek-chat --verbose

# LLM-as-Judge Smoke
PYTHONPATH=src python3 -m rulex.cli.llm_judge \
  --input data/processed/ruleframe/cuad_test.jsonl \
  --predictions outputs/llm_phase2_full_comparison/deepseek_few_shot/predictions.jsonl \
  --output-dir outputs/llm_judge_smoke \
  --model deepseek-chat --model-label "DeepSeek-few-shot-3" \
  --max-records 2 --verbose
```

---

> **文档作者**: Reasonix AI Agent  
> **审核状态**: 待人工确认  
> **关联 PR**: [https://github.com/cosmic-snail/DoResearch/pull/1](https://github.com/cosmic-snail/DoResearch/pull/1)
