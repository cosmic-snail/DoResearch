# CUAD 实验计划：法律条款抽取 → 营养规则抽取预备研究

> **目标**：在 CUAD（Contract Understanding Atticus Dataset）上系统评估法律条款抽取方法，为后续营养指南规则抽取提供方法基础和迁移见解。  
> **前提**：营养数据未就绪前，CUAD 作为同构任务（篇章→span 定位）的完整实验场。  
> **创建日期**：2026-06-09  
> **ScholarAIO 工作区**：`C:\Users\56501\scholaraio`  
> **CUAD 数据**：`C:\Users\56501\scholaraio\data\cuad\CUADv1.json`

---

## 一、RoBERTa-base 详解

### 1.1 是什么

RoBERTa（Robustly optimized BERT approach）是 Meta AI 在 2019 年对 BERT 的改进版本。`roberta-base` 是其中参数量为 125M 的版本。

| 属性 | 值 |
|------|-----|
| 全称 | Robustly optimized BERT approach |
| 发布时间 | 2019 年 7 月（Liu et al., arXiv:1907.11692） |
| 参数量 | 125M |
| 层数 | 12 层 Transformer Encoder |
| 隐藏维度 | 768 |
| 注意力头数 | 12 |
| 最大序列长度 | 512 tokens |
| 词汇表大小 | 50K（Byte-level BPE） |
| 预训练数据 | 160GB 文本（BookCorpus + CC-News + OpenWebText + Stories） |
| 训练时间 | ~1 天（1024 块 V100 GPU） |

### 1.2 与 BERT 的关键差异

| 改进点 | BERT | RoBERTa |
|--------|------|---------|
| **训练数据量** | 16GB | 160GB（10x） |
| **训练步数** | 1M steps | 500K steps（但每步 batch 更大） |
| **静态 Masking** | 是 | 否 — 动态 Masking（每个 epoch 重新随机 mask） |
| **Next Sentence Prediction (NSP)** | 是 | 否 — 去掉 NSP，只做 MLM |
| **序列长度** | 固定 512 | 短序列逐步增加到长序列 |
| **Batch Size** | 256 | 8K（更大的 batch） |
| **Tokenization** | WordPiece (30K) | Byte-level BPE (50K) |
| **学习率调度** | 固定 warmup | 不同阶段的 warmup 比例调优 |
| **Adam 参数** | epsilon=1e-8, beta2=0.999 | epsilon=1e-6, beta2=0.98 |

### 1.3 动态 Masking 为什么重要

BERT 在数据预处理时对每个样本做一次随机 mask，每个 epoch 看到同样的 masked text。  
RoBERTa 将原始文本复制 10 份，每份独立随机 mask，10 个 epoch 看到 10 种不同的 masked version。

效果：模型不能靠记忆"某个位置一定被 mask"来投机取巧，更充分地学习上下文信息。在 GLUE/SQuAD 等下游任务上平均提升 1-3%。

### 1.4 去掉 NSP 的影响

BERT 预训练包含两个任务：Masked Language Model (MLM) 和 Next Sentence Prediction (NSP — 判断 B 句是否是 A 句的下文)。

RoBERTa 发现去掉 NSP 后下游任务不降反升。原因：NSP 过于简单，模型没有从中获得有意义的语言知识。更优做法是使用连续文本片段（从同一文档中采样），序列可以跨越文档边界（DOC-SENTENCES），实验证明这是最优策略。

### 1.5 为什么 CUAD 用 RoBERTa

1. **SQuAD 亲缘性**：CUAD 任务格式完全等同于 SQuAD extractive QA，RoBERTa 在 SQuAD 上表现优异
2. **性价比**：125M 参数，单 GPU 可训练，推理速度快（比 deberta-large 快 3x，比 GPT-4 便宜 1000x）
3. **可复现性**：HuggingFace 有成熟的 SQuAD fine-tuning pipeline，CUAD 代码直接复用
4. **领域泛化**：RoBERTa 的 byte-level BPE tokenizer 对法律/合同中专有名词和格式标记有更好覆盖
5. **社区基准**：论文还比较了 RoBERTa-large (355M) 和 DeBERTa-xlarge (900M)，roberta-base 作为最小基线

### 1.6 RoBERTa-base 在 CUAD 上的性能（论文报告）

| Model | AUPR | Prec@80%R | Prec@90%R |
|-------|------|----------|----------|
| RoBERTa-base | 44.6 | 38.6 | 23.3 |
| RoBERTa-large | 45.8 | 39.7 | 26.3 |
| DeBERTa-xlarge | 49.0 | 44.8 | 30.1 |

数字看起来不高，但注意：这是 41 类 macro average，且 70% 的问题没有答案（任务本身极难）。人类律师 AUPR 约 80-90，44.6 意味着模型已具备"初级律师"的条款发现能力。

### 1.7 架构图解

```
输入文本: "The Tenant shall maintain insurance with limits of $1M"
           ↓ Byte-level BPE tokenization
Tokens:    [The, GTenant, Gshall, Gmaintain, Ginsurance, Gwith, Glimits, 
            Gof, G$, 1, M]
           ↓
Embedding: Token Embedding + Position Embedding (learned) + Segment Embedding
           ↓
Transformer Encoder x 12:
  Layer i = MultiHeadSelfAttention(LN(x)) + x  -->  FeedForward(LN(x)) + x
  每个 attention head: Q,K,V = x*W_q, x*W_k, x*W_v  -->  softmax(QK^T/sqrt(d_k))*V
  FeedForward: 2 层全连接 (768 -> 3072 -> 768)，GELU 激活
           ↓
输出:       每个 token 的 768 维上下文表示
           ↓
SQuAD Head: start_logits = h*W_s (1x768, 每个 token 是答案起点的概率)
            end_logits = h*W_e (1x768, 每个 token 是答案终点的概率)
           ↓
解码:       argmax(start_logits[i] + end_logits[j]), i<=j, j-i < max_answer_len
```

### 1.8 对 CUAD 微调时的关键超参数

```yaml
# CUAD 官方 train.py 使用的参数
model: "roberta-base"
learning_rate: 3e-5
batch_size: 16 (per GPU)
epochs: 2
max_seq_length: 512
doc_stride: 128
max_query_length: 64
max_answer_length: 150  # 法律条款比 SQuAD 答案长得多！
warmup_steps: 0
weight_decay: 0.01
optimizer: AdamW

# 数据处理关键点:
# - question 被截断到 64 tokens（CUAD 的 question 长达 150 词）
# - 合同文本用 sliding window (stride=128) 切分成 512-token chunks
# - 每个 chunk 独立预测 answer，评估时合并
# - 正负样本平衡：训练时删除 70% 的负样本使正负比约 1:1
```

---

## 二、CUAD 数据全景

### 2.1 数据概况

| 属性 | 值 |
|------|-----|
| 合同数量 | 510 份（来自 SEC EDGAR 的 Exhibit 10 附件） |
| 合同来源 | 上市公司提交的商业合同（分销协议、许可协议、服务协议等） |
| 平均合同长度 | ~52,000 字符 / ~15,000 tokens |
| 标注条款类型 | 41 种（clause types） |
| 总 QA pairs | 20,910（= 510 x 41） |
| 有答案的比例 | ~30%（大部分条款在大部分合同中不存在） |
| 标注团队 | 数十名律师（Attorney Advisors）+ 法学院学生，律师监督 |

### 2.2 数据格式

SQuAD JSON 格式。每份合同是一个 `paragraph`，包含整份合同全文作为 `context`，以及 41 个 QA pair（对应 41 个 clause type）。每个 QA pair 包含一个自然语言 question 和 0-N 个 answer span（`text` + `answer_start` 字符偏移量）。

### 2.3 41 个 Clause Type 完整列表

按难度分层：

**容易（短答案、高频、固定格式）**：Document Name, Parties, Agreement Date, Effective Date, Expiration Date

**中等（中等长度、有模式但有变化）**：Renewal Term, Notice Period To Terminate Renewal, Governing Law, Termination For Convenience, Change Of Control, Anti-Assignment, Ip Ownership Assignment, Joint Ip Ownership, License Grant, Non-Transferable License, Post-Termination Services, Audit Rights, Cap On Liability, Liquidated Damages, Warranty Duration, Insurance

**困难（长答案、罕见、语义依赖上下文）**：Most Favored Nation, Non-Compete, Exclusivity, No-Solicit Of Customers, Competitive Restriction Exception, No-Solicit Of Employees, Non-Disparagement, Rofr/Rofo/Rofn, Revenue/Profit Sharing, Price Restrictions, Minimum Commitment, Volume Restriction, Affiliate License-Licensor, Affiliate License-Licensee, Unlimited/All-You-Can-Eat-License, Irrevocable Or Perpetual License, Source Code Escrow, Uncapped Liability, Covenant Not To Sue, Third Party Beneficiary

---

## 三、三阶段实验方案

### 阶段一：基线复现与分析（~2 周）

#### 实验 1.1：RoBERTa-base 基线复现

复现 CUAD 论文 baseline。用 `roberta-base` + HuggingFace `AutoModelForQuestionAnswering`，CUAD train (410) / test (100) split，官方超参数。评估 AUPR、Precision@80%Recall、Precision@90%Recall、Token-level F1（IoU >= 0.5），与论文报告的 44.6 AUPR 对比。

#### 实验 1.2：41 类难度分层分析

逐类别计算 AUPR/F1，分析答案覆盖率、平均答案长度、答案分散度、位置熵、词汇多样性五个维度对难度的影响，用 KMeans (k=3) 聚类为 easy/medium/hard 三层。为营养规则类型设计提供参考——CUAD 中的 easy 类型对应格式固定的营养规则（"每日摄入量"），hard 类型需要对上下文语义推理。

#### 实验 1.3：负样本处理策略

对比四种处理 70% 无答案 case 的策略：SQuAD v2 null prediction、Binary pre-filter classifier、Confidence threshold rejection、纯文本分类器（不用 QA head）。评估 "no answer" accuracy、positive-only F1、overall F1。

### 阶段二：深度分析（~3 周）

#### 实验 2.1：Span 边界精度分析

IoU >= 0.5 太宽松。细粒度评估 Exact Match、Boundary F1、起止偏移分布、overshoot/undershoot ratio。分析长答案 vs 短答案的边界误差模式差异——这对营养规则（需要精确抽取数值阈值）至关重要。

#### 实验 2.2：Multi-Span vs Single-Span

多答案 clause（约占 15%）的 recall gap。对比 RoBERTa top-1、top-3 with non-max-suppression、Iterative masking、GPT-4 generative listing。核心指标：multi-span recall、precision@k、over-generation rate。

#### 实验 2.3：Cross-Type Generalization（最关键）

留出 K=5/10/15 个 clause type 不参与训练，测试 unseen types。三组留出策略：随机、按语义聚类（Licensing/Restrictions/Financial 三簇）、只留出 hard types。核心发现：语义相近的 unseen type 比随机 unseen 好多少？多少 shot 能让 unseen type 恢复到 seen type 的 80%？

**这对营养规则的直接启示**：当你定义新的营养规则类型时，每个类型最少需要标注多少条。

#### 实验 2.4：Question 表述的敏感性

对比 5 种 question 变体（原始长 question / 短 question / 纯关键词 / 中文 / 错误标签名），不重新训练——在已训练模型上直接 inference。度量 AUPR 下降幅度和各类别方差。验证"问法的稳定性"对后续营养规则的 query 设计至关重要。

### 阶段三：前沿方法对比（~3 周）

#### 实验 3.1：LLM Prompting 对比

4 个模型（GPT-4o, Claude Sonnet 4.5, DeepSeek-V3, LLaMA-3-70B）x 4 种 prompt 策略（Zero-shot, Few-shot(3), Few-shot(5), Chain-of-Thought）。核心发现：GPT-4 zero-shot 能否接近 fine-tuned RoBERTa？额外指标：$ cost per 100 contracts, latency, 5-run consistency。

#### 实验 3.2：LLM Scaling 效应

LLaMA 3.2 1B -> 3B -> 8B -> 70B, Mistral 7B, Qwen 14B/32B, Mixtral 8x7B 的全阶梯测试。分析是否存在 emergent ability threshold——在第几 B 参数时 clause extraction 能力跃升。

#### 实验 3.3：Fine-tune vs Prompt vs RAG

三维度系统对比：AUPR、速度、成本、可部署性、新类型适应。RAG pipeline：embed contract paragraphs -> retrieve top-5 -> LLM extracts span。这是后续营养规则系统（长指南 + 多规则类型）的架构预研。

#### 实验 3.4：Error Taxonomy

定义 4 大类错误（FN / FP / Boundary Error / Complete Misplacement），在 RoBERTa vs GPT-4 vs RAG 上做错误分布对比。分析营养规则中哪种错误后果最严重（FP 可能导致错误推荐，undershoot 可能截断关键数值）。

---

## 四、实验矩阵总览

| 阶段 | 实验 | 核心问题 | 时间 | 可发表性 |
|------|------|---------|------|:---:|
| **1.1** | Baseline 复现 | CUAD RoBERTa 基线能否复现？ | 3天 | 基础 |
| **1.2** | 难度分层 | 41 类难度差异多大？为营养规则提供参考 | 3天 | ★★★ |
| **1.3** | 负样本策略 | 70% 无答案，最优处理是什么？ | 4天 | ★★★ |
| **2.1** | Span 边界 | IoU 宽松->EM 严格，精度差了多少？ | 4天 | ★★★★ |
| **2.2** | Multi-Span | 多答案场景的 recall gap | 4天 | ★★★ |
| **2.3** | Cross-Type 泛化 | 训练过的类型能迁移到 unseen 吗？ | 5天 | ★★★★★ |
| **2.4** | Question 敏感性 | 问法对结果影响多大？ | 3天 | ★★★ |
| **3.1** | LLM Prompting | GPT-4 能省掉 fine-tune 吗？ | 5天 | ★★★★★ |
| **3.2** | LLM Scaling | 多大模型才够？ | 4天 | ★★★ |
| **3.3** | 三范式对比 | FT vs Prompt vs RAG 全对比 | 5天 | ★★★★ |
| **3.4** | Error Taxonomy | 所有方法错在哪？ | 4天 | ★★★★ |

---

## 五、推荐执行路线

### 最低可发表组合（~4 周）

Week 1: [1.1] Baseline 复现  
Week 2: [1.3] 负样本策略 + [2.3] Cross-Type 泛化  
Week 3: [3.1] LLM Prompting 对比  
Week 4: [3.4] Error Taxonomy + [2.1] Span 边界  

产出: short paper — "Revisiting CUAD: Negative Samples, Cross-Type Transfer, and the LLM Baseline"

### 完整论文路线（~8 周）

Week 1-2: [1.1] -> [1.2] -> [1.3]  
Week 3-4: [2.1] -> [2.2] -> [2.3] -> [2.4]  
Week 5-7: [3.1] -> [3.2] -> [3.3]  
Week 8: [3.4] + 论文写作  

产出: full paper — "A Systematic Study of Legal Clause Extraction: From RoBERTa to LLMs, from Classification to Cross-Type Generalization"

---

## 六、技术实现备忘

### 6.1 环境

CUAD 数据已在 `C:\Users\56501\scholaraio\data\cuad\CUADv1.json`（510 合同，41 clause types）。环境已有 torch 2.12.0+cpu、transformers 5.10.2、datasets、evaluate、seqeval。

### 6.2 核心评估指标

CUAD 评估基于 token-level Jaccard IoU >= 0.5：

```
IoU = |pred_tokens AND gt_tokens| / |pred_tokens OR gt_tokens|
命中判定: IoU >= 0.5

对所有 prediction 按 confidence 排序 -> 计算不同 threshold 下的 (prec, recall) -> AUPR
```

### 6.3 预期输出产物

```
experiments/
  1.1_baseline/     train.py + evaluate.py + results.json + README
  1.2_difficulty/   analysis notebook + difficulty_scores.csv
  1.3_negatives/    strategy comparison + results
  2.1_boundary/     boundary error distribution plots
  2.2_multispan/    multi-span recall analysis
  2.3_crosstype/    cross-type generalization results
  2.4_question/     question ablation results
  3.1_llm_prompt/   LLM prompting results
  3.2_scaling/      scaling curve plot
  3.3_paradigms/    three-paradigm comparison
  3.4_errors/       error taxonomy results
```

---

## 七、对营养规则抽取的映射表

| CUAD 概念 | 营养规则对应 | 是否可直接迁移 |
|-----------|------------|:---:|
| Clouse Type (41 类) | Nutrition Rule Type (N 类) | 是 |
| Full Contract (~52K chars) | Full Guideline PDF (~80K chars) | 是 |
| Question Template | Rule Type Query | 是 |
| Answer Span (text excerpt) | Rule Paragraph Span | 是 |
| Multi-Answer QA | Multi-Rule Guideline | 是 |
| No-Answer QA (70%) | No-Rule-For-This-Type | 是（更多） |
| Jaccard IoU >= 0.5 | 需要更紧的边界度量 | 需调整 |
| 跨法律类型泛化 (Exp 2.3) | 跨营养规则类型泛化 | 核心迁移问题 |
| LLM Prompting (Exp 3.1) | LLM 零样本营养抽取 | 方法直接迁移 |

---

## 附录：RoBERTa 家族速查

| 模型 | 参数 | 层 | 隐藏维度 | 注意力头 | HuggingFace ID |
|------|------|---|---------|---------|---------------|
| RoBERTa-base | 125M | 12 | 768 | 12 | `roberta-base` |
| RoBERTa-large | 355M | 24 | 1024 | 16 | `roberta-large` |
| DeBERTa-base | 140M | 12 | 768 | 12 | `microsoft/deberta-base` |
| DeBERTa-large | 400M | 24 | 1024 | 16 | `microsoft/deberta-large` |
| DeBERTa-v3-large | 435M | 24 | 1024 | 16 | `microsoft/deberta-v3-large` |
| DeBERTa-xlarge (CUAD SOTA) | 900M | 48 | 1024 | 16 | `microsoft/deberta-xlarge` |
