# 营养指南规则抽取预研：公开数据集实验执行计划

> **目标**：在营养领域标注数据尚未就绪前，先用公开数据集验证“长文档规范性文本 -> 规则片段抽取 -> 结构化规则编码 -> 可验证推荐逻辑”的实验框架，并形成后续迁移到营养指南的最小可复用系统。
>
> **整合来源**：`research-plan-nutrition-rule-extraction.md` 与 `cuad-experiment-plan.md`
>
> **建议主线**：以 CUAD 为主实验场，研究跨规则类型泛化、无答案处理、边界精度、LLM 与微调模型对比、规则结构化编码；以 DART/WebNLG/RAMS/Spider 做辅助验证，不把它们扩展成同等规模主线。

---

## 1. 研究定位

### 1.1 最终课题

最终面向营养指南，抽取如下形式的可执行规则：

```text
适用对象 / 条件集合 -> 推荐动作 -> 数值约束 -> 证据等级 / 强度 -> 例外条件 -> 来源片段
```

例：

```json
{
  "population": "成人2型糖尿病患者",
  "condition": ["超重或肥胖", "无肾功能禁忌"],
  "recommendation": "控制总能量摄入并增加膳食纤维",
  "numeric_constraint": [{"nutrient": "fiber", "operator": ">=", "value": 25, "unit": "g/day"}],
  "strength": "strong",
  "exception": ["急性并发症期间需个体化处理"],
  "source_span": "指南原文片段",
  "confidence": 0.82
}
```

### 1.2 当前约束

- 营养领域 gold labels 暂不可用。
- 直接做营养推荐系统会缺少可靠评估闭环。
- 因此先做“领域无关的规则抽取框架”，用公开数据集验证方法论。

### 1.3 阶段性论文问题

当前阶段不宜直接宣称“营养推荐系统完成”，而应聚焦：

> 在标注稀缺的规范性长文档中，如何可靠抽取可结构化编码的规则片段，并评估其跨规则类型、跨领域迁移能力？

这个问题能自然连接法律合同、流程/事件文本、结构化语义解析和未来营养指南。

---

## 2. 数据集策略

### 2.1 主数据集：CUAD

CUAD 最适合作为第一阶段主实验场，因为它同时具备四个和营养指南相似的特征：

- 长文档输入：整份合同对应未来的整份指南。
- 规范性文本：合同条款与指南建议都带有义务、限制、例外和条件。
- 多规则类型：41 个 clause types 对应未来的营养 rule types。
- 大量无答案问题：很多合同不包含某类条款，对应指南中某类规则不存在的情形。

CUAD 任务映射：

| CUAD | 营养指南 |
| --- | --- |
| Contract | Guideline |
| Clause type | Nutrition rule type |
| Question template | Rule query |
| Answer span | Recommendation / evidence span |
| No-answer QA | Rule absent |
| Multi-answer clause | 同类多条推荐 |
| Cross-type holdout | 新营养规则类型低样本迁移 |

### 2.2 辅助数据集

| 数据集 | 用途 | 是否纳入主实验 |
| --- | --- | --- |
| DART / WebNLG | 训练或测试 text -> triples / frame 的结构化编码能力 | 只做 Phase 3 辅助 |
| RAMS / WikiEvents | 评估跨句 argument role 填充能力 | 只做鲁棒性分析 |
| Spider | 借鉴 text -> executable logic / SQL 的评估思想 | 暂不训练，作为 DSL 设计参考 |
| 医学 IE 数据集 | NER 层迁移，如 disease/drug/nutrient entity 识别 | 等营养数据出现后再接入 |

### 2.3 不建议现在同时铺开的内容

- 不建议同时跑完 CUAD、RAMS、WebNLG、Spider 的完整 benchmark。
- 不建议一开始就做完整营养病例推荐评估。
- 不建议把模型规模比较做成主贡献，除非已有充足算力和 API 预算。

---

## 3. 统一任务定义

### 3.1 RuleFrame 中间表示

所有数据集先转换到统一的 RuleFrame，避免后续营养迁移时推倒重来：

```json
{
  "doc_id": "string",
  "domain": "legal | nutrition | event | data2text",
  "rule_type": "string",
  "query": "string",
  "source_text": "string",
  "gold_spans": [
    {
      "text": "string",
      "start": 0,
      "end": 10
    }
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
    "answer_count": 1,
    "source_dataset": "CUAD"
  }
}
```

### 3.2 两层任务

**Task A：规则片段定位**

输入：文档 + 规则类型查询。

输出：0-N 个原文 span。

指标：AUPR、Precision@80%Recall、Exact Match、Token F1、Boundary F1、No-answer accuracy、Multi-span recall。

**Task B：规则结构化编码**

输入：已定位 span。

输出：RuleFrame 中的 condition/action/value/exception/evidence 字段。

指标：schema compliance、slot-level F1、numeric exact match、unit normalization accuracy、logic consistency、round-trip consistency。

---

## 4. 核心研究问题与实验假设

### RQ1：跨规则类型泛化

**问题**：模型见过一部分类别后，能否抽取未见过的新规则类型？

**假设**：LLM/RAG 在 unseen rule types 上优于只靠 QA fine-tuning 的 RoBERTa；但在 seen types 上，小模型微调可能更稳定、更便宜。

**营养意义**：未来营养规则类型一定会不断扩展，不能每个类型都重标大量数据。

### RQ2：无答案与多答案鲁棒性

**问题**：系统能否知道“此文档没有该规则”，并在有多条规则时全部找出？

**假设**：二阶段架构（先判断存在性，再抽取 span）会提升 precision；LLM 生成式方法召回高但过生成风险大。

**营养意义**：错误地生成不存在的营养建议比漏掉一条建议风险更高。

### RQ3：边界与数值精度

**问题**：span 粗略命中不够，系统能否精确保留阈值、单位、例外和适用人群？

**假设**：传统 CUAD IoU >= 0.5 对营养规则过于宽松，需要 Boundary F1、numeric exact match 和 unit normalization。

**营养意义**：`< 2 g/day`、`2 g/kg/day`、`2 servings/day` 的错误会直接改变推荐。

### RQ4：片段抽取到结构化编码的对齐

**问题**：抽到原文后，能否稳定转成可执行 JSON/DSL？

**假设**：schema-constrained LLM 输出 + rule validator 比自由生成 JSON 更可靠；DART/WebNLG 可作为结构化语义映射的辅助预训练或评估材料。

**营养意义**：推荐系统需要的是可计算规则，不只是可读摘要。

---

## 5. 实验路线

### Phase 0：实验底座搭建（3-5 天）

**目标**：让所有后续实验都能复用同一套数据接口、预测接口和评估接口。

产物：

- `data/raw/cuad/`：CUAD 原始数据。
- `data/processed/ruleframe/cuad.jsonl`：统一 RuleFrame 格式。
- `src/data_adapters/cuad_adapter.py`：CUAD -> RuleFrame。
- `src/evaluation/span_metrics.py`：AUPR、Token F1、Boundary F1、No-answer accuracy。
- `src/evaluation/frame_metrics.py`：schema compliance、slot F1、numeric exact match。
- `experiments/registry.yaml`：记录每个实验的数据、模型、prompt、随机种子、输出路径。

验收标准：

- 能从 510 份合同生成 `doc_id, rule_type, query, context, answers`。
- 能对一个 dummy predictor 计算全套指标。
- 所有实验结果输出为 `results.json` 和 `predictions.jsonl`。

### Phase 1：CUAD 基线复现与诊断（1-2 周）

**目标**：建立可信基线，确认本地评估与公开 benchmark 不偏离。

实验：

1. RoBERTa-base extractive QA baseline。
2. Legal-BERT / DeBERTa 可选 baseline。
3. 41 类 clause difficulty profiling。
4. 无答案策略对比：SQuAD-v2 null prediction、二分类 pre-filter、confidence threshold。

关键指标：

- Macro AUPR。
- Precision@80%Recall。
- Positive-only F1。
- No-answer accuracy。
- 每类答案长度、答案数量、位置分布、训练样本数。

阶段判断：

- 如果 RoBERTa baseline 明显复现失败，先修数据切分、token offset、sliding window 合并和评估脚本。
- 如果复现接近公开结果，进入 Phase 2。

### Phase 2：核心 open question 实验（2-3 周）

**目标**：形成当前阶段最有论文价值的结果。

#### Exp 2.1 Cross-Type Generalization

设置：

- 留出 5/10/15 个 clause types 不参与训练。
- 三种留出方式：随机、按语义簇、只留 hard types。
- 对比：RoBERTa fine-tune、RAG + LLM、zero-shot LLM、few-shot LLM。

指标：

- Seen vs unseen AUPR/F1 gap。
- 达到 seen-type 80% 性能所需 examples 数。
- Unseen hard types 的失败模式。

预期贡献：

- 给出“新营养规则类型需要多少标注样本”的经验曲线。

#### Exp 2.2 Question / Query Sensitivity

设置：

- 原始长 question。
- 短 query。
- 关键词 query。
- schema-style query，如 `extract condition/action/value/exception for non-compete`。
- 错误或模糊 query。

指标：

- 性能下降比例。
- 类别间方差。
- LLM 与 fine-tuned QA 的稳定性差异。

预期贡献：

- 指导未来营养 rule query / ontology label 的写法。

#### Exp 2.3 No-answer 与 Multi-span

设置：

- 无答案：null prediction、pre-filter、threshold calibration。
- 多答案：top-k span、NMS、iterative masking、LLM list extraction。

指标：

- No-answer precision/recall。
- Multi-span recall@k。
- Over-generation rate。
- Calibration curve。

预期贡献：

- 给出医学/营养场景更重视安全性的阈值选择策略。

#### Exp 2.4 Boundary and Numeric Precision

设置：

- 对比 IoU >= 0.5、Exact Match、Boundary F1、起止偏移误差。
- 对包含数字、日期、金额、比例的 CUAD clause 做子集分析。
- 模拟营养数值规则：抽取 threshold/operator/unit/value。

指标：

- Boundary F1。
- Numeric exact match。
- Unit / operator accuracy。
- Overshoot vs undershoot ratio。

预期贡献：

- 论证传统 span-level F1 不足以评估可执行规则。

### Phase 3：结构化编码与可验证规则（2 周）

**目标**：从“抽片段”推进到“可执行规则中间表示”。

实验：

1. 对 CUAD span 做 frame filling：condition、obligation/action、threshold/value、exception、party/entity。
2. 使用 schema-constrained prompting 生成 JSON。
3. 使用 validator 检查字段类型、枚举值、单位、空值、逻辑矛盾。
4. 使用 DART/WebNLG 做 text -> triple/frame 的辅助 sanity check。

指标：

- JSON parse success。
- Schema compliance。
- Slot-level precision/recall/F1。
- Value/operator/unit exact match。
- Round-trip consistency：RuleFrame -> text -> RuleFrame 是否保持核心字段。

阶段产物：

- 初版跨领域 Rule DSL。
- 初版 validator。
- 结构化错误 taxonomy。

### Phase 4：营养领域冷启动迁移（1-2 周，等待资料可用后）

**目标**：用少量营养指南资料验证迁移，而不是立即追求完整临床推荐系统。

数据：

- 1-3 份开放营养指南 PDF。
- 优先选择带推荐汇总表、证据等级表、适用人群说明的指南。

方法：

- PDF/table extraction 生成 pseudo-gold。
- CUAD 上选出的最佳抽取方法 zero-shot 到营养指南。
- LLM-assisted 标注 50-100 条营养规则。
- 用人工小样本校准 LLM-as-judge 与内在指标。

指标：

- Table-derived pseudo recall。
- Human-checked precision。
- Schema compliance。
- Numeric exact match。
- Contradiction / exception detection。

阶段产物：

- 营养规则 schema v1。
- 50-100 条小规模 gold/pseudo-gold。
- “法律规范文本 -> 营养指南”的迁移分析。

---

## 6. 推荐最小可发表实验包

如果目标是 4-6 周内形成一篇 workshop/short paper，建议只做以下组合：

1. CUAD baseline 复现。
2. Cross-type generalization。
3. No-answer + multi-span calibration。
4. Boundary/numeric precision。
5. LLM vs fine-tuned QA vs RAG 三范式对比。
6. 小规模 RuleFrame 编码实验。

暂缓：

- 全量 LLM scaling。
- 多医学 IE 数据集迁移。
- 完整营养病例推荐系统。
- 医师多人一致性评估。

建议题目方向：

```text
From Normative Text to Executable Rules:
A Cross-Type Generalization Study for Low-Resource Rule Extraction
```

或更贴近营养终局：

```text
Preparing Nutrition Guideline Rule Extraction Without Nutrition Labels:
Lessons from Legal Clause Extraction and Structured Rule Encoding
```

---

## 7. 8 周执行排期

| 周次 | 任务 | 关键产物 | Go / No-Go |
| --- | --- | --- | --- |
| Week 1 | Phase 0 + CUAD adapter + metrics | RuleFrame 数据、dummy metrics | 指标脚本可跑 |
| Week 2 | RoBERTa baseline + 41 类诊断 | baseline results、difficulty table | 复现误差可解释 |
| Week 3 | Cross-type holdout | seen/unseen 曲线 | 出现清晰泛化差距 |
| Week 4 | Query sensitivity + no-answer | query ablation、calibration | 找到稳健 query 策略 |
| Week 5 | Multi-span + boundary/numeric | error taxonomy v1 | 证明宽松 F1 不够 |
| Week 6 | LLM/RAG 对比 | 三范式表格 | 明确成本-效果取舍 |
| Week 7 | RuleFrame 编码 + validator | schema compliance 结果 | JSON/DSL 可稳定生成 |
| Week 8 | 写作与营养迁移 pilot | paper draft、nutrition schema v1 | 能支撑 short paper |

---

## 8. 实验目录建议

```text
data/
  raw/
    cuad/
  processed/
    ruleframe/

src/
  data_adapters/
    cuad_adapter.py
    dart_adapter.py
    nutrition_adapter.py
  models/
    qa_extractor.py
    llm_extractor.py
    rag_extractor.py
  prompting/
    clause_prompts.yaml
    ruleframe_prompts.yaml
  evaluation/
    span_metrics.py
    frame_metrics.py
    calibration.py
  schema/
    ruleframe.schema.json
    validator.py

experiments/
  00_smoke_test/
  01_cuad_baseline/
  02_cross_type_generalization/
  03_query_sensitivity/
  04_no_answer_multispan/
  05_boundary_numeric/
  06_llm_rag_comparison/
  07_ruleframe_encoding/

reports/
  figures/
  tables/
  error_analysis/
```

---

## 9. 主要风险与规避

| 风险 | 表现 | 规避 |
| --- | --- | --- |
| 实验过散 | 每个数据集都跑一点，但没有主贡献 | CUAD 是主线，其他数据集只回答结构化编码或鲁棒性问题 |
| 只做 benchmark 没有科研问题 | 结果像工程复现 | 把 RQ1 跨类型泛化、RQ3 边界/数值精度作为论文主问题 |
| CUAD 与营养差异太大 | 法律条款不能代表营养推荐 | 只迁移“规范性长文档规则抽取方法”，不迁移领域实体语义 |
| LLM 成本不可控 | 多模型多 prompt 费用高 | 先抽 50-100 合同子集做筛选，再全量跑 2-3 个代表方法 |
| 指标不能说明可执行性 | span F1 高但 JSON 不能用 | Phase 3 加 schema compliance、numeric exact match、validator |
| 营养数据仍然延迟 | 无法展示最终领域 | 用 1-3 份公开指南 + pseudo-gold 做 pilot |

---

## 10. 立即行动清单

### 第 1 天

- 下载或确认 CUAD 数据路径。
- 建立项目目录结构。
- 写 `RuleFrame` schema。
- 写 CUAD adapter，输出 `data/processed/ruleframe/cuad.jsonl`。

### 第 2-3 天

- 实现 span metrics。
- 用 dummy predictions 跑通完整评估。
- 抽样 20 条检查 offset、答案边界、无答案样本。

### 第 4-7 天

- 跑 RoBERTa-base baseline。
- 生成每类 clause 的训练样本数、答案长度、无答案比例。
- 输出第一版 `reports/tables/cuad_baseline.csv`。

### 第 2 周

- 做 cross-type holdout split。
- 跑 seen/unseen 对比。
- 记录失败样本，建立 error taxonomy。

### 第 3-4 周

- 加 LLM zero-shot / few-shot / RAG。
- 做 query sensitivity。
- 做 no-answer threshold calibration。

### 第 5-6 周

- 做 boundary/numeric precision。
- 做 RuleFrame JSON 编码。
- 写 validator 与 schema compliance。

### 第 7-8 周

- 选 1-3 份营养指南做 pilot。
- 整理论文图表。
- 写 short paper 初稿。

---

## 11. 预期论文贡献表述

建议把贡献写成三点：

1. **问题贡献**：提出面向低资源规范性长文档的可执行规则抽取框架，将 span extraction、cross-type transfer、schema encoding 和 verification 放在同一评估闭环中。
2. **实证贡献**：在 CUAD 上系统分析跨规则类型泛化、无答案、多答案、query 敏感性和边界/数值精度，给出对营养指南迁移有直接意义的经验结论。
3. **系统贡献**：构建 RuleFrame 中间表示、validator 和可复现实验管线，为后续营养指南规则抽取与推荐系统提供可迁移底座。

---

## 12. 参考数据源

- CUAD：510 commercial contracts、13,000+ labels、41 clause types、CC BY 4.0。核验来源：The Atticus Project CUAD 页面、CUAD GitHub、CUAD arXiv paper。
- RAMS：document-level event argument extraction，可用于跨句 argument role 思路参考。核验来源：JHU RAMS 数据集页面。
- DART / WebNLG：text 与 structured triples 的双向映射，可用于结构化编码 sanity check。核验来源：DART NAACL 2021、WebNLG INLG 2017。
- Spider：text-to-SQL 的 executable logic 评估思想，可用于后续 nutrition DSL 设计参考。核验来源：Spider GitHub / EMNLP 2018。

公开链接：

- CUAD dataset page: https://www.atticusprojectai.org/cuad/
- CUAD GitHub: https://github.com/TheAtticusProject/cuad/
- CUAD paper: https://arxiv.org/abs/2103.06268
- RAMS: https://nlp.jhu.edu/rams/
- DART: https://aclanthology.org/2021.naacl-main.37/
- WebNLG: https://aclanthology.org/W17-3518/
- Spider: https://github.com/taoyds/spider
