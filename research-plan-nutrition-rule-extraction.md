# 营养指南规则抽取 → 结构化编码：科研课题全景文档

> **项目背景**：从临床营养指南中自动抽取条件规则，并将其结构化为可执行的推荐系统。  
> **核心 Pipeline**：`营养指南PDF → LLM/规则抽取 → 结构化规则 → 饮食推荐系统`  
> **关联研究方向**：IE 中的 Rule Extraction + Canonicalization + Neuro-Symbolic + Domain Transfer  
> **创建日期**：2026-06-09  


## 三、2026 研究方向全景

### 3.1 领域主线（5 条）

1. **LLM 取代传统 Rule-Based IE 范式**：不再手工写规则，而是让 LLM 生成规则，规则成为中间产物→再编译为结构化知识/代码
2. **Neural-Symbolic 融合**：LLM 生成自然语言概念模块，符号程序组合为可解释决策规则（LSP 范式）
3. **KG + LM 联合规则学习**：LM 辅助 KG 规则排序，嵌入+规则联合
4. **多模态/多源知识融合构建**：代码/KG/文本统一规则抽取
5. **Canonicalization 最后一步**：open tuples → fixed schema mapping

### 3.2 七大 Open Question

| 编号 | 问题 | 紧迫度 | 空白度 |
|------|------|--------|--------|
| **OQ1** | 如何让 LLM 抽取的规则保证事实正确性（Hallucination Grounding） | 🔴 极高 | 有竞争 |
| **OQ2** | 从确认性抽取→发现性抽取（Confirmation→Discovery） | 🔴 极高 | 🟢 空白 |
| **OQ3** | 规则→编码端到端对齐（Canonicalization + Schema Mapping） | 🟡 高 | 有人做 |
| **OQ4** | 多模态规则抽取 | 🟡 高 | 🟢 很少 |
| **OQ5** | 规则质量形式化验证（Symbolic Verification） | 🟡 高 | 🟢 空白 |
| **OQ6** | 增量/终身规则学习 | 🟢 中 | 🟢 空白 |
| **OQ7** | 规则质量多维评价体系 | 🔴 极高 | 🟢 空白 |

### 3.3 关键论文速查

| 论文 | arXiv ID | 核心贡献 |
|------|----------|---------|
| **Survey: OpenIE from Rule-based to LLM** | 2208.08690 | 2007→2024 全脉络综述 |
| **LLMs are Interpretable Learners (LSP)** | 2406.17224 | LLM→符号程序，可解释规则 |
| **Learning Rules from KGs Guided by LMs** | 2409.07869 | LM 辅助 KG 规则排序 |
| **Mapping & Cleaning Open KBs (Generative Translation)** | 2306.12766 | Open tuples→fixed schema |
| **OntoMetric** | 2512.01289 | Ontology-constrained LLM extraction |
| **UNO: User log-driven Optimization** | 2602.06470 | 日志→半结构化规则蒸馏 |
| **RepoDoc** | 2604.26523 | 代码→KG→文档生成 |
| **Phenotype-driven KG enrichment** | 2604.16982 | GNN+因果+LLM，novelty 发现 |
| **Explainable Rule Application** | 2506.16335 | 三步验证框架（实体→属性→规则） |
| **APRCOIE** | 2403.10758 | 中文 OIE，自动模式生成 |
| **IMoJIE** | 2005.08178 | Iterative memory-based OpenIE |

---

## 四、实验框架设计

### 4.1 Pipeline 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        四层 Pipeline                              │
├──────────────┬────────────────┬────────────────┬────────────────┤
│ ① 抽取层       │ ② 规则层         │ ③ 编码层         │ ④ 评估层         │
│ LLM/NER/OpenIE│ Rule Induction │ Schema Mapping  │ 自动+人工       │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ 输入: 指南文本  │ 中间产物: 规则    │ 输出: JSON/DAG   │ 多维度指标       │
│ →实体/关系      │ →逻辑条件        │ →可执行代码       │                 │
└──────────────┴────────────────┴────────────────┴────────────────┘
```

### 4.2 五组实验

| 实验 | 研究问题 | 核心方法 | 评估 |
|------|---------|---------|------|
| **Exp1** | 多方法对比：规则/神经/LLM 抽取营养规则 | GPT-4 vs BioBERT vs 正则 baseline | F1 on rule extraction |
| **Exp2** | 跨指南规则冲突检测 | 逐对比较 + 冲突分类 | 冲突 recall/precision |
| **Exp3** | Ontology-Constrained 验证 | UMLS/SNOMED + 文献检索 | 三层 hallucination rate |
| **Exp4** | 规则→结构化编码 | DSL 填充 + schema compliance | Type correctness |
| **Exp5** | 临床效用评估 | 50 个病例 + 3 名营养师对照 | Cohen's kappa |

### 4.3 论文结构建议

```
Title: "From Guidelines to Tables: LLM-Based Rule Extraction, Verification, 
        and Structured Encoding for Evidence-Based Dietary Recommendation"

1. Introduction（营养指南数量爆炸 + LLM hallucination 在医学致命）
2. Related Work（Medical IE, OpenIE, Neuro-Symbolic, Clinical DSS）
3. Method（3.1 Extraction / 3.2 Ontology Verification / 3.3 Schema Encoding / 3.4 Conflict Detection）
4. Experiments（Exp1-Exp5 + Ablation）
5. Discussion（Limitations, Ethics, Future Work）
6. Conclusion
```

### 4.4 三个差异化切入视角

| 视角 | 核心贡献 | 实验重点 | 难度 |
|------|---------|---------|------|
| **A: Safety-first** | 首个人工+自动双层验证保障 | Exp1 + Exp3 + Exp5 | ⭐⭐ |
| **B: Discovery-oriented** | 跨指南矛盾检测、隐性规则发现 | Exp2 + Exp4 | ⭐⭐⭐ |
| **C: Full pipeline** | PDF→可执行推荐，全链路开源 | 全部 | ⭐⭐⭐⭐ |

**推荐起始点**：A（Safety-first）——实验量适中、叙事清晰、临床审稿人认同度高。

---

## 五、无标注数据的评价路径

### 六条路径速查

| 路径 | 核心思想 | 零标注 | 可发表性 |
|------|---------|--------|---------|
| **路径一** | 指南汇总表作 Pseudo-Label | ✅ 是 | ⭐⭐⭐⭐ 高 |
| **路径二** | Round-Trip Self-Consistency | ✅ 是 | ⭐⭐⭐ 中高 |
| **路径三** | LLM-as-Judge + 小规模人工校准 | 🟡 需要 50 条 | ⭐⭐⭐⭐ 高 |
| **路径四** | 逻辑一致性作内在指标 | ✅ 是 | ⭐⭐⭐⭐ 高 |
| **路径五** | 跨领域迁移（已有标注集→营养 zero-shot） | ✅ 是 | ⭐⭐⭐⭐⭐ 极高 |
| **路径六** | LLM 预抽取 + 人工审核快速建标注集 | 🟡 需要 8-12h | ⭐⭐⭐⭐⭐ 极高 |

### 推荐三阶段路线

```
Phase 1（现在，0 标注，约 1 月）
├── 路径一：提取 PDF 汇总表 → pseudo-gold rules → 自动评测
├── 路径四：逻辑一致性 + 本体合规度 + 内部矛盾检测
└── 产出：初版系统 + intrinsic evaluation → 投 workshop

Phase 2（少量标注，约 1 周）
├── 路径三：50 条人工标注 → 校准 LLM-as-judge
├── 路径二：Round-trip self-consistency
└── 产出：calibrated evaluation framework → 投 short paper

Phase 3（完整标注，约 1 月）
├── 路径六：LLM-assisted 300 条标注
├── Exp1-Exp5 全部跑通
└── 产出：完整 benchmark → 投 ACL/EMNLP/AAAI full paper
```

---

## 六、数据集全景

### 6.1 医学 IE 数据集（格式不匹配，但实体层可用）

| 数据集 | 标注格式 | 规模 | 与规则抽取匹配度 | 获取 |
|--------|---------|------|:---:|------|
| **i2b2 2010** | `(treatment, treats, disease)` | 1.3K notes | ⭐⭐ | 需 DUA |
| **BioCreative V CDR** | `(Chemical, CID_relation, Disease)` | 1.5K abstracts | ⭐⭐⭐ (NER) | 开放 |
| **n2c2 2018** | `(Drug, dosage, frequency, reason)` | 505 notes | ⭐⭐⭐ (最接近) | 需注册 |
| **MedMentions** | UMLS entity span 标注 | 4.3K abstracts | ⭐⭐⭐ (NER) | 开放 |
| **NCBI Disease** | Disease span 标注 | 793 abstracts | ⭐⭐⭐ (NER) | 开放 |
| **DDI SemEval** | `(Drug1, interaction, Drug2)` | 1K docs | ⭐⭐ | 开放 |
| **ChemProt** | `(Chemical, relation, Protein)` | 1.8K abstracts | ⭐ | 开放 |

**核心认识**：医学 IE 数据集标注的是 `(entity, relation, entity)` 陈述性二元关系，而营养规则抽取需要的是 `(condition_set, action, value_set)` 规范性条件规则。输出格式本质不同→**无法直接评测**，但 NER 层和 Embedding 可迁移。

### 6.2 其他领域数据集（格式更接近）

| 领域 | 数据集 | 标注格式亮点 | 匹配度 | 获取 |
|------|--------|------------|:---:|------|
| **法律** 🥇 | **CUAD** | clause_type + obligation + threshold | ⭐⭐⭐⭐ | 开放 |
| | ContractEval | CUAD 的 LLM 评测基准 | ⭐⭐⭐⭐ | 开放 |
| | LEDGAR | 57 类法律条款，84K 条 | ⭐⭐⭐ | 开放 |
| **事件抽取** 🥈 | **RAMS** | event_type → argument roles (模板式) | ⭐⭐⭐ | 开放 |
| | WikiEvents | 篇章级事件参数 | ⭐⭐⭐ | 开放 |
| | ACE 2005 | 33 种事件类型 | ⭐⭐ | 需 LDC |
| **Data-to-Text(反向)** 🥉 | **WebNLG** | text↔triples pairs，可反向训练 | ⭐⭐⭐ | 开放 |
| | DART | 82K text-triple pairs | ⭐⭐⭐ | 开放 |
| | T-REx | 11M Wikidata-aligned triples | ⭐⭐ | 开放 |
| **协议/流程** | Protocols.io | 操作+条件+参数，和指南同构 | ⭐⭐⭐ | 半开放 |
| **Text-to-SQL** | Spider | text→SQL WHERE 子句 | ⭐⭐ | 开放 |
| **推理/问答** | DROP | 多文档数值推理 | ⭐⭐ | 开放 |

### 6.3 综合对比矩阵

```
                    条件逻辑  数值阈值  跨句   规范性  结构化输出  数据规模  可获得性
CUAD (法律)          █████    ████    ████  █████   ████       ███      ████
RAMS/WikiEvents      ███      ██      ████  ██      █████      ████     ████
WebNLG (反向)        ███      ███     ██    ██      █████      █████    █████
Protocols            █████    █████   ████  █████   ███        ██       ██
Spider (Text2SQL)    █████    █████   ██    ██      █████      ████     █████
i2b2/医疗IE           ██       ██      ██    ██      ███        ████     ███
```

### 6.4 推荐组合策略

```
Layer 1: 实体层 — NCBI Disease + MedMentions（医学实体）+ CUAD（条件/义务实体）
Layer 2: 条件-行动关系层 — CUAD（条件→义务）+ RAMS（事件模板）
Layer 3: 结构化编码层 — WebNLG 反转（text→triples）→ nutrition DSL
```

**首选论文视角**：「法律→营养 跨领域迁移」
- CUAD 格式和营养规则相似度最高
- 两个领域都是规范性文本
- 跨领域迁移是审稿人认可的研究范式
- 领域差距足够大，迁移失败分析也有价值

---

## 七、资源清单

| 资源 | 来源 | 状态 |
|------|------|------|
| 中国糖尿病膳食指南 2022 | 中国营养学会 | 需获取 PDF |
| ESPEN Guideline on Clinical Nutrition | ESPEN 官网 | 开放获取 |
| KDIGO CKD Nutrition Guideline | KDIGO 官网 | 开放获取 |
| UMLS / SNOMED CT 账号 | NIH 免费申请 | 需注册 |
| 3 名营养科医师标注者 | 医院合作 | 需伦理审批 |
| GPT-4 API | OpenAI | 付费 |
| CUAD 法律数据集 | GitHub | 开放 |
| BioCreative V CDR | Biocreative 官网 | 开放 |
| MedMentions | GitHub | 开放 |
| WebNLG | GitLab | 开放 |
| RAMS / WikiEvents | GitHub | 开放 |

---

## 八、后续可执行的任务清单

### 立即可做

- [ ] 获取一份带汇总表的营养指南 PDF（如 ESPEN guideline）
- [ ] 写 Python 脚本提取 PDF 表格 → pseudo-gold rules
- [ ] 用 GPT-4 从指南正文抽取规则 → 对比表格计算 recall
- [ ] 跑路径四：逻辑一致性 + 本体合规度 + 内部矛盾检测
- [ ] 下载 CUAD 数据集，跑法律→营养 zero-shot 迁移实验

### 后续

- [ ] 注册 n2c2 获取用药逻辑标注数据
- [ ] 设计营养规则 DSL schema
- [ ] 构建 LLM-assisted 标注协议
- [ ] 申请医院伦理审批（如需医师标注）
- [ ] 撰写 related work：搜 CUAD、RAMS、Protocols 方法论文

---

## 附录：使用 ScholarAIO 搜索的示例查询

```bash
cd C:\Users\56501\scholaraio

# 跨库搜索规则+医学
scholaraio explore search --name neuro-symbolic "rule extraction medical guideline"

# 搜索 ontology-constrained 方法
scholaraio explore search --name kb-construction "ontology constrained information extraction"

# 搜索 distant supervision 方法
scholaraio explore search --name relation-extract "distant supervision rule generation"

# 搜索 legal domain transfer
scholaraio explore search --name ie-foundation "legal clause extraction condition action"

# 拉取 arXiv 论文入主库
scholaraio arxiv fetch 2208.08690 --ingest  # OpenIE Survey
scholaraio arxiv fetch 2406.17224 --ingest  # LSP
scholaraio arxiv fetch 2409.07869 --ingest  # Rules from KGs guided by LMs
```
