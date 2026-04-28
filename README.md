# LLM_writing_evaluation — Methodology & Replication Package

基于 **Herbold 等 (2023)** ——《A large-scale comparison of human-written versus ChatGPT-generated essays》（*Scientific Reports*, DOI: 10.1038/s41598-023-45644-9）的实验设计复现与扩展，比较多个现代大语言模型（LLM）生成作文与高中生作文。

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.10+-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.5+-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-3776AB?logo=python&logoColor=white)](https://www.nltk.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?logo=python&logoColor=white)](https://matplotlib.org/)
[![pypdf](https://img.shields.io/badge/pypdf-3.0+-F37626?logo=python&logoColor=white)](https://pypdf.readthedocs.io/)

## 概述

本仓库包含完整的研究方法与可复现流水线：

1. **作文生成** —— 多个 LLM 对 15 个代表性主题（从 Herbold 等原始数据集分层采样）生成议论文
2. **AI 教师评分** —— 多个 LLM 评估器使用与 Herbold 等一致的 7 项评分细则进行盲评
3. **客观语言学特征提取** —— 自动计算词汇、句法、语篇层面的语言学特征
4. **评分聚合与主观-客观相关性分析** —— 跨评估器汇总评分，计算校正权重，分析主观评分与客观语言特征的关联
5. **可视化与分析** —— 生成与论文对齐的研究图表（排名、热力图、一致性分析等）

## 环境要求

- Python 3.10+
- 安装依赖：`pip install -r requirements.txt`
- spaCy 英文模型：`python -m spacy download en_core_web_sm`

## 仓库结构

```
eduandai-methodology/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── rubric.json                          # 7 项评分细则（0–6 分制，0.5 步长）
│   ├── representative_topics.json           # 15 个主题 + 学生范文 + ChatGPT-3/4 参考作文
│   ├── reference_stats.json                 # Herbold 等 (2023) 基线分数
│   ├── external_generation_prompts.md       # 外部/第三方 LLM 生成调用的提示词格式
│   ├── model_providers.example.json         # 模板 — 重命名为 model_providers.json
│   ├── model_providers.mock.json            # 模拟模式（无需真实 API）
│   ├── teacher_evaluators.example.json      # 模板 — 重命名为 teacher_evaluators.json
│   └── teacher_evaluators.mock.json         # 模拟模式（无需真实 API）
├── essay_benchmark/
│   ├── __init__.py
│   ├── study.py                             # 核心研究配置与资源加载
│   ├── grading.py                           # 作文评分逻辑（基于评分细则）
│   ├── openai_compatible.py                 # OpenAI 兼容 API 客户端
│   └── text_utils.py                        # 文本处理工具
├── scripts/
│   ├── generate_essays.py                   # 批量生成作文
│   ├── batch_grade_essays.py                # 批量 AI 教师评分
│   ├── calc_objective_features.py           # 客观语言学特征提取
│   ├── aggregate_ratings.py                 # 跨评估器评分聚合
│   ├── analyze_subjective_objective.py      # 主观-客观相关性分析
│   ├── extract_study_assets.py              # 从原始数据集提取题目与基线分数
│   ├── plot_research_results.py             # 生成研究图表（排名、热力图、一致性等）
│   ├── plot_aggregated.py                   # 聚合数据可视化
│   ├── render_objective_heatmap.py          # 客观特征热力图
│   └── render_objective_table.py            # 客观特征表格
└── outputs/                                 # 不会上传到本仓库，2026年4月数据集的下载方式见下文
    ├── essays-wide.csv                      # 完整作文数据集（宽表格式）
    ├── ratings_paper_aligned.csv            # 9 位 AI 教师的汇总评分
    ├── ratings_summary_by_source.csv        # 按来源汇总的评分统计
    ├── aggregated/                          # 评分聚合结果
    │   ├── weighted_scores_by_source.csv
    │   ├── teacher_weights.csv
    │   ├── teacher_agreement.csv
    │   ├── teacher_strictness.csv
    │   ├── self_bias.csv
    │   └── within_company_compare.csv
    ├── subjective-objective/                # 主观-客观相关性分析
    │   ├── subjective_objective_by_source.csv
    │   ├── subjective_objective_by_essay.csv
    │   ├── correlations_source_weighted_overall.csv
    │   ├── correlations_source_plain_overall.csv
    │   ├── correlations_essay_plain_overall.csv
    │   └── subjective_scores_by_essay.csv
    ├── objective-features/                  # 客观语言学特征
    │   ├── essays-with-objective-features.csv
    │   ├── objective_features_long.csv
    │   └── objective_features_summary_by_source.csv
    ├── ratings/                             # 各评估器的独立评分
    └── figures/                             # 预生成的研究图表
```

[点击此处可下载2026年4月部分模型的测试数据集](https://cloud.tsinghua.edu.cn/d/e4591ca9610a469cacdc/)

## 研究方法

### 1. 作文生成

**题目**：从 Herbold 等 (2023) 的 90 道主题中，按学生分数分布分层采样 15 个主题，涵盖教育、科技、文化、社会、健康、媒体等多个领域。

**提示词**（零样本）：

- **System Prompt**: `"You are a student writing an English argumentative essay for an upper-secondary writing task. Write directly and do not mention that you are an AI. Do not include a title, preface, bullet list, markdown, citations, notes, or explanations. Produce only the essay body."`

- **User Prompt**: `"Write an essay with about 200 words on \"{topic}\"."`

**生成模型**：可使用Open AI兼容的api接口接入大模型进行作文生成，每个模型生成 14 篇约 200 词的英文议论文。

对于不通过本流水线 API 调用的外部模型，使用 `config/external_generation_prompts.md` 中的标准化提示词格式手动生成，再合并回数据集。

### 2. AI 教师评分

**评分细则**：与 Herbold 等 (2023) 对齐的 7 项标准（0–6 分制，0.5 分步长）：

| 编号 | 英文名称 | 德文字段名（Herbold 对齐） |
|---|---|---|
| 1 | Topic and completeness | themenbezogenheit |
| 2 | Logic and composition | logik-des-aufbaus |
| 3 | Expressiveness and comprehensibility | ausfuehrlichkeit-aussagekraft |
| 4 | Language mastery | sprachbeherrschung |
| 5 | Complexity | komplexitaet |
| 6 | Vocabulary and text linking | wortschatz-textverknuepfung |
| 7 | Language constructs | gebrauch-sprachlicher-strukturen |

**评估方式**：每个 LLM 评估器（"教师"代理）独立评分，`temperature=0.2`。评估器仅看到作文原文（盲评），不知道来源是哪个模型/学生。输出格式为严格 JSON，含 7 项分值与 1 项综合评分。

**评分流程**：按题目批量调用——每题一次 API 请求包含该题目的所有待评分作文源，避免逐篇调用的位置偏差。支持断点续评，去重键为 `(thema, quelle, teacher_name)`。

### 3. 客观语言学特征

对每篇作文自动计算以下特征（扩展了 Herbold 等的特征集）：

| 类别 | 特征 |
|---|---|
| 词汇 | 词数、句数、平均句长、词汇多样性（MTLD） |
| 句法 | 词性标注分布、句法树深度、从句密度、名词化（每句） |
| 语篇 | 话语标记、连接词、情态动词频率（全部模态、epistemic markers、模態 type 2） |
| 可读性 | Flesch-Kincaid、Gunning Fog 可读性指标 |

### 4. 评分聚合与校正

**教师权重校准**：以 Herbold 等 (2023) 原始数据集中的学生作文真人评分为基线，计算每位 AI 教师的 RMSE（均方根误差），以其倒数作为聚合权重。与人类评分一致性越高的教师获得越高权重。

**校正总分**：`overall_score_weighted` = 各教师 overall 评分 × weight 的加权平均。

### 5. 主观-客观相关性分析

在 essay 级别和 source 级别计算各客观语言特征与 AI 教师主观评分的 Pearson 相关系数，识别哪些客观特征最能解释模型间的评分差异。

### 6. 质量控制

- 所有评分调用强制验证 JSON 格式，缺失字段触发自动重试
- 每次运行留存完整错误日志
- 出力目录追加式写入，重复运行不覆盖已有结果
- **模拟模式**（`mock.json`）：基于词长/词汇的启发式评分，无需 API 调用即可端到端验证流水线

## 设置与使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. 提取原始研究数据资产

```bash
python scripts/extract_study_assets.py
```

从 `sherbold-chatgpt-student-essay-study-3f09052/` 中重建 `config/representative_topics.json` 和 `config/reference_stats.json`。

### 3. 配置模型提供商

```bash
cp config/model_providers.example.json config/model_providers.json
```

编辑 `model_providers.json`，为各模型填写 `base_url`、`model`、`api_key_env`，并通过环境变量设置对应的 API Key。

### 4. 配置教师评估器

```bash
cp config/teacher_evaluators.example.json config/teacher_evaluators.json
```

编辑 `teacher_evaluators.json`，为各评估器填写 `base_url`、`model`、`api_key_env`。

### 5. 生成作文

```bash
python scripts/generate_essays.py --providers config/model_providers.json
```

### 6. 评分作文

```bash
python scripts/batch_grade_essays.py \
  --dataset outputs/essays-wide.csv \
  --teachers config/teacher_evaluators.json \
  --output-dir outputs/ratings/<teacher-folder> \
  --include-student
```

### 7. 提取客观特征

```bash
python scripts/calc_objective_features.py
```

### 8. 聚合评分

```bash
python scripts/aggregate_ratings.py
```

### 9. 主观-客观相关性分析

```bash
python scripts/analyze_subjective_objective.py
```

### 10. 生成研究图表

```bash
python scripts/plot_research_results.py
```

## 模拟模式验证

无需 API 调用即可端到端测试流水线：

```bash
python scripts/batch_grade_essays.py \
  --dataset outputs/essays-wide.csv \
  --teachers config/teacher_evaluators.mock.json \
  --output-dir /tmp/mock-test --include-student
```

`mock.json` 中的 `base_url: "mock"` 会触发 `essay_benchmark/grading.py::mock_grade`，基于词长与词汇启发式返回模拟分数。

## 与原论文的主要差异（2026年4月）

[数据在此处（同上）](https://cloud.tsinghua.edu.cn/d/e4591ca9610a469cacdc/)

| 维度 | Herbold 等 (2023) | 本研究 |
|---|---|---|
| 比较模型 | GPT-3.5 / GPT-4 | 多个 2024–2025 世代 LLM（9+ 模型） |
| 评估者 | 108 位真人教师 | LLM 教师代理（9 位 AI 评估器） |
| 主题数 | 90 | 15（代表性子集，按难度分层） |
| 语言学特征 | 7 项（6 类） | 扩展至含可读性与 POS 指标 |
| 教师校准 | 无 | 基于人类评分的 RMSE 权重校准 |

## 参考文献

Herbold, S., Hautli-Janisz, A., Heuer, U., Kikteva, Z., & Trautsch, A. (2023). A large-scale comparison of human-written versus ChatGPT-generated essays. *Scientific Reports*, 13, 18617. https://doi.org/10.1038/s41598-023-45644-9

原始复现包：[sherbold/chatgpt-student-essay-study](https://github.com/sherbold/chatgpt-student-essay-study)

## 许可

仅供研究与教育用途。

## 📜 鸣谢

本项目为\
**北京大学 2026 年春季《教育与人工智能》课程**\
作业成果，感谢课程主讲教师**贾积有教授**在研究选题、分析方法与平台设计上的悉心指导。

作者 · [**李涛**](https://github.com/Lee-Eee-Eee) · 清华大学工程物理系

---

<div align="center">

若本项目对你的研究或课程有帮助，欢迎 ⭐ Star 支持。
问题与建议请通过 [Issues](https://github.com/Lee-Eee-Eee/EduAnalytics/issues) 反馈。

</div>
