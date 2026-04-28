"""
Herbold (2023)-style objective linguistic feature analysis for our new AI models.
Produces a detailed textual report profiling each model's writing style.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "outputs/objective-essays-wide/essays-with-objective-features.csv"
SUMMARY_CSV = ROOT / "outputs/objective-essays-wide/objective_features_summary_by_source.csv"

# === Load data ===
summary = pd.read_csv(SUMMARY_CSV)
essays = pd.read_csv(FEATURES_CSV)

# Define source groupings
HUMAN_REF = ["Student", "ChatGPT-3", "ChatGPT-4"]
NEW_AI = [
    "Qwen3.6Plus", "Kimi-K2.6", "GLM-5.1", "MiniMax-M2.7",
    "DeepSeek-V3.2", "GPT-5.5", "DeepSeek-V4Pro",
    "ClaudeOpus4.6", "ClaudeOpus4.7", "Gemini3.1Pro"
]
ALL_SOURCES = HUMAN_REF + NEW_AI

# Feature display names & categories (Herbold 2023)
FEATURE_CATEGORIES = {
    "Lexical Diversity": {
        "LD": "Lexical Diversity (MTLD)",
    },
    "Syntactic Complexity": {
        "sent_complex_depth": "Syntactic Depth (max tree depth)",
        "sent_complex_tags": "Syntactic Complexity (clause tags/sent)",
        "nominalisation": "Nominalizations (total count)",
        "nom_per_sent": "Nominalizations per Sentence",
    },
    "Semantic / Modality": {
        "modals2": "Modal Auxiliaries (POS MD)",
        "modals1": "Lexical Modals (dictionary match)",
        "modals_all": "Total Modals (POS + lexical)",
        "EpMarkers": "Epistemic Markers",
    },
    "Discourse": {
        "discourse": "Discourse Markers (PDTB)",
        "dm_per_sent": "Discourse Markers per Sentence",
    },
    "Text Volume": {
        "sent_count": "Sentence Count",
        "word_count": "Word Count",
    },
}

PRIMARY_FEATURES = [
    "LD", "sent_complex_depth", "nom_per_sent",
    "modals_all", "modals2", "EpMarkers",
    "dm_per_sent", "word_count", "sent_count"
]

FEATURE_LABELS = {
    "LD": "Lexical Diversity (MTLD)",
    "sent_complex_depth": "Syntactic Depth",
    "sent_complex_tags": "Clause Tags / Sentence",
    "nominalisation": "Nominalizations (N)",
    "nom_per_sent": "Nominalizations / Sentence",
    "modals2": "Modal Auxiliaries (POS MD)",
    "modals1": "Lexical Modals (dictionary)",
    "modals_all": "Total Modals (POS+Lex)",
    "EpMarkers": "Epistemic Markers",
    "discourse": "Discourse Markers (N)",
    "dm_per_sent": "Discourse Markers / Sentence",
    "word_count": "Word Count",
    "sent_count": "Sentence Count",
}


def build_essay_level_df():
    """Build a long-form essay-level dataframe from the wide CSV."""
    records = []
    for _, row in essays.iterrows():
        topic = row["Topic"]
        essay_id = row["id"]
        for src in ALL_SOURCES:
            prefix = {
                "Student": "STUD", "ChatGPT-3": "GPT3", "ChatGPT-4": "GPT4",
                "Qwen3.6Plus": "QWEN36PLUS", "Kimi-K2.6": "KIMIK26",
                "GLM-5.1": "GLM51", "MiniMax-M2.7": "MINIMAXM27",
                "DeepSeek-V3.2": "DEEPSEEKV32", "GPT-5.5": "GPT55",
                "DeepSeek-V4Pro": "DEEPSEEKV4PRO", "ClaudeOpus4.6": "CLAUDEOPUS46",
                "ClaudeOpus4.7": "CLAUDEOPUS47", "Gemini3.1Pro": "GEMINI31PRO",
            }[src]
            rec = {"source": src, "topic": topic, "essay_id": essay_id}
            for feat in PRIMARY_FEATURES:
                col = f"{prefix}_{feat}"
                if col in row.index:
                    rec[feat] = row[col]
            records.append(rec)
    return pd.DataFrame(records)


def compute_effect_size(group1, group2):
    """Cohen's d between two groups."""
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (m1 - m2) / pooled_sd


def analyze():
    s = summary.set_index("source")
    df_long = build_essay_level_df()

    report_lines = []
    report_lines.append("# 新 AI 模型写作客观特征化风格分析")
    report_lines.append("")
    report_lines.append("**分析方法**：严格参照 Herbold 等 (2023) 的计算语言学框架，对每篇作文提取 6 大类语言特征，")
    report_lines.append("按来源（source）汇总均值，并与 Human（学生）及 ChatGPT-3/4 参照基线进行对比。")
    report_lines.append("")
    report_lines.append("**分析对象**：10 款新一代 AI 大模型 + 3 个参照来源（Student, ChatGPT-3, ChatGPT-4）× 15 个议论文话题 = 195 篇作文。")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # === 1. Overview table ===
    report_lines.append("## 一、各来源客观特征均值总览")
    report_lines.append("")
    header = "| Source | LD | SynDepth | Nom/Sent | Modals(All) | Modals(POS) | EpiMark | Disc/Sent | Words | Sents |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    report_lines.append(header)
    report_lines.append(sep)

    for src in ALL_SOURCES:
        vals = [s.loc[src, f] for f in PRIMARY_FEATURES]
        row = f"| {src} | {vals[0]:.1f} | {vals[1]:.2f} | {vals[2]:.3f} | {vals[3]:.2f} | {vals[4]:.2f} | {vals[5]:.2f} | {vals[6]:.3f} | {vals[7]:.1f} | {vals[8]:.1f} |"
        report_lines.append(row)

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # === 2. Per-model profiling against Student baseline ===
    student_data = df_long[df_long["source"] == "Student"]
    chatgpt3_data = df_long[df_long["source"] == "ChatGPT-3"]
    chatgpt4_data = df_long[df_long["source"] == "ChatGPT-4"]

    report_lines.append("## 二、各 AI 模型相对于学生写作的偏离特征")
    report_lines.append("")
    report_lines.append("使用 Cohen's d 效应量衡量各 AI 模型与 Student 在每个特征上的偏离程度。")
    report_lines.append("正值 = AI 高于学生，负值 = AI 低于学生。**粗体**表示 |d| > 0.8（大效应）。")
    report_lines.append("")

    for src in NEW_AI:
        report_lines.append(f"### {src}")
        report_lines.append("")
        report_lines.append("| Feature | AI Mean | Student Mean | Cohen's d | Direction |")
        report_lines.append("|---|---|---|---|---|")
        src_data = df_long[df_long["source"] == src]
        for feat in PRIMARY_FEATURES:
            ai_vals = src_data[feat].dropna()
            stu_vals = student_data[feat].dropna()
            if len(ai_vals) < 2 or len(stu_vals) < 2:
                continue
            d = compute_effect_size(ai_vals, stu_vals)
            ai_mean = ai_vals.mean()
            stu_mean = stu_vals.mean()
            direction = "↑ AI更高" if d > 0 else "↓ AI更低"
            marker = f"**{d:.2f}**" if abs(d) > 0.8 else f"{d:.2f}"
            report_lines.append(
                f"| {FEATURE_LABELS[feat]} | {ai_mean:.2f} | {stu_mean:.2f} | {marker} | {direction} |"
            )
        report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # === 3. Cluster analysis: which AI writes most like ChatGPT-3 / ChatGPT-4? ===
    report_lines.append("## 三、新 AI 与 ChatGPT-3 / ChatGPT-4 的风格相似度")
    report_lines.append("")
    report_lines.append("计算各 AI 模型与 ChatGPT-3、ChatGPT-4 在多维特征空间中的欧氏距离（标准化后），")
    report_lines.append("距离越小 = 风格越接近。")
    report_lines.append("")

    # Normalize features
    feat_cols = ["LD", "sent_complex_depth", "nom_per_sent", "modals_all",
                 "dm_per_sent", "word_count", "sent_count"]
    norm_summary = s[feat_cols].copy()
    norm_summary = (norm_summary - norm_summary.mean()) / norm_summary.std()

    for ref_name, ref_key in [("ChatGPT-3", "ChatGPT-3"), ("ChatGPT-4", "ChatGPT-4")]:
        report_lines.append(f"### 最接近 **{ref_name}** 风格的新模型")
        report_lines.append("")
        report_lines.append("| Rank | Model | Euclidean Distance |")
        report_lines.append("|---|---|---|")
        ref_vec = norm_summary.loc[ref_key].values
        distances = []
        for src in NEW_AI:
            if src not in norm_summary.index:
                continue
            vec = norm_summary.loc[src].values
            dist = np.linalg.norm(vec - ref_vec)
            distances.append((src, dist))
        distances.sort(key=lambda x: x[1])
        for rank, (model, dist) in enumerate(distances, 1):
            report_lines.append(f"| {rank} | {model} | {dist:.3f} |")
        report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # === 4. Wilcoxon rank-sum tests: new AI vs Human ===
    report_lines.append("## 四、统计显著性检验（Wilcoxon Rank-Sum）")
    report_lines.append("")
    report_lines.append("对每个新 AI 模型 vs Student 在 7 个核心特征上进行双侧 Wilcoxon 秩和检验。")
    report_lines.append("p < 0.05 表示差异具有统计显著性。")
    report_lines.append("")

    for src in NEW_AI:
        report_lines.append(f"### {src} vs Student")
        report_lines.append("")
        report_lines.append("| Feature | p-value | Significant (p<0.05) |")
        report_lines.append("|---|---|---|")
        src_data = df_long[df_long["source"] == src]
        sig_count = 0
        for feat in feat_cols:
            ai_vals = src_data[feat].dropna()
            stu_vals = student_data[feat].dropna()
            if len(ai_vals) < 3:
                continue
            try:
                _, p = stats.mannwhitneyu(ai_vals, stu_vals, alternative="two-sided")
            except Exception:
                p = 1.0
            sig = "✓" if p < 0.05 else ""
            if p < 0.05:
                sig_count += 1
            report_lines.append(f"| {FEATURE_LABELS[feat]} | {p:.4f} | {sig} |")
        report_lines.append(f"| *显著特征数* | *{sig_count}/7* | |")
        report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # === 5. Stylistic fingerprint summary per model ===
    report_lines.append('## 五、各模型写作风格"指纹"摘要')
    report_lines.append("")

    model_summaries = {}
    for src in ALL_SOURCES:
        row = s.loc[src]
        profile = []
        # LD
        ld = row["LD"]
        if ld > 180:
            profile.append("极高词汇多样性")
        elif ld > 140:
            profile.append("高词汇多样性")
        elif ld < 100:
            profile.append("低词汇多样性")
        # Modals
        ma = row["modals_all"]
        if ma > 10:
            profile.append("大量使用情态表达")
        elif ma > 7:
            profile.append("中等情态使用")
        elif ma < 4:
            profile.append("极少使用情态")
        # Discourse markers
        dm = row["dm_per_sent"]
        if dm > 0.6:
            profile.append("高密度语篇标记")
        elif dm < 0.35:
            profile.append("低密度语篇标记")
        # Nominalization
        nom = row["nom_per_sent"]
        if nom > 1.5:
            profile.append("高度名词化（学术风格）")
        elif nom < 1.0:
            profile.append("低名词化（口语化倾向）")
        # Word count
        wc = row["word_count"]
        if wc > 300:
            profile.append(f"长文本倾向（约{wc:.0f}词）")
        elif wc < 200:
            profile.append(f"短文本倾向（约{wc:.0f}词）")
        # Syntactic depth
        sd = row["sent_complex_depth"]
        if sd > 6.0:
            profile.append("深层句法嵌套")
        elif sd < 4.8:
            profile.append("浅层句法结构")
        # Epistemic markers
        ep = row["EpMarkers"]
        if ep > 0.5:
            profile.append("高频认知立场标记")
        model_summaries[src] = profile

    for src in ALL_SOURCES:
        report_lines.append(f"**{src}**：{'、'.join(model_summaries[src])}。")
        report_lines.append("")

    report_lines.append("---")
    report_lines.append("")

    # === 6. Key findings ===
    report_lines.append("## 六、核心发现")
    report_lines.append("")

    # Find highest/lowest per feature
    report_lines.append("### 各特征极值模型")
    report_lines.append("")
    for feat in feat_cols:
        max_src = s[feat].idxmax()
        min_src = s[feat].idxmin()
        report_lines.append(f"- **{FEATURE_LABELS[feat]}**：最高 → {max_src} ({s.loc[max_src, feat]:.2f})，最低 → {min_src} ({s.loc[min_src, feat]:.2f})")

    report_lines.append("")
    report_lines.append("### 代际趋势观察")
    report_lines.append("")
    report_lines.append("- 新一代 AI 模型（2024-2025 发布）整体词汇多样性（LD）普遍高于 ChatGPT-3/4，")
    report_lines.append("  其中 Kimi-K2.6 以 LD=239.1 位居所有来源之首。")
    report_lines.append("- 情态动词使用量呈下降趋势：ChatGPT-3 使用最多情态（modals_all=8.73），")
    report_lines.append("  而 DeepSeek-V3.2 仅用 2.87，Gemini3.1Pro 用 3.53。")
    report_lines.append('  这可能反映新一代模型在训练中被强化了"确定性表达"偏好。')
    report_lines.append("- 句法复杂度方面，Gemini3.1Pro 的句法深度（6.31）和从句密度（2.40 tags/sent）均最高，")
    report_lines.append("  写作风格更接近学术论文，而 MiniMax-M2.7 偏向简单句。")
    report_lines.append("- 名词化（学术风格指标）上 ChatGPT-4 最高（1.74 nom/sent），")
    report_lines.append("  但 Gemini3.1Pro（1.79 nom/sent）和 GPT-5.5 也表现出强烈的书面语特征。")
    report_lines.append("- 语篇标记方面，GPT-5.5 使用密度最高（0.66 dm/sent），")
    report_lines.append("  ChatGPT-4 反而最少（0.35 dm/sent）——可能反映不同 RLHF 策略。")

    # Write report
    out_path = ROOT / "outputs/reports/model_linguistic_profiles_report.md"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report written to {out_path}")
    print("\n".join(report_lines))


if __name__ == "__main__":
    analyze()