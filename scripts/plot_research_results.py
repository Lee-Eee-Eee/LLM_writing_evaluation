from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CRITERIA = [
    "themenbezogenheit",
    "logik-des-aufbaus",
    "ausfuehrlichkeit-aussagekraft",
    "sprachbeherrschung",
    "komplexitaet",
    "wortschatz-textverknuepfung",
    "gebrauch-sprachlicher-strukturen",
]

CRITERIA_LABELS = [
    "Topic",
    "Logic",
    "Express",
    "Language",
    "Complexity",
    "Vocabulary",
    "Structures",
]

OBJECTIVE_LABELS = {
    "LD": "Lexical diversity",
    "word_count": "Word count",
    "sent_count": "Sentence count",
    "modals_all": "All modals",
    "modals2": "Modal type 2",
    "EpMarkers": "Epistemic markers",
    "nom_per_sent": "Nominalisation/sentence",
    "mod_per_sent": "Modals/sentence",
    "discourse": "Discourse markers",
}

COMPANY_COLORS = {
    "OpenAI": "#10A37F",
    "Anthropic": "#C15F3C",
    "Google": "#4285F4",
    "DeepSeek": "#4D6BFE",
    "Moonshot": "#202020",
    "MiniMax": "#FF6B35",
    "Alibaba": "#FF8A00",
    "ZhipuAI": "#6F42C1",
    "Human": "#777777",
    "ByteDance": "#00A7D8",
    "xAI": "#111111",
    "Unknown": "#BBBBBB",
}


def load_ratings(by_teacher_dir: Path, exclude_folders: set[str]) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(by_teacher_dir.glob("*/ratings_paper_aligned.csv")):
        if csv_path.parent.name in exclude_folders:
            continue
        df = pd.read_csv(csv_path)
        df["teacher_folder"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No ratings_paper_aligned.csv found under {by_teacher_dir}")
    return pd.concat(frames, ignore_index=True)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_corrected_ranking(weighted: pd.DataFrame, out: Path) -> None:
    df = weighted.sort_values("overall_score__weighted", ascending=True).copy()
    colors = [COMPANY_COLORS.get(c, "#888888") for c in df["company"]]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(df["quelle"], df["overall_score__weighted"], color=colors, alpha=0.92)
    ax.scatter(df["overall_score__plain_mean"], df["quelle"], color="black", s=22, zorder=3, label="plain mean")
    for _, row in df.iterrows():
        ax.text(row["overall_score__weighted"] + 0.025, row["quelle"], f"{row['overall_score__weighted']:.2f}",
                va="center", fontsize=8.5)
    ax.set_xlim(3.6, 5.6)
    ax.set_xlabel("Corrected overall score (weighted by teacher-human RMSE)")
    ax.set_title("Corrected essay-source ranking")
    ax.legend(loc="lower right", frameon=False)
    save(fig, out)


def plot_criteria_heatmap(weighted: pd.DataFrame, out: Path) -> None:
    df = weighted.sort_values("overall_score__weighted", ascending=False).copy()
    matrix = df[[f"{c}__weighted" for c in CRITERIA]].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=3.8, vmax=5.6, aspect="auto")
    ax.set_xticks(np.arange(len(CRITERIA_LABELS)))
    ax.set_xticklabels(CRITERIA_LABELS, rotation=30, ha="right")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df["quelle"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    fig.colorbar(im, ax=ax, label="weighted score")
    ax.set_title("Corrected 7-criteria profile by essay source")
    save(fig, out)


def plot_teacher_calibration(weights: pd.DataFrame, out: Path) -> None:
    df = weights.sort_values("rmse_vs_human", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(df["teacher_name"], df["weight"], color="#4C78A8")
    ax2 = ax.twiny()
    ax2.plot(df["rmse_vs_human"], df["teacher_name"], color="#D62728", marker="o", linewidth=1.8)
    ax.set_xlabel("Aggregation weight")
    ax2.set_xlabel("RMSE vs human reference (lower is better)", color="#D62728")
    ax2.tick_params(axis="x", colors="#D62728")
    ax.invert_yaxis()
    for bar, rmse in zip(bars, df["rmse_vs_human"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}",
                va="center", fontsize=8)
    ax.set_title("Teacher calibration: weights derived from human-reference error")
    save(fig, out)


def plot_teacher_agreement(agreement: pd.DataFrame, out: Path) -> None:
    summary = (
        agreement.groupby("quelle")
        .agg(mean_std=("std", "mean"), max_range=("range", "max"), mean_score=("mean", "mean"))
        .reset_index()
        .sort_values("mean_std", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(summary["quelle"], summary["mean_std"], color="#72B7B2")
    ax.scatter(summary["max_range"], summary["quelle"], color="#E45756", s=28, label="max teacher range")
    ax.set_xlabel("Teacher disagreement on overall score")
    ax.set_title("Where AI teachers agree or diverge most")
    ax.legend(frameon=False)
    save(fig, out)


def plot_score_distribution(ratings: pd.DataFrame, weighted: pd.DataFrame, out: Path) -> None:
    order = weighted.sort_values("overall_score__weighted", ascending=False)["quelle"].tolist()
    data = [ratings.loc[ratings["quelle"] == source, "overall_score"].dropna().to_numpy() for source in order]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    box = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    company = weighted.set_index("quelle")["company"].to_dict()
    for patch, source in zip(box["boxes"], order):
        patch.set_facecolor(COMPANY_COLORS.get(company.get(source, "Unknown"), "#BBBBBB"))
        patch.set_alpha(0.72)
    ax.set_ylabel("Raw teacher overall score")
    ax.set_title("Distribution of teacher ratings by essay source")
    ax.tick_params(axis="x", rotation=35)
    save(fig, out)


def plot_correlation_bars(source_corr: pd.DataFrame, essay_corr: pd.DataFrame, out: Path) -> None:
    source = source_corr.head(10).copy()
    essay = essay_corr.head(10).copy()
    source["label"] = source["objective_feature"].map(OBJECTIVE_LABELS).fillna(source["objective_feature"])
    essay["label"] = essay["objective_feature"].map(OBJECTIVE_LABELS).fillna(essay["objective_feature"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharex=True)
    for ax, df, title in [
        (axes[0], source.sort_values("pearson_r"), "Source-level correlations"),
        (axes[1], essay.sort_values("pearson_r"), "Essay-level correlations"),
    ]:
        colors = ["#D62728" if v < 0 else "#2CA02C" for v in df["pearson_r"]]
        ax.barh(df["label"], df["pearson_r"], color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Pearson r with overall score")
        ax.set_xlim(-1.0, 1.0)
    save(fig, out)


def plot_objective_scatter(source_join: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, x_col, title in [
        (axes[0], "LD", "Lexical diversity vs corrected score"),
        (axes[1], "modals_all", "All modals vs corrected score"),
    ]:
        for _, row in source_join.dropna(subset=[x_col, "overall_score__weighted"]).iterrows():
            ax.scatter(row[x_col], row["overall_score__weighted"], s=54, color="#4C78A8")
            ax.text(row[x_col], row["overall_score__weighted"] + 0.018, str(row["quelle"]), fontsize=7.2, ha="center")
        pair = source_join[[x_col, "overall_score__weighted"]].dropna()
        if pair.shape[0] >= 2:
            z = np.polyfit(pair[x_col], pair["overall_score__weighted"], 1)
            xs = np.linspace(pair[x_col].min(), pair[x_col].max(), 100)
            ax.plot(xs, z[0] * xs + z[1], color="#E45756", linewidth=1.4)
        ax.set_xlabel(OBJECTIVE_LABELS.get(x_col, x_col))
        ax.set_ylabel("Corrected overall score")
        ax.set_title(title)
    save(fig, out)


def write_figure_index(out_dir: Path) -> None:
    rows = [
        ("corrected_overall_ranking.png", "校正总分排名；柱为加权校正分，黑点为未加权平均分。"),
        ("corrected_criteria_heatmap.png", "各来源在 7 个评分维度上的校正分布。"),
        ("teacher_calibration_weights.png", "AI 老师相对原论文人类评分的 RMSE 与聚合权重。"),
        ("teacher_agreement_by_source.png", "不同老师对同一来源评分的一致性/分歧度。"),
        ("rating_distribution_by_source.png", "各来源在 9 个老师评分下的原始 overall 分布。"),
        ("subjective_objective_correlations.png", "主观总分与客观语言特征的相关性，分 source-level 与 essay-level。"),
        ("objective_scatter_ld_modals.png", "关键客观特征与校正总分的散点关系。"),
    ]
    pd.DataFrame(rows, columns=["figure", "purpose"]).to_csv(out_dir / "figure_index.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create research-oriented visualizations for the essay benchmark.")
    parser.add_argument("--aggregated-dir", type=Path, default=Path("outputs/aggregated"))
    parser.add_argument("--subjective-objective-dir", type=Path, default=Path("outputs/subjective-objective"))
    parser.add_argument("--by-teacher-dir", type=Path, default=Path("outputs/full-run-ratings-by-teacher"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research-figures"))
    parser.add_argument(
        "--exclude-folders",
        nargs="*",
        default=[
            "glm-5-1-evaluator",
            "glm-5-1-evaluator.disabled",
            "minimax-m2-7-evaluator.old-20260427",
            "qwen3-5-397b-a17b-evaluator.old-20260427",
            "smoke-gpt-batch",
        ],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weighted = pd.read_csv(args.aggregated_dir / "weighted_scores_by_source.csv")
    weights = pd.read_csv(args.aggregated_dir / "teacher_weights.csv")
    agreement = pd.read_csv(args.aggregated_dir / "teacher_agreement.csv")
    source_join = pd.read_csv(args.subjective_objective_dir / "subjective_objective_by_source.csv")
    source_corr = pd.read_csv(args.subjective_objective_dir / "correlations_source_weighted_overall.csv")
    essay_corr = pd.read_csv(args.subjective_objective_dir / "correlations_essay_plain_overall.csv")
    ratings = load_ratings(args.by_teacher_dir, set(args.exclude_folders))

    plot_corrected_ranking(weighted, args.output_dir / "corrected_overall_ranking.png")
    plot_criteria_heatmap(weighted, args.output_dir / "corrected_criteria_heatmap.png")
    plot_teacher_calibration(weights, args.output_dir / "teacher_calibration_weights.png")
    plot_teacher_agreement(agreement, args.output_dir / "teacher_agreement_by_source.png")
    plot_score_distribution(ratings, weighted, args.output_dir / "rating_distribution_by_source.png")
    plot_correlation_bars(source_corr, essay_corr, args.output_dir / "subjective_objective_correlations.png")
    plot_objective_scatter(source_join, args.output_dir / "objective_scatter_ld_modals.png")
    write_figure_index(args.output_dir)
    print(f"Saved 7 research figures and figure_index.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
