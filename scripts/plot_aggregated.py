from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPANY_COLORS = {
    "OpenAI": "#10A37F",
    "Anthropic": "#D97757",
    "Google": "#4285F4",
    "DeepSeek": "#4D6BFE",
    "Moonshot": "#1F1F1F",
    "MiniMax": "#FF6B35",
    "Alibaba": "#FF6A00",
    "ZhipuAI": "#6F42C1",
    "xAI": "#0F0F0F",
    "ByteDance": "#00C8FF",
    "Human": "#888888",
    "Unknown": "#BBBBBB",
}

TEACHER_TO_COMPANY = {
    "GPT-5.4 Mini Evaluator": "OpenAI",
    "ClaudeHaiku4.5 Evaluator": "Anthropic",
    "DeepSeek-V3.2 Evaluator": "DeepSeek",
    "Gemini3.1FlashLite Evaluator": "Google",
    "Kimi-K2.5 Evaluator": "Moonshot",
    "MiniMax-M2.7-Highspeed Evaluator": "MiniMax",
    "Qwen3.5-Flash Evaluator": "Alibaba",
    "Grok-4.0-Fast Evaluator": "xAI",
    "Doubao-Seed-2.0-Pro Evaluator": "ByteDance",
}

CRITERIA_DE = [
    "themenbezogenheit",
    "logik-des-aufbaus",
    "ausfuehrlichkeit-aussagekraft",
    "sprachbeherrschung",
    "komplexitaet",
    "wortschatz-textverknuepfung",
    "gebrauch-sprachlicher-strukturen",
]
CRITERIA_SHORT = ["Topic", "Logic", "Express", "Lang", "Cmplx", "Vocab", "Struct"]


def plot_teacher_likeness(weights: pd.DataFrame, out: Path) -> None:
    df = weights.sort_values("rmse_vs_human", ascending=True)
    colors = [COMPANY_COLORS.get(TEACHER_TO_COMPANY.get(t, "Unknown"), "#888") for t in df["teacher_name"]]
    fig, ax = plt.subplots(figsize=(9, 4 + 0.3 * len(df)))
    bars = ax.barh(df["teacher_name"], df["rmse_vs_human"], color=colors)
    for bar, w in zip(bars, df["weight"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"w={w:.3f}", va="center", fontsize=9)
    ax.set_xlabel("RMSE vs human reference (lower = closer)")
    ax.set_title("Teacher likeness to human raters")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_weighted_vs_plain(weighted: pd.DataFrame, out: Path) -> None:
    df = weighted.sort_values("overall_score__weighted", ascending=True)
    y = np.arange(len(df))
    h = 0.4
    colors = [COMPANY_COLORS.get(c, "#888") for c in df["company"]]
    fig, ax = plt.subplots(figsize=(9, 4 + 0.3 * len(df)))
    ax.barh(y - h / 2, df["overall_score__weighted"], h, label="weighted", color=colors, edgecolor="black", linewidth=0.5)
    ax.barh(y + h / 2, df["overall_score__plain_mean"], h, label="plain mean", color=colors, alpha=0.4, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["quelle"])
    ax.set_xlabel("Overall score (0-6)")
    ax.set_title("Weighted vs plain-mean overall score by source")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_strictness(strictness: pd.DataFrame, out: Path) -> None:
    df = strictness.copy().sort_values("overall_mean", ascending=True)
    colors = [COMPANY_COLORS.get(c, "#888") for c in df["company"]]
    fig, ax = plt.subplots(figsize=(9, 4 + 0.3 * len(df)))
    ax.barh(df["teacher_name"], df["overall_mean"], color=colors,
            xerr=df["overall_std"], ecolor="black", capsize=4)
    ax.set_xlabel("Mean overall score given (lower = stricter)")
    ax.set_title("Teacher strictness ranking (error bars = std)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_self_bias(self_bias: pd.DataFrame, out: Path) -> None:
    df = self_bias.copy().sort_values("self_bias_delta", ascending=True)
    colors = ["#D62728" if v > 0 else "#2CA02C" for v in df["self_bias_delta"]]
    fig, ax = plt.subplots(figsize=(9, 4 + 0.3 * len(df)))
    ax.barh(df["teacher_name"], df["self_bias_delta"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ score (own_company − others)   |   red = teacher favors own company")
    ax.set_title("Self-bias: does each AI teacher rate its own company higher?")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_criteria_heatmap(ratings: pd.DataFrame, reference_path: Path, out: Path) -> None:
    by_source = ratings.groupby("quelle")[CRITERIA_DE].mean()
    ref_raw = json.loads(reference_path.read_text(encoding="utf-8"))
    en_to_de = {
        "topic_and_completeness": "themenbezogenheit",
        "logic_and_composition": "logik-des-aufbaus",
        "expressiveness_and_comprehensibility": "ausfuehrlichkeit-aussagekraft",
        "language_mastery": "sprachbeherrschung",
        "complexity": "komplexitaet",
        "vocabulary_and_text_linking": "wortschatz-textverknuepfung",
        "language_constructs": "gebrauch-sprachlicher-strukturen",
    }
    human_rows = {}
    for src, scores in ref_raw.items():
        human_rows[f"{src} (human)"] = [scores[en] for en in en_to_de]
    human_df = pd.DataFrame(human_rows, index=CRITERIA_DE).T
    human_df.columns = CRITERIA_DE
    combined = pd.concat([human_df, by_source])

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(combined) + 1.2))
    im = ax.imshow(combined.values, cmap="RdYlGn", vmin=2.5, vmax=5.5, aspect="auto")
    ax.set_xticks(range(len(CRITERIA_SHORT)))
    ax.set_xticklabels(CRITERIA_SHORT, rotation=30, ha="right")
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined.index)
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            ax.text(j, i, f"{combined.values[i, j]:.1f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="score")
    ax.set_title("7-criteria heatmap: human reference (top) + AI teacher mean per source")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from the aggregated rating CSVs.")
    parser.add_argument("--aggregated-dir", type=Path, default=Path("outputs/aggregated"))
    parser.add_argument("--by-teacher-dir", type=Path, default=Path("outputs/full-run-ratings-by-teacher"))
    parser.add_argument("--reference", type=Path, default=Path("config/reference_stats.json"))
    parser.add_argument("--exclude-folders", nargs="*", default=["glm-5-1-evaluator",
                                                                   "minimax-m2-7-evaluator.old-20260427",
                                                                   "qwen3-5-397b-a17b-evaluator.old-20260427"])
    args = parser.parse_args()

    out_dir = args.aggregated_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = pd.read_csv(args.aggregated_dir / "teacher_weights.csv")
    weighted = pd.read_csv(args.aggregated_dir / "weighted_scores_by_source.csv")
    strictness = pd.read_csv(args.aggregated_dir / "teacher_strictness.csv")
    self_bias = pd.read_csv(args.aggregated_dir / "self_bias.csv")

    frames = []
    for csv in sorted(args.by_teacher_dir.glob("*-evaluator/ratings_paper_aligned.csv")):
        if csv.parent.name in args.exclude_folders:
            continue
        frames.append(pd.read_csv(csv))
    ratings = pd.concat(frames, ignore_index=True)

    plot_teacher_likeness(weights, out_dir / "teacher_likeness.png")
    plot_weighted_vs_plain(weighted, out_dir / "weighted_vs_plain.png")
    plot_strictness(strictness, out_dir / "strictness.png")
    plot_self_bias(self_bias, out_dir / "self_bias.png")
    plot_criteria_heatmap(ratings, args.reference, out_dir / "criteria_heatmap.png")
    print(f"Saved 5 figures to {out_dir}")


if __name__ == "__main__":
    main()
