from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CRITERIA_DE_TO_EN = {
    "themenbezogenheit": "topic_and_completeness",
    "logik-des-aufbaus": "logic_and_composition",
    "ausfuehrlichkeit-aussagekraft": "expressiveness_and_comprehensibility",
    "sprachbeherrschung": "language_mastery",
    "komplexitaet": "complexity",
    "wortschatz-textverknuepfung": "vocabulary_and_text_linking",
    "gebrauch-sprachlicher-strukturen": "language_constructs",
}
CRITERIA_DE = list(CRITERIA_DE_TO_EN.keys())
CRITERIA_EN = list(CRITERIA_DE_TO_EN.values())
SCORE_COLS = CRITERIA_DE + ["overall_score"]

HUMAN_REFERENCE_SOURCES = ["Student", "ChatGPT-3", "ChatGPT-4"]

SOURCE_TO_COMPANY = {
    "Student": "Human",
    "ChatGPT-3": "OpenAI",
    "ChatGPT-4": "OpenAI",
    "GPT-5.5": "OpenAI",
    "ClaudeOpus4.6": "Anthropic",
    "ClaudeOpus4.7": "Anthropic",
    "DeepSeek-V3.2": "DeepSeek",
    "DeepSeek-V4Pro": "DeepSeek",
    "Gemini3.1Pro": "Google",
    "Kimi-K2.6": "Moonshot",
    "MiniMax-M2.7": "MiniMax",
    "Qwen3.6Plus": "Alibaba",
    "GLM-5.1": "ZhipuAI",
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


def load_all_ratings(by_teacher_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(by_teacher_dir.glob("*-evaluator/ratings_paper_aligned.csv")):
        df = pd.read_csv(csv_path)
        df["teacher_folder"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No ratings_paper_aligned.csv under {by_teacher_dir}")
    return pd.concat(frames, ignore_index=True)


def load_human_reference(reference_path: Path) -> pd.DataFrame:
    raw = json.loads(reference_path.read_text(encoding="utf-8"))
    rows = []
    for source, scores in raw.items():
        if source not in HUMAN_REFERENCE_SOURCES:
            continue
        de_row = {"quelle": source}
        for de, en in CRITERIA_DE_TO_EN.items():
            de_row[de] = float(scores[en])
        de_row["overall_score"] = float(scores["overall_score"])
        rows.append(de_row)
    return pd.DataFrame(rows)


def teacher_source_means(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.groupby(["teacher_name", "quelle"])[SCORE_COLS]
        .mean()
        .reset_index()
    )


def compute_teacher_weights(per_teacher_means: pd.DataFrame, human_means: pd.DataFrame) -> pd.DataFrame:
    human_indexed = human_means.set_index("quelle")
    rows = []
    for teacher, group in per_teacher_means.groupby("teacher_name"):
        teacher_indexed = group.set_index("quelle")
        sq_errors = []
        per_source_rmse = {}
        for source in HUMAN_REFERENCE_SOURCES:
            if source not in teacher_indexed.index:
                continue
            teacher_vec = teacher_indexed.loc[source, SCORE_COLS].astype(float).values
            human_vec = human_indexed.loc[source, SCORE_COLS].astype(float).values
            diff = teacher_vec - human_vec
            sq_errors.extend(diff * diff)
            per_source_rmse[source] = float(np.sqrt(np.mean(diff * diff)))
        rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")
        rows.append({
            "teacher_name": teacher,
            "rmse_vs_human": rmse,
            **{f"rmse_{src}": per_source_rmse.get(src, float("nan")) for src in HUMAN_REFERENCE_SOURCES},
        })
    weights = pd.DataFrame(rows)
    eps = 1e-3
    inv = 1.0 / (weights["rmse_vs_human"] + eps)
    weights["weight"] = inv / inv.sum()
    return weights.sort_values("rmse_vs_human").reset_index(drop=True)


def compute_weighted_aggregate(ratings: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    w = weights.set_index("teacher_name")["weight"]
    rows = []
    for source, group in ratings.groupby("quelle"):
        per_teacher = group.groupby("teacher_name")[SCORE_COLS].mean()
        aligned = per_teacher.reindex(w.index).dropna(how="all")
        local_w = w.reindex(aligned.index)
        local_w = local_w / local_w.sum()
        weighted = aligned.mul(local_w, axis=0).sum()
        plain = per_teacher.mean()
        out = {"quelle": source, "company": SOURCE_TO_COMPANY.get(source, "Unknown")}
        for col in SCORE_COLS:
            out[f"{col}__weighted"] = float(weighted[col])
            out[f"{col}__plain_mean"] = float(plain[col])
        rows.append(out)
    return pd.DataFrame(rows)


def compute_strictness(ratings: pd.DataFrame) -> pd.DataFrame:
    by_teacher = ratings.groupby("teacher_name")["overall_score"].agg(["mean", "std", "count"]).reset_index()
    by_teacher.columns = ["teacher_name", "overall_mean", "overall_std", "n_ratings"]
    by_teacher["company"] = by_teacher["teacher_name"].map(TEACHER_TO_COMPANY)
    return by_teacher.sort_values("overall_mean").reset_index(drop=True)


def compute_self_bias(ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for teacher, group in ratings.groupby("teacher_name"):
        teacher_company = TEACHER_TO_COMPANY.get(teacher)
        if teacher_company is None:
            continue
        own_mask = group["quelle"].map(SOURCE_TO_COMPANY) == teacher_company
        own = group.loc[own_mask, "overall_score"]
        other = group.loc[~own_mask & (group["quelle"].map(SOURCE_TO_COMPANY) != "Human"), "overall_score"]
        rows.append({
            "teacher_name": teacher,
            "teacher_company": teacher_company,
            "n_own_ratings": int(own.shape[0]),
            "own_mean": float(own.mean()) if not own.empty else float("nan"),
            "other_mean": float(other.mean()) if not other.empty else float("nan"),
            "self_bias_delta": float(own.mean() - other.mean()) if not own.empty and not other.empty else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("self_bias_delta", ascending=False).reset_index(drop=True)


def compute_within_company_compare(ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_source = ratings.groupby("quelle")["overall_score"].mean()
    for source, mean in by_source.items():
        company = SOURCE_TO_COMPANY.get(source, "Unknown")
        rows.append({"quelle": source, "company": company, "mean_overall": float(mean)})
    df = pd.DataFrame(rows).sort_values(["company", "mean_overall"], ascending=[True, False])
    df["rank_within_company"] = df.groupby("company")["mean_overall"].rank(ascending=False, method="min")
    return df.reset_index(drop=True)


def compute_teacher_agreement(ratings: pd.DataFrame) -> pd.DataFrame:
    pivoted = ratings.pivot_table(
        index=["thema", "quelle"],
        columns="teacher_name",
        values="overall_score",
        aggfunc="mean",
    )
    rows = []
    for (thema, quelle), row in pivoted.iterrows():
        scores = row.dropna()
        if scores.shape[0] < 2:
            continue
        rows.append({
            "thema": thema,
            "quelle": quelle,
            "n_teachers": int(scores.shape[0]),
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=0)),
            "range": float(scores.max() - scores.min()),
        })
    return pd.DataFrame(rows).sort_values("std", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted aggregation + analyses of per-teacher essay ratings.")
    parser.add_argument("--by-teacher-dir", type=Path, default=Path("outputs/full-run-ratings-by-teacher"))
    parser.add_argument("--reference", type=Path, default=Path("config/reference_stats.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/aggregated"))
    parser.add_argument("--exclude-teachers", nargs="*", default=["glm-5-1-evaluator"],
                        help="Teacher folder names to skip (e.g. glm-5-1-evaluator).")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ratings = load_all_ratings(args.by_teacher_dir)
    if args.exclude_teachers:
        ratings = ratings[~ratings["teacher_folder"].isin(args.exclude_teachers)]

    human_means = load_human_reference(args.reference)
    per_teacher_means = teacher_source_means(ratings)

    weights = compute_teacher_weights(per_teacher_means, human_means)
    weighted = compute_weighted_aggregate(ratings, weights)
    strictness = compute_strictness(ratings)
    self_bias = compute_self_bias(ratings)
    within_company = compute_within_company_compare(ratings)
    agreement = compute_teacher_agreement(ratings)

    weights.to_csv(args.output_dir / "teacher_weights.csv", index=False, encoding="utf-8-sig")
    weighted.to_csv(args.output_dir / "weighted_scores_by_source.csv", index=False, encoding="utf-8-sig")
    strictness.to_csv(args.output_dir / "teacher_strictness.csv", index=False, encoding="utf-8-sig")
    self_bias.to_csv(args.output_dir / "self_bias.csv", index=False, encoding="utf-8-sig")
    within_company.to_csv(args.output_dir / "within_company_compare.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(args.output_dir / "teacher_agreement.csv", index=False, encoding="utf-8-sig")

    print("=== Teacher weights (closer to human => higher weight) ===")
    print(weights.to_string(index=False))
    print("\n=== Weighted vs plain-mean overall_score by source ===")
    print(weighted[["quelle", "company", "overall_score__weighted", "overall_score__plain_mean"]]
          .sort_values("overall_score__weighted", ascending=False).to_string(index=False))
    print("\n=== Strictness (low overall_mean = strict) ===")
    print(strictness.to_string(index=False))
    print("\n=== Self-bias (positive = teacher rates own company higher) ===")
    print(self_bias.to_string(index=False))
    print(f"\nSaved 6 CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
