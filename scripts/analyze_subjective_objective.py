from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCORE_COLS = [
    "themenbezogenheit",
    "logik-des-aufbaus",
    "ausfuehrlichkeit-aussagekraft",
    "sprachbeherrschung",
    "komplexitaet",
    "wortschatz-textverknuepfung",
    "gebrauch-sprachlicher-strukturen",
    "overall_score",
]

OBJECTIVE_COLS = [
    "sent_count",
    "word_count",
    "sent_complex_tags",
    "sent_complex_depth",
    "LD",
    "discourse",
    "modals1",
    "modals2",
    "modals_all",
    "EpMarkers",
    "nominalisation",
    "dm_per_sent",
    "mod_per_sent",
    "ep_per_sent",
    "nom_per_sent",
]


def load_ratings(by_teacher_dir: Path, exclude_folders: set[str]) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(by_teacher_dir.glob("*/ratings_paper_aligned.csv")):
        if csv_path.parent.name in exclude_folders:
            continue
        df = pd.read_csv(csv_path)
        df["teacher_folder"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No rating files found in {by_teacher_dir}")
    return pd.concat(frames, ignore_index=True)


def corr_table(df: pd.DataFrame, x_cols: list[str], y_col: str, *, min_n: int = 4) -> pd.DataFrame:
    rows = []
    for col in x_cols:
        pair = df[[col, y_col]].dropna()
        if pair.shape[0] < min_n or pair[col].nunique() < 2 or pair[y_col].nunique() < 2:
            continue
        rows.append(
            {
                "objective_feature": col,
                "target": y_col,
                "n": int(pair.shape[0]),
                "pearson_r": float(pair[col].corr(pair[y_col], method="pearson")),
                "spearman_r": float(pair[col].corr(pair[y_col], method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relate subjective essay ratings to objective linguistic features.")
    parser.add_argument("--by-teacher-dir", type=Path, default=Path("outputs/full-run-ratings-by-teacher"))
    parser.add_argument("--objective-dir", type=Path, default=Path("outputs/objective-essays-wide"))
    parser.add_argument("--aggregated-dir", type=Path, default=Path("outputs/aggregated"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/subjective-objective"))
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
    exclude = set(args.exclude_folders)

    ratings = load_ratings(args.by_teacher_dir, exclude)
    objective_long = pd.read_csv(args.objective_dir / "objective_features_long.csv")
    objective_summary = pd.read_csv(args.objective_dir / "objective_features_summary_by_source.csv")
    weighted = pd.read_csv(args.aggregated_dir / "weighted_scores_by_source.csv")

    teacher_counts = (
        ratings.groupby(["teacher_name", "teacher_model"])["overall_score"]
        .agg(["count", "mean", "std"])
        .round(4)
        .reset_index()
    )
    teacher_counts.to_csv(args.output_dir / "teacher_rating_counts.csv", index=False, encoding="utf-8-sig")

    subjective_by_essay = (
        ratings.groupby(["thema", "quelle"])[SCORE_COLS]
        .agg(["mean", "std", "count"])
    )
    subjective_by_essay.columns = ["__".join(col).strip("_") for col in subjective_by_essay.columns]
    subjective_by_essay = subjective_by_essay.reset_index()
    subjective_by_essay.to_csv(args.output_dir / "subjective_scores_by_essay.csv", index=False, encoding="utf-8-sig")

    objective_for_join = objective_long.rename(columns={"Topic": "thema", "source": "quelle"})
    essay_join = subjective_by_essay.merge(
        objective_for_join[["thema", "quelle", "id", "File", *OBJECTIVE_COLS]],
        on=["thema", "quelle"],
        how="inner",
    )
    essay_join.to_csv(args.output_dir / "subjective_objective_by_essay.csv", index=False, encoding="utf-8-sig")

    source_subjective = (
        ratings.groupby("quelle")[SCORE_COLS]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={col: f"{col}__plain_teacher_mean" for col in SCORE_COLS})
    )
    source_join = source_subjective.merge(objective_summary, left_on="quelle", right_on="source", how="inner")
    weighted_cols = ["quelle"] + [col for col in weighted.columns if col.endswith("__weighted") or col.endswith("__plain_mean")]
    source_join = source_join.merge(weighted[weighted_cols], on="quelle", how="left")
    source_join.to_csv(args.output_dir / "subjective_objective_by_source.csv", index=False, encoding="utf-8-sig")

    corr_source_plain = corr_table(
        source_join,
        OBJECTIVE_COLS,
        "overall_score__plain_teacher_mean",
        min_n=5,
    )
    corr_source_plain.to_csv(args.output_dir / "correlations_source_plain_overall.csv", index=False, encoding="utf-8-sig")

    if "overall_score__weighted" in source_join.columns:
        corr_source_weighted = corr_table(source_join, OBJECTIVE_COLS, "overall_score__weighted", min_n=5)
        corr_source_weighted.to_csv(
            args.output_dir / "correlations_source_weighted_overall.csv",
            index=False,
            encoding="utf-8-sig",
        )

    corr_essay = corr_table(essay_join, OBJECTIVE_COLS, "overall_score__mean", min_n=20)
    corr_essay.to_csv(args.output_dir / "correlations_essay_plain_overall.csv", index=False, encoding="utf-8-sig")

    print("=== Teacher counts ===")
    print(teacher_counts.to_string(index=False))
    print("\n=== Top essay-level objective correlations with subjective overall ===")
    print(corr_essay.head(10).to_string(index=False))
    print("\n=== Top source-level objective correlations with plain subjective overall ===")
    print(corr_source_plain.head(10).to_string(index=False))
    print(f"\nSaved subjective/objective statistics to {args.output_dir}")


if __name__ == "__main__":
    main()
