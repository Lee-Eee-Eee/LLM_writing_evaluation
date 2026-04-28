from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from essay_benchmark.study import CONFIG_DIR, ORIGINAL_DATA_DIR, dump_json, ensure_dir

RATING_COLUMN_MAP = {
    "STUD_Topic": "topic_and_completeness",
    "STUD_Logic": "logic_and_composition",
    "STUD_Expressiveness": "expressiveness_and_comprehensibility",
    "STUD_LangMastery": "language_mastery",
    "STUD_Complexity": "complexity",
    "STUD_Vocab": "vocabulary_and_text_linking",
    "STUD_LangConstructs": "language_constructs",
}


def build_reference_stats() -> dict:
    ratings = pd.read_csv(ORIGINAL_DATA_DIR / "ratings.csv")
    grouped = (
        ratings.groupby("quelle")[
            [
                "themenbezogenheit",
                "logik-des-aufbaus",
                "ausfuehrlichkeit-aussagekraft",
                "sprachbeherrschung",
                "komplexitaet",
                "wortschatz-textverknuepfung",
                "gebrauch-sprachlicher-strukturen",
            ]
        ]
        .mean()
        .round(2)
    )
    payload = {}
    for source_name, row in grouped.iterrows():
        payload[source_name] = {
            "topic_and_completeness": float(row["themenbezogenheit"]),
            "logic_and_composition": float(row["logik-des-aufbaus"]),
            "expressiveness_and_comprehensibility": float(row["ausfuehrlichkeit-aussagekraft"]),
            "language_mastery": float(row["sprachbeherrschung"]),
            "complexity": float(row["komplexitaet"]),
            "vocabulary_and_text_linking": float(row["wortschatz-textverknuepfung"]),
            "language_constructs": float(row["gebrauch-sprachlicher-strukturen"]),
        }
        payload[source_name]["overall_score"] = round(
            sum(payload[source_name].values()) / 7,
            2,
        )
    return payload


def build_representative_topics() -> list[dict]:
    essays = pd.read_csv(ORIGINAL_DATA_DIR / "essays-with-linguistic-markers-and-ratings.csv")
    score_columns = list(RATING_COLUMN_MAP.keys())
    essays["student_overall"] = essays[score_columns].mean(axis=1)
    essays["student_word_count"] = essays["Student"].str.split().str.len()

    for column in ["student_overall", "student_word_count"]:
        mean_value = essays[column].mean()
        std_value = essays[column].std(ddof=0) or 1
        essays[f"{column}_z"] = (essays[column] - mean_value) / std_value

    selected_indices: list[int] = []
    essays["seed_distance"] = essays["student_overall_z"] ** 2 + essays["student_word_count_z"] ** 2
    selected_indices.append(int(essays["seed_distance"].idxmin()))

    while len(selected_indices) < 15:
        selected = essays.loc[selected_indices, ["student_overall_z", "student_word_count_z"]].to_numpy()
        remaining = essays.loc[~essays.index.isin(selected_indices)].copy()
        candidates = remaining[["student_overall_z", "student_word_count_z"]].to_numpy()
        distances = ((candidates[:, None, :] - selected[None, :, :]) ** 2).sum(axis=2) ** 0.5
        remaining["min_distance"] = distances.min(axis=1)
        selected_indices.append(int(remaining["min_distance"].idxmax()))

    selected = essays.loc[selected_indices].copy()
    selected["selection_rank"] = selected["student_overall"].rank(ascending=False, method="first")
    selected = selected.sort_values(["student_overall", "student_word_count"], ascending=[False, False])

    topics = []
    for _, row in selected.iterrows():
        topics.append(
            {
                "id": int(row["id"]),
                "file": row["File"],
                "topic": row["Topic"],
                "student_overall": round(float(row["student_overall"]), 2),
                "student_word_count": int(row["student_word_count"]),
                "student_scores": {
                    normalized_key: round(float(row[source_key]), 2)
                    for source_key, normalized_key in RATING_COLUMN_MAP.items()
                },
                "student_essay": row["Student"],
                "chatgpt3_essay": row["ChatGPT-3"],
                "chatgpt4_essay": row["ChatGPT-4"],
            }
        )
    return topics


def main() -> None:
    ensure_dir(CONFIG_DIR)
    dump_json(CONFIG_DIR / "reference_stats.json", build_reference_stats())
    dump_json(CONFIG_DIR / "representative_topics.json", build_representative_topics())
    print(f"Wrote study assets into {CONFIG_DIR}")


if __name__ == "__main__":
    main()
