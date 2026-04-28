from __future__ import annotations

import argparse
import csv
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from essay_benchmark.grading import flatten_paper_aligned_row, grade_essay, grade_essays_batch
from essay_benchmark.openai_compatible import ProviderError
from essay_benchmark.study import OUTPUT_DIR, load_json


def load_teachers(path: Path) -> list[dict[str, Any]]:
    teachers = load_json(path)
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("Teacher configuration must be a non-empty JSON list.")
    return teachers


def rating_key(row: dict[str, Any]) -> tuple[str, str, str]:
    topic = str(row.get("thema", row.get("topic", "")))
    return (topic, str(row.get("quelle", "")), str(row.get("teacher_name", "")))


def error_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("topic", "")), str(row.get("source_name", "")), str(row.get("teacher_name", "")))


def dedupe_rating_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        seen[rating_key(row)] = row
    return list(seen.values())


def dedupe_error_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        seen[error_key(row)] = row
    return list(seen.values())


def write_outputs(output_dir: Path, rating_rows: list[dict[str, Any]], error_rows: list[dict[str, Any]]) -> None:
    rating_rows = dedupe_rating_rows(rating_rows)
    error_rows = dedupe_error_rows(error_rows)

    if rating_rows:
        ratings_path = output_dir / "ratings_paper_aligned.csv"
        with ratings_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rating_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rating_rows)

        ratings_df = pd.DataFrame(rating_rows)
        summary = (
            ratings_df.groupby("quelle")[
                [
                    "themenbezogenheit",
                    "logik-des-aufbaus",
                    "ausfuehrlichkeit-aussagekraft",
                    "sprachbeherrschung",
                    "komplexitaet",
                    "wortschatz-textverknuepfung",
                    "gebrauch-sprachlicher-strukturen",
                    "overall_score",
                ]
            ]
            .mean()
            .round(2)
            .reset_index()
        )
        summary.to_csv(output_dir / "ratings_summary_by_source.csv", index=False, encoding="utf-8-sig")

    errors_path = output_dir / "grading_errors.csv"
    if error_rows:
        pd.DataFrame(error_rows).to_csv(errors_path, index=False, encoding="utf-8-sig")
    elif errors_path.exists():
        errors_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch grade essays with one or more OpenAI-compatible teacher models."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Wide CSV such as outputs/.../essays-wide.csv",
    )
    parser.add_argument(
        "--teachers",
        type=Path,
        default=Path("config/teacher_evaluators.example.json"),
        help="JSON list of evaluator configurations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for ratings output. Defaults to outputs/ratings-<dataset-name>.",
    )
    parser.add_argument(
        "--include-student",
        action="store_true",
        help="Also grade the original Student essays.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry previously failed combinations from grading_errors.csv.",
    )
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    base_columns = {"id", "File", "Topic", "Student"}
    source_columns = [column for column in dataset.columns if column not in {"id", "File", "Topic"}]
    if not args.include_student:
        source_columns = [column for column in source_columns if column != "Student"]

    teachers = load_teachers(args.teachers)
    output_dir = args.output_dir or OUTPUT_DIR / f"ratings-{args.dataset.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rating_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    completed: set[tuple[str, str, str]] = set()
    failed: set[tuple[str, str, str]] = set()

    ratings_path = output_dir / "ratings_paper_aligned.csv"
    if ratings_path.exists():
        existing_ratings = pd.read_csv(ratings_path).fillna("")
        rating_rows = dedupe_rating_rows(existing_ratings.to_dict("records"))
        completed = {rating_key(row) for row in rating_rows}

    errors_path = output_dir / "grading_errors.csv"
    if errors_path.exists():
        existing_errors = pd.read_csv(errors_path).fillna("")
        error_rows = dedupe_error_rows(existing_errors.to_dict("records"))
        failed = {error_key(row) for row in error_rows}

    for teacher in teachers:
        teacher_name = teacher.get("name", teacher.get("model", "Unnamed teacher"))
        teacher_model = teacher.get("model", "")
        print(f"Teacher batch start: {teacher_name}")

        for _, row in dataset.iterrows():
            topic = str(row["Topic"])
            pending_items: list[dict[str, str]] = []
            pending_source_names: list[str] = []

            for index, source_name in enumerate(source_columns, start=1):
                essay_text = str(row[source_name] or "").strip()
                if not essay_text or essay_text.lower() == "nan":
                    continue

                work_key = (topic, source_name, teacher_name)
                if work_key in completed or (not args.retry_errors and work_key in failed):
                    continue

                pending_items.append({"essay_id": f"e{index:02d}", "text": essay_text})
                pending_source_names.append(source_name)

            if not pending_items:
                continue

            print(
                f"Grading batch: topic={topic[:48]} | teacher={teacher_name} | essays={len(pending_items)}"
            )

            try:
                batch_results = grade_essays_batch(pending_items, topic, teacher)
                for item, source_name in zip(pending_items, pending_source_names):
                    result = batch_results[item["essay_id"]]
                    rating_row = flatten_paper_aligned_row(
                        session_id=uuid.uuid4().hex,
                        source_name=source_name,
                        topic=topic,
                        teacher_name=teacher_name,
                        teacher_model=teacher_model,
                        result=result,
                    )
                    rating_rows.append(rating_row)
                    completed.add(rating_key(rating_row))
                    row_error_key = (topic, source_name, teacher_name)
                    if row_error_key in failed:
                        failed.remove(row_error_key)
                    error_rows = [item_row for item_row in error_rows if error_key(item_row) != row_error_key]
            except Exception as batch_exc:
                # Fallback to per-essay grading if one batch fails.
                for item, source_name in zip(pending_items, pending_source_names):
                    try:
                        result = grade_essay(item["text"], topic, teacher)
                        rating_row = flatten_paper_aligned_row(
                            session_id=uuid.uuid4().hex,
                            source_name=source_name,
                            topic=topic,
                            teacher_name=teacher_name,
                            teacher_model=teacher_model,
                            result=result,
                        )
                        rating_rows.append(rating_row)
                        completed.add(rating_key(rating_row))
                        row_error_key = (topic, source_name, teacher_name)
                        if row_error_key in failed:
                            failed.remove(row_error_key)
                        error_rows = [item_row for item_row in error_rows if error_key(item_row) != row_error_key]
                    except Exception as exc:
                        error_row = {
                            "topic": topic,
                            "source_name": source_name,
                            "teacher_name": teacher_name,
                            "error": f"batch_failed={batch_exc}; item_failed={exc}",
                        }
                        error_rows.append(error_row)
                        failed.add(error_key(error_row))

            write_outputs(output_dir, rating_rows, error_rows)

    print(f"Saved batch grading outputs to {output_dir}")


if __name__ == "__main__":
    main()
