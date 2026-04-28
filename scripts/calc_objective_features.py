from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
from lexicalrichness import LexicalRichness

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from essay_benchmark.study import OUTPUT_DIR

MARKERS_DIR = ROOT_DIR / "sherbold-chatgpt-student-essay-study-3f09052" / "markers"

PREFIX_MAP = {
    "Student": "STUD",
    "ChatGPT-3": "GPT3",
    "ChatGPT-4": "GPT4",
}

TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:Title\s*:.*|#+\s.*)\r?\n(?:\s*\r?\n)*",
    flags=re.IGNORECASE,
)


def strip_title_prefix(text: str) -> str:
    return TITLE_PREFIX_RE.sub("", text, count=1)


def build_prefix(source_name: str) -> str:
    if source_name in PREFIX_MAP:
        return PREFIX_MAP[source_name]
    compact = re.sub(r"[^A-Za-z0-9]+", "", source_name).upper()
    return compact or "MODEL"


def num_of_words(text: str) -> int:
    return len(text.split())


def num_of_sent(doc: Any) -> int:
    return sum(1 for _ in doc.sents)


def sent_complexity_structure(doc: Any) -> int:
    return len(
        [
            token
            for token in doc
            if token.dep_ in {"acl", "conj", "advcl", "ccomp", "csubj", "discourse", "parataxis"}
        ]
    )


def calculate_dep_score(doc: Any) -> float:
    values = [sent_complexity_structure(sentence) for sentence in doc.sents]
    return float(sum(values) / len(values)) if values else 0.0


def walk_tree(node: Any, depth: int) -> int:
    if node.n_lefts + node.n_rights > 0:
        return max(walk_tree(child, depth + 1) for child in node.children)
    return depth


def calculate_dep_length(doc: Any) -> float:
    values = [walk_tree(sentence.root, 0) for sentence in doc.sents]
    return float(sum(values) / len(values)) if values else 0.0


def calculate_lex_richness_mtld(text: str) -> float:
    return float(LexicalRichness(text).mtld())


def load_discourse_markers() -> list[str]:
    discourse = pd.read_csv(
        MARKERS_DIR / "connectives_discourse_markers_PDTB.txt",
        sep="'",
        encoding="utf-8",
        header=None,
        usecols=[1, 3],
    )
    discourse[3] = discourse[3].apply(lambda x: x.replace("t_conn_", ""))
    discourse[1] = discourse[1].apply(lambda x: f" {x} ")
    discourse.sort_values(3, inplace=True, ascending=False)
    return discourse[1].tolist()


def count_discourse_markers(lemma_text: str, markers: list[str]) -> int:
    return sum(lemma_text.count(marker) for marker in markers if marker in lemma_text)


def load_modals() -> list[str]:
    modals = pd.read_csv(MARKERS_DIR / "modals.csv", sep=",", encoding="utf-8", header=None)
    modals[0] = modals[0].apply(lambda x: x.replace("_", " "))
    return modals[0].tolist()


def count_total_modals(lemma_text: str, modal_markers: list[str]) -> int:
    return sum(lemma_text.count(modal) for modal in modal_markers if modal in lemma_text)


def find_epistemic_markers(text: str) -> int:
    ep_markers: list[str] = []
    ep_markers.extend(
        re.findall(
            r"(?:I|We|we|One|one)(?:\s\w+)?(?:\s\w+)?\s(?:believes?|thinks?|means?|worry|worries|know|guesse?s?|assumes?)\s(?:that)?",
            text,
        )
    )
    ep_markers.extend(re.findall(r"(?:It|it)\sis\s(?:believed|known|assumed|thought)\s(?:that)?", text))
    ep_markers.extend(re.findall(r"(?:I|We|we)\s(?:am|are)\s(?:thinking|guessing)\s(?:that)?", text))
    ep_markers.extend(
        re.findall(r"(?:I|We|we|One|one)(?:\s\w+)?\s(?:do|does)\snot\s(?:believe?|think|know)\s(?:that)?", text)
    )
    ep_markers.extend(re.findall(r"(?:I|We|we|One|one)\swould(?:\s\w+)?(?:\snot)?\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"I\sam\s(?:afraid|sure|confident)\s(?:that)?", text))
    ep_markers.extend(
        re.findall(
            r"(?:My|my|Our|our)\s(?:experience|opinion|belief|knowledge|worry|worries|concerns?|guesse?s?)\s(?:is|are)\s(?:that)?",
            text,
        )
    )
    ep_markers.extend(re.findall(r"[In]n\s(?:my|our)(?:\s\w+)?\sopinion", text))
    ep_markers.extend(re.findall(r"As\sfar\sas\s(?:I|We|we)\s(?:am|are)\sconcerned", text))
    ep_markers.extend(re.findall(r"(?:I|We|we|One|one)\s(?:can|could|may|might)(?:\s\w+)?\sconclude\s(?:that)?", text))
    ep_markers.extend(re.findall(r"I\s(?:am\swilling\sto|must)\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"One\s(?:can|could|may|might)\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"[Oo]ne\s(?:can|could|may|might)\ssay\s(?:that)?", text))
    ep_markers.extend(re.findall(r"[Ii]t\sis\s(?:obvious|(?:un)?clear)", text))
    ep_markers.extend(re.findall(r"[Ii]t\s(?:seems|feels|looks)", text))
    return len(ep_markers)


def nominalisation_counter(doc: Any) -> int:
    suffixes_n = r"\b[A-Z]*\w+(?:tion|ment|ance|ence|ion|it(?:y|ies)|ness|ship)(?:s|es)?\b"
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    return len([noun for noun in nouns if re.match(suffixes_n, noun)])


def average_per_sentence(feature: float, sent_count: int) -> float:
    return float(feature / sent_count) if sent_count else 0.0


def compute_features(text: str, nlp: Any, discourse_markers: list[str], modal_markers: list[str]) -> dict[str, Any]:
    doc = nlp(text)
    lemma_text = " ".join(token.lemma_ for token in doc)
    pos_text = " ".join(token.tag_ for token in doc)
    sent_count = num_of_sent(doc)
    discourse_count = count_discourse_markers(lemma_text, discourse_markers)
    modals1 = count_total_modals(lemma_text, modal_markers)
    modals2 = pos_text.count("MD")
    modals_all = modals1 + modals2
    epistemic = find_epistemic_markers(text)
    nominalisation = nominalisation_counter(doc)

    return {
        "spacy": doc,
        "lemma": lemma_text,
        "pos": pos_text,
        "sent_count": sent_count,
        "word_count": num_of_words(text),
        "sent_complex_tags": calculate_dep_score(doc),
        "sent_complex_depth": calculate_dep_length(doc),
        "LD": calculate_lex_richness_mtld(text),
        "discourse": discourse_count,
        "modals1": modals1,
        "modals2": modals2,
        "modals_all": modals_all,
        "EpMarkers": epistemic,
        "nominalisation": nominalisation,
        "dm_per_sent": average_per_sentence(discourse_count, sent_count),
        "mod_per_sent": average_per_sentence(modals_all, sent_count),
        "ep_per_sent": average_per_sentence(epistemic, sent_count),
        "nom_per_sent": average_per_sentence(nominalisation, sent_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the objective linguistic features used in the Herbold et al. replication pipeline for a wide essay dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Wide CSV such as essays-wide.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for objective feature outputs. Defaults to outputs/objective-<dataset-name>.",
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy English model to load.")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    source_columns = [column for column in dataset.columns if column not in {"id", "File", "Topic"}]

    discourse_markers = load_discourse_markers()
    modal_markers = load_modals()
    nlp = spacy.load(args.spacy_model)

    output_dir = args.output_dir or OUTPUT_DIR / f"objective-{args.dataset.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []

    metric_columns = [
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

    for _, row in dataset.iterrows():
        wide_row: dict[str, Any] = {"id": row["id"], "File": row["File"], "Topic": row["Topic"]}
        for source_name in source_columns:
            essay_text = str(row[source_name] or "").strip()
            if not essay_text or essay_text.lower() == "nan":
                continue

            print(f"Objective features: {source_name} -> {row['Topic']}")
            features = compute_features(essay_text, nlp, discourse_markers, modal_markers)
            prefix = build_prefix(source_name)

            long_row = {
                "id": row["id"],
                "File": row["File"],
                "Topic": row["Topic"],
                "source": source_name,
                "essay_text": essay_text,
                "lemma": features["lemma"],
                "pos": features["pos"],
            }
            for metric in metric_columns:
                long_row[metric] = features[metric]
                wide_row[f"{prefix}_{metric}"] = features[metric]
            long_rows.append(long_row)
        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(output_dir / "objective_features_long.csv", index=False, encoding="utf-8-sig")

    wide_df = pd.DataFrame(wide_rows)
    merged = dataset.merge(wide_df, on=["id", "File", "Topic"], how="left")
    merged.to_csv(output_dir / "essays-with-objective-features.csv", index=False, encoding="utf-8-sig")

    summary = long_df.groupby("source")[metric_columns].mean().round(4).reset_index()
    summary.to_csv(output_dir / "objective_features_summary_by_source.csv", index=False, encoding="utf-8-sig")
    print(f"Saved objective feature outputs to {output_dir}")


if __name__ == "__main__":
    main()
