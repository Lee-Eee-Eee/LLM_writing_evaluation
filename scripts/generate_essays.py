from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from essay_benchmark.openai_compatible import ProviderError, call_chat_completion, extract_message_text
from essay_benchmark.study import OUTPUT_DIR, load_representative_topics, load_json


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return cleaned or "essay"


def load_providers(path: Path) -> list[dict[str, Any]]:
    providers = load_json(path)
    if not isinstance(providers, list) or not providers:
        raise ValueError("Provider configuration must be a non-empty JSON list.")
    return providers


def build_messages(provider: dict[str, Any], topic: str) -> list[dict[str, str]]:
    user_template = provider.get("user_prompt_template") or 'Write an essay with about 200 words on "{topic}".'
    user_prompt = user_template.format(topic=topic)
    messages: list[dict[str, str]] = []
    system_prompt = (provider.get("system_prompt") or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def mock_generate(topic: str, provider_name: str) -> str:
    return (
        f"{topic}\n\n"
        f"{provider_name} mock response: This is a placeholder essay written to test the pipeline. "
        "It introduces a position, gives two supporting reasons, and ends with a short conclusion. "
        "Replace the mock provider with a live OpenAI-compatible endpoint for real benchmarking."
    )


def generate_one(provider: dict[str, Any], topic: str) -> str:
    if provider.get("base_url", "").strip().lower() == "mock":
        return mock_generate(topic, provider.get("name", "Mock model"))

    payload = call_chat_completion(provider, messages=build_messages(provider, topic))
    text = extract_message_text(payload)
    if not text.strip():
        raise ProviderError(f"{provider.get('name', 'Provider')} returned an empty essay.")
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Herbold-style benchmark dataset where each model independently writes essays for the same topic set."
    )
    parser.add_argument(
        "--providers",
        type=Path,
        default=Path("config/model_providers.example.json"),
        help="JSON list of OpenAI-compatible generation providers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated outputs. Defaults to outputs/generation-<timestamp>.",
    )
    parser.add_argument(
        "--limit-topics",
        type=int,
        default=0,
        help="Only generate the first N representative topics for quick tests.",
    )
    parser.add_argument(
        "--only-provider",
        action="append",
        default=[],
        help="Generate only providers whose name matches this value. May be passed more than once.",
    )
    args = parser.parse_args()

    providers = load_providers(args.providers)
    if args.only_provider:
        requested = set(args.only_provider)
        providers = [provider for provider in providers if provider.get("name") in requested]
        if not providers:
            raise ValueError(f"No providers matched --only-provider: {', '.join(sorted(requested))}")

    topics = load_representative_topics()
    if args.limit_topics > 0:
        topics = topics[: args.limit_topics]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or OUTPUT_DIR / f"generation-{timestamp}"
    essays_dir = output_dir / "essays"
    essays_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for topic_item in topics:
        row = {
            "id": topic_item["id"],
            "File": topic_item["file"],
            "Topic": topic_item["topic"],
            "Student": topic_item["student_essay"],
            "ChatGPT-3": topic_item.get("chatgpt3_essay", ""),
            "ChatGPT-4": topic_item.get("chatgpt4_essay", ""),
        }
        for provider in providers:
            provider_name = provider["name"]
            print(f"Generating: {provider_name} -> {topic_item['topic']}")
            try:
                essay_text = generate_one(provider, topic_item["topic"])
            except ProviderError as exc:
                error_rows.append(
                    {
                        "topic_id": topic_item["id"],
                        "topic_file": topic_item["file"],
                        "topic": topic_item["topic"],
                        "source_name": provider_name,
                        "model": provider["model"],
                        "base_url": provider["base_url"],
                        "error": str(exc),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                row[provider_name] = ""
                continue

            provider_slug = slugify(provider_name)
            essay_file = essays_dir / provider_slug / topic_item["file"]
            essay_file.parent.mkdir(parents=True, exist_ok=True)
            essay_file.write_text(essay_text, encoding="utf-8")

            long_rows.append(
                {
                    "topic_id": topic_item["id"],
                    "topic_file": topic_item["file"],
                    "topic": topic_item["topic"],
                    "source_name": provider_name,
                    "model": provider["model"],
                    "base_url": provider["base_url"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "prompt": (provider.get("user_prompt_template") or 'Write an essay with about 200 words on "{topic}".').format(
                        topic=topic_item["topic"]
                    ),
                    "essay_text": essay_text,
                    "essay_path": str(essay_file),
                }
            )
            row[provider_name] = essay_text
        wide_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ai_essays_long.jsonl").open("w", encoding="utf-8") as handle:
        for item in long_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    if long_rows:
        with (output_dir / "ai_essays_long.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
            writer.writeheader()
            writer.writerows(long_rows)

    if error_rows:
        with (output_dir / "generation_errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(error_rows[0].keys()))
            writer.writeheader()
            writer.writerows(error_rows)

    if wide_rows:
        with (output_dir / "essays-wide.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(wide_rows[0].keys()))
            writer.writeheader()
            writer.writerows(wide_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topics": len(topics),
        "providers": [provider["name"] for provider in providers],
        "provider_models": {provider["name"]: provider["model"] for provider in providers},
        "errors": len(error_rows),
        "notes": [
            "Each provider independently generated one essay per topic.",
            "The default prompt mirrors the original paper: Write an essay with about 200 words on \"{topic}\".",
            "The Student, ChatGPT-3, and ChatGPT-4 columns come directly from the original Herbold replication package."
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
