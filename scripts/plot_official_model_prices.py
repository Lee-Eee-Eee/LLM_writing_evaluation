from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "outputs" / "reports" / "assets" / "model-icons"
OUTPUT_DIR = ROOT / "outputs" / "reports" / "assets" / "model-price-charts"

USD_PER_CNY = 1 / 7.2


@dataclass(frozen=True)
class ModelPrice:
    name: str
    icon_file: str
    input_price: float
    output_price: float
    currency: str
    source_note: str

    @property
    def input_plot_value(self) -> float:
        return self.input_price if self.currency == "USD" else self.input_price * USD_PER_CNY

    @property
    def output_plot_value(self) -> float:
        return self.output_price if self.currency == "USD" else self.output_price * USD_PER_CNY

    @property
    def input_label(self) -> str:
        return self._format_price(self.input_price)

    @property
    def output_label(self) -> str:
        return self._format_price(self.output_price)

    def _format_price(self, value: float) -> str:
        if self.currency == "USD":
            return f"${value:.2f}"
        return f"¥{value:.2f}"


GENERATION_MODELS: list[ModelPrice] = [
    ModelPrice(
        name="GPT-5.5",
        icon_file="openai.png",
        input_price=5.00,
        output_price=30.00,
        currency="USD",
        source_note="OpenAI pricing page",
    ),
    ModelPrice(
        name="Claude Opus 4.7",
        icon_file="claude-color.png",
        input_price=5.00,
        output_price=25.00,
        currency="USD",
        source_note="Anthropic pricing page",
    ),
    ModelPrice(
        name="Claude Opus 4.6",
        icon_file="claude-color.png",
        input_price=5.00,
        output_price=25.00,
        currency="USD",
        source_note="Anthropic pricing page",
    ),
    ModelPrice(
        name="Gemini 3.1 Pro",
        icon_file="gemini-color.png",
        input_price=2.00,
        output_price=12.00,
        currency="USD",
        source_note="Google pricing page",
    ),
    ModelPrice(
        name="DeepSeek-V4Pro",
        icon_file="deepseek-color.png",
        input_price=0.14,
        output_price=0.55,
        currency="USD",
        source_note="DeepSeek pricing page",
    ),
    ModelPrice(
        name="DeepSeek-V3.2",
        icon_file="deepseek-color.png",
        input_price=0.07,
        output_price=0.28,
        currency="USD",
        source_note="DeepSeek pricing page",
    ),
    ModelPrice(
        name="Qwen 3.6 Plus",
        icon_file="qwen-color.png",
        input_price=0.50,
        output_price=3.00,
        currency="USD",
        source_note="Alibaba Cloud Model Studio",
    ),
    ModelPrice(
        name="Kimi K2.6",
        icon_file="kimi-color.png",
        input_price=6.50,
        output_price=27.00,
        currency="CNY",
        source_note="Alibaba Cloud Model Studio",
    ),
    ModelPrice(
        name="GLM-5.1",
        icon_file="zhipu-color.png",
        input_price=6.00,
        output_price=24.00,
        currency="CNY",
        source_note="BigModel pricing page",
    ),
    ModelPrice(
        name="MiniMax-M2.7",
        icon_file="minimax-color.png",
        input_price=2.10,
        output_price=8.40,
        currency="CNY",
        source_note="MiniMax pricing page",
    ),
]


EVALUATION_MODELS: list[ModelPrice] = [
    ModelPrice(
        name="DeepSeek-V3.2",
        icon_file="deepseek-color.png",
        input_price=0.07,
        output_price=0.28,
        currency="USD",
        source_note="DeepSeek pricing page",
    ),
    ModelPrice(
        name="Claude Haiku 4.5",
        icon_file="claude-color.png",
        input_price=1.00,
        output_price=5.00,
        currency="USD",
        source_note="Anthropic pricing page",
    ),
    ModelPrice(
        name="Doubao Seed 2.0 Pro",
        icon_file="doubao-color.png",
        input_price=3.20,
        output_price=16.00,
        currency="CNY",
        source_note="Volcengine pricing page",
    ),
    ModelPrice(
        name="Gemini 3.1 Flash Lite",
        icon_file="gemini-color.png",
        input_price=0.25,
        output_price=1.50,
        currency="USD",
        source_note="Google pricing page",
    ),
    ModelPrice(
        name="GPT-5.4 mini",
        icon_file="openai.png",
        input_price=0.75,
        output_price=4.50,
        currency="USD",
        source_note="OpenAI pricing page",
    ),
    ModelPrice(
        name="Grok 4 Fast",
        icon_file="grok.png",
        input_price=0.20,
        output_price=0.50,
        currency="USD",
        source_note="xAI pricing page",
    ),
    ModelPrice(
        name="Kimi K2.5",
        icon_file="kimi-color.png",
        input_price=4.00,
        output_price=21.00,
        currency="CNY",
        source_note="Alibaba Cloud Model Studio",
    ),
    ModelPrice(
        name="MiniMax-M2.7 Highspeed",
        icon_file="minimax-color.png",
        input_price=4.20,
        output_price=16.80,
        currency="CNY",
        source_note="MiniMax pricing page",
    ),
    ModelPrice(
        name="Qwen 3.5 Flash",
        icon_file="qwen-color.png",
        input_price=0.10,
        output_price=0.40,
        currency="USD",
        source_note="Alibaba Cloud Model Studio",
    ),
]


def load_icon(icon_name: str) -> Image.Image:
    icon_path = ICON_DIR / icon_name
    if not icon_path.exists():
        raise FileNotFoundError(f"Missing icon: {icon_path}")
    return Image.open(icon_path).convert("RGBA")


def prepare_icon(image: Image.Image, model_name: str, canvas_size: int = 220) -> Image.Image:
    alpha = image.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        image = image.crop(bbox)

    is_kimi = "kimi" in model_name.lower()
    icon_ratio = 0.62 if is_kimi else 0.78
    target_side = int(canvas_size * icon_ratio)

    resize_scale = target_side / max(image.width, image.height)
    resized = image.resize(
        (max(1, int(image.width * resize_scale)), max(1, int(image.height * resize_scale))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    if is_kimi:
        draw = ImageDraw.Draw(canvas)
        margin = int(canvas_size * 0.12)
        radius = int(canvas_size * 0.18)
        draw.rounded_rectangle(
            (margin, margin, canvas_size - margin, canvas_size - margin),
            radius=radius,
            fill=(0, 0, 0, 235),
        )

    paste_x = (canvas_size - resized.width) // 2
    paste_y = (canvas_size - resized.height) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def add_icon(ax, image: Image.Image, x: float, y: float, zoom: float = 0.095) -> None:
    artist = AnnotationBbox(OffsetImage(image, zoom=zoom), (x, y), frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(artist)


def make_chart(title: str, subtitle: str, models: Iterable[ModelPrice], output_path: Path) -> None:
    rows = sorted(models, key=lambda model: model.output_plot_value, reverse=True)
    max_plot_value = max(max(model.input_plot_value, model.output_plot_value) for model in rows)
    max_name_len = max(len(model.name) for model in rows)
    label_width_units = 1.05 + max_name_len * 0.22
    left_margin = max(max_plot_value * 0.32, label_width_units, 3.8)
    right_margin = max_plot_value * 0.18
    total_width = left_margin + max_plot_value + right_margin

    fig, ax = plt.subplots(figsize=(16, max(7.5, len(rows) * 0.68)))
    fig.patch.set_facecolor("#f6f3ed")
    ax.set_facecolor("#fbfaf7")

    y_positions = list(range(len(rows)))[::-1]
    input_color = "#2f8f9d"
    output_color = "#d18f25"
    text_x = -left_margin + 1.05
    icon_x = text_x - 0.55

    for y, model in zip(y_positions, rows):
        ax.barh(y + 0.16, model.input_plot_value, height=0.26, color=input_color, alpha=0.9)
        ax.barh(y - 0.16, model.output_plot_value, height=0.26, color=output_color, alpha=0.9)

        icon = prepare_icon(load_icon(model.icon_file), model.name)
        add_icon(ax, icon, icon_x, y, zoom=0.095)

        ax.text(
            text_x,
            y,
            model.name,
            va="center",
            ha="left",
            fontsize=12.2,
            color="#1f2937",
            fontweight="semibold",
        )

        ax.text(
            model.input_plot_value + max_plot_value * 0.02,
            y + 0.16,
            model.input_label,
            va="center",
            ha="left",
            fontsize=10.4,
            color="#14515a",
            fontweight="bold",
        )
        ax.text(
            model.output_plot_value + max_plot_value * 0.02,
            y - 0.16,
            model.output_label,
            va="center",
            ha="left",
            fontsize=10.4,
            color="#855c12",
            fontweight="bold",
        )

    ax.set_xlim(-left_margin, total_width)
    ax.set_ylim(-1, len(rows))
    ax.set_yticks([])
    ax.set_xlabel("Equivalent US$ per 1M tokens for bar length; labels show each provider's official list price", fontsize=11)
    ax.set_title(title, fontsize=16.5, fontweight="bold", loc="left", pad=28)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9.7, color="#4b5563")

    ax.grid(axis="x", color="#d8d3c6", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c8c2b3")

    legend_handles = [
        plt.Line2D([0], [0], color=input_color, lw=8, label="Input"),
        plt.Line2D([0], [0], color=output_color, lw=8, label="Output"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.95, facecolor="#fffdf9", edgecolor="#e1dbc9")

    note = "CNY rows are converted with a fixed 7.2 CNY/USD rate for visual comparability only."
    fig.text(0.01, 0.01, note, fontsize=9.5, color="#6b7280")

    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_chart(
        title="Official model pricing for essay generation models",
        subtitle="Ten models used for generation. Chinese-list prices are converted to US$ only for bar length; the labels retain the official list price.",
        models=GENERATION_MODELS,
        output_path=OUTPUT_DIR / "generation_model_prices.png",
    )
    make_chart(
        title="Official model pricing for essay evaluation models",
        subtitle="Nine models used as graders. Chinese-list prices are converted to US$ only for bar length; the labels retain the official list price.",
        models=EVALUATION_MODELS,
        output_path=OUTPUT_DIR / "evaluation_model_prices.png",
    )


if __name__ == "__main__":
    main()