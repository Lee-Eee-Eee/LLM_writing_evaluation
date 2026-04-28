"""
Render the objective features as a normalized heatmap, matching the
visual style of corrected_criteria_heatmap.png (RdYlGn, imshow, cell text).
Values are min-max normalized per feature column so disparate scales are comparable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── Load data ──────────────────────────────────────────────
csv_path = Path("outputs/objective-essays-wide/objective_features_summary_by_source.csv")
df = pd.read_csv(csv_path)
df = df.set_index("source")

# ── Feature selection & column mapping ─────────────────────
FEATURE_MAP = {
    "LD":               "Lexical Diversity",
    "sent_complex_depth": "Syntactic Depth",
    "nom_per_sent":     "Nominalisation/Sent",
    "modals_all":       "Modals (All)",
    "modals2":          "Modals (POS)",
    "EpMarkers":        "Epistemic Markers",
    "dm_per_sent":      "Discourse/Sent",
    "word_count":       "Word Count",
    "sent_count":       "Sentence Count",
}

COLUMNS = list(FEATURE_MAP.keys())
LABELS  = [FEATURE_MAP[c] for c in COLUMNS]

# ── Source display order (same as original table) ──────────
SOURCE_ORDER = [
    "Student", "ChatGPT-3", "ChatGPT-4",
    "Qwen3.6Plus", "Kimi-K2.6", "GLM-5.1", "MiniMax-M2.7",
    "DeepSeek-V3.2", "GPT-5.5", "DeepSeek-V4Pro",
    "ClaudeOpus4.6", "ClaudeOpus4.7", "Gemini3.1Pro",
]

# ── Build matrix ───────────────────────────────────────────
available = [s for s in SOURCE_ORDER if s in df.index]
matrix_raw = df.loc[available, COLUMNS].astype(float).to_numpy()

# Min-max normalize each column to [0, 1]
matrix_norm = np.zeros_like(matrix_raw)
for j in range(matrix_raw.shape[1]):
    col = matrix_raw[:, j]
    cmin, cmax = col.min(), col.max()
    if cmax - cmin > 1e-9:
        matrix_norm[:, j] = (col - cmin) / (cmax - cmin)
    else:
        matrix_norm[:, j] = 0.5  # constant column → midpoint gray

# ── Model colors (same as Claude-style report) ──────────────
model_colors = {
    "Student":       "#7c746a",
    "ChatGPT-3":     "#10a37f",
    "ChatGPT-4":     "#10a37f",
    "Qwen3.6Plus":   "#ff7a00",
    "Kimi-K2.6":     "#242424",
    "GLM-5.1":       "#7661d8",
    "MiniMax-M2.7":  "#f36b2d",
    "DeepSeek-V3.2": "#4059d6",
    "GPT-5.5":       "#10a37f",
    "DeepSeek-V4Pro":"#4059d6",
    "ClaudeOpus4.6": "#d97757",
    "ClaudeOpus4.7": "#d97757",
    "Gemini3.1Pro":  "#4285f4",
}

# ── Color palette ───────────────────────────────────────────
PAPER  = "#f8f3ea"
INK    = "#2f2924"
MUTED  = "#776d63"
ACCENT = "#cc785c"

# ── Setup (match corrected_criteria_heatmap proportions) ────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "Times New Roman"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

fig, ax = plt.subplots(figsize=(10.5, 7.2), facecolor=PAPER)
ax.set_facecolor(PAPER)

# ── Heatmap ─────────────────────────────────────────────────
n_rows, n_cols = matrix_norm.shape
im = ax.imshow(matrix_norm, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

# X ticks: feature names
ax.set_xticks(np.arange(n_cols))
ax.set_xticklabels(LABELS, rotation=30, ha="right", fontsize=8.5, color=INK)

# Y ticks: source names with colored labels
ax.set_yticks(np.arange(n_rows))
ax.set_yticklabels(available, fontsize=8.5, color=INK)

# Color each y-tick label
for i, source in enumerate(available):
    c = model_colors.get(source, INK)
    ax.get_yticklabels()[i].set_color(c)
    ax.get_yticklabels()[i].set_fontweight("bold")

# ── Cell text annotations ───────────────────────────────────
for i in range(n_rows):
    for j in range(n_cols):
        raw_val = matrix_raw[i, j]
        norm_val = matrix_norm[i, j]
        # Format intelligently
        if abs(raw_val) < 0.01:
            txt = "0.00"
        elif abs(raw_val) < 1:
            txt = f"{raw_val:.3f}"
        elif abs(raw_val) < 10:
            txt = f"{raw_val:.2f}"
        else:
            txt = f"{raw_val:.1f}"
        # Use white text on dark cells, dark text on light cells
        text_color = "white" if norm_val < 0.3 or norm_val > 0.7 else INK
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.2,
                color=text_color, fontweight="bold")

# ── Colorbar ─────────────────────────────────────────────────
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label("Normalized value (min-max per feature)", color=MUTED, fontsize=8)
cbar.ax.tick_params(labelsize=7, colors=MUTED)

# ── Title & subtitle ─────────────────────────────────────────
ax.set_title(
    "Objective Linguistic Features by Essay Source\n(Normalized Heatmap)",
    fontsize=15, fontweight="bold", color=INK, pad=16, loc="left"
)
# Subtitle as figure text
fig.text(0.125, 0.935,
         "Herbold et al. (2023) computational-linguistics framework · 9 features × 13 sources · min-max normalized per column",
         fontsize=8, color=MUTED, fontstyle="italic")

# ── Footer ───────────────────────────────────────────────────
fig.text(0.125, 0.03,
         "Source: outputs/objective-essays-wide/objective_features_summary_by_source.csv  ·  "
         "Computed via calc_objective_features.py, rendered with RdYlGn colormap (green = high, red = low)",
         fontsize=7, color=MUTED, fontstyle="italic")

# ── Save ─────────────────────────────────────────────────────
out_dir = Path("outputs/research-figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "objective_features_heatmap.png"
fig.savefig(str(out_path), dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none",
            pad_inches=0.35)
plt.close(fig)
print(f"Saved → {out_path}")