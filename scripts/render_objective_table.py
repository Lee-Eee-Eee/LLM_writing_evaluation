"""
Render the objective features summary table as a Claude-styled PNG image.
Matches the aesthetic of essay_model_writing_evaluation_claude_style.html.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pathlib

# ── Data ──────────────────────────────────────────────
headers = [
    "Source", "LD", "SynDepth", "Nom/Sent", "Modals(All)",
    "Modals(POS)", "EpiMark", "Disc/Sent", "Words", "Sents"
]

rows = [
    ("Student",       83.8,  5.69, 0.887, 14.40, 11.60, 0.80, 0.541, 442.6, 23.9),
    ("ChatGPT-3",     74.9,  5.77, 1.220,  8.73,  7.80, 0.27, 0.489, 244.5, 12.7),
    ("ChatGPT-4",    107.1,  6.03, 1.736,  4.80,  3.73, 0.00, 0.345, 262.9, 12.9),
    ("Qwen3.6Plus",  216.1,  5.04, 1.426,  5.67,  4.33, 0.20, 0.562, 236.7, 13.9),
    ("Kimi-K2.6",    239.1,  4.88, 1.380,  5.00,  3.53, 0.07, 0.574, 195.5, 11.7),
    ("GLM-5.1",      156.8,  5.13, 1.094,  4.29,  3.21, 0.50, 0.492, 196.1, 11.4),
    ("MiniMax-M2.7", 170.0,  4.70, 1.313,  6.33,  4.60, 0.40, 0.425, 224.1, 15.0),
    ("DeepSeek-V3.2",140.8,  5.16, 1.187,  2.87,  2.07, 0.07, 0.441, 206.5, 12.8),
    ("GPT-5.5",      171.3,  4.64, 1.026,  9.07,  8.40, 0.27, 0.664, 186.3, 12.6),
    ("DeepSeek-V4Pro",180.4, 5.25, 1.391,  5.40,  4.20, 0.07, 0.513, 216.8, 12.3),
    ("ClaudeOpus4.6",140.7,  4.75, 1.170,  6.60,  5.67, 0.67, 0.469, 212.6, 13.8),
    ("ClaudeOpus4.7",183.5,  4.94, 0.979,  5.73,  5.00, 0.40, 0.436, 207.4, 12.9),
    ("Gemini3.1Pro", 155.4,  6.31, 1.793,  3.53,  2.53, 0.00, 0.604, 192.3,  8.7),
]

# ── Model-specific colors (from Claude-style HTML) ───
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

# ── Color palette ─────────────────────────────────────
PAPER       = "#f8f3ea"
INK         = "#2f2924"
MUTED       = "#776d63"
LINE        = "#e3d8c8"
ACCENT      = "#cc785c"
HIGHLIGHT_BG = "#efe3d3"

# ── Setup ─────────────────────────────────────────────
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Georgia", "DejaVu Serif", "Times New Roman"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"

fig, ax = plt.subplots(figsize=(14, 7.5), facecolor=PAPER)
ax.set_facecolor(PAPER)
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.5)
ax.axis("off")

# ── Title ─────────────────────────────────────────────
ax.text(0.5, 7.15, "Objective Features Summary by Source",
        fontsize=18, fontweight="bold", color=INK, va="center", ha="left",
        fontfamily="serif")
ax.text(0.5, 6.85, "Herbold-style computational linguistics framework · 9 features across 13 sources",
        fontsize=9, color=MUTED, va="center", ha="left")

# ── Layout calculation ────────────────────────────────
n_rows = len(rows)
n_cols = len(headers)

# Column widths: Source col wider
col_widths = [2.4] + [1.12] * (n_cols - 1)
x_starts = [0.0]
for w in col_widths[:-1]:
    x_starts.append(x_starts[-1] + w)
# Adjust x_starts to start at margin
x_offset = 0.5
x_starts = [x + x_offset for x in x_starts]
total_width = x_starts[-1] + col_widths[-1] - x_offset

row_height = 0.37
header_height = 0.45
y_top = 6.65
data_start_y = y_top - header_height

def get_y(row_idx):
    """Top y of data row (0-indexed)."""
    return data_start_y - (row_idx + 1) * row_height

# ── Draw header background ────────────────────────────
header_rect = FancyBboxPatch(
    (x_offset - 0.06, data_start_y - row_height + 0.01),
    total_width + 0.12, header_height,
    boxstyle="round,pad=0.06", facecolor=HIGHLIGHT_BG,
    edgecolor="none", zorder=1
)
ax.add_patch(header_rect)

# ── Draw header text ──────────────────────────────────
for ci, (header, x0, w) in enumerate(zip(headers, x_starts, col_widths)):
    align = "left" if ci == 0 else "center"
    x_pos = x0 if ci == 0 else x0 + w / 2
    ax.text(x_pos, data_start_y - row_height/2 + 0.02, header,
            fontsize=8, fontweight="bold", color=MUTED,
            va="center", ha=align, zorder=3)

# ── Draw data rows ────────────────────────────────────
for ri, row_data in enumerate(rows):
    source_name = row_data[0]
    y = get_y(ri)
    color = model_colors.get(source_name, INK)

    # Row background alternation
    if ri % 2 == 1:
        row_bg = FancyBboxPatch(
            (x_offset - 0.06, y - 0.01),
            total_width + 0.12, row_height,
            boxstyle="round,pad=0.04",
            facecolor="white", edgecolor="none",
            alpha=0.5, zorder=1
        )
        ax.add_patch(row_bg)

    # Bottom border line
    ax.plot([x_offset - 0.04, x_offset + total_width + 0.04],
            [y + 0.02, y + 0.02],
            color=LINE, linewidth=0.8, zorder=2)

    # Source name with color dot
    dot_x = x_starts[0] + 0.06
    dot_r = 0.09
    circle = plt.Circle((dot_x + dot_r, y + row_height/2), dot_r,
                        facecolor=color, edgecolor="none",
                        zorder=3, clip_on=False)
    ax.add_patch(circle)

    ax.text(x_starts[0] + 0.36, y + row_height/2, source_name,
            fontsize=8.5, color=INK, va="center", ha="left",
            fontweight="bold", zorder=3)

    # Numeric cells
    for ci, (val, x0, w) in enumerate(zip(row_data[1:], x_starts[1:], col_widths[1:]), start=1):
        x_pos = x0 + w / 2
        # Format
        if isinstance(val, float):
            if abs(val) < 0.01:
                text = "0.00"
            elif abs(val) < 1:
                text = f"{val:.3f}"
            elif abs(val) < 10:
                text = f"{val:.2f}"
            else:
                text = f"{val:.1f}"
        elif isinstance(val, int):
            text = str(val)
        else:
            text = str(val)

        ax.text(x_pos, y + row_height/2, text,
                fontsize=8, color=INK, va="center", ha="center",
                zorder=3)

# ── Footer note ───────────────────────────────────────
last_y = get_y(n_rows - 1)
footer_y = min(last_y - 0.55, 0.6)
ax.text(x_offset, footer_y,
        "Source data: outputs/objective-essays-wide/objective_features_summary_by_source.csv  ·  "
        "Computed via calc_objective_features.py following Herbold et al. (2023)",
        fontsize=7, color=MUTED, va="top", ha="left", style="italic")

# ── Legend for reference groups ───────────────────────
legend_y = footer_y - 0.3
ax.text(x_offset, legend_y, "Reference groups:",
        fontsize=7.5, fontweight="bold", color=MUTED, va="center", ha="left")

legend_items = [
    ("Student",       "#7c746a", "Human-written essays (baseline)"),
    ("ChatGPT-3",     "#10a37f", "ChatGPT-3.5 (original Herbold study)"),
    ("ChatGPT-4",     "#10a37f", "ChatGPT-4 (original Herbold study)"),
]
for i, (label, lc, desc) in enumerate(legend_items):
    lx = x_offset + i * 4.2
    circle = plt.Circle((lx + 0.08, legend_y), 0.07,
                        facecolor=lc, edgecolor="none",
                        zorder=3, clip_on=False)
    ax.add_patch(circle)
    ax.text(lx + 0.28, legend_y, f"{label}  ",
            fontsize=7.5, fontweight="bold", color=INK, va="center", ha="left")

# ── Save ──────────────────────────────────────────────
out_dir = pathlib.Path("outputs/research-figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "objective_features_table.png"
fig.savefig(str(out_path), dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor(), edgecolor="none",
            pad_inches=0.3)
plt.close(fig)
print(f"Saved → {out_path}")