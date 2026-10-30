"""
Visualization utilities — attention heatmaps and metric charts.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FixedLocator
import seaborn as sns
import pandas as pd
import warnings

# Suppress Telugu glyph-missing warnings — cosmetic only, text still renders
warnings.filterwarnings("ignore", message="Glyph.*missing from font")


def plot_attention_heatmap(
    attention: np.ndarray,
    src_tokens: List[str],
    tgt_tokens: List[str],
    title: str = "Bahdanau Attention Heatmap",
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """
    Render a Bahdanau attention heatmap.

    Args:
        attention:   2-D array of shape [T_tgt, T_src] — attention weights.
        src_tokens:  English source tokens (columns).
        tgt_tokens:  Telugu output tokens (rows).
        title:       Figure title.
        figsize:     Override figure size. Auto-computed if None.
    Returns:
        matplotlib Figure object.
    """
    if attention.ndim != 2:
        raise ValueError(f"Expected 2-D attention, got shape {attention.shape}")

    n_rows, n_cols = attention.shape
    if figsize is None:
        w = max(8, n_cols * 0.7)
        h = max(4, n_rows * 0.55)
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    # Colour map — perceptually uniform, high contrast on dark background
    cmap = sns.color_palette("rocket", as_cmap=True)
    im = ax.imshow(attention, aspect="auto", cmap=cmap, interpolation="nearest")

    # Axes labels
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=40, ha="right", fontsize=9, color="white")
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=9, color="white")

    ax.set_xlabel("English input tokens", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Generated Telugu tokens", color="#94a3b8", fontsize=10)
    ax.set_title(title, color="white", fontsize=13, pad=12, fontweight="bold")

    # Tick colour
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Attention weight", color="#94a3b8", fontsize=9)

    fig.tight_layout()
    return fig


def plot_metric_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Render a grouped bar chart comparing BLEU and chrF for both models.

    Args:
        metrics_df: DataFrame with columns [Model, BLEU, chrF].
    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#0f172a")

    palette = ["#6366f1", "#10b981"]   # indigo for GRU, emerald for NLLB
    metric_configs = [
        ("BLEU", axes[0], "BLEU Score (higher = better)"),
        ("chrF", axes[1], "chrF Score (higher = better)"),
    ]

    for metric, ax, ylabel in metric_configs:
        ax.set_facecolor("#1e293b")
        bars = ax.bar(
            metrics_df["Model"],
            metrics_df[metric],
            color=palette,
            width=0.5,
            edgecolor="#334155",
            linewidth=0.8,
        )
        # Value labels
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.3,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=10)
        ax.set_ylim(0, metrics_df[metric].max() * 1.3)
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.set_major_locator(FixedLocator(range(len(metrics_df))))
        ax.set_xticklabels(
            [m.replace(" ", "\n") for m in metrics_df["Model"]],
            fontsize=8,
            color="white",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.yaxis.label.set_color("#94a3b8")

    fig.suptitle(
        "Model Evaluation Comparison",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_score_radar(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Render a horizontal comparison bar for quick visual ranking.

    Args:
        metrics_df: DataFrame with columns [Model, BLEU, chrF].
    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    models = metrics_df["Model"].tolist()
    bleu = metrics_df["BLEU"].tolist()
    chrf = metrics_df["chrF"].tolist()

    x = np.arange(len(models))
    width = 0.35

    b1 = ax.barh(x - width / 2, bleu, width, label="BLEU", color="#6366f1", alpha=0.85)
    b2 = ax.barh(x + width / 2, chrf, width, label="chrF", color="#10b981", alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels([m.replace(" ", "\n") for m in models], color="white", fontsize=8)
    ax.tick_params(colors="white")
    ax.set_xlabel("Score", color="#94a3b8")
    ax.set_title("BLEU vs chrF by Model", color="white", fontweight="bold")
    ax.legend(facecolor="#1e293b", labelcolor="white", edgecolor="#334155")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    fig.tight_layout()
    return fig
