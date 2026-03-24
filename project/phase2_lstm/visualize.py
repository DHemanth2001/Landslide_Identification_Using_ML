"""
Visualization functions for Bi-LSTM + Attention model.

Includes:
  - Attention heatmap: shows which past events the model focuses on
  - Forecast probability bars: multi-step type predictions
  - Training curves: loss and accuracy over epochs
  - Type distribution: per-country predicted type probabilities
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.plot_utils import _save_or_show


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    title: str = "Temporal Attention Weights",
    head_idx: int = 0,
    save_path: str = None,
) -> None:
    """
    Plot attention heatmap showing which past events the model focuses on.

    Args:
        attention_weights: (num_heads, T, T) attention weight matrix.
        title:             Figure title.
        head_idx:          Which attention head to visualize (or -1 for average).
        save_path:         If provided, save the figure.
    """
    if head_idx == -1:
        # Average across all heads
        attn = attention_weights.mean(axis=0)
        title += " (averaged across heads)"
    else:
        attn = attention_weights[head_idx]
        title += f" (head {head_idx})"

    T = attn.shape[0]
    fig_size = max(6, T * 0.3)

    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        attn,
        cmap="YlOrRd",
        vmin=0,
        vmax=attn.max(),
        xticklabels=range(1, T + 1),
        yticklabels=range(1, T + 1),
        linewidths=0.1,
    )
    plt.xlabel("Key (Past Event Index)")
    plt.ylabel("Query (Event Index)")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_attention_over_time(
    attention_weights: np.ndarray,
    event_labels: list = None,
    title: str = "Attention Focus for Final Prediction",
    save_path: str = None,
) -> None:
    """
    Bar chart showing how much attention the final prediction gives to each past event.
    This is the most interpretable visualization — shows WHY the model predicted a type.
    """
    # Average across heads, take the last row (prediction attends to past)
    attn_avg = attention_weights.mean(axis=0)  # (T, T)
    final_attn = attn_avg[-1, :]  # (T,) — what the last step attends to

    T = len(final_attn)
    if event_labels is None:
        event_labels = [f"Event {i + 1}" for i in range(T)]

    plt.figure(figsize=(max(8, T * 0.4), 4))
    colors = plt.cm.YlOrRd(final_attn / final_attn.max())
    plt.bar(range(T), final_attn, color=colors, edgecolor="gray", linewidth=0.5)
    plt.xticks(range(T), event_labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Attention Weight")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_forecast_probabilities(
    forecast: list,
    title: str = "Multi-Step Forecast",
    save_path: str = None,
) -> None:
    """
    Stacked bar chart showing type probabilities for each forecast step.
    """
    n_steps = len(forecast)
    type_names = list(forecast[0]["all_probs"].keys())
    n_types = len(type_names)

    # Build matrix
    prob_matrix = np.zeros((n_steps, n_types))
    for i, f in enumerate(forecast):
        for j, tn in enumerate(type_names):
            prob_matrix[i, j] = f["all_probs"].get(tn, 0)

    colors = plt.cm.Set3(np.linspace(0, 1, n_types))
    x = range(n_steps)

    plt.figure(figsize=(max(6, n_steps * 2), 5))
    bottom = np.zeros(n_steps)
    for j in range(n_types):
        plt.bar(x, prob_matrix[:, j], bottom=bottom, label=type_names[j],
                color=colors[j], edgecolor="white", linewidth=0.5)
        bottom += prob_matrix[:, j]

    plt.xticks(x, [f"Step {f['step']}" for f in forecast])
    plt.ylabel("Probability")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_type_probabilities(
    type_probs: dict,
    title: str = "Predicted Landslide Type Distribution",
    save_path: str = None,
) -> None:
    """Horizontal bar chart of type probabilities."""
    names = list(type_probs.keys())
    probs = [type_probs[n] for n in names]

    # Sort by probability
    sorted_pairs = sorted(zip(probs, names), reverse=True)
    probs, names = zip(*sorted_pairs)

    colors = plt.cm.RdYlGn_r(np.array(probs))

    plt.figure(figsize=(8, max(3, len(names) * 0.5)))
    bars = plt.barh(range(len(names)), probs, color=colors, edgecolor="gray", linewidth=0.5)
    plt.yticks(range(len(names)), names)
    plt.xlabel("Probability")
    plt.title(title)
    plt.xlim(0, 1.05)

    for bar, p in zip(bars, probs):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{p * 100:.1f}%", va="center", fontsize=9)

    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_multi_head_attention(
    attention_weights: np.ndarray,
    title: str = "Multi-Head Attention Patterns",
    save_path: str = None,
) -> None:
    """Plot attention patterns for all heads side by side."""
    n_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))

    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        attn = attention_weights[h]
        sns.heatmap(
            attn, ax=axes[h], cmap="YlOrRd",
            xticklabels=False, yticklabels=False,
            cbar=h == n_heads - 1,
        )
        axes[h].set_title(f"Head {h + 1}")
        axes[h].set_xlabel("Key")
        if h == 0:
            axes[h].set_ylabel("Query")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path)
