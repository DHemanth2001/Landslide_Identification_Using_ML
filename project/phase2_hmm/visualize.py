"""
Visualisation functions for HMM parameters and predictions.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.plot_utils import _save_or_show


def plot_transition_matrix(
    transmat: np.ndarray,
    state_labels: list = None,
    save_path: str = None,
) -> None:
    """Heatmap of HMM transition probabilities."""
    if state_labels is None:
        state_labels = [config.HMM_STATE_LABELS.get(i, f"S{i}") for i in range(len(transmat))]

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=state_labels,
        yticklabels=state_labels,
        linewidths=0.5,
    )
    plt.title("HMM Transition Matrix")
    plt.ylabel("From State")
    plt.xlabel("To State")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_emission_probabilities(
    emissionprob: np.ndarray,
    state_labels: list = None,
    observation_labels: list = None,
    save_path: str = None,
) -> None:
    """
    Heatmap showing which observation types each hidden state tends to emit.
    """
    if state_labels is None:
        state_labels = [config.HMM_STATE_LABELS.get(i, f"S{i}") for i in range(len(emissionprob))]
    if observation_labels is None:
        observation_labels = [f"Obs {i}" for i in range(emissionprob.shape[1])]

    plt.figure(figsize=(max(8, emissionprob.shape[1] * 1.2), 5))
    sns.heatmap(
        emissionprob,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=observation_labels,
        yticklabels=state_labels,
        linewidths=0.5,
    )
    plt.title("HMM Emission Probabilities")
    plt.ylabel("Hidden State")
    plt.xlabel("Observed Disaster Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_or_show(save_path)


def plot_state_sequence(
    states: np.ndarray,
    labels: list = None,
    title: str = "Hidden State Sequence",
    save_path: str = None,
) -> None:
    """Timeline plot of hidden states over observed events."""
    if labels is None:
        labels = [config.HMM_STATE_LABELS.get(int(s), str(s)) for s in states]

    fig, ax = plt.subplots(figsize=(max(10, len(states) * 0.6), 3))
    ax.step(range(len(states)), states, where="post", linewidth=2, color="steelblue")
    ax.scatter(range(len(states)), states, zorder=5, s=60, color="darkorange")
    ax.set_yticks(sorted(set(int(s) for s in states)))
    ax.set_yticklabels(
        [config.HMM_STATE_LABELS.get(i, f"S{i}") for i in sorted(set(int(s) for s in states))]
    )
    ax.set_xlabel("Event Index")
    ax.set_ylabel("Hidden State")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_probability_distribution(
    probabilities: list,
    title: str = "Occurrence Probability Distribution",
    save_path: str = None,
) -> None:
    """Histogram + KDE of occurrence probabilities for all events."""
    plt.figure(figsize=(7, 4))
    plt.hist(probabilities, bins=15, color="steelblue", edgecolor="white", alpha=0.7, density=True)

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(probabilities)
    x = np.linspace(0, 1, 200)
    plt.plot(x, kde(x), color="darkorange", linewidth=2, label="KDE")

    plt.xlabel("Occurrence Probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_start_probabilities(
    startprob: np.ndarray,
    state_labels: list = None,
    save_path: str = None,
) -> None:
    """Bar chart of HMM initial state probabilities."""
    if state_labels is None:
        state_labels = [config.HMM_STATE_LABELS.get(i, f"S{i}") for i in range(len(startprob))]

    plt.figure(figsize=(6, 4))
    plt.bar(state_labels, startprob, color="steelblue", edgecolor="white")
    plt.ylabel("Probability")
    plt.title("Initial State Probabilities")
    plt.ylim(0, 1)
    for i, p in enumerate(startprob):
        plt.text(i, p + 0.02, f"{p:.3f}", ha="center")
    plt.tight_layout()
    _save_or_show(save_path)
