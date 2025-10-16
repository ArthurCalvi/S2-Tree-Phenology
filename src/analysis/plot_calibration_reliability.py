#!/usr/bin/env python3
"""Plot reliability diagrams for embedding and harmonic RF models."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from src.utils import apply_science_style
apply_science_style()
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from src.utils import science_style


def load_curve(curve_path: Path) -> pd.DataFrame:
    df = pd.read_csv(curve_path)
    df['mid'] = (df['bin_lower'] + df['bin_upper']) / 2.0
    return df


def load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text())


def plot_panel(ax, curve: pd.DataFrame, metrics: dict, title: str, color: str) -> None:
    ax.plot([0, 1], [0, 1], linestyle='--', color='0.7', linewidth=1.0)

    bars = ax.bar(curve['mid'], curve['fraction'], width=0.08, color=color, alpha=0.18, align='center')
    mean_line, = ax.plot(curve['mid'], curve['mean_confidence'], marker='o', color=color, linewidth=2.0)
    empirical_line, = ax.plot(curve['mid'], curve['empirical_probability'], marker='s', color='black', linewidth=2.0)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted probability (Evergreen)')
    ax.set_ylabel('Fraction / Accuracy (Evergreen)')
    ax.set_title(f"{title}\nECE={metrics['ece']:.03f}, MCE={metrics['mce']:.03f}")
    ax.grid(alpha=0.2)

    handles = [
        Line2D([], [], linestyle='--', color='0.7', label='Ideal'),
        Line2D([], [], marker='o', color=color, linewidth=2.0, label='Mean confidence'),
        Line2D([], [], marker='s', color='black', linewidth=2.0, label='Empirical accuracy'),
        Patch(facecolor=color, alpha=0.18, label='Sample fraction')
    ]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot calibration reliability diagrams for embeddings and harmonics.')
    parser.add_argument('--embeddings-dir', required=True, type=Path, help='Directory with embeddings calibration outputs.')
    parser.add_argument('--harmonic-dir', required=True, type=Path, help='Directory with harmonic calibration outputs.')
    parser.add_argument('--output', required=True, type=Path, help='Output image path (PNG).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedding_curve = load_curve(args.embeddings_dir / 'reliability_curve.csv')
    harmonic_curve = load_curve(args.harmonic_dir / 'reliability_curve.csv')

    embedding_metrics = load_metrics(args.embeddings_dir / 'calibration_metrics.json')
    harmonic_metrics = load_metrics(args.harmonic_dir / 'calibration_metrics.json')

    with science_style():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)

        plot_panel(axes[0], embedding_curve, embedding_metrics, 'Embedding RF (Top-14)', color='#1f77b4')
        plot_panel(axes[1], harmonic_curve, harmonic_metrics, 'Harmonic RF (Top-14)', color='#ff7f0e')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=300)


if __name__ == '__main__':
    main()
