#!/usr/bin/env python3
"""Builds a dual-panel figure summarising ancillary drivers for tile subsets.

For each covariate the script reports both the percent difference relative to the
parity bucket and the Spearman correlation (ρ) with Δaccuracy. Outputs a figure
with two rows (embedding wins, harmonic wins) and two columns (percent shift,
Spearman ρ) saved under ``article/images/embedding_harmonic_driver_dual.png``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import numpy as np
import pandas as pd

from src.utils import apply_science_style

apply_science_style()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "analysis_context" / "tile_heterogeneity" / "subset_analysis"
OUTPUT_PATH = ROOT / "article" / "images" / "embedding_harmonic_driver_dual.png"

# Ordered list of covariates to display with human-readable labels
FEATURES = [
    ("shannon_diversity", "Shannon diversity"),
    ("deciduous_ratio", "Deciduous share"),
    ("era5_precip_total_mm_2023", "ERA5 rainfall 2023"),
    ("era5_temp_range_degC_2023", "ERA5 temperature range 2023"),
]


def _load_driver_stats(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}_driver_stats.csv")
    df.set_index("covariate", inplace=True)
    return df


def _percent_diff(row: pd.Series) -> float:
    ref = row["reference_mean"]
    if pd.isna(ref) or np.isclose(ref, 0.0):
        return np.nan
    return 100.0 * row["mean_difference"] / ref


def _prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for cov, label in FEATURES:
        if cov not in df.index:
            raise KeyError(f"Covariate {cov} missing from driver stats")
        row = df.loc[cov]
        records.append(
            {
                "covariate": cov,
                "label": label,
                "percent_diff": _percent_diff(row),
                "rho": row["spearman_delta_accuracy"],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    # Load data
    embedding_stats = _load_driver_stats("embedding_advantage")
    harmonic_stats = _load_driver_stats("harmonics_win")

    summary = json.loads((DATA_DIR / "subset_driver_summary.json").read_text())
    embed_count = summary.get("embedding_advantage", {}).get("tiles", len(embedding_stats))
    harmonic_count = summary.get("harmonics_win", {}).get("tiles", len(harmonic_stats))

    embed_rows = _prepare_rows(embedding_stats)
    harm_rows = _prepare_rows(harmonic_stats)

    # Figure layout
    with plt.style.context(["science", "no-latex"]):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex=False, sharey=True)

        rows = [
            (embed_rows, embed_count, "Embedding win", "#1b9e77"),
            (harm_rows, harmonic_count, "Harmonic win", "#d95f02"),
        ]

        for row_idx, (data, count, title, color) in enumerate(rows):
            ax = axes[row_idx]
            ax2 = ax.twiny()
            y = np.arange(len(FEATURES))

            # Percent difference bars (primary axis)
            ax.barh(y, data["percent_diff"], height=0.4, color=color, alpha=0.8, label="%Δ vs parity")
            ax.axvline(0, color="0.4", linewidth=0.8)
            ax.set_xlim(-60, 60)
            ax.set_yticks(y)
            ax.set_yticklabels(data["label"])
            ax.set_xlabel("Relative difference vs parity (%)")
            ax.set_title(f"{title} (N={count})", loc="left", fontsize=11, fontweight="bold")
            for val, yy in zip(data["percent_diff"], y):
                ax.text(val + (2 if val >= 0 else -2), yy, f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=9)

            # Spearman rho bars (secondary axis)
            ax2.barh(y - 0.4, data["rho"], height=0.3, color="#4c566a", alpha=0.85, label="Spearman rho")
            ax2.set_xlim(-0.6, 0.6)
            ax2.axvline(0, color="0.5", linewidth=0.6, linestyle="--")
            ax2.set_xlabel("Spearman rho with accuracy delta")
            ax2.set_yticks([])
            for val, yy in zip(data["rho"], y):
                ax2.text(val + (0.05 if val >= 0 else -0.05), yy - 0.4, f"rho={val:+.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

        fig.tight_layout(h_pad=1.5)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_PATH, dpi=300)


if __name__ == "__main__":
    main()
