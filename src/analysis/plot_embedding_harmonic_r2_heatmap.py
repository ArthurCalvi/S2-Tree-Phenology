"""Plot ridge R² scores (harmonic → embedding) as stacked bar panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from src.utils import apply_science_style


def prettify_feature_name(name: str) -> str:
    parts = name.split("_")
    if not parts:
        return name
    idx = parts[0].upper()
    remainder = parts[1:]
    replacements = {
        "amplitude": "amp",
        "phase": "phase",
        "offset": "offset",
        "var": "var",
        "residual": "res",
        "h1": "h1",
        "h2": "h2",
    }
    tokens = [idx]
    for chunk in remainder:
        tokens.append(replacements.get(chunk, chunk))
    return " ".join(tokens)


def load_correlation_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Correlation summary not found: {path}")
    corr_df = pd.read_csv(path)
    agg = (
        corr_df.groupby(["embedding", "best_feature"])
        .agg(
            corr_abs_mean=("corr_abs", "mean"),
            corr_abs_max=("corr_abs", "max"),
            n_regions=("eco_region", "nunique"),
            corr_signed_mean=("corr_signed", "mean"),
        )
        .reset_index()
    )
    top = (
        agg.sort_values(["embedding", "corr_abs_mean"], ascending=[True, False])
        .groupby("embedding", as_index=False)
        .first()
    )
    top["feature_pretty"] = top["best_feature"].apply(prettify_feature_name)
    return top.set_index("embedding")


def fetch_region_order(region_summary: Path, available: list[str]) -> list[str]:
    if region_summary.exists():
        df = pd.read_csv(region_summary)
        ordered = [eco for eco in df["eco_region"].tolist() if eco in available]
        remaining = [eco for eco in available if eco not in ordered]
        return ordered + sorted(remaining)
    return sorted(available)


def plot_bar_panels(
    r2_df: pd.DataFrame,
    corr_summary: pd.DataFrame,
    embedding_cols: list[str],
    output: Path,
) -> None:
    apply_science_style()

    values = r2_df.to_numpy(dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    span = max(abs(vmin), abs(vmax))
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)
    cmap = cm.get_cmap("RdBu_r")

    eco_regions = r2_df.index.tolist()
    n_regions = len(eco_regions)
    fig_height = 4 + 0.9 * n_regions

    heat_h = [1.0] * n_regions
    spacer_h = 0.35
    corr_h = 0.7
    emb_label_h = 0.8
    harmonic_label_h = 1.0
    fig, axes = plt.subplots(
        n_regions + 3,
        1,
        figsize=(12.7, fig_height + 3.2),
        sharex=True,
        gridspec_kw={
            "height_ratios": heat_h + [emb_label_h, corr_h, harmonic_label_h],
            "hspace": 0.18,
        },
    )

    x_positions = np.arange(len(embedding_cols))
    y_min = float(np.floor(vmin * 10) / 10) - 0.05
    y_max = float(np.ceil(vmax * 10) / 10) + 0.05

    for idx_region, (ax, eco) in enumerate(zip(axes[:n_regions], eco_regions)):
        vals = r2_df.loc[eco].to_numpy(dtype=float)
        bar_colors = [cmap(norm(v)) if np.isfinite(v) else (0.8, 0.8, 0.8, 1.0) for v in vals]
        ax.bar(x_positions, vals, color=bar_colors, width=0.75)
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([y_min, 0.0, y_max])
        ax.tick_params(axis="y", labelsize=9)
        ax.set_ylabel(eco, rotation=0, ha="right", va="center", fontsize=10, labelpad=30)
        ax.tick_params(axis="x", labelbottom=False)

    label_ax_emb = axes[n_regions]
    label_ax_emb.set_xlim(-0.5, len(embedding_cols) - 0.5)
    label_ax_emb.set_ylim(0, 1)
    label_ax_emb.axis("off")
    for idx, emb in enumerate(embedding_cols):
        label_ax_emb.text(
            x_positions[idx],
            0.5,
            emb.replace("embedding_", ""),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    label_ax_emb.text(
        0.5 * (len(embedding_cols) - 1),
        -0.05,
        "Embedding dimension",
        fontsize=10,
        rotation=0,
        va="top",
        ha="center",
        fontweight="bold",
    )

    # Bottom two axes: pearson correlations + labels
    corr_vals = []
    cues = []
    for emb in embedding_cols:
        if emb in corr_summary.index:
            info = corr_summary.loc[emb]
            corr_vals.append(info["corr_signed_mean"])
            cues.append(info["feature_pretty"])
        else:
            corr_vals.append(np.nan)
            cues.append("n/a")

    corr_ax = axes[-2]
    corr_ax.axhline(0, color="0.5", linewidth=0.8)
    corr_ax.set_xlim(-0.5, len(embedding_cols) - 0.5)
    corr_ax.set_ylim(-1, 1)
    corr_ax.set_ylabel("Pearson r", fontsize=10)
    corr_ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    corr_ax.set_xticks(x_positions)
    corr_ax.tick_params(axis="x", labelbottom=False)

    bar_colors_corr = ["#4c72b0" if val >= 0 else "#c44e52" for val in corr_vals]
    corr_ax.bar(
        x_positions,
        corr_vals,
        width=0.6,
        color=bar_colors_corr,
        alpha=0.8,
    )
    corr_ax.set_title(
        "Dominant harmonic cue per embedding (signed mean Pearson r across eco-regions)",
        fontsize=11,
        pad=12,
    )

    label_ax = axes[-1]
    label_ax.set_xlim(-0.5, len(embedding_cols) - 0.5)
    label_ax.set_ylim(0, 1)
    label_ax.axis("off")
    for idx, cue in enumerate(cues):
        label_ax.text(
            x_positions[idx],
            0.15,
            cue if cue != "n/a" else "",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=90,
        )
    label_ax.text(
        0.5 * (len(embedding_cols) - 1),
        0.0,
        "Closest harmonic descriptor",
        fontsize=10,
        rotation=0,
        va="bottom",
        ha="center",
        fontweight="bold",
    )

    axes[0].set_title(
        "How much of each embedding dimension can harmonics explain? (Ridge $R^2$)",
        fontsize=13,
        pad=14,
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes[:-1], orientation="vertical", fraction=0.015, pad=0.01)
    cbar.ax.set_ylabel("$R^2$", fontsize=11)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-embedding R² heatmap.")
    parser.add_argument(
        "--r2-data",
        type=Path,
        default=Path("results/analysis_similarity/embedding_r2_per_region.csv"),
        help="CSV with per-embedding R² per eco-region (output of plot_embedding_harmonic_r2.py).",
    )
    parser.add_argument(
        "--correlation-summary",
        type=Path,
        default=Path("results/analysis_similarity/closest_harmonic_region.csv"),
        help="CSV listing dominant harmonic features per embedding and region.",
    )
    parser.add_argument(
        "--region-summary",
        type=Path,
        default=Path("results/analysis_similarity/summary/summary_region.csv"),
        help="Optional CSV to enforce eco-region ordering.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("images/embedding_harmonic_r2_heatmap.png"),
        help="Destination for the figure.",
    )
    args = parser.parse_args()

    if not args.r2_data.exists():
        raise FileNotFoundError(
            f"R² CSV not found: {args.r2_data}. Run plot_embedding_harmonic_r2.py first."
        )
    r2_df = pd.read_csv(args.r2_data).set_index("eco_region")
    region_order = fetch_region_order(args.region_summary, r2_df.index.tolist())
    r2_df = r2_df.loc[region_order]

    embedding_cols = [col for col in r2_df.columns if col.startswith("embedding_")]
    corr_summary = load_correlation_summary(args.correlation_summary)

    plot_bar_panels(r2_df, corr_summary, embedding_cols, args.out)


if __name__ == "__main__":
    main()
