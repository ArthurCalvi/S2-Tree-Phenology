#!/usr/bin/env python3
"""
Compare tile-level performance deltas between embeddings and harmonic RF models.

The script expects out-of-fold prediction parquet files produced by the training
pipelines (one for embeddings, one for harmonics). It aggregates metrics per tile,
computes Δ accuracy / Δ F1 (macro) and joins ancillary covariates from
`results/analysis_context/tile_context_features.parquet` (plus optional coherence
metrics). Outputs:

- tile_metrics_delta.parquet: per tile metrics for both models with delta columns.
- eco_region_summary.csv: aggregate deltas + win counts per eco-region.
- correlation_summary.csv: Pearson/Spearman correlations between Δ metrics and
  selected contextual covariates (elevation, climate, soils, class ratios, Shannon).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repo root on path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DISPLAY_NAMES = {
    "elevation_mean": "Mean elevation (m)",
    "slope_mean": "Mean slope (°)",
    "era5_temp_mean_degC_2023": "ERA5 mean temp 2023 (°C)",
    "era5_temp_range_degC_2023": "ERA5 temp range 2023 (°C)",
    "era5_precip_total_mm_2023": "ERA5 total precip 2023 (mm)",
    "era5_soil_moisture_mean_2023": "ERA5 soil moisture 2023 (m³/m³)",
    "soil_organic_carbon_mean_gkg": "Soil organic carbon (g/kg)",
    "soil_clay_mean_gkg": "Soil clay fraction (g/kg)",
    "soil_sand_mean_gkg": "Soil sand fraction (g/kg)",
    "deciduous_ratio": "Deciduous share",
    "shannon_diversity": "Shannon diversity",
}

# Colour palette and typography (consistent across figures)
PALETTE = {
    "Emb Δ>+1σ": "#2A9D8F",
    "Parity |Δ|<1σ": "#264653",
    "Harm Δ<-1σ": "#E76F51",
}
BACKGROUND_COLOR = "#f7f7f7"
GRID_COLOR = "#d0d0d0"
FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 11
DPI = 250


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 1,
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.facecolor": "white",
            "axes.edgecolor": "#555555",
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "figure.facecolor": "white",
        }
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse tile-level performance differences between embeddings and harmonics."
    )
    parser.add_argument(
        "--embeddings-predictions",
        required=True,
        type=Path,
        help="Path to embeddings CV predictions parquet (columns: tile_id, y_true, y_pred, eco_region, ...).",
    )
    parser.add_argument(
        "--harmonics-predictions",
        required=True,
        type=Path,
        help="Path to harmonics CV predictions parquet (same schema).",
    )
    parser.add_argument(
        "--context-parquet",
        required=True,
        type=Path,
        help="Tile context parquet (elevation, ERA5, soils, class ratios, Shannon, eco_region).",
    )
    parser.add_argument(
        "--coherence-parquet",
        type=Path,
        help="Optional coherence metrics parquet with tile_index column to merge.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where analysis artefacts will be written.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{context} missing required columns: {missing}")


def aggregate_tile_metrics(df: pd.DataFrame, model_tag: str) -> pd.DataFrame:
    """Aggregate classification metrics per tile."""
    ensure_columns(df, ["tile_id", "y_true", "y_pred", "eco_region"], context=model_tag)

    rows = []
    for tile_id, group in df.groupby("tile_id"):
        metrics = compute_binary_metrics(group["y_true"].values, group["y_pred"].values)
        row = {
            "tile_id": int(tile_id),
            "eco_region": group["eco_region"].iloc[0],
            "n_samples": int(len(group)),
            f"accuracy_{model_tag}": metrics["accuracy"],
            f"f1_macro_{model_tag}": metrics["f1_macro"],
            f"f1_weighted_{model_tag}": metrics["f1_weighted"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_delta_metrics(
    emb_df: pd.DataFrame, harm_df: pd.DataFrame
) -> pd.DataFrame:
    merged = emb_df.merge(
        harm_df,
        on=["tile_id", "eco_region"],
        suffixes=("_embeddings", "_harmonics"),
        how="inner",
    )
    merged["delta_accuracy"] = (
        merged["accuracy_embeddings"] - merged["accuracy_harmonics"]
    )
    merged["delta_f1_macro"] = (
        merged["f1_macro_embeddings"] - merged["f1_macro_harmonics"]
    )
    std_delta = merged["delta_accuracy"].std()
    if std_delta == 0:
        merged["delta_z"] = 0.0
    else:
        merged["delta_z"] = merged["delta_accuracy"] / std_delta
    merged["embedding_advantage"] = merged["delta_z"] >= 1.0
    merged["harmonics_win"] = merged["delta_z"] <= -1.0
    merged["performance_bucket"] = np.select(
        [
            merged["embedding_advantage"],
            merged["harmonics_win"],
        ],
        ["embedding_advantage", "harmonics_win"],
        default="rough_parity",
    )
    bucket_labels = {
        "embedding_advantage": "Emb Δ>+1σ",
        "harmonics_win": "Harm Δ<-1σ",
        "rough_parity": "Parity |Δ|<1σ",
    }
    merged["performance_bucket_label"] = merged["performance_bucket"].map(bucket_labels)
    return merged


def build_correlation_summary(
    df: pd.DataFrame,
    target_cols: Iterable[str],
    feature_exprs: Dict[str, pd.Series],
) -> pd.DataFrame:
    records = []
    for target in target_cols:
        target_series = df[target]
        for feat_name, feat_values in feature_exprs.items():
            aligned = pd.DataFrame(
                {"target": target_series, "feature": feat_values}
            ).dropna()
            if len(aligned) < 3:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = aligned["target"].corr(aligned["feature"], method="pearson")
                spearman = aligned["target"].corr(
                    aligned["feature"], method="spearman"
                )
            records.append(
                {
                    "target": target,
                    "feature": feat_name,
                    "pearson": float(pearson) if not np.isnan(pearson) else np.nan,
                    "spearman": float(spearman) if not np.isnan(spearman) else np.nan,
                    "n": len(aligned),
                }
            )
    return pd.DataFrame(records)


def eco_region_summary(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("eco_region")
    summary = group.agg(
        tiles=("tile_id", "count"),
        mean_delta_accuracy=("delta_accuracy", "mean"),
        median_delta_accuracy=("delta_accuracy", "median"),
        std_delta_accuracy=("delta_accuracy", "std"),
        mean_delta_f1=("delta_f1_macro", "mean"),
        harmonics_win_tiles=("harmonics_win", "sum"),
        embedding_advantage_tiles=("embedding_advantage", "sum"),
    ).reset_index()
    summary["harmonics_win_share"] = (
        summary["harmonics_win_tiles"] / summary["tiles"]
    )
    summary["embedding_advantage_share"] = (
        summary["embedding_advantage_tiles"] / summary["tiles"]
    )
    return summary


def summarise_by_bucket(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    records = []
    for bucket, sub in df.groupby("performance_bucket"):
        record = {
            "performance_bucket": bucket,
            "performance_bucket_label": sub["performance_bucket_label"].iloc[0],
            "tiles": int(len(sub)),
            "mean_delta_accuracy": sub["delta_accuracy"].mean(),
            "median_delta_accuracy": sub["delta_accuracy"].median(),
            "std_delta_accuracy": sub["delta_accuracy"].std(),
            "mean_delta_f1_macro": sub["delta_f1_macro"].mean(),
        }
        for col in feature_cols:
            if col in sub.columns:
                record[f"median_{col}"] = sub[col].median()
                record[f"p25_{col}"] = sub[col].quantile(0.25)
                record[f"p75_{col}"] = sub[col].quantile(0.75)
        records.append(record)
    return pd.DataFrame(records)


def add_context_to_eco_summary(summary: pd.DataFrame, df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    for col in feature_cols:
        if col in df.columns:
            summary[f"median_{col}"] = df.groupby("eco_region")[col].median().values
            summary[f"p25_{col}"] = df.groupby("eco_region")[col].quantile(0.25).values
            summary[f"p75_{col}"] = df.groupby("eco_region")[col].quantile(0.75).values
    return summary


def create_scatter_plots(df: pd.DataFrame, feature_cols: Iterable[str], output_path: Path) -> None:
    features = [col for col in feature_cols if col in df.columns]
    if not features:
        return

    apply_plot_style()
    cols = 2
    rows = math.ceil(len(features) / cols)
    category_order = ["Emb Δ>+1σ", "Parity |Δ|<1σ", "Harm Δ<-1σ"]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 4.2), squeeze=False)

    y_min = float(df["delta_accuracy"].min())
    y_max = float(df["delta_accuracy"].max())
    padding = 0.02
    y_lower = min(-0.2, y_min - padding)
    y_upper = max(0.2, y_max + padding)

    for ax, feat in zip(axes.flat, features):
        display_name = DISPLAY_NAMES.get(feat, feat.replace("_", " "))
        subset = df[[feat, "delta_accuracy", "performance_bucket_label"]].dropna()
        pearson = subset["delta_accuracy"].corr(subset[feat], method="pearson")
        spearman = subset["delta_accuracy"].corr(subset[feat], method="spearman")

        for category in category_order:
            mask = subset["performance_bucket_label"] == category
            if not mask.any():
                continue
            ax.scatter(
                subset.loc[mask, feat],
                subset.loc[mask, "delta_accuracy"],
                s=34,
                alpha=0.72,
                color=PALETTE[category],
                edgecolor="white",
                linewidth=0.4,
                label=category if feat == features[0] else "_nolegend_",
            )
        ax.axhline(0.0, color="#444", linestyle="--", linewidth=0.8)
        ax.set_xlabel(display_name)
        ax.set_ylabel("Δ accuracy (Embeddings − Harmonics)")
        ax.set_ylim(y_lower, y_upper)
        title = display_name
        if not np.isnan(pearson):
            title = f"{display_name}\nPearson r={pearson:.2f}, Spearman ρ={spearman:.2f}"
        ax.set_title(title, fontsize=10)
        ax.set_facecolor(BACKGROUND_COLOR)

    for ax in axes.flat[len(features):]:
        ax.set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Tile-level Δ accuracy vs. contextual drivers", fontsize=BASE_FONT_SIZE + 4)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)

def create_boxplots(df: pd.DataFrame, feature_cols: Iterable[str], output_path: Path) -> None:
    features = [col for col in feature_cols if col in df.columns]
    if not features:
        return

    apply_plot_style()
    cols = 2
    rows = math.ceil(len(features) / cols)
    ordered_labels = ["Harm Δ<-1σ", "Parity |Δ|<1σ", "Emb Δ>+1σ"]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 4.2), squeeze=False)
    for ax, feat in zip(axes.flat, features):
        display_name = DISPLAY_NAMES.get(feat, feat.replace("_", " "))
        data = [
            df.loc[df["performance_bucket_label"] == label, feat].dropna()
            for label in ordered_labels
        ]
        if not any(len(d) for d in data):
            ax.set_visible(False)
            continue

        positions = np.arange(1, len(ordered_labels) + 1)
        violins = ax.violinplot(
            data,
            positions=positions,
            showmeans=False,
            showextrema=False,
            showmedians=False,
            widths=0.7,
        )
        for body, label in zip(violins["bodies"], ordered_labels):
            body.set_facecolor(PALETTE[label])
            body.set_edgecolor("#333333")
            body.set_alpha(0.6)
            body.set_linewidth(0.6)

        medians = [np.median(d) if len(d) else np.nan for d in data]
        ax.scatter(positions, medians, color="#222222", s=22, zorder=5, label="Median")
        ax.set_xticks(positions)
        ax.set_xticklabels(ordered_labels, rotation=10)
        ax.set_title(display_name, fontsize=BASE_FONT_SIZE + 1)
        ax.set_ylabel("Value")
        ax.set_facecolor(BACKGROUND_COLOR)

    for ax in axes.flat[len(features):]:
        ax.set_visible(False)

    fig.legend(
        handles=[mpl.lines.Line2D([], [], color="#222222", marker="o", linestyle="", label="Median")],
        loc="upper right",
        frameon=False,
    )
    fig.suptitle("Context distributions by performance regime", fontsize=BASE_FONT_SIZE + 4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)

def create_delta_histogram(df: pd.DataFrame, output_path: Path) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["delta_accuracy"], bins=30, color=PALETTE["Emb Δ>+1σ"], alpha=0.75, edgecolor="white")
    std_delta = df["delta_accuracy"].std()
    upper = std_delta
    lower = -std_delta
    ax.axvline(0.0, color="#444", linestyle="--", linewidth=0.8, label="Parity (Δ=0)")
    ax.axvline(upper, color=PALETTE["Emb Δ>+1σ"], linestyle=":", linewidth=0.8, label="+1σ threshold")
    ax.axvline(lower, color=PALETTE["Harm Δ<-1σ"], linestyle=":", linewidth=0.8, label="-1σ threshold")
    ax.set_xlabel("Δ accuracy (Embeddings − Harmonics)")
    ax.set_ylabel("Tile count")
    ax.set_title("Distribution of tile-level accuracy deltas")
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def create_eco_histograms(df: pd.DataFrame, output_path: Path) -> None:
    eco_regions = sorted(df["eco_region"].dropna().unique())
    if not eco_regions:
        return

    apply_plot_style()
    cols = 3
    rows = math.ceil(len(eco_regions) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.8, rows * 3.4), sharex=True, sharey=True)
    std_delta = df["delta_accuracy"].std()

    for ax, eco in zip(axes.flat, eco_regions):
        subset = df[df["eco_region"] == eco]["delta_accuracy"].dropna()
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.hist(subset, bins=15, color=PALETTE["Emb Δ>+1σ"], alpha=0.7, edgecolor="white")
        ax.axvline(0.0, color="#444", linestyle="--", linewidth=0.8)
        ax.axvline(std_delta, color=PALETTE["Emb Δ>+1σ"], linestyle=":", linewidth=0.8)
        ax.axvline(-std_delta, color=PALETTE["Harm Δ<-1σ"], linestyle=":", linewidth=0.8)
        mean_pp = subset.mean() * 100
        median_pp = subset.median() * 100
        ax.set_title(f"{eco}\nmean={mean_pp:+.1f} pp, median={median_pp:+.1f} pp", fontsize=BASE_FONT_SIZE)
        ax.set_xlabel("Δ accuracy (Embeddings − Harmonics)")
        ax.set_ylabel("Tile count")
        ax.set_facecolor(BACKGROUND_COLOR)

    for ax in axes.flat[len(eco_regions):]:
        ax.set_visible(False)

    fig.suptitle("Δ accuracy distribution per eco-region", fontsize=BASE_FONT_SIZE + 4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def create_eco_context_heatmap(summary: pd.DataFrame, context_features: Iterable[str], output_path: Path) -> None:
    feature_cols = [f"median_{col}" for col in context_features if f"median_{col}" in summary.columns]
    if not feature_cols:
        return

    apply_plot_style()
    data = summary.set_index("eco_region")[feature_cols]
    if data.empty:
        return

    normalized = (data - data.mean()) / data.std(ddof=0)
    normalized = normalized.fillna(0.0)

    fig, ax = plt.subplots(figsize=(len(feature_cols) * 1.9, len(data.index) * 0.55 + 2))
    im = ax.imshow(normalized.values, cmap="coolwarm", aspect="auto", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    display_labels = [
        DISPLAY_NAMES.get(col.replace("median_", ""), col.replace("median_", "").replace("_", " "))
        for col in feature_cols
    ]
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(display_labels, rotation=32, ha="right")
    ax.set_title("Eco-region median context z-scores", fontsize=BASE_FONT_SIZE + 4, pad=14)

    for i in range(len(data.index)):
        for j in range(len(feature_cols)):
            raw_val = data.iloc[i, j]
            z = normalized.iloc[i, j]
            color = "black" if abs(z) < 1.25 else "white"
            ax.text(j, i, f"{raw_val:.2f}", ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("z-score relative to national median", rotation=90, fontsize=BASE_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def export_context_zscores(summary: pd.DataFrame, context_features: Iterable[str], output_path: Path) -> None:
    feature_cols = [f"median_{col}" for col in context_features if f"median_{col}" in summary.columns]
    if not feature_cols:
        return
    data = summary.set_index("eco_region")[feature_cols]
    if data.empty:
        return
    normalized = (data - data.mean()) / data.std(ddof=0)
    normalized = normalized.fillna(0.0)
    deviation = normalized.abs().mean(axis=1).rename("mean_abs_z")
    export_df = normalized.copy()
    export_df["mean_abs_z"] = deviation
    export_df.to_csv(output_path)
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    if len(y_true) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

    labels = [1, 2]
    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    return {"accuracy": accuracy, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def main() -> None:
    args = parse_args()

    embeddings_df = pd.read_parquet(args.embeddings_predictions)
    harmonics_df = pd.read_parquet(args.harmonics_predictions)

    emb_tile = aggregate_tile_metrics(embeddings_df, model_tag="embeddings")
    harm_tile = aggregate_tile_metrics(harmonics_df, model_tag="harmonics")

    delta_df = compute_delta_metrics(emb_tile, harm_tile)
    if delta_df.empty:
        raise RuntimeError(
            "No overlapping tiles between embeddings and harmonics predictions."
        )

    context_df = pd.read_parquet(args.context_parquet)
    ensure_columns(context_df, ["tile_id"], "context parquet")

    enriched = delta_df.merge(context_df, on=["tile_id", "eco_region"], how="left")

    if args.coherence_parquet:
        coherence = pd.read_parquet(args.coherence_parquet)
        if "tile_index" not in coherence.columns:
            raise KeyError("Coherence parquet missing tile_index column.")
        coherence = coherence.rename(columns={"tile_index": "tile_id"})
        # Keep only 2023 coherence entries if multi-year
        if "year" in coherence.columns:
            coherence = coherence[coherence["year"] == 2023]
        enriched = enriched.merge(
            coherence, on=["tile_id", "eco_region"], how="left"
        )

    # Derived covariates
    if "era5_temp_max_degC_2023" in enriched.columns and "era5_temp_min_degC_2023" in enriched.columns:
        enriched["era5_temp_range_degC_2023"] = (
            enriched["era5_temp_max_degC_2023"]
            - enriched["era5_temp_min_degC_2023"]
        )
    if "era5_temp_max_degC_2020" in enriched.columns and "era5_temp_min_degC_2020" in enriched.columns:
        enriched["era5_temp_range_degC_2020"] = (
            enriched["era5_temp_max_degC_2020"]
            - enriched["era5_temp_min_degC_2020"]
        )

    context_features = [
        "elevation_mean",
        "slope_mean",
        "era5_temp_mean_degC_2023",
        "era5_temp_range_degC_2023",
        "era5_precip_total_mm_2023",
        "era5_soil_moisture_mean_2023",
        "soil_organic_carbon_mean_gkg",
        "soil_clay_mean_gkg",
        "soil_sand_mean_gkg",
        "deciduous_ratio",
        "shannon_diversity",
    ]

    feature_exprs: Dict[str, pd.Series] = {}
    for col in context_features:
        if col in enriched.columns:
            feature_exprs[col] = enriched[col]

    correlation_df = build_correlation_summary(
        enriched, target_cols=["delta_accuracy", "delta_f1_macro"], feature_exprs=feature_exprs
    )

    eco_summary_df = eco_region_summary(enriched)
    eco_summary_df = add_context_to_eco_summary(eco_summary_df, enriched, context_features)

    bucket_summary_df = summarise_by_bucket(enriched, context_features)

    national_stats = pd.DataFrame(
        [
            {
                "metric": "delta_accuracy",
                "mean": enriched["delta_accuracy"].mean(),
                "median": enriched["delta_accuracy"].median(),
                "std": enriched["delta_accuracy"].std(),
                "p25": enriched["delta_accuracy"].quantile(0.25),
                "p75": enriched["delta_accuracy"].quantile(0.75),
            },
            {
                "metric": "delta_f1_macro",
                "mean": enriched["delta_f1_macro"].mean(),
                "median": enriched["delta_f1_macro"].median(),
                "std": enriched["delta_f1_macro"].std(),
                "p25": enriched["delta_f1_macro"].quantile(0.25),
                "p75": enriched["delta_f1_macro"].quantile(0.75),
            },
        ]
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_dir / "tile_metrics_delta.parquet", index=False)
    eco_summary_df.to_csv(output_dir / "eco_region_summary.csv", index=False)
    correlation_df.to_csv(output_dir / "correlation_summary.csv", index=False)
    bucket_summary_df.to_csv(output_dir / "performance_bucket_summary.csv", index=False)
    national_stats.to_csv(output_dir / "national_delta_summary.csv", index=False)

    create_scatter_plots(
        enriched,
        context_features,
        output_dir / "delta_accuracy_vs_context.png",
    )
    create_boxplots(
        enriched,
        context_features,
        output_dir / "context_boxplots_by_bucket.png",
    )

    create_delta_histogram(
        enriched,
        output_dir / "delta_accuracy_histogram.png"
    )
    create_eco_histograms(
        enriched,
        output_dir / "eco_region_delta_histograms.png",
    )
    create_eco_context_heatmap(
        eco_summary_df,
        context_features,
        output_dir / "eco_region_context_heatmap.png",
    )
    export_context_zscores(
        eco_summary_df,
        context_features,
        output_dir / "eco_region_context_zscores.csv",
    )

    print(
        f"Saved tile metrics with context to {output_dir / 'tile_metrics_delta.parquet'} "
        f"({len(enriched)} tiles, "
        f"{enriched['embedding_advantage'].sum()} embedding-advantage tiles (Δ ≥ +1σ), "
        f"{enriched['harmonics_win'].sum()} harmonic-advantage tiles (Δ ≤ −1σ)."
    )


if __name__ == "__main__":
    main()
