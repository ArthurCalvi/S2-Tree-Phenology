#!/usr/bin/env python3
"""
Summarise ancillary covariates for tile subsets where one model clearly
outperforms the other (|Δ accuracy| >= 1σ).

Inputs (from analyze_tile_heterogeneity.py):
    - tile_metrics_delta.parquet
    - tile_context_features.parquet

Outputs:
    - <bucket>_driver_stats.csv per requested performance bucket
    - subset_driver_summary.json with headline metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

EXCLUDE_COLUMNS = {
    "tile_id",
    "eco_region",
    "performance_bucket",
    "performance_bucket_label",
    "delta_accuracy",
    "delta_f1_macro",
    "delta_f1_weighted",
    "accuracy_embeddings",
    "accuracy_harmonics",
    "f1_macro_embeddings",
    "f1_macro_harmonics",
    "f1_weighted_embeddings",
    "f1_weighted_harmonics",
    "n_samples_embeddings",
    "n_samples_harmonics",
    "n_samples",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse ancillary drivers for tile performance buckets."
    )
    parser.add_argument(
        "--tile-metrics",
        required=True,
        type=Path,
        help="Parquet produced by analyze_tile_heterogeneity.py (tile_metrics_delta.parquet).",
    )
    parser.add_argument(
        "--context",
        required=True,
        type=Path,
        help="Parquet with tile-level ancillary features (tile_context_features.parquet).",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["embedding_advantage", "harmonics_win"],
        help="Performance buckets to analyse (default: embedding_advantage harmonics_win).",
    )
    parser.add_argument(
        "--reference-bucket",
        default="rough_parity",
        help="Bucket used as baseline when computing mean differences (default: rough_parity).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for CSV/JSON outputs.",
    )
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing columns: {missing}")


def load_data(tile_metrics_path: Path, context_path: Path) -> pd.DataFrame:
    tile_df = pd.read_parquet(tile_metrics_path)
    context_df = pd.read_parquet(context_path)
    ensure_columns(tile_df, ["tile_id", "performance_bucket"], "tile metrics")
    ensure_columns(context_df, ["tile_id"], "context features")

    merged = tile_df.merge(context_df, on="tile_id", how="left", suffixes=("", "_ctx"))
    if merged["tile_id"].duplicated().any():
        raise ValueError("Duplicate tile_id detected after merge.")
    return merged


def identify_covariates(df: pd.DataFrame) -> List[str]:
    numeric_cols = [
        c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in EXCLUDE_COLUMNS
    ]
    return numeric_cols


def compute_stats(
    df: pd.DataFrame,
    bucket: str,
    reference_df: pd.DataFrame,
    covariates: Iterable[str],
) -> pd.DataFrame:
    subset = df[df["performance_bucket"] == bucket].copy()
    if subset.empty:
        raise ValueError(f"No tiles found for bucket '{bucket}'.")

    rows = []
    for cov in covariates:
        subset_series = subset[cov].dropna()
        reference_series = reference_df[cov].dropna()

        rows.append(
            {
                "covariate": cov,
                "subset_mean": subset_series.mean(),
                "reference_mean": reference_series.mean() if not reference_series.empty else np.nan,
                "mean_difference": subset_series.mean() - (reference_series.mean() if not reference_series.empty else np.nan),
                "pearson_delta_accuracy": subset_series.corr(subset["delta_accuracy"], method="pearson"),
                "spearman_delta_accuracy": subset_series.corr(subset["delta_accuracy"], method="spearman"),
                "pearson_delta_f1_macro": subset_series.corr(subset["delta_f1_macro"], method="pearson"),
                "spearman_delta_f1_macro": subset_series.corr(subset["delta_f1_macro"], method="spearman"),
            }
        )

    stats = pd.DataFrame(rows)
    stats.sort_values(by="mean_difference", key=lambda s: s.abs(), inplace=True, ascending=False)
    stats.reset_index(drop=True, inplace=True)
    return subset, stats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged = load_data(args.tile_metrics, args.context)
    covariates = identify_covariates(merged)

    if args.reference_bucket not in merged["performance_bucket"].unique():
        raise ValueError(f"Reference bucket '{args.reference_bucket}' not present.")
    reference_df = merged[merged["performance_bucket"] == args.reference_bucket]

    summary = {}

    for bucket in args.buckets:
        subset, stats_df = compute_stats(merged, bucket, reference_df, covariates)
        stats_path = args.output_dir / f"{bucket}_driver_stats.csv"
        stats_df.to_csv(stats_path, index=False)

        summary[bucket] = {
            "tiles": int(len(subset)),
            "mean_delta_accuracy": float(subset["delta_accuracy"].mean()),
            "median_delta_accuracy": float(subset["delta_accuracy"].median()),
            "mean_delta_f1_macro": float(subset["delta_f1_macro"].mean()),
            "median_delta_f1_macro": float(subset["delta_f1_macro"].median()),
            "top_covariates_by_mean_difference": stats_df.head(5)["covariate"].tolist(),
        }

    summary_path = args.output_dir / "subset_driver_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
