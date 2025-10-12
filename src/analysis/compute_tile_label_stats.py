"""Compute per-tile class composition metrics from the label parquet."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate deciduous/evergreen ratios and Shannon diversity per tile."
    )
    parser.add_argument(
        "--labels-parquet",
        required=True,
        type=Path,
        help="Path to the parquet containing per-pixel labels (needs tile_id, phenology, eco_region).",
    )
    parser.add_argument(
        "--context-parquet",
        required=True,
        type=Path,
        help="Existing tile context parquet to enrich (will be created if missing).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination parquet with merged context + class composition metrics.",
    )
    return parser.parse_args()


def safe_log(prob: float) -> float:
    if prob <= 0.0:
        return 0.0
    return prob * math.log(prob)


def compute_tile_metrics(df: pd.DataFrame) -> pd.DataFrame:
    tile_col = "tile_id"
    class_col = "phenology"
    eco_col = "eco_region"

    counts = (
        df.groupby([tile_col, class_col]).size().unstack(fill_value=0).astype("int64")
    )
    for label in (1, 2):
        if label not in counts.columns:
            counts[label] = 0
    counts = counts[[1, 2]].rename(columns={1: "n_deciduous", 2: "n_evergreen"})
    totals = counts.sum(axis=1).rename("n_samples")
    ratios = counts.div(totals.replace({0: np.nan}), axis=0)
    ratios = ratios.fillna(0.0).rename(
        columns={
            "n_deciduous": "deciduous_ratio",
            "n_evergreen": "evergreen_ratio",
        }
    )
    probs = ratios.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        shannon_arr = -np.where(probs > 0.0, probs * np.log(probs), 0.0).sum(axis=1)
    shannon = pd.Series(shannon_arr, index=ratios.index, name="shannon_diversity")
    eco_region = df.groupby(tile_col)[eco_col].first()
    summary = pd.concat([counts, totals, ratios, shannon, eco_region], axis=1)
    summary = summary.rename(columns={eco_col: "eco_region"})
    summary["deciduous_ratio"] = summary["deciduous_ratio"].astype("float64")
    summary["evergreen_ratio"] = summary["evergreen_ratio"].astype("float64")
    summary["shannon_diversity"] = summary["shannon_diversity"].astype("float64")
    summary = summary.reset_index()
    return summary


def main() -> None:
    args = parse_args()
    if not args.labels_parquet.exists():
        raise FileNotFoundError(f"Labels parquet not found: {args.labels_parquet}")

    labels = pd.read_parquet(
        args.labels_parquet, columns=["tile_id", "phenology", "eco_region"]
    )
    tile_metrics = compute_tile_metrics(labels)

    if args.context_parquet.exists():
        context = pd.read_parquet(args.context_parquet)
        context = context.set_index("tile_id")
        metric_cols = [
            "n_samples",
            "n_deciduous",
            "n_evergreen",
            "deciduous_ratio",
            "evergreen_ratio",
            "shannon_diversity",
            "eco_region",
        ]
        drop_cols = [col for col in metric_cols if col in context.columns]
        if drop_cols:
            context = context.drop(columns=drop_cols)
    else:
        context = pd.DataFrame(columns=["tile_id"]).set_index("tile_id")

    tile_metrics = tile_metrics.set_index("tile_id")
    merged = context.join(tile_metrics, how="outer")
    for col in ("n_samples", "n_deciduous", "n_evergreen"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype("int64")
    for col in ("deciduous_ratio", "evergreen_ratio", "shannon_diversity"):
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0).astype("float64")
    merged = merged.sort_index().reset_index()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)


if __name__ == "__main__":
    main()
