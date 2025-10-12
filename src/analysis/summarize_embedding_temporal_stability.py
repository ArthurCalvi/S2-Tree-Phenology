#!/usr/bin/env python3
"""
Summarize AlphaEarth embedding evaluation metrics across multiple years.

This script scans evaluation directories (e.g., results/evaluation/embeddings_2023)
and aggregates overall metrics plus eco-region metrics into consolidated tables. It
also computes deltas relative to a baseline year (default: 2023) so temporal stability
can be assessed quickly.

Outputs
-------
- embedding_temporal_overview.csv: overall metrics per year + deltas vs baseline.
- embedding_temporal_overview.parquet: same as CSV for downstream use.
- embedding_temporal_eco_deltas.csv: eco-region metrics per year with deltas vs baseline.
- embedding_temporal_eco_deltas.parquet: parquet version of the eco table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize embedding evaluation metrics across years.")
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("results/evaluation"),
        help="Root directory containing embeddings_<year> evaluation folders.",
    )
    parser.add_argument(
        "--baseline-year",
        type=int,
        default=2023,
        help="Year used as stability baseline for delta computations.",
    )
    parser.add_argument(
        "--pattern",
        default="embeddings_",
        help="Prefix used to detect evaluation directories (default: embeddings_).",
    )
    parser.add_argument(
        "--output-prefix",
        default="embedding_temporal",
        help="Prefix for output summary files.",
    )
    return parser.parse_args()


def load_metrics_json(metrics_path: Path) -> Dict[str, float]:
    with metrics_path.open("r") as f:
        payload = json.load(f)
    overall = payload["overall_metrics"]
    keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
    ]
    row = {k: overall[k] for k in keys}
    row["n_samples"] = overall["n_samples"]
    return row


def load_eco_metrics_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    keep = [
        "eco_region",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
        "n_samples",
    ]
    return df[keep]


def main() -> None:
    args = parse_args()
    root = args.evaluation_root
    if not root.exists():
        raise FileNotFoundError(f"Evaluation root {root} not found.")

    overall_rows: List[Dict[str, float]] = []
    eco_frames: List[pd.DataFrame] = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        if not subdir.name.startswith(args.pattern):
            continue
        year_str = subdir.name[len(args.pattern) :]
        if not year_str.isdigit():
            continue
        year = int(year_str)

        metrics_path = subdir / f"metrics_{subdir.name}.json"
        if not metrics_path.exists():
            print(f"Warning: metrics json missing for {subdir.name}, skipping.")
            continue

        overall_row = load_metrics_json(metrics_path)
        overall_row["year"] = year
        overall_rows.append(overall_row)

        eco_csv = subdir / f"eco_metrics_{subdir.name}.csv"
        if eco_csv.exists():
            eco_df = load_eco_metrics_csv(eco_csv)
            eco_df["year"] = year
            eco_frames.append(eco_df)
        else:
            print(f"Warning: eco metrics missing for {subdir.name}, skipping eco-region deltas.")

    if not overall_rows:
        raise RuntimeError("No evaluation directories found matching the provided pattern.")

    overall_df = pd.DataFrame(overall_rows).sort_values("year").reset_index(drop=True)
    # Optionally append a CV baseline row if the embeddings model directory is available.
    cv_path = Path("results/final_model/latest_embeddings_topk_k14/fold_metrics_embeddings_topk_k14.csv")
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
    ]
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        cv_means = cv_df[metric_cols].mean().to_dict()
        cv_row = {"year": args.baseline_year, "source": "cv_2023"}
        cv_row.update(cv_means)
        overall_df = pd.concat([overall_df, pd.DataFrame([cv_row])], ignore_index=True, sort=False)

    baseline_mask = (overall_df["year"] == args.baseline_year) & (overall_df.get("source").fillna("") != "cv_2023")
    if not baseline_mask.any():
        raise ValueError(f"Baseline year {args.baseline_year} not present among evaluation metrics.")

    baseline_row = overall_df[baseline_mask].iloc[0]
    for col in metric_cols:
        overall_df[f"{col}_delta_vs_{args.baseline_year}"] = overall_df[col] - baseline_row[col]

    overall_df = overall_df.sort_values(["year", "source"]).reset_index(drop=True)
    output_overview_csv = root / f"{args.output_prefix}_overview.csv"
    output_overview_parquet = root / f"{args.output_prefix}_overview.parquet"
    overall_df.to_csv(output_overview_csv, index=False)
    overall_df.to_parquet(output_overview_parquet, index=False)
    print(f"Wrote overall summary to {output_overview_csv}")

    if eco_frames:
        eco_df = pd.concat(eco_frames, ignore_index=True)
        pivot_cols = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
        ]

        baseline_eco = eco_df[eco_df["year"] == args.baseline_year].set_index("eco_region")
        delta_frames = []
        for year, group in eco_df.groupby("year"):
            group = group.set_index("eco_region")
            aligned = group.join(
                baseline_eco[pivot_cols],
                how="left",
                lsuffix="",
                rsuffix=f"_baseline_{args.baseline_year}",
            )
            for col in pivot_cols:
                aligned[f"{col}_delta_vs_{args.baseline_year}"] = (
                    aligned[col] - aligned[f"{col}_baseline_{args.baseline_year}"]
                )
            aligned["year"] = year
            delta_frames.append(aligned.reset_index())
        eco_summary = pd.concat(delta_frames, ignore_index=True)

        eco_csv = root / f"{args.output_prefix}_eco_deltas.csv"
        eco_parquet = root / f"{args.output_prefix}_eco_deltas.parquet"
        eco_summary.to_csv(eco_csv, index=False)
        eco_summary.to_parquet(eco_parquet, index=False)
        print(f"Wrote eco-region summary to {eco_csv}")
    else:
        print("No eco-region metrics found; skipping eco summary.")


if __name__ == "__main__":
    main()
