#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare eco-region weighting schemes between baseline and embeddings datasets.

For each parquet:
- Computes dataset fraction per eco_region
- Uses EFFECTIVE_FOREST_AREA_BY_REGION to compute area fractions
- Derives weights as (area_frac / dataset_frac), then normalizes to sum to N

Prints a side-by-side summary and high-level differences.

Usage:
  python src/analysis/compare_eco_region_weights.py \
    --baseline results/datasets/training_datasets_pixels.parquet \
    --embeddings results/datasets/training_datasets_pixels_embedding.parquet
"""

import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.constants import EFFECTIVE_FOREST_AREA_BY_REGION


def summarize(path: str) -> dict:
    df = pd.read_parquet(path, columns=['eco_region'])
    total = len(df)
    counts = df['eco_region'].value_counts()
    ds_frac = (counts / total).to_dict()
    total_area = sum(EFFECTIVE_FOREST_AREA_BY_REGION.values()) or 1.0
    area_frac = {r: EFFECTIVE_FOREST_AREA_BY_REGION.get(r, 0.0) / total_area for r in counts.index}
    weights = {r: (area_frac.get(r, 0.0) / ds_frac.get(r, 1.0)) if ds_frac.get(r, 0.0) > 0 else 1.0 for r in counts.index}
    # normalize weights to sum to N
    # (normalization factor doesn't affect relative per-region ratios)
    norm = total / sum(weights[r] * counts[r] for r in counts.index)
    weights = {r: weights[r] * norm for r in counts.index}
    return {
        'total': total,
        'counts': counts,
        'ds_frac': ds_frac,
        'area_frac': area_frac,
        'weights': weights,
    }


def main():
    ap = argparse.ArgumentParser(description='Compare eco-region weighting between baseline and embeddings parquets')
    ap.add_argument('--baseline', default='results/datasets/training_datasets_pixels.parquet')
    ap.add_argument('--embeddings', default='results/datasets/training_datasets_pixels_embedding.parquet')
    args = ap.parse_args()

    base = summarize(args.baseline)
    emb = summarize(args.embeddings)

    regions = sorted(set(base['counts'].index).union(set(emb['counts'].index)))
    print(f"Baseline rows:   {base['total']:,}")
    print(f"Embeddings rows: {emb['total']:,}")
    print("\nEco-region comparison (dataset% | area% | weight baseline -> embeddings):")
    for r in regions:
        ds_b = base['ds_frac'].get(r, 0.0) * 100
        ds_e = emb['ds_frac'].get(r, 0.0) * 100
        ar = EFFECTIVE_FOREST_AREA_BY_REGION.get(r, 0.0)
        ar_p = (ar / sum(EFFECTIVE_FOREST_AREA_BY_REGION.values()) * 100) if ar > 0 else 0.0
        wb = base['weights'].get(r, float('nan'))
        we = emb['weights'].get(r, float('nan'))
        print(f"- {r:40s} ds% {ds_b:6.2f} -> {ds_e:6.2f} | area% {ar_p:6.2f} | w {wb:7.4f} -> {we:7.4f}")

    # Optional: summarize absolute mean weight difference across regions
    diffs = []
    for r in regions:
        if r in base['weights'] and r in emb['weights']:
            diffs.append(abs(base['weights'][r] - emb['weights'][r]))
    if diffs:
        print(f"\nMean abs per-region weight diff: {sum(diffs)/len(diffs):.6f}")


if __name__ == '__main__':
    main()

