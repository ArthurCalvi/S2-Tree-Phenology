#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add eco-region based sample weights to the embeddings parquet.

This mirrors the logic used for the harmonic baseline (see add_weights.py):
- Compute dataset fraction per eco_region
- Compute effective forest area fraction per region (from constants)
- Weight = area_fraction / dataset_fraction
- Normalize weights to sum to N samples

Usage:
  python src/dataset_creation/add_weights_embeddings.py \
    --input results/datasets/training_datasets_pixels_embedding.parquet \
    --output results/datasets/training_datasets_pixels_embedding.parquet
"""

import os
import argparse
import logging
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.constants import EFFECTIVE_FOREST_AREA_BY_REGION


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def compute_weights(df: pd.DataFrame, eco_col: str = 'eco_region') -> pd.DataFrame:
    counts = df[eco_col].value_counts()
    total = len(df)
    total_area = sum(EFFECTIVE_FOREST_AREA_BY_REGION.values()) or 1.0

    region_w = {}
    logging.info("Eco-region weights (area/dataset):")
    for region, cnt in counts.items():
        ds_frac = cnt / total if total > 0 else 0
        area = EFFECTIVE_FOREST_AREA_BY_REGION.get(region, 0.0)
        area_frac = area / total_area
        w = (area_frac / ds_frac) if ds_frac > 0 else 1.0
        region_w[region] = w
        logging.info(f" - {region}: n={cnt}, ds%={ds_frac*100:.2f}, area%={area_frac*100:.2f}, w={w:.4f}")

    df['weight'] = df[eco_col].map(region_w).fillna(1.0)
    # Normalize to keep effective sample size
    scale = (total / df['weight'].sum()) if df['weight'].sum() > 0 else 1.0
    df['weight'] = df['weight'] * scale
    logging.info(f"Weight sum after normalization: {df['weight'].sum():.1f} (expected ~{total})")
    return df


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description='Add eco-region weights to embeddings parquet')
    ap.add_argument('--input', default='results/datasets/training_datasets_pixels_embedding.parquet')
    ap.add_argument('--output', default='results/datasets/training_datasets_pixels_embedding.parquet')
    ap.add_argument('--eco-col', default='eco_region')
    args = ap.parse_args()

    logging.info(f"Loading {args.input}")
    df = pd.read_parquet(args.input)
    if args.eco_col not in df.columns:
        raise SystemExit(f"Column {args.eco_col} missing in {args.input}")

    df = compute_weights(df, args.eco_col)

    out_dir = Path(args.output).parent
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Writing with weights to {args.output}")
    df.to_parquet(args.output, index=False)
    logging.info("Done.")


if __name__ == '__main__':
    main()

