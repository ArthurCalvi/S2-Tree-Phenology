#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect the embeddings training parquet produced at
results/datasets/training_datasets_pixels_embedding.parquet.

Shows a quick head, columns, basic label/weight coverage, and a peek at
the first few embedding feature columns.

Usage:
  python src/analysis/inspect_embeddings_parquet.py \
    --path results/datasets/training_datasets_pixels_embedding.parquet \
    --limit 10
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Inspect embeddings parquet head and basic stats")
    ap.add_argument('--path', default='results/datasets/training_datasets_pixels_embedding.parquet',
                    help='Path to embeddings parquet')
    ap.add_argument('--limit', type=int, default=10, help='Number of head rows to print')
    ap.add_argument('--sample_n', type=int, default=None, help='Optional sample size for quick stats')
    args = ap.parse_args()

    df = pd.read_parquet(args.path)
    print(f"Loaded: {args.path}")
    print(f"Rows: {len(df):,}  Cols: {len(df.columns)}")

    # Show columns overview
    emb_cols = [c for c in df.columns if c.startswith('embedding_')]
    id_cols = [c for c in ['tile_id', 'row', 'col'] if c in df.columns]
    meta_cols = [c for c in ['phenology', 'genus', 'species', 'source', 'year', 'eco_region', 'weight'] if c in df.columns]

    print("\nColumns:")
    print("- ids:", id_cols)
    print("- embeddings (count):", len(emb_cols))
    print("- meta:", meta_cols)

    print("\nHead:")
    # Order a concise view
    preview_cols = id_cols + ['phenology'] + emb_cols[:5] + [c for c in ['eco_region', 'weight', 'year', 'genus', 'species', 'source'] if c in meta_cols]
    print(df[preview_cols].head(args.limit).to_string(index=False))

    # Basic stats (optionally on a sample)
    dfx = df.sample(args.sample_n, random_state=42) if args.sample_n and args.sample_n < len(df) else df
    if 'weight' in dfx.columns:
        miss = dfx['weight'].isna().mean() * 100
        print(f"\nWeight missing (overall): {miss:.2f}%")
        # Top tiles by missing share
        grp = dfx.groupby('tile_id')['weight'].apply(lambda s: s.isna().mean()*100).sort_values(ascending=False)
        print("Top tiles by missing weight (%):")
        print(grp.head(10).round(2).to_string())

    if 'phenology' in dfx.columns:
        vc = dfx['phenology'].value_counts(normalize=True).mul(100).round(2)
        print("\nPhenology distribution (%):")
        print(vc.to_string())

    if emb_cols:
        desc = dfx[emb_cols[:5]].describe().T[['mean','std','min','max']]
        print("\nEmbedding columns (first 5) summary:")
        print(desc.round(4).to_string())


if __name__ == '__main__':
    main()

