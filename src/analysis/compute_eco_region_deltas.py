#!/usr/bin/env python3
"""Compute eco-region performance deltas between harmonic and embedding RF models."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute eco-region accuracy/F1 deltas between harmonic and embedding models.")
    parser.add_argument('--embedding-metrics', required=True, type=Path,
                        help='CSV with embedding eco metrics (results/final_model/eco_metrics_embeddings_topk_k14.csv).')
    parser.add_argument('--harmonic-metrics', required=True, type=Path,
                        help='CSV with harmonic eco metrics (results/final_model/phenology_eco_metrics_selected_features.csv).')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output CSV path for merged metrics and deltas.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    emb = pd.read_csv(args.embedding_metrics)
    harm = pd.read_csv(args.harmonic_metrics)

    emb = emb[emb['eco_region'] != 'Unknown']
    emb_group = emb.groupby('eco_region').agg({'accuracy': 'mean', 'f1_macro': 'mean'})
    emb_group = emb_group.rename(columns={'accuracy': 'accuracy_emb', 'f1_macro': 'f1_emb'})

    harm_group = harm.set_index('eco_region')[['accuracy_mean', 'f1_macro_mean']]
    harm_group = harm_group.rename(columns={'accuracy_mean': 'accuracy_harm', 'f1_macro_mean': 'f1_harm'})

    merged = emb_group.join(harm_group, how='inner')
    merged['delta_accuracy'] = merged['accuracy_emb'] - merged['accuracy_harm']
    merged['delta_f1'] = merged['f1_emb'] - merged['f1_harm']
    merged = merged.reset_index().sort_values('delta_f1', ascending=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
