#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Random Forest on embedding features with eco-region balanced CV.

Two configs:
 - all: use all embedding_* features (64)
 - topk: select top-K features via RF feature_importances_ (K set by --k)

Usage:
  python src/training/train_rf_embeddings.py \
    --dataset_path results/datasets/training_datasets_pixels_embedding.parquet \
    --config all --output_dir results/final_model

  python src/training/train_rf_embeddings.py \
    --dataset_path results/datasets/training_datasets_pixels_embedding.parquet \
    --config topk --k 14 --output_dir results/final_model
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# Ensure project root is in sys.path (for SLURM/remote execution)
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# project imports
from src.utils import (
    create_eco_balanced_folds_df,
    compute_metrics,
)


def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/train_rf_embeddings.log'),
            logging.StreamHandler()
        ]
    )


def select_topk_features(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    topk = list(importances.sort_values(ascending=False).head(k).index)
    return topk


def run_cv(df: pd.DataFrame, feature_cols: list[str], n_splits: int = 5,
           n_estimators: int = 50, max_depth: int = 30, min_samples_split: int = 30):
    X = df[feature_cols]
    y = df['phenology']

    folds = create_eco_balanced_folds_df(df, n_splits=n_splits, random_state=42)

    results_per_fold = []
    results_per_ecoregion: dict[str, list[dict]] = defaultdict(list)
    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc='CV folds')):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        sample_weights = None
        if 'weight' in df.columns:
            sample_weights = df.iloc[tr_idx]['weight'].values

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=max(1, min_samples_split // 2)
        )
        clf.fit(Xtr, ytr, sample_weight=sample_weights)
        ypred = clf.predict(Xva)
        metrics = compute_metrics(yva, ypred)
        metrics['fold'] = fold + 1
        results_per_fold.append(metrics)

        # Per eco-region metrics on validation split
        if 'eco_region' in df.columns:
            eco_regions_in_val = df.iloc[va_idx]['eco_region'].unique()
            for eco in eco_regions_in_val:
                mask = (df.iloc[va_idx]['eco_region'] == eco)
                if mask.any():
                    m = compute_metrics(yva[mask], ypred[mask])
                    m['fold'] = fold + 1
                    m['eco_region'] = str(eco)
                    m['n_samples'] = int(mask.sum())
                    results_per_ecoregion[str(eco)].append(m)

    res = pd.DataFrame(results_per_fold)
    # Aggregate fold metrics
    summary = {
        'n_features': len(feature_cols),
        'features': feature_cols,
        'metrics_mean': res.mean(numeric_only=True).to_dict(),
        'metrics_std': res.std(numeric_only=True).to_dict(),
    }
    # Aggregate eco-region metrics
    eco_rows = []
    for eco, lst in results_per_ecoregion.items():
        dfe = pd.DataFrame(lst)
        avg = dfe.mean(numeric_only=True).to_dict()
        avg['eco_region'] = eco
        avg['n_samples_total'] = int(dfe['n_samples'].sum()) if 'n_samples' in dfe else None
        eco_rows.append(avg)
    eco_df = pd.DataFrame(eco_rows) if eco_rows else pd.DataFrame()
    return summary, eco_df


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description='Train RF on embeddings with eco-region CV')
    ap.add_argument('--dataset_path', required=True, help='Path to training_datasets_pixels_embedding.parquet')
    ap.add_argument('--config', choices=['all','topk'], default='all', help='Use all features or top-K by RF importance')
    ap.add_argument('--k', type=int, default=14, help='K for topk config (match article RFECV size)')
    ap.add_argument('--features_file', type=str, default=None, help='Optional path to a file listing selected features (one per line) for topk config')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--output_dir', default='results/final_model')
    ap.add_argument('--sample_n', type=int, default=None, help='Optional sample size for quick run')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.dataset_path)

    # ensure eco_region exists (should be added by converter)
    if 'eco_region' not in df.columns:
        logging.warning('eco_region missing; CV may not be eco-balanced')

    if args.sample_n and args.sample_n < len(df):
        df = df.sample(args.sample_n, random_state=42)

    # Features: embedding_0..embedding_63
    feature_cols = sorted([c for c in df.columns if c.startswith('embedding_')])
    # Guard against inf values at training time (allow NaN for sklearn trees)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    assert feature_cols, 'No embedding_* columns found'

    chosen_k = None
    if args.config == 'topk':
        if args.features_file and Path(args.features_file).exists():
            logging.info(f'Loading selected features from {args.features_file}')
            with open(args.features_file, 'r') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            # validate presence in dataframe
            missing = [c for c in lines if c not in df.columns]
            if missing:
                logging.warning(f'{len(missing)} features from file are missing in dataset and will be ignored: {missing[:5]}')
            selected = [c for c in lines if c in df.columns]
            if not selected:
                raise SystemExit('No valid features after reading features_file')
            feature_cols = selected
            chosen_k = len(feature_cols)
            logging.info(f'Using {chosen_k} features from file')
        else:
            logging.info(f'Selecting top-{args.k} features by RF importance')
            topk = select_topk_features(df[feature_cols], df['phenology'], args.k)
            feature_cols = topk
            chosen_k = len(feature_cols)

    logging.info(f'Using {len(feature_cols)} features')
    summary, eco_df = run_cv(df, feature_cols, n_splits=args.n_splits)

    # Save outputs
    if args.config == 'all':
        tag = 'embeddings_all'
    else:
        tag = f"embeddings_topk_k{chosen_k if chosen_k is not None else args.k}"
    # enrich summary with config + timestamp
    summary_out = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'dataset_path': args.dataset_path,
        'config': args.config,
        'k': (chosen_k if args.config == 'topk' else None),
        'features_file': args.features_file if args.config == 'topk' else None,
        'n_splits': args.n_splits,
        'rf_params': {
            'n_estimators': 50,
            'max_depth': 30,
            'min_samples_split': 30,
            'min_samples_leaf': max(1, 30 // 2),
            'class_weight': 'balanced',
            'random_state': 42
        },
        **summary
    }
    with open(Path(args.output_dir) / f"metrics_{tag}.json", 'w') as f:
        json.dump(summary_out, f, indent=2)
    with open(Path(args.output_dir) / f"features_{tag}.txt", 'w') as f:
        f.write("\n".join(summary['features']))
    # Save eco-region metrics CSV if available
    if not eco_df.empty:
        eco_df.to_csv(Path(args.output_dir) / f"eco_metrics_{tag}.csv", index=False)

    logging.info(f"Saved metrics and features under {args.output_dir}")

if __name__ == '__main__':
    main()
