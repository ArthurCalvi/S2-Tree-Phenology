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
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

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

    res = pd.DataFrame(results_per_fold)
    summary = {
        'n_features': len(feature_cols),
        'features': feature_cols,
        'metrics_mean': res.mean(numeric_only=True).to_dict(),
        'metrics_std': res.std(numeric_only=True).to_dict(),
    }
    return summary


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description='Train RF on embeddings with eco-region CV')
    ap.add_argument('--dataset_path', required=True, help='Path to training_datasets_pixels_embedding.parquet')
    ap.add_argument('--config', choices=['all','topk'], default='all', help='Use all features or top-K by RF importance')
    ap.add_argument('--k', type=int, default=14, help='K for topk config (match article RFECV size)')
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
    assert feature_cols, 'No embedding_* columns found'

    if args.config == 'topk':
        logging.info(f'Selecting top-{args.k} features by RF importance')
        topk = select_topk_features(df[feature_cols], df['phenology'], args.k)
        feature_cols = topk

    logging.info(f'Using {len(feature_cols)} features')
    summary = run_cv(df, feature_cols, n_splits=args.n_splits)

    # Save outputs
    tag = f"embeddings_{args.config}{'' if args.config=='all' else '_k'+str(args.k)}"
    with open(Path(args.output_dir) / f"metrics_{tag}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    with open(Path(args.output_dir) / f"features_{tag}.txt", 'w') as f:
        f.write("\n".join(summary['features']))

    logging.info(f"Saved metrics and features under {args.output_dir}")

if __name__ == '__main__':
    main()
