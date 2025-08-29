#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursive Feature Selection for Embeddings (RF + eco-region CV)

Selects the best subset of embedding_* features using recursive elimination.
Saves selection curve, best feature list, overall metrics JSON (with config and timestamp),
and per–eco‑region metrics CSV for the best subset.

Usage:
  python src/training/recursive_feature_selection_embeddings.py \
    --dataset_path results/datasets/training_datasets_pixels_embedding.parquet \
    --output_dir results/feature_selection_embeddings \
    --min_features 6 --n_splits 5
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# Ensure project root in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import (
    compute_metrics,
    create_eco_balanced_folds_df,
)


def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rfe_embeddings.log'),
            logging.StreamHandler()
        ]
    )


def parse_step_schedule(step_arg: str, n_features: int, min_features: int) -> List[int]:
    if not step_arg:
        return [max(1, n_features // 10)]  # fallback
    steps = [int(s.strip()) for s in step_arg.split(',') if s.strip()]
    # ensure the sum of steps can reach min_features
    return steps


def cv_importance_and_score(df: pd.DataFrame, features: List[str], n_splits: int) -> tuple[pd.Series, dict]:
    X = df[features]
    y = df['phenology']
    folds = create_eco_balanced_folds_df(df, n_splits=n_splits, random_state=42)

    importances = []
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc=f'CV {len(features)} feats')):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        sample_weights = None
        if 'weight' in df.columns:
            sample_weights = df.iloc[tr_idx]['weight'].values
        clf = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=30,
            min_samples_split=30,
            min_samples_leaf=15
        )
        clf.fit(Xtr, ytr, sample_weight=sample_weights)
        importances.append(pd.Series(clf.feature_importances_, index=features))
        ypred = clf.predict(Xva)
        m = compute_metrics(yva, ypred)
        m['fold'] = fold + 1
        fold_metrics.append(m)

    imp_avg = pd.concat(importances, axis=1).mean(axis=1)
    mdf = pd.DataFrame(fold_metrics)
    score = {
        'f1_macro_mean': float(mdf['f1_macro'].mean()),
        'f1_macro_std': float(mdf['f1_macro'].std()),
        'accuracy_mean': float(mdf['accuracy'].mean()),
        'accuracy_std': float(mdf['accuracy'].std()),
    }
    return imp_avg, score


def eco_region_metrics(df: pd.DataFrame, features: List[str], n_splits: int) -> pd.DataFrame:
    X = df[features]
    y = df['phenology']
    folds = create_eco_balanced_folds_df(df, n_splits=n_splits, random_state=42)
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc='Eco metrics CV')):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        w = df.iloc[tr_idx]['weight'].values if 'weight' in df.columns else None
        clf = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=30,
            min_samples_split=30,
            min_samples_leaf=15
        )
        clf.fit(Xtr, ytr, sample_weight=w)
        ypred = clf.predict(Xva)
        if 'eco_region' in df.columns:
            eco_vals = df.iloc[va_idx]['eco_region']
            for eco in eco_vals.unique():
                mask = (eco_vals == eco)
                if mask.any():
                    m = compute_metrics(yva[mask], ypred[mask])
                    m['fold'] = fold + 1
                    m['eco_region'] = str(eco)
                    m['n_samples'] = int(mask.sum())
                    rows.append(m)
    if not rows:
        return pd.DataFrame()
    dfe = pd.DataFrame(rows)
    out_rows = []
    for eco, grp in dfe.groupby('eco_region'):
        avg = grp.mean(numeric_only=True).to_dict()
        avg['eco_region'] = eco
        avg['n_samples_total'] = int(grp['n_samples'].sum()) if 'n_samples' in grp else None
        out_rows.append(avg)
    return pd.DataFrame(out_rows)


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description='Recursive feature selection for embeddings')
    ap.add_argument('--dataset_path', default='results/datasets/training_datasets_pixels_embedding.parquet')
    ap.add_argument('--output_dir', default='results/feature_selection_embeddings')
    ap.add_argument('--min_features', type=int, default=6)
    ap.add_argument('--step_schedule', default='6,5,4,3,2,1', help='Comma-separated elimination steps')
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--sample_n', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.dataset_path)
    if args.sample_n and args.sample_n < len(df):
        df = df.sample(args.sample_n, random_state=42)

    features = sorted([c for c in df.columns if c.startswith('embedding_')])
    assert features, 'No embedding_* columns found'

    n_total = len(features)
    steps = parse_step_schedule(args.step_schedule, n_total, args.min_features)
    logging.info(f"Total features: {n_total}; min_features: {args.min_features}; steps: {steps}")

    # Selection loop
    current = features.copy()
    curve = []
    best = {'features': current, 'score': -1.0, 'summary': None}

    for i, step in enumerate(steps):
        imp, score = cv_importance_and_score(df, current, args.n_splits)
        curve.append({'n_features': len(current), **score})
        # Track best by f1_macro_mean
        if score.get('f1_macro_mean', -1) > best['score']:
            best['score'] = score['f1_macro_mean']
            best['features'] = current.copy()
            best['summary'] = score

        if len(current) <= args.min_features:
            logging.info("Reached min_features; stopping elimination")
            break
        # Eliminate lowest-importance features for next iteration
        drop_k = min(step, len(current) - args.min_features)
        to_drop = imp.sort_values(ascending=True).head(drop_k).index.tolist()
        current = [f for f in current if f not in to_drop]
        logging.info(f"Step {i+1}: dropped {drop_k} features; remaining {len(current)}")

    # Final best set eco metrics
    eco_df = eco_region_metrics(df, best['features'], args.n_splits)

    # Save artifacts
    ts = datetime.utcnow().isoformat() + 'Z'
    with open(Path(args.output_dir) / 'selection_curve_embeddings.json', 'w') as f:
        json.dump({'timestamp': ts, 'curve': curve}, f, indent=2)
    with open(Path(args.output_dir) / 'features_embeddings_selected.txt', 'w') as f:
        f.write("\n".join(best['features']))

    summary = {
        'timestamp': ts,
        'dataset_path': args.dataset_path,
        'n_splits': args.n_splits,
        'min_features': args.min_features,
        'step_schedule': args.step_schedule,
        'rf_params': {
            'n_estimators': 50,
            'max_depth': 30,
            'min_samples_split': 30,
            'min_samples_leaf': 15,
            'class_weight': 'balanced',
            'random_state': 42
        },
        'best': {
            'n_features': len(best['features']),
            'features': best['features'],
            **(best['summary'] or {})
        }
    }
    with open(Path(args.output_dir) / 'metrics_embeddings_selected.json', 'w') as f:
        json.dump(summary, f, indent=2)

    if not eco_df.empty:
        eco_df.to_csv(Path(args.output_dir) / 'eco_metrics_embeddings_selected.csv', index=False)

    logging.info(f"Saved selection artifacts under {args.output_dir}")


if __name__ == '__main__':
    main()
