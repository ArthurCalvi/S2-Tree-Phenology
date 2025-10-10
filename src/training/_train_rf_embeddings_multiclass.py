#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared utilities to train Random Forest embedding models on multi-class targets."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# project imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in os.sys.path:
    os.sys.path.append(str(project_root))

from src.utils import create_eco_balanced_folds_df  # noqa: E402

RF_DEFAULTS: Dict[str, int | float] = {
    'n_estimators': 50,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'max_depth': 30,
    'min_samples_split': 30,
}


def setup_logging(log_name: str) -> None:
    os.makedirs('logs', exist_ok=True)
    log_path = Path('logs') / log_name
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging to %s", log_path)


def build_rf_classifier(n_estimators: int = 50,
                        max_depth: int = 30,
                        min_samples_split: int = 30) -> RandomForestClassifier:
    params = RF_DEFAULTS.copy()
    params.update({
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
    })
    params['min_samples_leaf'] = max(1, params['min_samples_split'] // 2)
    return RandomForestClassifier(**params)


def select_topk_features(X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    return list(importances.sort_values(ascending=False).head(k).index)


def compute_multiclass_metrics(y_true: Sequence[int],
                               y_pred: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if len(y_true) == 0:
        metrics.update({
            'accuracy': float('nan'),
            'balanced_accuracy': float('nan'),
            'precision_macro': float('nan'),
            'recall_macro': float('nan'),
            'f1_macro': float('nan'),
            'f1_weighted': float('nan'),
        })
        metrics['support'] = 0
        return metrics

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['support'] = int(y_true.size)
    return metrics


def to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def run_cv(df: pd.DataFrame,
           feature_cols: List[str],
           target_col: str,
           labels: List[int],
           n_splits: int = 5,
           n_estimators: int = 50,
           max_depth: int = 30,
           min_samples_split: int = 30) -> Tuple[Dict, pd.DataFrame, Dict, Dict]:
    X = df[feature_cols]
    y = df[target_col]

    folds = create_eco_balanced_folds_df(df, n_splits=n_splits, random_state=42)

    results_per_fold: List[Dict[str, float]] = []
    results_per_ecoregion: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc='CV folds')):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        sample_weights = None
        if 'weight' in df.columns:
            sample_weights = df.iloc[tr_idx]['weight'].values

        clf = build_rf_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        clf.fit(Xtr, ytr, sample_weight=sample_weights)
        ypred = clf.predict(Xva)

        metrics = compute_multiclass_metrics(yva, ypred)
        metrics['fold'] = fold + 1
        results_per_fold.append(metrics)

        if 'eco_region' in df.columns:
            eco_regions_in_val = df.iloc[va_idx]['eco_region'].unique()
            for eco in eco_regions_in_val:
                mask = (df.iloc[va_idx]['eco_region'] == eco)
                if mask.any():
                    eco_metrics = compute_multiclass_metrics(yva[mask], ypred[mask])
                    eco_metrics['fold'] = fold + 1
                    eco_metrics['eco_region'] = str(eco)
                    results_per_ecoregion[str(eco)].append(eco_metrics)

        all_true.append(yva.to_numpy())
        all_pred.append(ypred)

    res = pd.DataFrame(results_per_fold)
    metric_columns = [c for c in res.columns if c not in {'fold', 'support'}]
    summary = {
        'n_features': len(feature_cols),
        'features': feature_cols,
        'metrics_mean': res[metric_columns].mean(numeric_only=True).to_dict(),
        'metrics_std': res[metric_columns].std(numeric_only=True).to_dict(),
        'support_total': int(res['support'].sum())
    }

    eco_rows = []
    for eco, entries in results_per_ecoregion.items():
        eco_df = pd.DataFrame(entries)
        metric_cols = [c for c in eco_df.columns if c not in {'fold', 'eco_region', 'support'}]
        agg = eco_df[metric_cols].mean(numeric_only=True).to_dict()
        agg['eco_region'] = eco
        agg['n_samples_total'] = int(eco_df['support'].sum()) if 'support' in eco_df else None
        eco_rows.append(agg)
    eco_metrics_df = pd.DataFrame(eco_rows) if eco_rows else pd.DataFrame()

    y_true_full = np.concatenate(all_true) if all_true else np.array([])
    y_pred_full = np.concatenate(all_pred) if all_pred else np.array([])
    if y_true_full.size == 0:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        report = {}
    else:
        cm = confusion_matrix(y_true_full, y_pred_full, labels=labels)
        report = classification_report(
            y_true_full,
            y_pred_full,
            labels=labels,
            zero_division=0,
            output_dict=True
        )

    return summary, eco_metrics_df, {
        'labels': labels,
        'matrix': cm
    }, report


def parse_args(target_column: str,
               target_name: str,
               default_output_dir: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f'Train RF on embeddings for {target_name} classification'
    )
    parser.add_argument('--dataset_path', required=True,
                        help='Path to training_datasets_pixels_embedding.parquet')
    parser.add_argument('--config', choices=['all', 'topk'], default='all',
                        help='Use all features or top-K by RF importance')
    parser.add_argument('--k', type=int, default=14,
                        help='K for topk config (default: 14)')
    parser.add_argument('--features_file', type=str, default=None,
                        help='Optional path to a file listing selected features (one per line)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of eco-balanced CV splits (default: 5)')
    parser.add_argument('--output_dir', default=default_output_dir,
                        help=f'Output directory (default: {default_output_dir})')
    parser.add_argument('--sample_n', type=int, default=None,
                        help='Optional random sample size for quick runs')
    return parser.parse_args()


def run(target_column: str,
        target_name: str,
        default_tag: str,
        log_filename: str) -> None:
    args = parse_args(
        target_column=target_column,
        target_name=target_name,
        default_output_dir=f'results/final_model_{default_tag}',
    )

    setup_logging(log_filename)

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.dataset_path)

    if target_column not in df.columns:
        raise SystemExit(f'Target column {target_column} missing from dataset')

    initial_rows = len(df)
    df = df[df[target_column].notna()]
    df = df[df[target_column] > 0]
    df[target_column] = df[target_column].astype(int)
    logging.info('Filtered dataset from %d to %d rows by %s > 0', initial_rows, len(df), target_column)

    if args.sample_n and args.sample_n < len(df):
        df = df.sample(args.sample_n, random_state=42)
        logging.info('Sampled down to %d rows', len(df))

    feature_cols = sorted([c for c in df.columns if c.startswith('embedding_')])
    if not feature_cols:
        raise SystemExit('No embedding_* columns found in dataset')
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)

    labels = sorted(df[target_column].unique().tolist())
    class_counts = df[target_column].value_counts().sort_index().to_dict()
    logging.info('%s classes: %d (%s)', target_name, len(labels), class_counts)

    chosen_features = feature_cols
    chosen_k: Optional[int] = None
    if args.config == 'topk':
        base_features = chosen_features
        if args.features_file and Path(args.features_file).exists():
            with open(args.features_file, 'r') as f:
                requested = [ln.strip() for ln in f.readlines() if ln.strip()]
            present = [c for c in requested if c in df.columns]
            missing = [c for c in requested if c not in df.columns]
            if missing:
                logging.warning('Ignoring %d missing features from %s (e.g. %s)', len(missing), args.features_file, missing[:5])
            if not present:
                raise SystemExit('No valid features after reading features_file')
            chosen_features = present
            chosen_k = len(chosen_features)
            logging.info('Using %d features provided via %s', chosen_k, args.features_file)
        else:
            chosen_features = select_topk_features(df[base_features], df[target_column], args.k)
            chosen_k = len(chosen_features)
            logging.info('Selected top-%d features via RF importance', chosen_k)
    logging.info('Training with %d features', len(chosen_features))

    rf_n_estimators = RF_DEFAULTS['n_estimators']
    rf_max_depth = RF_DEFAULTS['max_depth']
    rf_min_samples_split = RF_DEFAULTS['min_samples_split']

    summary, eco_df, confusion_info, class_report = run_cv(
        df,
        chosen_features,
        target_column,
        labels,
        n_splits=args.n_splits,
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
    )

    if args.config == 'all':
        tag = f'{default_tag}_embeddings_all'
    else:
        k_tag = chosen_k if chosen_k is not None else args.k
        tag = f'{default_tag}_embeddings_topk_k{k_tag}'

    logging.info('Training final RandomForest on full dataset')
    final_model = build_rf_classifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
    )
    sample_weights_full = df['weight'].values if 'weight' in df.columns else None
    final_model.fit(df[chosen_features], df[target_column], sample_weight=sample_weights_full)

    model_ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    model_filename = f'rf_embeddings_{tag}_{model_ts}.joblib'
    model_path = Path(args.output_dir) / model_filename
    joblib.dump(final_model, model_path)
    logging.info('Saved model to %s', model_path)

    metadata = {
        'model_file': model_filename,
        'created_at_utc': model_ts,
        'dataset_path': args.dataset_path,
        'config': args.config,
        'k': chosen_k if args.config == 'topk' else None,
        'features_file': args.features_file if args.config == 'topk' else None,
        'features': chosen_features,
        'classes': to_serializable(final_model.classes_.tolist()),
        'class_counts': to_serializable(class_counts),
        'target_column': target_column,
        'target_name': target_name,
        'rf_params': {
            'n_estimators': rf_n_estimators,
            'max_depth': rf_max_depth,
            'min_samples_split': rf_min_samples_split,
            'min_samples_leaf': final_model.min_samples_leaf,
            'class_weight': 'balanced',
            'random_state': RF_DEFAULTS['random_state'],
            'n_jobs': final_model.n_jobs,
        },
    }
    with open(Path(args.output_dir) / f'model_metadata_{tag}.json', 'w') as f:
        json.dump(to_serializable(metadata), f, indent=2)

    summary_out = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'dataset_path': args.dataset_path,
        'target_column': target_column,
        'target_name': target_name,
        'config': args.config,
        'k': chosen_k if args.config == 'topk' else None,
        'features_file': args.features_file if args.config == 'topk' else None,
        'generated_features_file': f'features_{tag}.txt',
        'n_splits': args.n_splits,
        'rf_params': metadata['rf_params'],
        'model_file': model_filename,
        'classes': to_serializable(final_model.classes_.tolist()),
        'class_counts': to_serializable(class_counts),
        **summary,
        'confusion_matrix': {
            'labels': to_serializable(confusion_info['labels']),
            'matrix': to_serializable(confusion_info['matrix'].tolist()),
        },
        'classification_report': to_serializable(class_report),
    }
    with open(Path(args.output_dir) / f'metrics_{tag}.json', 'w') as f:
        json.dump(to_serializable(summary_out), f, indent=2)

    with open(Path(args.output_dir) / f'classification_report_{tag}.json', 'w') as f:
        json.dump(to_serializable(class_report), f, indent=2)

    with open(Path(args.output_dir) / f'confusion_matrix_{tag}.json', 'w') as f:
        json.dump({
            'labels': to_serializable(confusion_info['labels']),
            'matrix': to_serializable(confusion_info['matrix'].tolist()),
        }, f, indent=2)

    with open(Path(args.output_dir) / f'features_{tag}.txt', 'w') as f:
        f.write('\n'.join(chosen_features))

    if not eco_df.empty:
        eco_df.to_csv(Path(args.output_dir) / f'eco_metrics_{tag}.csv', index=False)

    logging.info('Saved artefacts to %s', args.output_dir)


__all__ = ['run']
