#!/usr/bin/env python3
"""Compute calibration and threshold sensitivity metrics for RF phenology models.

This script reproduces cross-validated predictions using the eco-region
balanced folds so that every pixel receives an out-of-fold probability.
The predictions are then aggregated to report:
  * Macro-F1, balanced accuracy, class-wise recall & precision
  * Confusion counts for a set of decision thresholds (default 0.45, 0.50, 0.55)
  * Reliability curve statistics (10-bin histogram) and Expected Calibration Error
  * Per-eco-region summary statistics (mean ± std across folds)

Usage example
-------------
python src/analysis/compute_calibration_metrics.py \
    --dataset results/datasets/training_datasets_pixels_embedding.parquet \
    --features-file results/final_model/features_embeddings_topk_k14.txt \
    --output-dir results/analysis_calibration/embeddings \
    --model-type embeddings

Notes
-----
- Currently supports harmonic and embedding datasets. Provide a features
  list via --features-file; if omitted for harmonics, the script uses the
  24 Fourier feature columns automatically.
- The RandomForest hyper-parameters mirror the training defaults used in
  the repo (n_estimators=50, max_depth=30, min_samples_split=30,
  min_samples_leaf=15, class_weight='balanced').
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils import (
    create_eco_balanced_folds_df,
    compute_metrics,
    unscale_feature,
    transform_circular_features,
)
from src.constants import AVAILABLE_INDICES, FEATURE_SUFFIX_TO_TYPE

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_FEATURES_HARMONIC = [
    'ndvi_amplitude_h1', 'ndvi_amplitude_h2', 'ndvi_phase_h1', 'ndvi_phase_h2', 'ndvi_offset', 'ndvi_var_residual',
    'evi_amplitude_h1', 'evi_amplitude_h2', 'evi_phase_h1', 'evi_phase_h2', 'evi_offset', 'evi_var_residual',
    'nbr_amplitude_h1', 'nbr_amplitude_h2', 'nbr_phase_h1', 'nbr_phase_h2', 'nbr_offset', 'nbr_var_residual',
    'crswir_amplitude_h1', 'crswir_amplitude_h2', 'crswir_phase_h1', 'crswir_phase_h2', 'crswir_offset', 'crswir_var_residual'
]

THRESHOLDS_DEFAULT = [0.45, 0.50, 0.55]
NUM_BINS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute calibration metrics for phenology classifiers.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the pixel-level parquet dataset.")
    parser.add_argument("--features-file", type=Path, help="Optional text file with feature names (one per line).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to store metrics outputs.")
    parser.add_argument("--model-type", choices=["harmonic", "embeddings"], required=True, help="Model flavour (only for metadata).")
    parser.add_argument("--label-column", default="phenology", help="Name of the target column (default: phenology).")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of eco-balanced folds (default: 5).")
    parser.add_argument("--thresholds", type=float, nargs="*", default=None, help="Decision thresholds for hard labels (defaults to 0.45 0.50 0.55).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on number of rows for quick tests.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for fold shuffling.")
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=30)
    parser.add_argument("--min-samples-split", type=int, default=30)
    parser.add_argument("--min-samples-leaf", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs for RandomForest (default: -1).")
    return parser.parse_args()


def load_feature_names(args: argparse.Namespace, available_columns: Iterable[str]) -> List[str]:
    if args.features_file:
        features = [line.strip() for line in args.features_file.read_text().splitlines() if line.strip()]
        if not features:
            raise ValueError("Features file is empty.")
        return features

    # Fallback for harmonic dataset: autodetect the 24 Fourier features
    features = [col for col in DEFAULT_FEATURES_HARMONIC if col in available_columns]
    if not features:
        raise ValueError("No features provided and harmonic defaults missing from dataset.")
    return features


def read_dataset(path: Path, columns: List[str], max_rows: int | None = None) -> pd.DataFrame:
    LOGGER.info("Loading dataset %s with columns: %s", path, columns)
    df = pd.read_parquet(path, columns=columns)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        LOGGER.warning("Downsampled dataset to %d rows for quick test.", max_rows)
    return df


def init_threshold_containers(thresholds: List[float]) -> Dict[float, Dict[str, float]]:
    containers = {}
    for thr in thresholds:
        containers[thr] = {metric: 0.0 for metric in ("tp", "fp", "tn", "fn")}
    return containers


def update_threshold_counts(counts: Dict[float, Dict[str, float]], thresholds: List[float], y_true: np.ndarray, prob_evergreen: np.ndarray) -> None:
    decid_mask = (y_true == 1).astype(np.int8)
    ever_mask = (y_true == 2).astype(np.int8)
    for thr in thresholds:
        preds_ever = prob_evergreen >= thr
        tp = np.sum(preds_ever & (y_true == 2))
        fp = np.sum(preds_ever & (y_true == 1))
        fn = np.sum((~preds_ever) & (y_true == 2))
        tn = np.sum((~preds_ever) & (y_true == 1))
        counts[thr]["tp"] += tp
        counts[thr]["fp"] += fp
        counts[thr]["fn"] += fn
        counts[thr]["tn"] += tn


def metrics_from_conf_matrix(tn: float, fp: float, fn: float, tp: float) -> Dict[str, float]:
    total = tn + fp + fn + tp
    if total == 0:
        return {key: float("nan") for key in [
            "accuracy", "precision_deciduous", "precision_evergreen", "recall_deciduous", "recall_evergreen",
            "f1_deciduous", "f1_evergreen", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted"
        ]}

    precision_ever = tp / (tp + fp) if (tp + fp) else 0.0
    recall_ever = tp / (tp + fn) if (tp + fn) else 0.0
    f1_ever = (2 * precision_ever * recall_ever / (precision_ever + recall_ever)) if (precision_ever + recall_ever) else 0.0

    # Treat deciduous as positive by symmetry
    tp_decid = tn
    fp_decid = fn
    fn_decid = fp
    precision_decid = tp_decid / (tp_decid + fp_decid) if (tp_decid + fp_decid) else 0.0
    recall_decid = tp_decid / (tp_decid + fn_decid) if (tp_decid + fn_decid) else 0.0
    f1_decid = (2 * precision_decid * recall_decid / (precision_decid + recall_decid)) if (precision_decid + recall_decid) else 0.0

    precision_macro = (precision_decid + precision_ever) / 2
    recall_macro = (recall_decid + recall_ever) / 2
    f1_macro = (f1_decid + f1_ever) / 2

    support_decid = tp_decid + fn_decid  # equals tn + fp
    support_ever = tp + fn
    accuracy = (tp + tn) / total

    precision_weighted = ((precision_decid * support_decid) + (precision_ever * support_ever)) / total if total else 0.0
    recall_weighted = ((recall_decid * support_decid) + (recall_ever * support_ever)) / total if total else 0.0
    f1_weighted = ((f1_decid * support_decid) + (f1_ever * support_ever)) / total if total else 0.0

    return {
        "accuracy": accuracy,
        "precision_deciduous": precision_decid,
        "precision_evergreen": precision_ever,
        "recall_deciduous": recall_decid,
        "recall_evergreen": recall_ever,
        "f1_deciduous": f1_decid,
        "f1_evergreen": f1_ever,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def finalize_threshold_metrics(counts: Dict[float, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    results = {}
    for thr, c in counts.items():
        metrics = metrics_from_conf_matrix(c["tn"], c["fp"], c["fn"], c["tp"])
        metrics.update({key: c[key] for key in ("tn", "fp", "fn", "tp")})
        results[f"{thr:.2f}"] = metrics
    return results


def compute_reliability(probabilities: np.ndarray, labels: np.ndarray, num_bins: int = NUM_BINS) -> Dict[str, np.ndarray]:
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    idx = np.digitize(probabilities, edges, right=True) - 1
    idx = np.clip(idx, 0, num_bins - 1)

    bin_counts = np.bincount(idx, minlength=num_bins)
    bin_prob_sum = np.bincount(idx, weights=probabilities, minlength=num_bins)
    bin_true_sum = np.bincount(idx, weights=(labels == 2).astype(float), minlength=num_bins)

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_confidence = np.where(bin_counts > 0, bin_prob_sum / bin_counts, 0.0)
        empirical_prob = np.where(bin_counts > 0, bin_true_sum / bin_counts, 0.0)

    total = bin_counts.sum()
    weights = bin_counts / total if total else np.zeros_like(bin_counts, dtype=float)
    ece = np.sum(weights * np.abs(avg_confidence - empirical_prob))
    mce = np.max(np.abs(avg_confidence - empirical_prob)) if bin_counts.any() else 0.0

    reliability_df = pd.DataFrame({
        "bin_lower": edges[:-1],
        "bin_upper": edges[1:],
        "count": bin_counts,
        "fraction": weights,
        "mean_confidence": avg_confidence,
        "empirical_probability": empirical_prob,
    })

    return {
        "reliability_df": reliability_df,
        "ece": float(ece),
        "mce": float(mce),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine columns to load
    schema = pq.read_schema(args.dataset)
    available_columns = schema.names

    base_columns = [args.label_column, 'tile_id', 'eco_region']
    has_weight = 'weight' in available_columns
    if has_weight:
        base_columns.append('weight')

    # Determine feature list before loading all columns
    if args.features_file:
        features = load_feature_names(args, available_columns)
    else:
        features = load_feature_names(args, available_columns)

    additional_columns = set()
    if args.model_type == 'harmonic':
        for feat in features:
            if '_phase_h1_cos' in feat or '_phase_h1_sin' in feat:
                additional_columns.add(feat.replace('_phase_h1_cos', '_phase_h1').replace('_phase_h1_sin', '_phase_h1'))
            if '_phase_h2_cos' in feat or '_phase_h2_sin' in feat:
                additional_columns.add(feat.replace('_phase_h2_cos', '_phase_h2').replace('_phase_h2_sin', '_phase_h2'))

    features_in_dataset = [feat for feat in features if feat in available_columns]
    missing_before_processing = [feat for feat in features if feat not in available_columns]

    columns_to_load = list(dict.fromkeys(base_columns + features_in_dataset + list(additional_columns)))

    df = read_dataset(args.dataset, columns_to_load, max_rows=args.max_rows)
    df = df.reset_index(drop=True)

    if args.model_type == 'harmonic':
        LOGGER.info("Unscaling harmonic features and applying circular transforms...")
        df = df.copy()
        for index in AVAILABLE_INDICES:
            for suffix, feature_type in FEATURE_SUFFIX_TO_TYPE.items():
                col_name = f"{index}{suffix}"
                if col_name in df.columns:
                    index_name = index if feature_type in ('amplitude', 'offset') else None
                    df[col_name] = unscale_feature(df[col_name].to_numpy(), feature_type=feature_type, index_name=index_name)
        df = transform_circular_features(df, AVAILABLE_INDICES)

    # Validate feature presence after transformations
    missing_features = [feat for feat in features if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Dataset missing requested features after preprocessing: {missing_features[:5]}" + ("..." if len(missing_features) > 5 else ""))

    if args.model_type == 'embeddings' and 'weight' not in df.columns:
        LOGGER.warning("Embeddings dataset missing 'weight' column; proceeding without weights.")

    # Ensure eco_region column exists
    if 'eco_region' not in df.columns:
        raise ValueError("Dataset must contain an 'eco_region' column.")

    tile_col = df['tile_id'].to_numpy()
    eco_region_values = df['eco_region'].to_numpy()
    y = df[args.label_column].to_numpy()
    weights = df['weight'].to_numpy() if 'weight' in df.columns else None

    X = df[features].to_numpy(dtype=np.float32)
    del df  # free DataFrame memory

    thresholds = sorted(set(args.thresholds)) if args.thresholds else THRESHOLDS_DEFAULT
    threshold_counts = init_threshold_containers(thresholds)

    fold_stats = []
    eco_region_stats: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    prob_all = []
    label_all = []

    # Create folds using a lightweight DataFrame as the helper expects
    fold_df = pd.DataFrame({'tile_id': tile_col, 'eco_region': eco_region_values})
    folds = create_eco_balanced_folds_df(fold_df, n_splits=args.n_splits, random_state=args.random_state)

    LOGGER.info("Starting %d-fold cross-validation over %d samples (%d features).", len(folds), len(tile_col), len(features))

    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(folds, desc="Folds"), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        sample_weight = weights[train_idx] if weights is not None else None

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            class_weight='balanced',
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        proba_val = clf.predict_proba(X_val)
        classes = clf.classes_
        if 2 not in classes:
            raise ValueError("Trained classifier is missing class '2' in predict_proba output.")
        evergreen_idx = int(np.where(classes == 2)[0][0])
        prob_evergreen = proba_val[:, evergreen_idx]

        # Default threshold at 0.5 for fold metrics
        y_pred = np.where(prob_evergreen >= 0.5, 2, 1)
        metrics_fold = compute_metrics(y_val, y_pred)
        metrics_fold['fold'] = fold_idx
        fold_stats.append(metrics_fold)

        # Update threshold counts
        update_threshold_counts(threshold_counts, thresholds, y_val, prob_evergreen)

        # Reliability arrays
        prob_all.append(prob_evergreen)
        label_all.append(y_val)

        # Eco-region metrics (mean across folds later)
        eco_val = eco_region_values[val_idx]
        unique_regions = np.unique(eco_val)
        for region in unique_regions:
            mask = eco_val == region
            region_name = str(region)
            region_metrics = compute_metrics(y_val[mask], y_pred[mask])
            region_metrics['fold'] = fold_idx
            region_metrics['n_samples'] = int(mask.sum())
            eco_region_stats[region_name].append(region_metrics)

    prob_concat = np.concatenate(prob_all)
    labels_concat = np.concatenate(label_all)

    reliability_info = compute_reliability(prob_concat, labels_concat, num_bins=NUM_BINS)
    reliability_df = reliability_info['reliability_df']
    ece = reliability_info['ece']
    mce = reliability_info['mce']

    threshold_metrics = finalize_threshold_metrics(threshold_counts)

    # Persist outputs
    fold_df_out = pd.DataFrame(fold_stats)
    fold_path = args.output_dir / 'fold_metrics.csv'
    fold_df_out.to_csv(fold_path, index=False)

    eco_rows = []
    metric_keys = [
        'accuracy', 'precision_deciduous', 'recall_deciduous', 'f1_deciduous',
        'precision_evergreen', 'recall_evergreen', 'f1_evergreen',
        'precision_macro', 'recall_macro', 'f1_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted'
    ]
    for region, metrics_list in eco_region_stats.items():
        df_region = pd.DataFrame(metrics_list)
        summary = {'eco_region': region, 'n_samples_mean': df_region['n_samples'].mean(), 'n_samples_total': df_region['n_samples'].sum()}
        for key in metric_keys:
            if key in df_region:
                summary[f'{key}_mean'] = df_region[key].mean()
                summary[f'{key}_std'] = df_region[key].std()
        eco_rows.append(summary)

    eco_df = pd.DataFrame(eco_rows).sort_values('f1_macro_mean', ascending=False)
    eco_path = args.output_dir / 'eco_region_metrics.csv'
    eco_df.to_csv(eco_path, index=False)

    reliability_path = args.output_dir / 'reliability_curve.csv'
    reliability_df.to_csv(reliability_path, index=False)

    metrics_json = {
        'dataset': str(args.dataset),
        'model_type': args.model_type,
        'n_samples': int(len(prob_concat)),
        'n_features': len(features),
        'rf_params': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf': args.min_samples_leaf,
            'class_weight': 'balanced',
        },
        'threshold_metrics': threshold_metrics,
        'ece': ece,
        'mce': mce,
        'num_bins': NUM_BINS,
    }

    json_path = args.output_dir / 'calibration_metrics.json'
    json_path.write_text(json.dumps(metrics_json, indent=2))

    LOGGER.info("Saved fold metrics to %s", fold_path)
    LOGGER.info("Saved eco-region metrics to %s", eco_path)
    LOGGER.info("Saved reliability curve to %s (ECE=%.4f, MCE=%.4f)", reliability_path, ece, mce)
    LOGGER.info("Saved aggregated calibration metrics to %s", json_path)


if __name__ == '__main__':
    main()
