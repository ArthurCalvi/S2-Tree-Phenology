import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import HalvingGridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

# Ensure project root is on the import path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.constants import GENUS_MAPPING, PHENOLOGY_MAPPING
from src.utils import (
    compute_metrics,
    compute_multiclass_metrics,
    transform_circular_features,
    unscale_feature,
    create_eco_balanced_folds_df,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DATASET_PATH = "results/datasets/training_datasets_pixels.parquet"
INDICES = ["ndvi", "evi", "nbr", "crswir"]
FEATURE_TYPES_TO_UNSCALE = {
    "amplitude_h1": "amplitude",
    "amplitude_h2": "amplitude",
    "phase_h1": "phase",
    "phase_h2": "phase",
    "offset": "offset",
    "var_residual": "variance",
}

DEFAULT_METRICS = [
    "accuracy",
    "precision_deciduous",
    "recall_deciduous",
    "f1_deciduous",
    "precision_evergreen",
    "recall_evergreen",
    "f1_evergreen",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
]

GENUS_METRICS = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
    "recall_macro",
    "recall_weighted",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def setup_logger(estimator_name: str, target: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/tune_tabular_{estimator_name}_{target}.log"

    logger = logging.getLogger(f"tune_tabular_{estimator_name}_{target}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the script runs multiple times
    if not logger.handlers:
        handler_stream = logging.StreamHandler()
        handler_file = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s"
        )
        handler_stream.setFormatter(formatter)
        handler_file.setFormatter(formatter)
        logger.addHandler(handler_stream)
        logger.addHandler(handler_file)

    return logger


def read_feature_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    features: List[str] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            features.append(line)
    return features


def serialize_for_json(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {serialize_for_json(key): serialize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(obj)
    return str(obj)


def build_estimator_and_grid(
    estimator_name: str,
) -> Tuple[object, List[Dict[str, object]], str]:
    if estimator_name == "logreg":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="saga",
                        penalty="l2",
                        C=1.0,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        )
        param_grid: List[Dict[str, object]] = [
            {"clf__penalty": ["l2"], "clf__C": [0.01, 0.1, 1.0, 10.0]},
            {
                "clf__penalty": ["elasticnet"],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__l1_ratio": [0.0, 0.5, 0.9],
            },
        ]
        fit_param = "clf__sample_weight"
    elif estimator_name == "linear_svm":
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=42,
                        max_iter=15000,
                        dual=True,
                        loss="squared_hinge",
                    ),
                ),
            ]
        )
        param_grid = [
            {
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__loss": ["hinge", "squared_hinge"],
            }
        ]
        fit_param = "clf__sample_weight"
    elif estimator_name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        param_grid = [
            {
                "n_estimators": [300, 600],
                "max_depth": [None, 20, 30],
                "max_features": ["sqrt", "log2", 0.5],
                "min_samples_leaf": [1, 5, 10],
            }
        ]
        fit_param = "sample_weight"
    elif estimator_name == "histgb":
        estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.1,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            early_stopping=True,
            random_state=42,
            max_bins=255,
        )
        param_grid = [
            {
                "learning_rate": [0.05, 0.1, 0.2],
                "max_leaf_nodes": [31, 63, 127],
                "min_samples_leaf": [20, 50],
                "l2_regularization": [0.0, 1e-4, 1e-3],
            }
        ]
        fit_param = "sample_weight"
    else:
        raise ValueError(f"Unsupported estimator: {estimator_name}")

    return estimator, param_grid, fit_param


def evaluate_estimator_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    fold_splits: List[Tuple[np.ndarray, np.ndarray]],
    sample_weight: Optional[np.ndarray],
    fit_param_name: Optional[str],
    metrics_func,
    metrics_kwargs: Dict,
    logger: logging.Logger,
) -> Dict[str, float]:
    results_per_fold: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(fold_splits, desc="Evaluating best model", leave=False), start=1
    ):
        estimator_fold = clone(estimator)
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        fit_kwargs = {}
        if sample_weight is not None and fit_param_name is not None:
            fit_kwargs[fit_param_name] = sample_weight[train_idx]

        estimator_fold.fit(X_train, y_train, **fit_kwargs)

        y_pred = estimator_fold.predict(X_val)
        metrics = metrics_func(y_val, y_pred, **metrics_kwargs)
        metrics["fold"] = fold_idx
        results_per_fold.append(metrics)

    results_df = pd.DataFrame(results_per_fold)

    metric_keys = [k for k in results_df.columns if k != "fold"]
    summary: Dict[str, float] = {}
    for metric in metric_keys:
        summary[f"mean_{metric}"] = results_df[metric].mean()
        summary[f"std_{metric}"] = results_df[metric].std()

    summary["quantiles"] = {
        "q25": results_df[metric_keys].quantile(0.25).to_dict(),
        "median": results_df[metric_keys].quantile(0.50).to_dict(),
        "q75": results_df[metric_keys].quantile(0.75).to_dict(),
    }

    logger.info("Cross-validation summary (best estimator):")
    for metric in metric_keys:
        logger.info(
            f"  {metric}: {summary[f'mean_{metric}']:.4f} ± {summary[f'std_{metric}']:.4f}"
        )

    return summary


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Tune or train tabular models (logistic, linear SVM, Extra Trees, HistGB) with eco-balanced CV."
    )
    parser.add_argument(
        "--estimator",
        required=True,
        choices=["logreg", "linear_svm", "extra_trees", "histgb"],
        help="Estimator to train/tune.",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        required=True,
        help="Path to a file containing one feature name per line.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help=f"Path to the parquet dataset (default: {DATASET_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tuning_tabular",
        help="Directory to store tuning results and trained models.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default=None,
        help="Optional filename for tuning metrics JSON.",
    )
    parser.add_argument(
        "--best-model-name",
        type=str,
        default=None,
        help="Optional filename for the trained model artifact.",
    )
    parser.add_argument(
        "--param-json",
        type=str,
        default=None,
        help="Path to JSON with pre-defined hyperparameters (skip tuning).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="phenology",
        choices=["phenology", "genus"],
        help="Target column to predict.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of eco-balanced cross-validation splits.",
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=3,
        help="HalvingGridSearchCV factor.",
    )
    parser.add_argument(
        "--min-resources",
        type=int,
        default=700000,
        help="Minimum resources (samples) for the first halving iteration.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode with stratified sampling.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=10000,
        help="Number of samples to use in test mode.",
    )

    args = parser.parse_args()

    logger = setup_logger(args.estimator, args.target_column)
    logger.info("=== Tabular model tuning/training script ===")
    logger.info(f"Estimator: {args.estimator}")
    logger.info(f"Target column: {args.target_column}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Features file: {args.features_file}")
    if args.param_json:
        logger.info(f"Loading hyperparameters from: {args.param_json}")
    if args.test:
        logger.info(f"Test mode enabled (sample size: {args.test_size})")

    # Load dataset
    try:
        df = pd.read_parquet(args.dataset_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found: {args.dataset_path}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Failed to load dataset: {exc}")
        sys.exit(1)

    logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")

    if args.target_column not in df.columns:
        logger.error(
            f"Target column '{args.target_column}' not found. Available columns: {df.columns.tolist()}"
        )
        sys.exit(1)

    # Apply test sampling if requested
    if args.test and len(df) > args.test_size:
        logger.info("Sampling dataset for test mode...")
        n_classes = df[args.target_column].nunique()
        sample_per_class = max(1, args.test_size // max(1, n_classes))
        df = (
            df.groupby(args.target_column, group_keys=False)
            .apply(lambda grp: grp.sample(min(len(grp), sample_per_class), random_state=42))
            .reset_index(drop=True)
        )
        logger.info(f"Test subset size: {len(df)}")

    # Unscale and transform harmonic features
    logger.info("Unscaling harmonic features to physical ranges...")
    df_copy = df.copy()
    unscaled_columns = 0
    skipped_columns: List[str] = []
    for index_name in INDICES:
        for suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            column_name = f"{index_name}_{suffix}"
            if column_name not in df_copy.columns:
                continue
            try:
                df_copy[column_name] = unscale_feature(
                    df_copy[column_name],
                    feature_type=feature_type,
                    index_name=index_name,
                )
                unscaled_columns += 1
            except Exception as exc:
                logger.warning(f"Failed to unscale column {column_name}: {exc}")
                skipped_columns.append(column_name)

    if skipped_columns:
        logger.info(f"Skipped {len(skipped_columns)} columns during unscaling (not present or failed).")
    logger.info(f"Unscaled {unscaled_columns} columns.")
    df = transform_circular_features(df_copy, INDICES)
    logger.info("Applied circular feature transformation.")

    # Load feature list
    feature_file = Path(args.features_file)
    selected_features = read_feature_list(feature_file)
    logger.info(f"Loaded {len(selected_features)} features from {feature_file.name}")

    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        logger.error(f"Missing features in dataset: {missing_features}")
        sys.exit(1)

    X = df[selected_features]
    y = df[args.target_column]

    sample_weight = df["weight"].to_numpy() if "weight" in df.columns else None
    if sample_weight is not None:
        logger.info(
            "Sample weights detected. Stats -> "
            f"min: {sample_weight.min():.4f}, max: {sample_weight.max():.4f}, "
            f"mean: {sample_weight.mean():.4f}"
        )

    fold_splits = create_eco_balanced_folds_df(df, n_splits=args.n_splits, random_state=42)
    logger.info(f"Generated {len(fold_splits)} eco-balanced splits.")

    # Target-specific configuration
    if args.target_column == "phenology":
        scorer_average = "weighted"
        metrics_func = compute_metrics
        metrics_kwargs: Dict = {}
        metrics_list = DEFAULT_METRICS
    else:
        scorer_average = "macro"
        metrics_func = compute_multiclass_metrics
        metrics_kwargs = {
            "labels": sorted(GENUS_MAPPING.keys()),
            "target_names": [GENUS_MAPPING[k] for k in sorted(GENUS_MAPPING.keys())],
        }
        metrics_list = GENUS_METRICS

    estimator, param_grid, fit_param_name = build_estimator_and_grid(args.estimator)
    logger.info(f"Estimator configured: {estimator}")
    total_candidates = sum(len(ParameterGrid(grid)) for grid in param_grid)
    logger.info(f"Total hyperparameter combinations: {total_candidates}")

    best_estimator = None
    best_params: Dict[str, object]
    tuning_results: Optional[Dict[str, object]] = None

    if args.param_json:
        with open(args.param_json, "r") as handle:
            param_payload = json.load(handle)
        best_params = param_payload.get("best_params", param_payload)
        estimator.set_params(**best_params)
        logger.info(f"Loaded parameters: {best_params}")
    else:
        logger.info("Starting HalvingGridSearchCV...")
        scorer = make_scorer(f1_score, average=scorer_average)

        actual_min_resources = args.min_resources
        if args.test:
            actual_min_resources = max(100, min(len(X), int(len(X) * 0.1)))
            logger.info(f"Adjusted min_resources for test mode -> {actual_min_resources}")

        search = HalvingGridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scorer,
            factor=args.factor,
            min_resources=actual_min_resources,
            aggressive_elimination=False,
            cv=fold_splits,
            random_state=42,
            n_jobs=-1,
            verbose=2,
        )

        fit_kwargs = {}
        if sample_weight is not None and fit_param_name is not None:
            fit_kwargs[fit_param_name] = sample_weight

        start = time.time()
        search.fit(X, y, **fit_kwargs)
        duration = time.time() - start
        logger.info(f"HalvingGridSearchCV complete in {duration/60:.2f} minutes.")
        logger.info(f"Best params: {search.best_params_}")
        logger.info(f"Best {scorer_average} F1 (CV): {search.best_score_:.4f}")

        best_estimator = search.best_estimator_
        best_params = search.best_params_

        tuning_results = {
            "best_params": best_params,
            f"best_score_f1_{scorer_average}": search.best_score_,
            "halving_factor": args.factor,
            "halving_min_resources": args.min_resources,
            "runtime_seconds": round(duration, 2),
            "cv_results": {
                key: serialize_for_json(val)
                for key, val in search.cv_results_.items()
            },
        }

    if best_estimator is None:
        estimator.set_params(**best_params)
        best_estimator = estimator

    cv_summary = evaluate_estimator_cv(
        best_estimator,
        X,
        y,
        fold_splits,
        sample_weight,
        fit_param_name,
        metrics_func,
        metrics_kwargs,
        logger,
    )

    # Fit final model on the full dataset
    logger.info("Fitting best estimator on the full dataset...")
    final_estimator = clone(best_estimator)
    fit_kwargs_full = {}
    if sample_weight is not None and fit_param_name is not None:
        fit_kwargs_full[fit_param_name] = sample_weight
    final_estimator.fit(X, y, **fit_kwargs_full)

    output_dir = Path(args.output_dir) / args.estimator / args.target_column
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_name = (
        args.results_name
        if args.results_name
        else f"{args.estimator}_{args.target_column}_metrics_{run_timestamp}.json"
    )
    model_name = (
        args.best_model_name
        if args.best_model_name
        else f"{args.estimator}_{args.target_column}_model_{run_timestamp}.joblib"
    )

    results_payload = {
        "estimator": args.estimator,
        "target_column": args.target_column,
        "selected_features": selected_features,
        "best_params": best_params,
        "cv_metrics": cv_summary,
        "n_splits": args.n_splits,
        "factor": args.factor,
        "min_resources": args.min_resources,
        "test_mode": args.test,
    }
    if tuning_results:
        results_payload["tuning"] = tuning_results

    results_path = output_dir / results_name
    with results_path.open("w") as handle:
        json.dump(results_payload, handle, indent=4, default=serialize_for_json)
    logger.info(f"Saved metrics to {results_path}")

    model_path = output_dir / model_name
    joblib.dump(final_estimator, model_path)
    logger.info(f"Saved model to {model_path}")
    logger.info("=== Script complete ===")


if __name__ == "__main__":
    main()
