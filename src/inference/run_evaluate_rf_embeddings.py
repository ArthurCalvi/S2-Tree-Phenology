#!/usr/bin/env python3
"""
Evaluate a trained RandomForest embedding model on an alternative embeddings parquet.

This script applies a serialized RandomForest model (trained on AlphaEarth embeddings)
to a new embeddings parquet (e.g., a different year), re-uses the eco-balanced cross
validation folds from training, and emits a consistent set of metrics/artefacts:

- Overall metrics JSON with fold statistics (mean/std/quantiles) and bookkeeping fields.
- Per-fold metrics CSV mirroring the training summary.
- Per-eco-region metrics CSV.
- Per-tile metrics parquet for downstream heterogeneity analysis.
- Per-sample predictions parquet (incl. fold assignment and probabilities).

Typical usage:
    python src/inference/run_evaluate_rf_embeddings.py \
        --dataset_path results/datasets/training_datasets_pixels_embedding_2022.parquet \
        --model_path results/final_model/rf_embeddings_embeddings_topk_k14_*.joblib \
        --features_file results/final_model/features_embeddings_topk_k14.txt \
        --fold_manifest results/final_model/cv_predictions_embeddings_topk_k14.parquet \
        --tag embeddings_2022 \
        --output_dir results/evaluation/embeddings_2022
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path (important for HPC / cron jobs)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import compute_metrics  # noqa: E402


LOGGER = logging.getLogger("evaluate_rf_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RandomForest embeddings model on alternate embeddings parquet."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to embeddings parquet to evaluate (must contain embedding_* columns and phenology labels).",
    )
    parser.add_argument(
        "--model_dir",
        help="Directory containing latest embeddings artefacts (e.g., results/final_model/latest_embeddings_topk_k14). "
             "If provided, model/features/manifest paths are auto-detected.",
    )
    parser.add_argument(
        "--model_path",
        help="Path to the trained RandomForest joblib file (mutually exclusive with --model_dir).",
    )
    parser.add_argument(
        "--features_file",
        help="Path to text file listing embedding features (one per line) to use for inference (mutually exclusive with --model_dir).",
    )
    parser.add_argument(
        "--fold_manifest",
        help="Parquet generated during training containing columns tile_id,row,col,fold (e.g., cv_predictions_*.parquet). "
             "(Mutually exclusive with --model_dir).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where evaluation artefacts will be written.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag used to prefix output filenames (e.g., embeddings_2022).",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Optional limit on number of rows for quick smoke tests.",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_feature_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    features = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not features:
        raise ValueError(f"Features file {path} is empty.")
    for feat in features:
        if not feat.startswith("embedding_"):
            raise ValueError(f"Unexpected feature name '{feat}'. Expected 'embedding_<idx>'.")
    return features


def ensure_columns(df: pd.DataFrame, columns: List[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"{context} missing required columns: {missing}")


def resolve_artifacts_from_dir(model_dir: Path) -> Tuple[Path, Path, Path]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    def pick_latest(pattern: str) -> Path:
        candidates = sorted(model_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No files matching '{pattern}' in {model_dir}")
        return candidates[0]

    model_path = pick_latest("*.joblib")
    features_file = pick_latest("features_*.txt")
    manifest = pick_latest("cv_predictions_*.parquet")

    logging.info("Auto-detected artefacts in %s:", model_dir)
    logging.info("  model:    %s", model_path.name)
    logging.info("  features: %s", features_file.name)
    logging.info("  manifest: %s", manifest.name)

    return model_path, features_file, manifest


def metrics_dict_to_row(metrics: dict, extra: Optional[dict] = None) -> dict:
    """Convert metrics dict (output of compute_metrics) to JSON-serialisable row."""
    row = {k: (float(v) if isinstance(v, (np.floating, np.float64, np.float32)) else v)
           for k, v in metrics.items()}
    if extra:
        row.update(extra)
    return row


def build_group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, sub_df in df.groupby(group_col):
        metrics = compute_metrics(sub_df["y_true"].values, sub_df["y_pred"].values)
        extra = {
            group_col: key,
            "n_samples": int(len(sub_df)),
        }
        rows.append(metrics_dict_to_row(metrics, extra))
    if not rows:
        return pd.DataFrame(columns=[group_col, "n_samples"])
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    setup_logging(args.loglevel)

    dataset_path = Path(args.dataset_path)
    model_dir = Path(args.model_dir) if args.model_dir else None

    if model_dir:
        if any([args.model_path, args.features_file, args.fold_manifest]):
            raise SystemExit("When using --model_dir, do not supply --model_path/--features_file/--fold_manifest.")
        model_path, features_path, manifest_path = resolve_artifacts_from_dir(model_dir)
    else:
        if not (args.model_path and args.features_file and args.fold_manifest):
            raise SystemExit("Provide either --model_dir or all of --model_path/--features_file/--fold_manifest.")
        model_path = Path(args.model_path)
        features_path = Path(args.features_file)
        manifest_path = Path(args.fold_manifest)

    output_dir = Path(args.output_dir)

    LOGGER.info("Loading dataset from %s", dataset_path)
    df = pd.read_parquet(dataset_path)
    LOGGER.info("Dataset loaded with %d rows and %d columns", len(df), len(df.columns))

    if args.sample_n and args.sample_n < len(df):
        LOGGER.info("Sampling %d rows for quick evaluation", args.sample_n)
        df = df.sample(args.sample_n, random_state=42).reset_index(drop=True)

    required_cols = ["phenology", "tile_id", "row", "col"]
    ensure_columns(df, required_cols, "Evaluation dataset")
    if "eco_region" not in df.columns:
        LOGGER.warning("eco_region column not found; eco metrics will be skipped.")

    features = load_feature_list(features_path)
    ensure_columns(df, features, "Evaluation dataset (features)")

    LOGGER.info("Loading manifest from %s", manifest_path)
    manifest = pd.read_parquet(manifest_path)
    ensure_columns(manifest, ["tile_id", "row", "col", "fold"], "Fold manifest")
    manifest = manifest.drop_duplicates(subset=["tile_id", "row", "col"])
    LOGGER.info("Manifest contains %d unique samples (fold assignments)", len(manifest))

    LOGGER.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    LOGGER.info("Preparing feature matrix")
    X = df[features].replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    y_true = df["phenology"].astype(int)

    LOGGER.info("Running inference (predict_proba)")
    probas = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    prob_df = pd.DataFrame(
        probas,
        columns=[f"prob_class_{cls}" for cls in classes],
        index=df.index,
    )
    y_pred = model.predict(X).astype(int)

    LOGGER.info("Merging predictions with fold manifest")
    result_cols = ["tile_id", "row", "col", "phenology"]
    optional_cols = [col for col in ["eco_region", "weight", "NomSER", "utm_x", "utm_y"] if col in df.columns]
    result_df = df[result_cols + optional_cols].copy()
    result_df = result_df.rename(columns={"phenology": "y_true"})
    result_df["y_pred"] = y_pred
    result_df = pd.concat([result_df, prob_df], axis=1)

    merged = result_df.merge(
        manifest[["tile_id", "row", "col", "fold"]],
        on=["tile_id", "row", "col"],
        how="left",
        validate="one_to_one",
    )
    missing_folds = merged["fold"].isna().sum()
    if missing_folds > 0:
        LOGGER.warning("Fold manifest missing for %d samples; these will be dropped.", missing_folds)
        merged = merged.dropna(subset=["fold"])
    merged["fold"] = merged["fold"].astype(int)

    LOGGER.info("Computing overall metrics")
    overall_metrics = compute_metrics(merged["y_true"].values, merged["y_pred"].values)
    overall_row = metrics_dict_to_row(overall_metrics, {"n_samples": int(len(merged))})

    LOGGER.info("Computing per-fold metrics")
    fold_metrics_df = build_group_metrics(merged, "fold").sort_values("fold")

    LOGGER.info("Computing per-eco-region metrics")
    eco_metrics_df = pd.DataFrame()
    if "eco_region" in merged.columns:
        eco_metrics_df = build_group_metrics(merged, "eco_region").sort_values("eco_region")

    LOGGER.info("Computing per-tile metrics")
    tile_metrics_df = build_group_metrics(merged, "tile_id").sort_values("tile_id")

    LOGGER.info("Calculating fold summary statistics (mean/std/quantiles)")
    numeric_cols = [col for col in fold_metrics_df.columns if col not in ("fold",)]
    fold_metrics_mean = fold_metrics_df[numeric_cols].mean().to_dict() if not fold_metrics_df.empty else {}
    fold_metrics_std = fold_metrics_df[numeric_cols].std().to_dict() if not fold_metrics_df.empty else {}
    fold_metrics_quantiles = {}
    if not fold_metrics_df.empty:
        for quant, label in zip([0.25, 0.5, 0.75], ["q25", "median", "q75"]):
            fold_metrics_quantiles[label] = fold_metrics_df[numeric_cols].quantile(quant).to_dict()

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag

    LOGGER.info("Writing predictions parquet")
    preds_path = output_dir / f"predictions_{tag}.parquet"
    merged.to_parquet(preds_path, index=False, coerce_timestamps="ms")

    LOGGER.info("Writing fold metrics CSV")
    fold_metrics_path = output_dir / f"fold_metrics_{tag}.csv"
    fold_metrics_df.to_csv(fold_metrics_path, index=False, float_format="%.6f")

    if not eco_metrics_df.empty:
        eco_metrics_path = output_dir / f"eco_metrics_{tag}.csv"
        eco_metrics_df.to_csv(eco_metrics_path, index=False, float_format="%.6f")
    else:
        eco_metrics_path = None
        LOGGER.warning("Eco-region metrics dataframe empty; skipping CSV.")

    LOGGER.info("Writing tile metrics parquet")
    tile_metrics_path = output_dir / f"tile_metrics_{tag}.parquet"
    tile_metrics_df.to_parquet(tile_metrics_path, index=False)

    LOGGER.info("Creating summary JSON")
    def _jsonify(val):
        if isinstance(val, (np.integer, np.int64, np.int32, np.uint8, np.uint16)):
            return int(val)
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        return val

    summary = {
        "timestamp": pd.Timestamp.utcnow().isoformat() + "Z",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "features_file": str(features_path),
        "fold_manifest": str(manifest_path),
        "tag": tag,
        "n_samples": int(len(merged)),
        "class_labels": classes,
        "overall_metrics": {k: _jsonify(v) for k, v in overall_row.items()},
        "fold_metrics_mean": {k: _jsonify(v) for k, v in fold_metrics_mean.items()},
        "fold_metrics_std": {k: _jsonify(v) for k, v in fold_metrics_std.items()},
        "fold_metrics_quantiles": {
            q: {k: _jsonify(v) for k, v in vals.items()}
            for q, vals in fold_metrics_quantiles.items()
        },
        "fold_metrics_file": fold_metrics_path.name,
        "tile_metrics_file": tile_metrics_path.name,
        "predictions_file": preds_path.name,
        "eco_metrics_file": Path(eco_metrics_path).name if eco_metrics_path else None,
    }

    summary_path = output_dir / f"metrics_{tag}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Summary JSON written to %s", summary_path)

    LOGGER.info("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
