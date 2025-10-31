"""Visualise ridge R² per embedding dimension and eco-region.

This script recomputes the ridge-regression overlap between harmonic descriptors
and the Top-14 AlphaEarth embeddings, storing the per-embedding R² per eco-region
and rendering a multi-row histogram. Each row corresponds to one eco-region,
the x-axis enumerates embedding dimensions, and the bar height indicates the
cross-validated weighted R². The bottom panel annotates the dominant harmonic
cue (largest |Pearson r|) for each embedding dimension together with the mean
absolute correlation across eco-regions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.dataset as ds
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

from src.utils import apply_science_style

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = lambda x, **kwargs: x  # type: ignore


META_COLUMNS = {
    "tile_id",
    "row",
    "col",
    "phenology",
    "genus",
    "species",
    "source",
    "year",
    "eco_region",
    "weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ridge R² per embedding dimension and eco-region."
    )
    parser.add_argument(
        "--harmonics",
        type=Path,
        default=Path("results/datasets/training_datasets_pixels.parquet"),
        help="Parquet file containing harmonic descriptors and metadata.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("results/datasets/training_datasets_pixels_embedding.parquet"),
        help="Parquet file containing embedding features and metadata.",
    )
    parser.add_argument(
        "--topk",
        type=Path,
        default=Path("results/final_model/features_embeddings_topk_k14.txt"),
        help="Text file listing the selected embedding column names (one per line).",
    )
    parser.add_argument(
        "--out-data",
        type=Path,
        default=Path("results/analysis_similarity/embedding_r2_per_region.csv"),
        help="Destination CSV storing the per-embedding R² per eco-region.",
    )
    parser.add_argument(
        "--out-figure",
        type=Path,
        default=Path("images/embedding_harmonic_r2_by_region.png"),
        help="Destination path for the generated figure.",
    )
    parser.add_argument(
        "--correlation-summary",
        type=Path,
        default=Path("results/analysis_similarity/closest_harmonic_region.csv"),
        help="CSV with the dominant harmonic feature and |r| per embedding & region.",
    )
    parser.add_argument(
        "--region-summary",
        type=Path,
        default=Path("results/analysis_similarity/summary/summary_region.csv"),
        help="CSV providing eco-region ordering (optional).",
    )
    return parser.parse_args()


def weighted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    w = sample_weight.astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
    if not np.any(mask):
        return np.nan
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    w = w[mask]
    w_sum = w.sum()
    if w_sum <= 0:
        w = np.ones_like(y_true, dtype=float)
        w_sum = w.size
    y_mean = np.sum(w * y_true) / w_sum
    sst = np.sum(w * (y_true - y_mean) ** 2)
    if sst <= 0:
        return np.nan
    sse = np.sum(w * (y_true - y_pred) ** 2)
    return 1.0 - sse / sst


def load_topk_embeddings(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def extract_feature_columns(schema: Iterable[str]) -> List[str]:
    return [name for name in schema if name not in META_COLUMNS]


def prettify_feature_name(name: str) -> str:
    parts = name.split("_")
    if not parts:
        return name
    idx = parts[0].upper()
    remainder = parts[1:]
    replacements = {
        "amplitude": "amp",
        "phase": "phase",
        "offset": "offset",
        "var": "var",
        "residual": "res",
    }
    pretty_parts: List[str] = [idx]
    for chunk in remainder:
        pretty_parts.append(replacements.get(chunk, chunk))
    return " ".join(pretty_parts)


def fetch_region_order(region_summary: Path, available_regions: Iterable[str]) -> List[str]:
    if region_summary.exists():
        df = pd.read_csv(region_summary)
        ordered = [eco for eco in df["eco_region"].tolist() if eco in available_regions]
        remaining = [eco for eco in available_regions if eco not in ordered]
        return ordered + sorted(remaining)
    return sorted(available_regions)


def compute_region_r2(
    eco: str,
    harm_ds: ds.Dataset,
    emb_ds: ds.Dataset,
    harmonic_cols: List[str],
    embedding_cols: List[str],
    alphas: np.ndarray,
    min_tile_samples: int = 20,
) -> Tuple[str, Dict[str, float]]:
    filter_expr = ds.field("eco_region") == eco

    harm_table = harm_ds.to_table(
        columns=["tile_id", "row", "col", "eco_region", "weight"] + harmonic_cols,
        filter=filter_expr,
    )
    emb_table = emb_ds.to_table(
        columns=["tile_id", "row", "col"] + embedding_cols,
        filter=filter_expr,
    )
    if harm_table.num_rows == 0 or emb_table.num_rows == 0:
        return eco, {}

    harm_df = harm_table.to_pandas()
    emb_df = emb_table.to_pandas()

    merged = harm_df.merge(
        emb_df,
        on=["tile_id", "row", "col"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return eco, {}

    # Drop rows with missing or non-finite values
    cols_to_check = ["weight"] + harmonic_cols + embedding_cols
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=cols_to_check)
    if merged.empty:
        return eco, {}

    groups = merged["tile_id"].to_numpy()
    weights = merged["weight"].to_numpy(dtype=np.float64)
    X_full = merged[harmonic_cols].to_numpy(dtype=np.float64)

    unique_tiles = np.unique(groups)
    n_tiles = unique_tiles.size

    r2_per_embedding: Dict[str, float] = {}

    for emb in embedding_cols:
        y = merged[emb].to_numpy(dtype=np.float64)
        if n_tiles >= 2:
            n_splits = min(5, n_tiles)
            gkf = GroupKFold(n_splits=n_splits)
            y_pred = np.zeros_like(y)
            mask = np.zeros_like(y, dtype=bool)

            for train_idx, test_idx in gkf.split(X_full, y, groups):
                X_tr = X_full[train_idx]
                X_te = X_full[test_idx]
                y_tr = y[train_idx]
                w_tr = weights[train_idx]

                scaler = StandardScaler(with_mean=True, with_std=True)
                X_trs = scaler.fit_transform(X_tr)
                X_tes = scaler.transform(X_te)

                if X_trs.shape[0] >= 2:
                    cv = min(5, X_trs.shape[0])
                    ridge_cv = RidgeCV(alphas=alphas, cv=cv, fit_intercept=True)
                    ridge_cv.fit(X_trs, y_tr)
                    alpha = float(ridge_cv.alpha_)
                else:
                    alpha = 1.0

                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(X_trs, y_tr, sample_weight=w_tr)
                y_pred[test_idx] = ridge.predict(X_tes)
                mask[test_idx] = True

            r2 = weighted_r2_score(y[mask], y_pred[mask], weights[mask])
        else:
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xs = scaler.fit_transform(X_full)
            if Xs.shape[0] >= 2:
                cv = min(5, Xs.shape[0])
                ridge_cv = RidgeCV(alphas=alphas, cv=cv, fit_intercept=True)
                ridge_cv.fit(Xs, y)
                alpha = float(ridge_cv.alpha_)
            else:
                alpha = 1.0
            ridge = Ridge(alpha=alpha, fit_intercept=True)
            ridge.fit(Xs, y, sample_weight=weights)
            y_pred = ridge.predict(Xs)
            r2 = weighted_r2_score(y, y_pred, weights)

        r2_per_embedding[emb] = r2

    return eco, r2_per_embedding


def summarise_correlations(corr_path: Path) -> pd.DataFrame:
    if not corr_path.exists():
        raise FileNotFoundError(f"Correlation summary not found: {corr_path}")
    corr_df = pd.read_csv(corr_path)
    agg = (
        corr_df.groupby(["embedding", "best_feature"])["corr_abs"]
        .agg(["mean", "max", "count"])
        .reset_index()
    )
    top = (
        agg.sort_values(["embedding", "mean"], ascending=[True, False])
        .groupby("embedding", as_index=False)
        .first()
    )
    top = top.rename(
        columns={
            "mean": "corr_abs_mean",
            "max": "corr_abs_max",
            "count": "n_regions",
        }
    )
    top["feature_pretty"] = top["best_feature"].apply(prettify_feature_name)
    return top.set_index("embedding")


def plot_r2_matrix(
    r2_df: pd.DataFrame,
    corr_summary: pd.DataFrame,
    embedding_cols: List[str],
    output_path: Path,
) -> None:
    apply_science_style()

    # Compute global limits for consistent y-range
    y_min = float(np.floor(r2_df.values.min() * 10) / 10) - 0.05
    y_max = float(np.ceil(r2_df.values.max() * 10) / 10) + 0.05

    eco_regions = r2_df.index.tolist()
    n_regions = len(eco_regions)
    fig_height = max(2.5 + 0.9 * n_regions, 6.0)

    fig, axes = plt.subplots(
        n_regions + 1,
        1,
        figsize=(12, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0] * n_regions + [0.6], "hspace": 0.15},
    )

    x_positions = np.arange(len(embedding_cols))

    for ax, eco in zip(axes[:-1], eco_regions):
        values = r2_df.loc[eco].to_numpy(dtype=float)
        ax.bar(x_positions, values, color="#1f77b4", edgecolor="none")
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_ylabel(eco, rotation=0, ha="right", va="center", fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(
            np.linspace(y_min, y_max, num=3)
        )
        ax.tick_params(axis="y", labelsize=9)

    bottom_ax = axes[-1]
    bottom_ax.set_xlim(-0.5, len(embedding_cols) - 0.5)
    bottom_ax.set_ylim(0, 1)
    bottom_ax.axis("off")

    for idx, emb in enumerate(embedding_cols):
        label = emb.replace("embedding_", "")
        info = corr_summary.loc[emb] if emb in corr_summary.index else None
        if info is not None:
            text = f"{info['feature_pretty']}\n|r|={info['corr_abs_mean']:.2f}"
        else:
            text = "n/a"
        bottom_ax.text(
            idx,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=9,
        )
        bottom_ax.text(
            idx,
            0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    axes[0].set_title(
        "Ridge $R^2$ of embedding dimensions explained by harmonic descriptors",
        fontsize=13,
        pad=12,
    )
    axes[-2].set_xlabel("Embedding dimension (Top-14 subset)", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    harm_ds = ds.dataset(args.harmonics)
    emb_ds = ds.dataset(args.embeddings)

    topk_embeddings = load_topk_embeddings(args.topk)
    if not topk_embeddings:
        raise ValueError(f"No embedding names found in {args.topk}")

    harm_schema = harm_ds.schema.names
    harmonic_cols = extract_feature_columns(harm_schema)

    available_regions = (
        pd.read_parquet(args.embeddings, columns=["eco_region"])["eco_region"]
        .dropna()
        .unique()
        .tolist()
    )
    region_order = fetch_region_order(args.region_summary, available_regions)

    alphas = np.logspace(-3, 3, num=13)

    records = []
    iterator = tqdm(region_order, desc="Eco-regions", unit="region")
    for eco in iterator:
        eco_name, r2_per_embedding = compute_region_r2(
            eco,
            harm_ds,
            emb_ds,
            harmonic_cols,
            topk_embeddings,
            alphas,
        )
        if not r2_per_embedding:
            continue
        records.append({"eco_region": eco_name, **r2_per_embedding})

    if not records:
        raise RuntimeError("No R² scores were computed; check data inputs.")

    r2_df = pd.DataFrame.from_records(records).set_index("eco_region")

    missing_regions = [eco for eco in region_order if eco not in r2_df.index]
    if missing_regions:
        print(
            "Warning: the following eco-regions produced no R² values and will be skipped:",
            ", ".join(missing_regions),
        )
    ordered_regions = [eco for eco in region_order if eco in r2_df.index]
    r2_df = r2_df.reindex(ordered_regions)

    for emb in topk_embeddings:
        if emb not in r2_df.columns:
            r2_df[emb] = np.nan
    r2_df = r2_df[topk_embeddings]

    args.out_data.parent.mkdir(parents=True, exist_ok=True)
    r2_df.to_csv(args.out_data)

    corr_summary = summarise_correlations(args.correlation_summary)
    plot_r2_matrix(r2_df, corr_summary, topk_embeddings, args.out_figure)


if __name__ == "__main__":
    main()
