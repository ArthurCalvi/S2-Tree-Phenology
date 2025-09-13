import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from shapely import wkb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import time

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# ----------------------------
# Utilities
# ----------------------------


META_EMBED_COLS = {"tile_id", "row", "col", "phenology", "eco_region"}
META_HARM_COLS = {
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


def read_topk_embeddings_list(path: str) -> List[str]:
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def load_tiles_centroids(tiles_parquet: str) -> pd.DataFrame:
    df = pd.read_parquet(tiles_parquet)
    # geometry stored as WKB bytes; compute centroids in EPSG:2154
    xs, ys = [], []
    for g in df["geometry"].tolist():
        geom = wkb.loads(g)
        c = geom.centroid
        xs.append(c.x)
        ys.append(c.y)
    out = pd.DataFrame({
        "tile_id": np.arange(len(df), dtype=int),  # tile_id assumed = row index
        "x2154": xs,
        "y2154": ys,
    })
    return out


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)
    # Center y_true with weighted mean
    w = sample_weight.astype(float)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan
    y_mean = np.sum(w * y_true) / w_sum
    sse = np.sum(w * (y_true - y_pred) ** 2)
    sst = np.sum(w * (y_true - y_mean) ** 2)
    if sst <= 0:
        return np.nan
    return 1.0 - sse / sst


def select_harmonic_feature_columns(harm_cols: List[str]) -> List[str]:
    # Keep all non-meta columns as the full harmonic base
    return [c for c in harm_cols if c not in META_HARM_COLS]


def select_embedding_columns(embed_cols: List[str], topk: List[str]) -> List[str]:
    # Ensure requested names exist
    present = [c for c in topk if c in embed_cols]
    missing = [c for c in topk if c not in embed_cols]
    if missing:
        raise ValueError(f"Missing embedding columns in parquet: {missing}")
    return present


@dataclass
class SimilarityConfig:
    harmonics_parquet: str
    embeddings_parquet: str
    tiles_parquet: str
    topk_embeddings_file: str
    output_dir: str
    n_splits: int = 5
    use_weights: bool = True
    sample_per_region: Optional[int] = None
    min_tile_samples: int = 20
    random_state: int = 42


def compute_similarity(cfg: SimilarityConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load data columns lazily
    harm = pd.read_parquet(cfg.harmonics_parquet)

    emb = pd.read_parquet(cfg.embeddings_parquet)

    # Join on pixel key
    key = ["tile_id", "row", "col"]
    cols_harm = select_harmonic_feature_columns(harm.columns.tolist())
    # Keep eco_region & weight for grouping/metrics
    keep_harm = key + ["eco_region", "weight"] + cols_harm
    harm = harm[keep_harm]

    # Embedding columns: top-k subset
    topk = read_topk_embeddings_list(cfg.topk_embeddings_file)
    emb_cols = select_embedding_columns(emb.columns.tolist(), topk)
    # Keep only pixel keys and embedding columns; we will use eco_region from harmonics to avoid mismatched labels
    keep_emb = key + emb_cols
    emb = emb[keep_emb]

    # Join on exact pixel keys only
    df = harm.merge(emb, on=key, how="inner", validate="one_to_one")
    if df.empty:
        raise RuntimeError("Join produced empty DataFrame. Check key alignment and filters.")

    # Basic logging
    print(f"Loaded harmonics rows: {len(harm):,}; embeddings rows: {len(emb):,}; merged rows: {len(df):,}")
    print(f"Harmonic features used: {len(cols_harm)}; Top-k embeddings: {len(emb_cols)}")
    print("Eco-region counts (merged):", df['eco_region'].value_counts().to_dict())
    # Tiles per region
    tiles_per_region = df.groupby('eco_region')['tile_id'].nunique().to_dict()
    print("Tiles per eco-region (merged):", tiles_per_region)

    # Optional sampling per region for faster debugging; default None (use all)
    if cfg.sample_per_region is not None and cfg.sample_per_region > 0:
        def _sample(g: pd.DataFrame) -> pd.DataFrame:
            n = min(cfg.sample_per_region, len(g))
            return g.sample(n=n, random_state=cfg.random_state)
        before = len(df)
        df = df.groupby('eco_region', group_keys=False).apply(_sample)
        print(f"Applied sampling per region: {before:,} -> {len(df):,} rows")

    # Prepare groups
    groups = df["tile_id"].to_numpy()
    eco_regions = df["eco_region"].to_numpy()
    weights = df["weight"].to_numpy() if cfg.use_weights and "weight" in df.columns else np.ones(len(df))

    X = df[cols_harm].to_numpy(dtype=float)
    # Standardize features once; within CV we refit scaler to avoid leakage
    # Here we will fit scaler in each fold pipeline.

    # Cross-validated ridge with a mild grid
    alphas = np.logspace(-3, 3, 7)

    gkf = GroupKFold(n_splits=cfg.n_splits)

    # Storage for per-tile and per-region records
    per_tile_records: List[Dict] = []
    per_region_records: List[Dict] = []
    coef_records_feature: List[Dict] = []
    closest_region_records: List[Dict] = []
    closest_tile_records: List[Dict] = []

    # Preload tile centroids
    tile_xy = load_tiles_centroids(cfg.tiles_parquet)

    # For efficiency, iterate by eco-region; each tile belongs to one eco-region in the joined dataset
    eco_list = sorted(df["eco_region"].unique().tolist())
    start_time = time.time()

    # Helper: parse harmonic feature to group label (index:component)
    def feature_group_label(feat: str) -> str:
        # Expect pattern like 'ndvi_amplitude_h1', 'evi_phase_h2', 'nbr_offset', 'crswir_var_residual'
        parts = feat.split("_")
        if len(parts) >= 2:
            idx = parts[0]
            component = "_".join(parts[1:])
            return f"{idx}:{component}"
        return f"other:{feat}"
    for eco in tqdm(eco_list, desc="Eco-regions", unit="region"):
        mask_eco = df["eco_region"] == eco
        df_e = df.loc[mask_eco]
        if df_e.empty:
            continue

        X_e = df_e[cols_harm].to_numpy(dtype=float)
        W_e = df_e["weight"].to_numpy() if cfg.use_weights and "weight" in df_e.columns else np.ones(len(df_e))
        G_e = df_e["tile_id"].to_numpy()

        # Initialize containers for tile-wise R2 across embeddings
        # We'll aggregate per embedding then compute mean/std per tile
        tile_r2s: Dict[int, List[float]] = {}

        # Region-level logging
        unique_tiles = np.unique(G_e)
        n_groups = len(unique_tiles)
        print(f"[Region] {eco}: rows={len(df_e):,}, tiles={n_groups}, using GroupKFold splits={min(cfg.n_splits, n_groups) if n_groups>=2 else 0}")

        # For collecting normalized coefficients across folds for region-level aggregation
        coef_norm_accumulator: Dict[str, List[float]] = {f: [] for f in cols_harm}

        # Determine number of unique groups (tiles) in this region
        unique_tiles = np.unique(G_e)
        n_groups = len(unique_tiles)

        for emb_col in tqdm(emb_cols, desc=f"Embeddings ({eco})", leave=False):
            y_e = df_e[emb_col].to_numpy(dtype=float)

            if n_groups >= 2:
                # Fold loop with group split on tiles (n_splits limited by n_groups)
                gkf_local = GroupKFold(n_splits=min(cfg.n_splits, n_groups))
                for train_idx, test_idx in gkf_local.split(X_e, y_e, groups=G_e):
                    X_tr, X_te = X_e[train_idx], X_e[test_idx]
                    y_tr, y_te = y_e[train_idx], y_e[test_idx]
                    w_tr = W_e[train_idx]
                    w_te = W_e[test_idx]
                    g_te = G_e[test_idx]

                    scaler = StandardScaler(with_mean=True, with_std=True)
                    X_trs = scaler.fit_transform(X_tr)
                    X_tes = scaler.transform(X_te)

                    n_train = len(y_tr)
                    if n_train >= 2:
                        cv_ridge = min(5, n_train)
                        model = RidgeCV(alphas=alphas, cv=cv_ridge, scoring=None, fit_intercept=True)
                        # RidgeCV does not accept sample_weight directly; fit then refit with chosen alpha including weights
                        model.fit(X_trs, y_tr)
                        alpha = model.alpha_
                    else:
                        alpha = 1.0
                    # Refit weighted ridge with chosen alpha
                    from sklearn.linear_model import Ridge

                    ridge = Ridge(alpha=alpha, fit_intercept=True)
                    ridge.fit(X_trs, y_tr, sample_weight=w_tr)
                    y_pred = ridge.predict(X_tes)

                    # Compute per-tile R2 for test tiles
                    df_fold = pd.DataFrame({
                        "tile_id": g_te,
                        "y": y_te,
                        "yhat": y_pred,
                        "w": w_te,
                    })
                    for t_id, grp in df_fold.groupby("tile_id"):
                        r2 = weighted_r2_score(grp["y"].to_numpy(), grp["yhat"].to_numpy(), grp["w"].to_numpy())
                        if not np.isfinite(r2):
                            continue
                        tile_r2s.setdefault(int(t_id), []).append(float(r2))

                    # Store normalized absolute coefficients for interpretability
                    coefs = ridge.coef_.ravel()
                    abs_sum = np.sum(np.abs(coefs))
                    if abs_sum > 0:
                        norm_abs = np.abs(coefs) / abs_sum
                        for fname, val in zip(cols_harm, norm_abs):
                            coef_norm_accumulator[fname].append(float(val))
            else:
                # Fallback: Fit on all data and evaluate in-sample per tile (no CV possible)
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs = scaler.fit_transform(X_e)
                n_all = len(y_e)
                if n_all >= 2:
                    cv_ridge = min(5, n_all)
                    model = RidgeCV(alphas=alphas, cv=cv_ridge, scoring=None, fit_intercept=True)
                    model.fit(Xs, y_e)
                    alpha = model.alpha_
                else:
                    alpha = 1.0
                from sklearn.linear_model import Ridge
                ridge = Ridge(alpha=alpha, fit_intercept=True)
                ridge.fit(Xs, y_e, sample_weight=W_e)
                y_pred = ridge.predict(Xs)

                df_all = pd.DataFrame({
                    "tile_id": G_e,
                    "y": y_e,
                    "yhat": y_pred,
                    "w": W_e,
                })
                for t_id, grp in df_all.groupby("tile_id"):
                    r2 = weighted_r2_score(grp["y"].to_numpy(), grp["yhat"].to_numpy(), grp["w"].to_numpy())
                    if not np.isfinite(r2):
                        continue
                    tile_r2s.setdefault(int(t_id), []).append(float(r2))

                coefs = ridge.coef_.ravel()
                abs_sum = np.sum(np.abs(coefs))
                if abs_sum > 0:
                    norm_abs = np.abs(coefs) / abs_sum
                    for fname, val in zip(cols_harm, norm_abs):
                        coef_norm_accumulator[fname].append(float(val))

        # Aggregate per tile: mean/std across embeddings and folds
        # Augment with centroids
        tdf = (
            pd.DataFrame([
                {"tile_id": t, "r2_linear_mean": float(np.mean(v)), "r2_linear_std": float(np.std(v)), "n_measures": len(v)}
                for t, v in tile_r2s.items()
            ])
            .merge(tile_xy, on="tile_id", how="left")
        )
        tdf["eco_region"] = eco
        per_tile_records.append(tdf)

        # Aggregate per eco-region
        # Compute an observation-level out-of-sample R2 across all test predictions (micro-average)
        # R2 is not additive; so we recompute by concatenating all folds/embeddings
        # For simplicity, compute macro: average of tile means
        region_r2_mean = tdf["r2_linear_mean"].mean()
        region_r2_std = tdf["r2_linear_mean"].std(ddof=0)
        per_region_records.append({
            "eco_region": eco,
            "r2_linear_mean": float(region_r2_mean),
            "r2_linear_std": float(region_r2_std),
            "n_tiles": int(len(tdf)),
        })

        # Region-level coefficients: average normalized abs coefficients across folds (and implicitly across embeddings)
        for fname, vals in coef_norm_accumulator.items():
            if not vals:
                continue
            coef_records_feature.append({
                "eco_region": eco,
                "feature": fname,
                "group": feature_group_label(fname),
                "coef_norm_mean": float(np.mean(vals)),
                "coef_norm_std": float(np.std(vals)),
                "n": int(len(vals)),
            })

        # "Closest harmonic" diagnostic per eco-region (per embedding)
        # Use simple Pearson correlation over all rows in the region
        H_e = df_e[cols_harm]
        for emb_col in emb_cols:
            y = df_e[emb_col]
            # Compute corr with each harmonic feature
            corrs = H_e.apply(lambda h: y.corr(h), axis=0)
            # handle NaNs
            corrs = corrs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            best_feat = corrs.abs().idxmax()
            best_corr = float(corrs.loc[best_feat])
            closest_region_records.append({
                "eco_region": eco,
                "embedding": emb_col,
                "best_feature": best_feat,
                "corr_abs": abs(best_corr),
                "corr_signed": best_corr,
            })

        # "Closest harmonic" per tile within region
        for t_id, dft in df_e.groupby("tile_id"):
            # Minimum observations threshold for stability
            if len(dft) < cfg.min_tile_samples:
                continue
            H_t = dft[cols_harm]
            for emb_col in emb_cols:
                y = dft[emb_col]
                corrs = H_t.apply(lambda h: y.corr(h), axis=0)
                corrs = corrs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                best_feat = corrs.abs().idxmax()
                best_corr = float(corrs.loc[best_feat])
                row = {
                    "tile_id": int(t_id),
                    "eco_region": eco,
                    "embedding": emb_col,
                    "best_feature": best_feat,
                    "corr_abs": abs(best_corr),
                    "corr_signed": best_corr,
                }
                closest_tile_records.append(row)

    # Concatenate and save outputs
    per_tile_df = pd.concat(per_tile_records, ignore_index=True) if per_tile_records else pd.DataFrame()
    per_region_df = pd.DataFrame(per_region_records)
    coef_feat_df = pd.DataFrame(coef_records_feature)
    closest_region_df = pd.DataFrame(closest_region_records)
    closest_tile_df = pd.DataFrame(closest_tile_records)

    out_tile = os.path.join(cfg.output_dir, "embedding_harmonic_similarity_tile.parquet")
    out_region = os.path.join(cfg.output_dir, "embedding_harmonic_similarity_region.csv")
    per_tile_df.to_parquet(out_tile, index=False)
    per_region_df.to_csv(out_region, index=False)
    print(f"Wrote per-tile similarity to {out_tile} ({len(per_tile_df)} rows)")
    print(f"Wrote per-region similarity to {out_region} ({len(per_region_df)} rows)")

    # Join tile centroids to closest-tile and save
    if not closest_tile_df.empty:
        closest_tile_df = closest_tile_df.merge(tile_xy, on="tile_id", how="left")
    closest_tile_out = os.path.join(cfg.output_dir, "closest_harmonic_tile.parquet")
    closest_region_out = os.path.join(cfg.output_dir, "closest_harmonic_region.csv")
    coef_feat_out = os.path.join(cfg.output_dir, "embedding_harmonic_coefficients_region.csv")
    closest_tile_df.to_parquet(closest_tile_out, index=False)
    closest_region_df.to_csv(closest_region_out, index=False)
    coef_feat_df.to_csv(coef_feat_out, index=False)
    print(f"Wrote per-tile closest harmonics to {closest_tile_out} ({len(closest_tile_df)} rows)")
    print(f"Wrote per-region closest harmonics to {closest_region_out} ({len(closest_region_df)} rows)")
    print(f"Wrote region-level coefficients to {coef_feat_out} ({len(coef_feat_df)} rows)")

    # Also persist config used
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    elapsed = time.time() - start_time
    print(f"Done in {elapsed/60:.1f} min")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compute linear similarity (R^2) between Top-14 embeddings and full harmonic base, with per-tile and per-eco-region aggregation.")
    ap.add_argument("--harmonics", default="results/datasets/training_datasets_pixels.parquet", help="Parquet path with harmonic features")
    ap.add_argument("--embeddings", default="results/datasets/training_datasets_pixels_embedding.parquet", help="Parquet path with embedding features")
    ap.add_argument("--tiles", default="results/datasets/tiles_2_5_km_final.parquet", help="Parquet path with tiles polygons (EPSG:2154)")
    ap.add_argument("--topk", default="results/final_model/features_embeddings_topk_k14.txt", help="Text file listing Top-14 embedding column names")
    ap.add_argument("--out", default="results/analysis_similarity", help="Output directory for similarity metrics")
    ap.add_argument("--splits", type=int, default=5, help="GroupKFold splits over tiles")
    ap.add_argument("--no-weights", action="store_true", help="Disable sample weights in R^2 computation")
    ap.add_argument("--sample-per-region", type=int, default=None, help="Optional sampling per eco-region for debugging (use all if omitted)")
    ap.add_argument("--min-tile-samples", type=int, default=20, help="Minimum samples per tile to compute correlations")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    cfg = SimilarityConfig(
        harmonics_parquet=args.harmonics,
        embeddings_parquet=args.embeddings,
        tiles_parquet=args.tiles,
        topk_embeddings_file=args.topk,
        output_dir=args.out,
        n_splits=args.splits,
        use_weights=not args.no_weights,
        sample_per_region=args.sample_per_region,
        min_tile_samples=args.min_tile_samples,
    )
    compute_similarity(cfg)


if __name__ == "__main__":
    main()
