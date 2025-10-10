#!/usr/bin/env python3
"""Compute spatial coherence metrics for phenology maps.

This script iterates over the 2.5 km training tiles and extracts
co-located windows from the embedding and harmonic classification maps.
For each tile it computes:
  * Edge density (m of class boundary per km^2)
  * Patch density (8-connected components per 100 km^2) for total forest,
    deciduous, and evergreen classes

The harmonic map is evaluated both in its raw form and after a 3x3
median filter to quantify the effect of simple denoising.

Outputs
-------
- Parquet file with per-tile metrics for each map variant
- CSV summary aggregating metrics nationally and per eco-region

Example
-------
python src/analysis/compute_coherence_metrics.py \
    --tiles results/datasets/tiles_2_5_km_final.parquet \
    --embedding results/postprocessing/embeddings/embedding_classes_masked.tif \
    --harmonic /Users/arthurcalvi/Data/phenology/forest_classification_harmonic.tif \
    --output-parquet results/analysis_coherence/coherence_metrics.parquet \
    --summary-csv results/analysis_coherence/coherence_summary.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.mask import mask
from scipy.ndimage import median_filter, label
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

PIXEL_AREA_SCALE = 1e6  # Convert m^2 to km^2
PATCH_DENSITY_SCALE = 100  # Express patch counts per 100 km^2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute spatial coherence metrics for phenology maps.")
    parser.add_argument("--tiles", required=True, type=Path, help="Path to the tiles parquet (GeoParquet).")
    parser.add_argument("--embedding", required=True, type=Path, help="Path to the embedding classification raster (GeoTIFF).")
    parser.add_argument("--harmonic", required=True, type=Path, help="Path to the harmonic classification raster (GeoTIFF).")
    parser.add_argument("--output-parquet", required=True, type=Path, help="Destination parquet for per-tile metrics.")
    parser.add_argument("--summary-csv", required=True, type=Path, help="Destination CSV for aggregated summaries.")
    parser.add_argument("--median-size", type=int, default=3, help="Median filter size for harmonic map (odd integer, default: 3).")
    parser.add_argument("--sample-frac", type=float, default=1.0, help="Optional fraction of tiles to process (0 < frac <= 1).")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed when sampling tiles.")
    parser.add_argument("--eco-region", nargs="*", default=None, help="Optional list of eco-region names (NomSER) to include.")
    parser.add_argument("--all-touched", action="store_true", help="Use all_touched=True when sampling pixels (expands coverage).")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_tiles(path: Path, eco_regions: Iterable[str] | None, sample_frac: float, seed: int) -> gpd.GeoDataFrame:
    tiles = gpd.read_parquet(path)
    if eco_regions:
        tiles = tiles[tiles["NomSER"].isin(eco_regions)].copy()
        LOGGER.info("Filtered to %d tiles after eco-region selection.", len(tiles))
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1].")
    if sample_frac < 1:
        tiles = tiles.sample(frac=sample_frac, random_state=seed).copy()
        LOGGER.info("Subsampled tiles to %d (frac=%.3f).", len(tiles), sample_frac)
    tiles = tiles.reset_index(drop=True)
    return tiles


def extract_window(dataset: rasterio.io.DatasetReader, geometry, *, all_touched: bool) -> np.ndarray:
    try:
        data, _ = mask(dataset, [geometry], crop=True, filled=True, all_touched=all_touched)
        arr = data[0]
    except ValueError:
        # No overlap between geometry and raster
        return np.empty((0, 0), dtype=np.int16)
    return arr


def apply_median(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.size == 0:
        return arr
    if size <= 1:
        return arr.copy()
    # Apply median filter but keep non-forest pixels unchanged
    filtered = median_filter(arr, size=size, mode="nearest")
    forest_mask = arr > 0
    result = arr.copy()
    result[forest_mask] = np.where(filtered[forest_mask] > 0, filtered[forest_mask], arr[forest_mask])
    return result


def compute_edge_density(labels: np.ndarray, pixel_size: float) -> Tuple[float, float]:
    if labels.size == 0:
        return np.nan, 0.0
    valid = labels > 0
    valid_count = int(valid.sum())
    if valid_count == 0:
        return np.nan, 0.0

    # Horizontal edges
    horiz = (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    # Vertical edges
    vert = (labels[:-1, :] != labels[1:, :]) & valid[:-1, :] & valid[1:, :]
    edge_pixels = int(horiz.sum() + vert.sum())
    edge_length_m = edge_pixels * pixel_size

    area_km2 = valid_count * ((pixel_size ** 2) / PIXEL_AREA_SCALE)
    if area_km2 <= 0:
        return np.nan, area_km2
    edge_density = edge_length_m / area_km2
    return edge_density, area_km2


def compute_patch_metrics(labels: np.ndarray, area_km2: float) -> Dict[str, float]:
    if labels.size == 0 or np.isnan(area_km2) or area_km2 <= 0:
        return {
            "patch_count_total": np.nan,
            "patch_density_total_per_100km2": np.nan,
            "patch_count_deciduous": np.nan,
            "patch_density_deciduous_per_100km2": np.nan,
            "patch_count_evergreen": np.nan,
            "patch_density_evergreen_per_100km2": np.nan,
        }

    structure = np.ones((3, 3), dtype=int)  # 8-connected

    deciduous = labels == 1
    evergreen = labels == 2

    _, patches_decid = label(deciduous, structure=structure)
    _, patches_ever = label(evergreen, structure=structure)

    patch_count_decid = int(patches_decid)
    patch_count_ever = int(patches_ever)
    patch_count_total = patch_count_decid + patch_count_ever

    scale = PATCH_DENSITY_SCALE / area_km2
    return {
        "patch_count_total": patch_count_total,
        "patch_density_total_per_100km2": patch_count_total * scale,
        "patch_count_deciduous": patch_count_decid,
        "patch_density_deciduous_per_100km2": patch_count_decid * scale,
        "patch_count_evergreen": patch_count_ever,
        "patch_density_evergreen_per_100km2": patch_count_ever * scale,
    }


def compute_metrics(labels: np.ndarray, pixel_size: float) -> Dict[str, float]:
    edge_density, area_km2 = compute_edge_density(labels, pixel_size)
    patches = compute_patch_metrics(labels, area_km2)
    valid_fraction = float(np.count_nonzero(labels > 0) / labels.size) if labels.size else np.nan
    metrics = {
        "area_km2": area_km2,
        "edge_density_m_per_km2": edge_density,
        "forest_fraction": valid_fraction,
    }
    metrics.update(patches)
    return metrics


def process_tile(
    geometry,
    embedding_ds: rasterio.io.DatasetReader,
    harmonic_ds: rasterio.io.DatasetReader,
    *,
    pixel_size: float,
    median_size: int,
    all_touched: bool,
) -> Dict[str, Dict[str, float]]:
    embedding_arr = extract_window(embedding_ds, geometry, all_touched=all_touched)
    harmonic_arr = extract_window(harmonic_ds, geometry, all_touched=all_touched)

    results: Dict[str, Dict[str, float]] = {
        "embedding": compute_metrics(embedding_arr, pixel_size),
        "harmonic": compute_metrics(harmonic_arr, pixel_size),
    }

    harmonic_median = apply_median(harmonic_arr, median_size)
    results["harmonic_median"] = compute_metrics(harmonic_median, pixel_size)
    return results


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col for col in df.columns
        if any(col.startswith(prefix) for prefix in ("embedding_", "harmonic_"))
    ]
    groups = [("national", df)]
    if "eco_region" in df.columns:
        groups.extend(sorted(df.groupby("eco_region"), key=lambda item: item[0]))

    rows = []
    for label, group in groups:
        summary = {"group": label, "n_tiles": len(group), "area_km2_sum": group.get("geometry_area_km2", pd.Series(dtype=float)).sum()}
        for col in metric_cols:
            summary[f"{col}_mean"] = group[col].mean()
            summary[f"{col}_median"] = group[col].median()
        rows.append(summary)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    ensure_parent(args.output_parquet)
    ensure_parent(args.summary_csv)

    tiles = load_tiles(args.tiles, args.eco_region, args.sample_frac, args.sample_seed)
    if tiles.empty:
        LOGGER.error("No tiles to process after filtering. Exiting.")
        return

    try:
        embedding_ds = rasterio.open(args.embedding)
        harmonic_ds = rasterio.open(args.harmonic)
    except RasterioIOError as exc:
        LOGGER.error("Failed to open raster: %s", exc)
        raise SystemExit(1)

    pixel_size = float(abs(embedding_ds.transform.a))
    LOGGER.info("Using pixel size %.2f m", pixel_size)

    results = []

    for idx, row in tqdm(tiles.iterrows(), total=len(tiles), desc="Tiles"):
        tile_metrics = {
            "tile_index": int(idx),
            "eco_region": row.get("NomSER"),
            "year": row.get("year"),
            "perc": row.get("perc"),
            "perc_deciduous": row.get("perc_deciduous"),
            "perc_evergreen": row.get("perc_evergreen"),
            "effective_pixels": row.get("effective_pixels"),
            "geometry_area_km2": row.geometry.area / PIXEL_AREA_SCALE,
        }

        variant_metrics = process_tile(
            row.geometry,
            embedding_ds,
            harmonic_ds,
            pixel_size=pixel_size,
            median_size=args.median_size,
            all_touched=args.all_touched,
        )

        for variant, metrics in variant_metrics.items():
            for key, value in metrics.items():
                tile_metrics[f"{variant}_{key}"] = value

        results.append(tile_metrics)

    embedding_ds.close()
    harmonic_ds.close()

    if not results:
        LOGGER.error("No metrics computed.")
        return

    df = pd.DataFrame(results)
    df.to_parquet(args.output_parquet, index=False)
    LOGGER.info("Saved per-tile metrics to %s", args.output_parquet)

    summary = aggregate_metrics(df)
    summary.to_csv(args.summary_csv, index=False)
    LOGGER.info("Saved summary metrics to %s", args.summary_csv)


if __name__ == "__main__":
    main()
