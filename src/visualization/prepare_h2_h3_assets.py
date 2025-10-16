#!/usr/bin/env python3
"""
Prepare raster assets for manuscript Figures 4 (H2) and 5 (H3).

Steps automated:
  1. Crop national embedding and harmonic maps to the selected tile extents.
  2. Run the EMB-14 RandomForest on historical AlphaEarth embedding tiles for the
     requested years, producing both probability stacks and class rasters.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import pyproj
import rasterio
from rasterio.windows import Window, from_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference.inference_rf_embeddings import (
    configure_warning_filters,
    load_feature_list,
    process_tile,
)
from src.visualization.figure_tile_selection import H2_H3_TILE_SELECTION

LOGGER = logging.getLogger("prepare_h2_h3_assets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop national rasters and run EMB-14 inference for figure tiles."
    )
    parser.add_argument(
        "--tiles-selection",
        type=Path,
        default=None,
        help="Optional JSON file listing tile metadata; defaults to embedded H2/H3 tiles.",
    )
    parser.add_argument(
        "--embedding-map",
        type=Path,
        default=Path("results/postprocessing/embeddings/embedding_classes_masked.tif"),
        help="Path to the national embedding classification raster (Lambert-93).",
    )
    parser.add_argument(
        "--harmonic-map",
        type=Path,
        default=Path("/Users/arthurcalvi/Data/phenology/forest_classification_harmonic.tif"),
        help="Path to the national harmonic classification raster.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Base directory where figure assets will be written.",
    )
    parser.add_argument(
        "--embeddings-root",
        type=Path,
        default=Path("data/embeddings"),
        help="Root directory containing yearly AlphaEarth embedding tiles.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "results/final_model_embeddings/rf_embeddings_embeddings_topk_k14_20250919T105507Z.joblib"
        ),
        help="Trained RandomForest model to load for EMB-14 inference.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        default=Path("results/feature_selection_embeddings/features_embeddings_topk_k14.txt"),
        help="Text file listing embedding features used by the model.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2018, 2020, 2022, 2023],
        help="Years to process for EMB-14 inference (historical AlphaEarth tiles).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Window size for block-wise inference (matches inference_rf_embeddings).",
    )
    parser.add_argument(
        "--band-min",
        type=float,
        default=-1.0,
        help="Minimum embedding value used when tiles are stored as uint16.",
    )
    parser.add_argument(
        "--band-max",
        type=float,
        default=1.0,
        help="Maximum embedding value used when tiles are stored as uint16.",
    )
    parser.add_argument(
        "--missing-fill",
        type=float,
        default=0.0,
        help="Fill value for missing embedding bands during inference.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Raise an error if requested embedding bands are absent in a tile.",
    )
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=None,
        help="Override RandomForest n_jobs before inference (defaults to model setting).",
    )
    parser.add_argument(
        "--skip-crops",
        action="store_true",
        help="Skip cropping the national rasters (Figure 4 assets).",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip EMB-14 inference on historical tiles (Figure 5 assets).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs even if target files already exist.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_tile_selection(selection_path: Path | None) -> list[dict]:
    if selection_path is None:
        LOGGER.info("Using built-in tile selection for H2/H3 figures")
        tiles = [dict(tile) for tile in H2_H3_TILE_SELECTION]
    else:
        if not selection_path.exists():
            raise FileNotFoundError(f"Tile selection not found: {selection_path}")
        tiles = json.loads(selection_path.read_text())
    if not isinstance(tiles, list) or not tiles:
        origin = selection_path or "embedded default"
        raise ValueError(f"Tile selection must be a non-empty list ({origin})")
    for tile in tiles:
        for key in ("tile_id", "eco_region", "bbox_wgs84"):
            if key not in tile:
                raise ValueError(f"Tile entry missing key '{key}' ({selection_path or 'default'})")
    return tiles


def transform_bbox(
    bbox: Sequence[float],
    src_crs: str,
    dst_crs: str | rasterio.crs.CRS,
) -> tuple[float, float, float, float]:
    """Transform an axis-aligned bounding box between CRSs."""
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    min_lon, min_lat, max_lon, max_lat = bbox
    corners = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]
    xs: list[float] = []
    ys: list[float] = []
    for lon, lat in corners:
        x, y = transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def crop_raster_to_bbox(
    src_path: Path,
    bbox_wgs84: Sequence[float],
    dst_path: Path,
    overwrite: bool = False,
) -> None:
    """Crop a raster to the provided WGS84 bounding box and save to dst_path."""
    if dst_path.exists() and not overwrite:
        LOGGER.info("Skipping crop for %s (already exists)", dst_path.name)
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        minx, miny, maxx, maxy = transform_bbox(bbox_wgs84, "EPSG:4326", src.crs)
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        window = window.round_offsets().round_lengths()

        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": rasterio.windows.transform(window, src.transform),
                "driver": "GTiff",
            }
        )

        LOGGER.debug(
            "Cropping %s -> %s | window=%s size=%sx%s",
            src_path.name,
            dst_path.name,
            window,
            profile["width"],
            profile["height"],
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)


def resolve_embedding_tile(
    embeddings_root: Path,
    tile_id: int,
    year: int,
) -> Path:
    """Return the expected embedding tile path for a tile/year combination."""
    fname = f"emb_tile_{tile_id:03d}_{year}.tif"
    candidates = [
        embeddings_root / str(year) / fname,
        embeddings_root / fname,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def run_emb_inference(
    tiles: Iterable[dict],
    embeddings_root: Path,
    years: Sequence[int],
    model_path: Path,
    features_file: Path,
    output_dir: Path,
    block_size: int,
    band_min: float,
    band_max: float,
    missing_fill: float,
    fail_on_missing: bool,
    rf_n_jobs: int | None,
    overwrite: bool,
) -> None:
    configure_warning_filters()
    LOGGER.info("Loading EMB-14 model from %s", model_path)
    model = joblib.load(model_path)
    if rf_n_jobs is not None and hasattr(model, "n_jobs"):
        LOGGER.info("Overriding RandomForest n_jobs to %s", rf_n_jobs)
        model.n_jobs = rf_n_jobs

    feature_names = load_feature_list(features_file)
    LOGGER.info("Loaded %d embedding features", len(feature_names))

    for year in years:
        year_dir = output_dir / "figure_h3" / f"{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        for tile in tiles:
            tile_id = int(tile["tile_id"])
            tile_path = resolve_embedding_tile(embeddings_root, tile_id, year)
            if not tile_path.exists():
                LOGGER.warning(
                    "Embedding tile missing for tile_id=%03d year=%d -> %s",
                    tile_id,
                    year,
                    tile_path,
                )
                continue

            target_path = year_dir / tile_path.name
            if target_path.exists() and not overwrite:
                LOGGER.info(
                    "Skipping inference for tile %03d year %d (already exists)",
                    tile_id,
                    year,
                )
                continue

            LOGGER.info(
                "Running EMB-14 inference for tile %03d year %d",
                tile_id,
                year,
            )
            process_tile(
                tile_path=tile_path,
                output_dir=year_dir,
                model=model,
                feature_names=feature_names,
                block_size=block_size,
                save_classes=True,
                band_min=band_min,
                band_max=band_max,
                missing_fill=missing_fill,
                fail_on_missing=fail_on_missing,
            )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tiles = load_tile_selection(args.tiles_selection)
    LOGGER.info("Loaded %d tiles for processing", len(tiles))

    figure_base = args.output_dir
    h2_dir_embed = figure_base / "figure_h2" / "embedding"
    h2_dir_harm = figure_base / "figure_h2" / "harmonic"
    h2_dir_embed.mkdir(parents=True, exist_ok=True)
    h2_dir_harm.mkdir(parents=True, exist_ok=True)

    if not args.skip_crops:
        LOGGER.info("Cropping national rasters for Figure 4 assets")
        for tile in tiles:
            tile_id = int(tile["tile_id"])
            bbox = tile["bbox_wgs84"]
            embed_out = h2_dir_embed / f"H2_emb14_tile_{tile_id:03d}_2023.tif"
            harm_out = h2_dir_harm / f"H2_harm14_tile_{tile_id:03d}_2023.tif"

            crop_raster_to_bbox(
                args.embedding_map,
                bbox,
                embed_out,
                overwrite=args.overwrite,
            )
            crop_raster_to_bbox(
                args.harmonic_map,
                bbox,
                harm_out,
                overwrite=args.overwrite,
            )
    else:
        LOGGER.info("Skipping raster cropping as requested")

    if not args.skip_inference:
        LOGGER.info("Running EMB-14 inference for Figure 5 assets")
        run_emb_inference(
            tiles=tiles,
            embeddings_root=args.embeddings_root,
            years=args.years,
            model_path=args.model_path,
            features_file=args.features_file,
            output_dir=args.output_dir,
            block_size=args.block_size,
            band_min=args.band_min,
            band_max=args.band_max,
            missing_fill=args.missing_fill,
            fail_on_missing=args.fail_on_missing,
            rf_n_jobs=args.rf_n_jobs,
            overwrite=args.overwrite,
        )
    else:
        LOGGER.info("Skipping EMB-14 inference as requested")


if __name__ == "__main__":
    main()
