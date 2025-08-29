#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Preparation for Embeddings

Builds training tiles by clipping the 64‑band embedding VRT to each 2.5 km tile
and appending 5 categorical/meta bands: phenology, genus, species, source, year.

Key differences vs. harmonic pipeline:
- Keeps feature dtype as float32 (for embeddings) and writes label bands as float32
- Does not try to set harmonic band names; optionally labels embedding bands A00..A63

Usage example:
  python src/sampling/dataset_preparation_embedding.py \
    --tiles_path results/datasets/tiles_2_5_km_final.parquet \
    --dataset_path results/datasets/final_dataset.parquet \
    --features_vrt data/embeddings/features.vrt \
    --output_dir data/training/embeddings \
    --loglevel INFO --limit 3
"""

import argparse
import datetime
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import tqdm
from rasterio.features import rasterize
from typing import Optional
from rasterio.windows import from_bounds
from shapely.geometry import box


def setup_logging(loglevel: str) -> str:
    os.makedirs("logs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/dataset_preparation_embedding_{ts}.log"
    lvl = getattr(logging, loglevel.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)]
    )
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.info(f"Logging to {log_file}")
    return log_file


def get_tile_extent(geometry, src_crs, dst_crs):
    if src_crs == dst_crs:
        minx, miny, maxx, maxy = geometry.bounds
        return box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame(geometry=[geometry], crs=src_crs)
    gdf = gdf.to_crs(dst_crs)
    minx, miny, maxx, maxy = gdf.geometry.iloc[0].bounds
    return box(minx, miny, maxx, maxy)


def create_category_mappings(gdf: gpd.GeoDataFrame) -> dict:
    mappings = {
        "phenology": {"deciduous": 1, "evergreen": 2}
    }
    for col in ["genus", "species", "source"]:
        vals = sorted(gdf[col].dropna().unique())
        mappings[col] = {v: i + 1 for i, v in enumerate(vals)}
        logging.info(f"Mapping {col}: {len(vals)} unique")
    return mappings


def rasterize_attr(gdf: gpd.GeoDataFrame, attribute: str, mapping: Optional[dict], transform, shape, crs):
    valid = gdf[~gdf[attribute].isna()].copy()
    if len(valid) == 0:
        return np.zeros(shape, dtype=np.float32)
    if mapping is not None:
        valid['value'] = valid[attribute].map(mapping)
    else:
        # Year or raw numeric fields may be strings like '2010.0'; coerce robustly
        import pandas as pd
        # Try numeric conversion first
        coerced = pd.to_numeric(valid[attribute], errors='coerce')
        # If still all NaN, extract digits
        if coerced.isna().all():
            coerced = valid[attribute].astype(str).str.extract(r'(\d+)')[0]
            coerced = pd.to_numeric(coerced, errors='coerce')
        valid['value'] = coerced.fillna(0).round().astype(np.uint32)
    # Fill any unmapped values with 0
    valid['value'] = valid['value'].fillna(0).astype(np.uint32)
    if valid.crs != crs:
        valid = valid.to_crs(crs)
    shapes = [(geom, int(val)) for geom, val in zip(valid.geometry, valid['value'])]
    arr = rasterize(
        shapes=shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint32,
        all_touched=False
    ).astype(np.float32)
    return arr


def main():
    ap = argparse.ArgumentParser(description="Prepare training tiles from embedding VRT + labels")
    ap.add_argument('--tiles_path', required=True, help='Parquet of 2.5km tiles')
    ap.add_argument('--dataset_path', required=True, help='final_dataset.parquet with labels')
    ap.add_argument('--features_vrt', required=True, help='Path to embeddings VRT (64 bands)')
    ap.add_argument('--output_dir', required=True, default='data/training/embeddings')
    ap.add_argument('--sanitize', action='store_true', help='Clip embeddings to [-1,1] and replace non-finite with 0.0 before writing')
    ap.add_argument('--loglevel', default='INFO')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--tile_ids', type=str, default=None)
    args = ap.parse_args()

    setup_logging(args.loglevel)
    os.makedirs(args.output_dir, exist_ok=True)

    # Open VRT once to get CRS and metadata
    with rasterio.open(args.features_vrt) as src_vrt:
        raster_crs = src_vrt.crs
        logging.info(f"Embeddings VRT CRS: {raster_crs}")

    tiles = gpd.read_parquet(args.tiles_path)
    data = gpd.read_parquet(args.dataset_path)

    if 'tile_id' in data.columns:
        data = data.set_index('tile_id', drop=False)

    # Reproject tiles/data to raster CRS
    if tiles.crs != raster_crs:
        tiles = tiles.to_crs(raster_crs)
    if data.crs != raster_crs:
        data = data.to_crs(raster_crs)

    # Filter
    if args.tile_ids:
        keep = [int(t.strip()) for t in args.tile_ids.split(',')]
        tiles = tiles[tiles.index.isin(keep)]
    if args.limit and args.limit < len(tiles):
        tiles = tiles.head(args.limit)
        logging.info(f"Limiting to {len(tiles)} tiles")

    mappings = create_category_mappings(data)
    with open(Path(args.output_dir)/'categorical_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)

    processed = 0
    skipped = 0
    skipped_info = defaultdict(list)

    for idx, row in tqdm.tqdm(tiles.iterrows(), total=len(tiles), desc='Tiles'):
        tile_id = row.tile_id if 'tile_id' in tiles.columns else idx
        geom = row.geometry

        # Subset dataset polygons by tile_id when possible, else intersects
        if 'tile_id' in data.index.names and tile_id in data.index:
            tile_df = data.loc[[tile_id]].copy()
        else:
            tile_df = data[data.intersects(geom)].copy()
        if len(tile_df) == 0:
            skipped += 1
            skipped_info[tile_id].append({'reason': 'No polygons for tile'})
            continue

        # Clip embeddings VRT to tile bounds (aligned to pixel grid)
        with rasterio.open(args.features_vrt) as src:
            geom_r = get_tile_extent(geom, tiles.crs, src.crs)
            minx, miny, maxx, maxy = geom_r.bounds
            T = src.transform
            c0, r0 = ~T * (minx, miny)
            c1, r1 = ~T * (maxx, maxy)
            c0, r0 = int(c0), int(r0)
            c1, r1 = int(c1 + 0.5), int(r1 + 0.5)
            ax, ay = T * (c0, r0)
            bx, by = T * (c1, r1)
            window = from_bounds(ax, ay, bx, by, src.transform)
            feat = src.read(window=window).astype(np.float32)  # (64,H,W)
            if args.sanitize:
                # Clip to documented range and sanitize non-finite values
                np.clip(feat, -1.0, 1.0, out=feat)
                feat[~np.isfinite(feat)] = 0.0
            out_transform = src.window_transform(window)

            out_meta = src.meta.copy()
            # Ensure float32 for all bands, avoid nodata issues with uint
            out_meta.update({
                'driver': 'GTiff',
                'height': feat.shape[1],
                'width': feat.shape[2],
                'transform': out_transform,
                'count': feat.shape[0] + 5,
                'dtype': 'float32',
                'compress': 'lzw',
                'predictor': 2,
                'nodata': 0.0
            })

        H, W = feat.shape[1], feat.shape[2]
        shape = (H, W)

        # Rasterize labels (float32 output)
        phen = rasterize_attr(tile_df, 'phenology', mappings['phenology'], out_transform, shape, raster_crs)
        gen = rasterize_attr(tile_df, 'genus', mappings['genus'], out_transform, shape, raster_crs)
        spe = rasterize_attr(tile_df, 'species', mappings['species'], out_transform, shape, raster_crs)
        srcb = rasterize_attr(tile_df, 'source', mappings['source'], out_transform, shape, raster_crs)
        yr = rasterize_attr(tile_df, 'year', None, out_transform, shape, raster_crs)

        out_path = os.path.join(args.output_dir, f"tile_{tile_id}_training.tif")
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            # embeddings 64 bands
            for i in range(feat.shape[0]):
                dst.write(feat[i], i + 1)
                dst.set_band_description(i + 1, f"A{i:02d}")
            # labels
            dst.write(phen, feat.shape[0] + 1); dst.set_band_description(feat.shape[0] + 1, 'phenology')
            dst.write(gen, feat.shape[0] + 2); dst.set_band_description(feat.shape[0] + 2, 'genus')
            dst.write(spe, feat.shape[0] + 3); dst.set_band_description(feat.shape[0] + 3, 'species')
            dst.write(srcb, feat.shape[0] + 4); dst.set_band_description(feat.shape[0] + 4, 'source')
            dst.write(yr, feat.shape[0] + 5); dst.set_band_description(feat.shape[0] + 5, 'year')

        processed += 1
        if processed % 10 == 0:
            logging.info(f"Processed {processed} tiles ...")

    if skipped:
        rep = Path(args.output_dir)/'skipped_tiles_embedding.json'
        with open(rep, 'w') as f:
            json.dump({'skipped': skipped, 'info': skipped_info}, f, indent=2)
        logging.info(f"Saved skipped report: {rep}")
    logging.info(f"Done. Processed={processed}, skipped={skipped}")


if __name__ == '__main__':
    main()
