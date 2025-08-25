#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download Google Satellite Embeddings (64-D) per training tile locally using the
Earth Engine Python API, guided by a .env file.

Reads tiles from a local Parquet (EPSG:2154), fetches the 2023 image from
GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, reprojects to EPSG:2154 at 10 m, and
writes one GeoTIFF per tile under data/embeddings/ named emb_tile_###_YEAR.tif.

.env variables (optional; CLI args override):
- EE_SERVICE_ACCOUNT=<email>                 # Optional service account
- EE_PRIVATE_KEY_FILE=path/to/key.json       # Optional key file for service account
- EE_PRIVATE_KEY=path/to/key.json            # (alt) key file path
- GOOGLE_APPLICATION_CREDENTIALS=path/to/key # (alt) key file path
- EE_PROJECT=your-project-id                  # Earth Engine project
- EARTHENGINE_PROJECT=your-project-id         # (alt) project env name
- EE_TILES_PARQUET=results/datasets/tiles_2_5_km_final.parquet
- EE_OUTPUT_DIR=data/embeddings
- EE_YEAR=2023
- EE_CRS=EPSG:2154
- EE_SCALE=10

Usage:
  python src/gee/download_embeddings.py \
    --tiles results/datasets/tiles_2_5_km_final.parquet \
    --output-dir data/embeddings --year 2023 --crs EPSG:2154 --scale 10
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import ee
import geopandas as gpd
import requests
from shapely.geometry import mapping
from dotenv import load_dotenv
from tqdm import tqdm
# Add chunking/mosaic imports
import math
import tempfile
from shapely.geometry import box as shapely_box
import rasterio
from rasterio.merge import merge as rio_merge
import rasterio.warp

from typing import Optional


def init_ee_from_env(project: Optional[str] = None):
    sa = os.getenv('EE_SERVICE_ACCOUNT')
    key = (
        os.getenv('EE_PRIVATE_KEY_FILE')
        or os.getenv('EE_PRIVATE_KEY')
        or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    )
    # Fallback project env var name
    if not project:
        project = os.getenv('EE_PROJECT') or os.getenv('EARTHENGINE_PROJECT')
    if sa and key and Path(key).exists():
        creds = ee.ServiceAccountCredentials(sa, key)
        ee.Initialize(credentials=creds, project=project)
        logging.info(f"Initialized EE with service account: {sa}")
    else:
        try:
            ee.Initialize(project=project)
            logging.info("Initialized EE with existing credentials")
        except Exception:
            logging.info("Authenticating with EE interactively...")
            ee.Authenticate()
            # If ADC exists but no quota project is set, EE now requires a project.
            # Pass it explicitly if provided.
            ee.Initialize(project=project)


def get_embedding_image(year: int):
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')
    band_list = [f'A{i:02d}' for i in range(64)]
    ic = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
        .filterDate(start, end)
    # Mosaic to ensure coverage across image footprints
    img = ic.mosaic().select(band_list)
    return img


def get_embedding_image_for_region(year: int, region_wgs84_geom) -> ee.Image:
    """Return the single embedding image that covers the given region."""
    logging.debug(f"Getting embedding image for region: {region_wgs84_geom}")
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, 'year')

    band_list = [f'A{i:02d}' for i in range(64)]
    logging.debug(f"Requesting {len(band_list)} bands")

    # Convert shapely geometry to ee.Geometry
    geom_dict = mapping(region_wgs84_geom)
    logging.debug(f"Region GeoJSON: {geom_dict}")

    # First, let's check what images are available for this region
    ic = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
        .filterDate(start, end) \
        .filterBounds(ee.Geometry(geom_dict))

    collection_size = ic.size().getInfo()
    logging.debug(f"Image collection size: {collection_size}")

    if collection_size == 0:
        logging.warning(f"No embedding images found for region in {year}")
        # Let's try a broader search to see what's available
        ic_all = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL') \
            .filterDate(start, end)
        total_count = ic_all.size().getInfo()
        logging.debug(f"Total images available globally in {year}: {total_count}")

        # Get some sample image info
        sample = ic_all.first()
        sample_info = sample.getInfo()
        logging.debug(f"Sample image info: {sample_info['properties'] if 'properties' in sample_info else 'No properties'}")
        return None

    img = ic.first().select(band_list).toFloat()

    # Log projection detail
    proj_info = img.select('A00').projection().getInfo()
    logging.debug(f"Image projection: {proj_info}")

    # Log image properties
    img_props = img.getInfo()['properties']
    logging.debug(f"Image properties: {img_props}")

    return img


def save_and_reproject_tile(
    img: ee.Image,
    tile_geom_2154: gpd.GeoSeries,
    target_crs: str,
    scale: int,
    out_path: Path,
    bands: int,
):
    """
    Downloads a tile in its native projection and reprojects it locally to the target CRS.
    """
    # 1. Determine Native Projection from GEE Image
    proj_info = img.select('A00').projection().getInfo()
    native_crs = proj_info.get('crs')
    if not native_crs:
        raise ValueError("Could not determine native CRS from GEE Image")
    logging.debug(f"Image native CRS: {native_crs}")

    # 2. Prepare Region (WGS84) rectangle and server-side clip to constrain export exactly
    # Build a precise 4326 rectangle for the region
    tile_wgs84 = tile_geom_2154.to_crs('EPSG:4326')
    wgs84_minx, wgs84_miny, wgs84_maxx, wgs84_maxy = tile_wgs84.iloc[0].geometry.bounds
    rect_wgs84 = ee.Geometry.Rectangle([wgs84_minx, wgs84_miny, wgs84_maxx, wgs84_maxy], None, False)
    logging.debug(f"Region rectangle (EPSG:4326): {[wgs84_minx, wgs84_miny, wgs84_maxx, wgs84_maxy]}")
    # Clip first on server side
    img = img.clip(rect_wgs84)

    # The crs and crsTransform define the output grid, which should be in the native CRS
    tile_native = tile_geom_2154.to_crs(native_crs)
    native_minx, native_miny, native_maxx, native_maxy = tile_native.iloc[0].geometry.bounds
    
    # Define the exact pixel grid for the download in the native projection
    crs_transform = [
        float(scale), 0.0, float(native_minx),
        0.0, -float(scale), float(native_maxy)
    ]
    logging.debug(f"CRS Transform (in {native_crs}): {crs_transform}")

    # 3. Download the image data in its native projection
    req = {
        # Keep it simple: specify region and scale only; let GEE choose sane defaults
        'region': [wgs84_minx, wgs84_miny, wgs84_maxx, wgs84_maxy],
        'scale': int(scale),
        'format': 'GEO_TIFF',
        'filePerBand': False,
    }

    url = img.getDownloadURL(req)
    logging.debug(f"Download URL: {url}")
    
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_native_tif:
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(tmp_native_tif.name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            logging.debug(f"Downloaded native raster to {tmp_native_tif.name}")

            # 4. Reproject the downloaded raster to the target CRS using rasterio
            with rasterio.open(tmp_native_tif.name) as src:
                # Calculate transform and dimensions for the output raster
                dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                
                # Copy metadata and update for reprojection
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'compress': 'deflate',
                    'tiled': True,
                })
                
                logging.debug(f"Reprojecting from {src.crs} to {target_crs}")
                logging.debug(f"Output dimensions: {dst_width}x{dst_height}")

                with rasterio.open(out_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        rasterio.warp.reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=dst_transform,
                            dst_crs=target_crs,
                            resampling=rasterio.warp.Resampling.nearest
                        )
            logging.info(f"Successfully saved reprojected tile to {out_path}")

        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_native_tif.name):
                os.remove(tmp_native_tif.name)


def main():
    load_dotenv()
    # Basic logging; will be updated after parsing args
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    ap = argparse.ArgumentParser(description='Download GEE Satellite Embeddings per tile')
    ap.add_argument('--tiles', default=os.getenv('EE_TILES_PARQUET', 'results/datasets/tiles_2_5_km_final.parquet'))
    ap.add_argument('--output-dir', default=os.getenv('EE_OUTPUT_DIR', 'data/embeddings'))
    ap.add_argument('--year', type=int, default=int(os.getenv('EE_YEAR', '2023')))
    ap.add_argument('--crs', default=os.getenv('EE_CRS', 'EPSG:2154'))
    ap.add_argument('--scale', type=int, default=int(os.getenv('EE_SCALE', '10')))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--resume', action='store_true', help='Skip tiles whose files already exist')
    ap.add_argument('--project', default=os.getenv('EE_PROJECT'), help='Earth Engine Cloud project id')
    ap.add_argument('--window-px', type=int, default=64, help='Window size (pixels per side) for chunked downloads (default: 64)')
    ap.add_argument('--tile-start', type=int, default=0, help='Start processing from this tile_id (default: 0)')
    ap.add_argument('--loglevel', default=os.getenv('LOGLEVEL', 'INFO'), help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = ap.parse_args()

    # Set log level
    try:
        logging.getLogger().setLevel(getattr(logging, args.loglevel.upper()))
    except Exception:
        pass
    logging.debug(f"Args: {args}")

    # Informative hint if project missing
    if not args.project:
        logging.warning("No EE project provided. If initialization fails, set EE_PROJECT in .env or pass --project.")
    init_ee_from_env(project=args.project)

    tiles = gpd.read_parquet(args.tiles)
    # Ensure tiles in target metric CRS for transform computation
    if tiles.crs is None or str(tiles.crs) != args.crs:
        tiles = tiles.to_crs(args.crs)
    # Also prepare WGS84 geometries for the EE region parameter
    tiles_wgs84 = tiles.to_crs('EPSG:4326')
    logging.debug(f"Tiles count={len(tiles)}, CRS={tiles.crs}")

    if 'tile_id' not in tiles.columns:
        tiles = tiles.reset_index(drop=True)
        tiles['tile_id'] = tiles.index.astype(int)

    # Also ensure tiles_wgs84 has tile_id
    if 'tile_id' not in tiles_wgs84.columns:
        tiles_wgs84 = tiles_wgs84.reset_index(drop=True)
        tiles_wgs84['tile_id'] = tiles_wgs84.index.astype(int)

    # Filter by tile_start and limit
    tiles = tiles[tiles['tile_id'] >= args.tile_start]
    tiles_wgs84 = tiles_wgs84[tiles_wgs84['tile_id'] >= args.tile_start]

    if args.limit:
        tiles = tiles.head(args.limit)
        tiles_wgs84 = tiles_wgs84.head(args.limit)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(list(tiles.iterrows()), total=len(tiles), desc='Tiles'):
        tid = int(row['tile_id'])
        # Create a GeoDataFrame for this single tile for easier CRS handling
        tile_geom_2154 = gpd.GeoDataFrame([row], crs=tiles.crs)
        geom_wgs84 = tiles_wgs84.loc[idx].geometry

        out_path = out_dir / f"emb_tile_{tid:03d}_{args.year}.tif"
        
        logging.debug(f"Processing tile_id={tid}, geom_2154_bounds={tile_geom_2154.total_bounds}")
        
        if args.resume and out_path.exists():
            logging.debug(f"Skipping tile_id={tid} (already exists)")
            continue
        try:
            img = get_embedding_image_for_region(args.year, geom_wgs84)
            if img is None:
                logging.warning(f"No embedding data available for tile_id={tid}, skipping")
                continue

            # Iterate windows of 64x64 px to keep requests small and predictable
            windows = []
            geom = tile_geom_2154.iloc[0].geometry
            xmin, ymin, xmax, ymax = geom.bounds
            step = float(args.scale) * float(args.window_px)
            x = xmin
            while x < xmax:
                y = ymin
                while y < ymax:
                    x1 = min(x + step, xmax)
                    y1 = min(y + step, ymax)
                    windows.append(shapely_box(x, y, x1, y1))
                    y = y1
                x = x1

            with tempfile.TemporaryDirectory() as tmpdir:
                part_files = []
                for wi, w in enumerate(windows):
                    window_gdf = gpd.GeoDataFrame([{'geometry': w}], crs=args.crs)
                    part_path = Path(tmpdir) / f"tile{tid:03d}_part{wi:03d}.tif"
                    save_and_reproject_tile(
                        img=img,
                        tile_geom_2154=window_gdf,
                        target_crs=args.crs,
                        scale=args.scale,
                        out_path=part_path,
                        bands=64,
                    )
                    if part_path.exists():
                        part_files.append(str(part_path))

                if not part_files:
                    logging.warning(f"No parts created for tile {tid}")
                else:
                    # Mosaic parts
                    srcs = [rasterio.open(fp) for fp in part_files]
                    mosaic, mosaic_transform = rio_merge(srcs)
                    meta = srcs[0].meta.copy()
                    for s in srcs:
                        s.close()
                    meta.update({
                        'driver': 'GTiff',
                        'height': mosaic.shape[1],
                        'width': mosaic.shape[2],
                        'transform': mosaic_transform,
                        'compress': 'deflate',
                        'tiled': True,
                    })
                    with rasterio.open(out_path, 'w', **meta) as dst:
                        dst.write(mosaic)
                    logging.info(f"Saved mosaicked tile to {out_path}")
        except Exception as e:
            logging.error(f"Failed tile_id={tid}: {e}", exc_info=True)

    logging.info(f"Done. Files saved in {out_dir}")


if __name__ == '__main__':
    main()
