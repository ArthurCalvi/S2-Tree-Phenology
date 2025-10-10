#!/usr/bin/env python3
"""Create a VRT mosaic from embedding class tiles and report alignment with a forest mask."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import rasterio

try:
    from osgeo import gdal  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    gdal = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("build_embedding_class_vrt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a VRT mosaic from *_classes.tif tiles and compare its grid with a forest mask."
    )
    parser.add_argument("--tiles-dir", required=True, help="Directory containing per-tile class rasters")
    parser.add_argument("--forest-mask", required=True, help="Forest mask raster for alignment check")
    parser.add_argument("--output-dir", default="results/postprocessing/embeddings", help="Directory for the VRT")
    parser.add_argument("--pattern", default="*_classes.tif", help="Glob pattern for class tiles")
    parser.add_argument("--vrt-name", default="embedding_classes.vrt", help="Output VRT filename")
    return parser.parse_args()


def build_vrt(tile_paths: List[Path], output_path: Path) -> None:
    if gdal is None:
        raise SystemExit("GDAL Python bindings (osgeo.gdal) are required to build the VRT")
    LOGGER.info("Building VRT %s from %d tiles", output_path, len(tile_paths))
    options = gdal.BuildVRTOptions(resolution='highest')
    ds = gdal.BuildVRT(str(output_path), [str(p) for p in tile_paths], options=options)
    if ds is None:
        raise RuntimeError(f"Failed to build VRT at {output_path}")
    ds = None


def report_alignment(vrt_path: Path, mask_path: Path) -> None:
    with rasterio.open(vrt_path) as vrt, rasterio.open(mask_path) as mask:
        LOGGER.info("Class VRT: CRS=%s, size=%sx%s, transform=%s",
                    vrt.crs, vrt.width, vrt.height, vrt.transform)
        LOGGER.info("Forest mask: CRS=%s, size=%sx%s, transform=%s",
                    mask.crs, mask.width, mask.height, mask.transform)

        same_crs = vrt.crs == mask.crs
        same_transform = vrt.transform == mask.transform
        same_dims = (vrt.width == mask.width) and (vrt.height == mask.height)

        if same_crs and same_transform and same_dims:
            LOGGER.info("Alignment check: VRT and mask grids match exactly.")
        else:
            LOGGER.warning("Alignment check: grids differ (mask will be resampled when applying it).")


def main() -> None:
    args = parse_args()
    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.is_dir():
        raise SystemExit(f"Tiles directory not found: {tiles_dir}")

    tile_paths = [p for p in sorted(tiles_dir.glob(args.pattern))
                  if p.is_file() and p.suffix.lower() == '.tif']
    if not tile_paths:
        raise SystemExit(f"No .tif tiles matching pattern '{args.pattern}' in {tiles_dir}")

    if not Path(args.forest_mask).exists():
        raise SystemExit(f"Forest mask not found: {args.forest_mask}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vrt_path = output_dir / args.vrt_name

    build_vrt(tile_paths, vrt_path)
    LOGGER.info("VRT written to %s", vrt_path)

    report_alignment(vrt_path, Path(args.forest_mask))


if __name__ == '__main__':
    main()
