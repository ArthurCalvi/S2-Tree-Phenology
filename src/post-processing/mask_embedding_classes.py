#!/usr/bin/env python3
"""Apply a forest mask to an embedding class mosaic (VRT or GeoTIFF)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("mask_embedding_classes")

DEFAULT_BLOCK_SIZE = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask a class mosaic with a forest mask")
    parser.add_argument("--class-raster", required=True, help="Class VRT/GeoTIFF to mask")
    parser.add_argument("--forest-mask", required=True, help="Forest mask raster (1=forest)")
    parser.add_argument("--output", required=True, help="Output masked GeoTIFF path")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="Window size for streaming")
    return parser.parse_args()


def generate_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)
            yield Window(col_off, row_off, win_w, win_h)


def main() -> None:
    args = parse_args()
    class_path = Path(args.class_raster)
    mask_path = Path(args.forest_mask)
    output_path = Path(args.output)
    if not class_path.exists():
        raise SystemExit(f"Class raster not found: {class_path}")
    if not mask_path.exists():
        raise SystemExit(f"Forest mask not found: {mask_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(class_path) as class_src:
        profile = class_src.profile
        profile.update(
            driver='GTiff',
            dtype='uint8',
            count=1,
            nodata=0,
            compress='lzw',
            predictor=2,
            tiled=True,
            blockxsize=min(256, class_src.width),
            blockysize=min(256, class_src.height),
            BIGTIFF='IF_SAFER',
        )
        with rasterio.open(output_path, 'w', **profile) as dst:
            with rasterio.open(mask_path) as mask_src:
                with WarpedVRT(
                    mask_src,
                    crs=class_src.crs,
                    transform=class_src.transform,
                    width=class_src.width,
                    height=class_src.height,
                    resampling=Resampling.nearest,
                ) as mask_vrt:
                    mask_nodata = mask_vrt.nodata
                    total_windows = (class_src.width + args.block_size - 1) // args.block_size
                    total_windows *= (class_src.height + args.block_size - 1) // args.block_size
                    with tqdm(total=total_windows, desc="Masking", unit="window", dynamic_ncols=True) as progress:
                        for window in generate_windows(class_src.width, class_src.height, args.block_size):
                            class_block = class_src.read(1, window=window)
                            mask_block = mask_vrt.read(1, window=window)
                            if mask_block.dtype != np.uint8:
                                mask_block = mask_block.astype(np.uint8, copy=False)
                            if mask_nodata is not None:
                                valid = (mask_block != mask_nodata) & (mask_block > 0)
                            else:
                                valid = mask_block > 0
                            out_block = np.where(valid, class_block.astype(np.uint8, copy=False), 0).astype(np.uint8)
                            dst.write(out_block, 1, window=window)
                            progress.update(1)

    LOGGER.info("Masked raster written to %s", output_path)


if __name__ == '__main__':
    main()
