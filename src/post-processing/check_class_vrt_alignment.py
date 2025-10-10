#!/usr/bin/env python3
"""Validate class VRT alignment against forest mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check geometric alignment between class VRT and forest mask")
    parser.add_argument("--class-vrt", required=True, help="Path to class mosaic VRT")
    parser.add_argument("--forest-mask", required=True, help="Path to forest mask raster")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_path = Path(args.class_vrt)
    mask_path = Path(args.forest_mask)
    if not class_path.exists():
        raise SystemExit(f"Class VRT not found: {class_path}")
    if not mask_path.exists():
        raise SystemExit(f"Forest mask not found: {mask_path}")

    with rasterio.open(class_path) as class_src, rasterio.open(mask_path) as mask_src:
        print("Class VRT:")
        print(f"  CRS: {class_src.crs}")
        print(f"  Width x Height: {class_src.width} x {class_src.height}")
        print(f"  Transform: {class_src.transform}")
        print(f"  Resolution: {abs(class_src.transform.a)} x {abs(class_src.transform.e)}")

        print("Forest Mask:")
        print(f"  CRS: {mask_src.crs}")
        print(f"  Width x Height: {mask_src.width} x {mask_src.height}")
        print(f"  Transform: {mask_src.transform}")
        print(f"  Resolution: {abs(mask_src.transform.a)} x {abs(mask_src.transform.e)}")

        aligned = (
            class_src.crs == mask_src.crs and
            class_src.transform == mask_src.transform and
            class_src.width == mask_src.width and
            class_src.height == mask_src.height
        )
        print(f"Aligned: {'YES' if aligned else 'NO'}")
        if not aligned:
            print("Use the masking script, which will resample the mask on the fly.")


if __name__ == '__main__':
    main()
