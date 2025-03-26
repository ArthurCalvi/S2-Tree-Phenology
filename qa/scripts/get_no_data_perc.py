#!/usr/bin/env python3
"""
Compute the no-data percentage of a mosaic .vrt file.

This script searches the given directory for the only .vrt file (assumed to be a monthly mosaic),
reads it in blocks (to manage memory for large Sentinel-2 mosaics over France), computes the percentage
of pixels equal to the no-data value (read from the raster's profile), and updates a JSON file with
an entry mapping the directory basename (e.g., a date) to the computed percentage.

Usage:
    python collect_dem_info.py <input_dir> <output_json> [--debug]
"""

import argparse
import glob
import json
import os
from typing import Tuple

import numpy as np
import rasterio


def compute_nodata_percentage(vrt_path: str, debug: bool = False) -> Tuple[float, float]:
    """
    Compute the percentage of no-data pixels in the given .vrt file.

    Reads the raster block-by-block to avoid loading the entire dataset into memory.

    Args:
        vrt_path: Path to the .vrt file.
        debug: If True, print debugging information.

    Returns:
        A tuple (nodata_value, percentage) where:
         - nodata_value: The nodata value read from the raster profile.
         - percentage: The percentage of pixels equal to the nodata value.
    """
    total_pixels = 0
    nodata_pixels = 0

    with rasterio.open(vrt_path) as src:
        nodata = src.nodata
        if nodata is None:
            raise ValueError(f"No nodata value found in the raster profile of {vrt_path}")
        if debug:
            print(f"[DEBUG] Nodata value: {nodata}")

        # Iterate over the block windows for band 1
        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            total = data.size
            count_nd = np.count_nonzero(data == nodata)
            total_pixels += total
            nodata_pixels += count_nd
            if debug:
                print(f"[DEBUG] Window {window}: total pixels = {total}, nodata pixels = {count_nd}")

    if total_pixels == 0:
        raise ValueError("No pixels were read from the raster.")

    percentage = (nodata_pixels / total_pixels) * 100.0
    if debug:
        print(f"[DEBUG] Total pixels: {total_pixels}, Total nodata pixels: {nodata_pixels}")
        print(f"[DEBUG] No-data percentage: {percentage:.2f}%")
    return nodata, percentage


def update_json(date: str, percentage: float, output_json: str, debug: bool = False) -> None:
    """
    Update the JSON file with an entry mapping the given date to the no-data percentage.

    If the JSON file exists, it is loaded and updated; otherwise, a new dictionary is created.

    Args:
        date: The date string (typically the basename of the input directory).
        percentage: The computed no-data percentage.
        output_json: Path to the JSON file to update.
        debug: If True, print debugging information.
    """
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            try:
                data = json.load(f)
            except Exception:
                data = {}
    else:
        data = {}
    data[date] = percentage
    if debug:
        print(f"[DEBUG] Updating JSON: {date} -> {percentage:.2f}%")
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute no-data percentage from a mosaic .vrt file.")
    parser.add_argument("input_dir", help="Input directory containing a single .vrt file")
    parser.add_argument("output_json", help="Output JSON file to update with no-data percentage")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    # Find the only .vrt file in the input directory
    vrt_files = glob.glob(os.path.join(args.input_dir, "*.vrt"))
    if len(vrt_files) != 1:
        raise ValueError(f"Expected exactly one .vrt file in {args.input_dir}, found {len(vrt_files)}")
    vrt_path = vrt_files[0]
    if args.debug:
        print(f"[DEBUG] Found VRT file: {vrt_path}")

    # Compute the no-data percentage
    nodata, percentage = compute_nodata_percentage(vrt_path, debug=args.debug)
    # Use the input directory basename (assumed to be a date string) as the key
    date_str = os.path.basename(os.path.dirname(os.path.normpath(args.input_dir)))

    if args.debug:
        print(f"[DEBUG] Using date key: {date_str}")

    # Update the JSON file
    update_json(date_str, percentage, args.output_json, debug=args.debug)

    # Print result to stdout
    result = {"nodata": nodata, "percentage": percentage}
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()