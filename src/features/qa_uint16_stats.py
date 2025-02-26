#!/usr/bin/env python3
"""
Compute basic QA for UInt16 TIF files, by blocks:
  - No-data percentage
  - Min, Max, Mean, Std
We then scale min, max, mean, std by dividing by 65535.

All "features_*.tif" in --input-dir will be processed. We output
a CSV with stats or simply print them out.

Author: YourName
Date: 2023-xx-yy
"""

import os
import argparse
from pathlib import Path
import rasterio
from rasterio.windows import Window
import csv
import math
import json

def qa_for_one_file(tif_path: Path, results_dict=None) -> dict:
    """
    Read the TIF in blocks, compute no-data%, min, max, mean, std for each band.
    We assume the file is UInt16. If nodata is set in the dataset, we skip it.
    
    :param tif_path: Path to the input .tif
    :param results_dict: Dictionary to store results if provided
    :return: Dictionary with stats for this file
    """
    file_stats = {"bands": []}
    
    # Define band names mapping
    band_names = [
        "ndvi_amplitude_h1",
        "ndvi_amplitude_h2",
        "ndvi_phase_h1",
        "ndvi_phase_h2",
        "ndvi_offset",
        "ndvi_variance",
        "evi_amplitude_h1",
        "evi_amplitude_h2",
        "evi_phase_h1",
        "evi_phase_h2",
        "evi_offset",
        "evi_variance",
        "nbr_amplitude_h1",
        "nbr_amplitude_h2",
        "nbr_phase_h1",
        "nbr_phase_h2",
        "nbr_offset",
        "nbr_variance",
        "crswir_amplitude_h1",
        "crswir_amplitude_h2",
        "crswir_phase_h1",
        "crswir_phase_h2",
        "crswir_offset",
        "crswir_variance"
    ]
    
    with rasterio.open(tif_path, 'r') as src:
        profile = src.profile
        nodata_val = src.nodata if src.nodata is not None else 0  # set to 0 if none
        print(f"Nodata value: {nodata_val}")
        
        n_bands = profile['count']
        width   = profile['width']
        height  = profile['height']
        block_w = profile.get('blockxsize', 1024)
        block_h = profile.get('blockysize', 1024)

        # Each band => track min, max, sum, sum_sq, count_data, count_nodata
        stats = []
        for _ in range(n_bands):
            stats.append({
                'min': 65535,       # large initial min
                'max': 0,           # small initial max
                'sum': 0.0,         # double for accumulation
                'sum_sq': 0.0,
                'count_data': 0,    # # of valid (non-nodata) px
                'count_nodata': 0
            })

        # Read file block by block
        for y in range(0, height, block_h):
            for x in range(0, width, block_w):
                w = min(block_w, width - x)
                h = min(block_h, height - y)
                window = Window(x, y, w, h)

                # shape => (n_bands, h, w)
                arr = src.read(window=window)  # still UInt16

                # For each band, update stats
                for b_i in range(n_bands):
                    band_data = arr[b_i]
                    if nodata_val is not None:
                        # valid_mask => True where valid data
                        valid_mask = (band_data != nodata_val)
                    else:
                        # everything is valid if nodata is None
                        valid_mask = None

                    # Count no-data
                    if valid_mask is not None:
                        count_nodata = (~valid_mask).sum()
                        stats[b_i]['count_nodata'] += count_nodata
                        # Filter to valid data
                        valid_values = band_data[valid_mask]
                    else:
                        valid_values = band_data

                    n_valid = valid_values.size
                    stats[b_i]['count_data'] += n_valid

                    if n_valid > 0:
                        bmin = valid_values.min()
                        bmax = valid_values.max()
                        stats[b_i]['min'] = min(stats[b_i]['min'], bmin)
                        stats[b_i]['max'] = max(stats[b_i]['max'], bmax)
                        s   = valid_values.sum(dtype='float64')
                        s2  = (valid_values**2).sum(dtype='float64')
                        stats[b_i]['sum']    += s
                        stats[b_i]['sum_sq'] += s2

        # Now compute final stats for each band
        total_pixels = width * height
        for b_i in range(n_bands):
            st = stats[b_i]
            nd_count = st['count_nodata']
            dt_count = st['count_data']  # valid data count

            # Avoid any zero division
            if dt_count == 0:
                # Then min, max, mean, std = 0?
                frac_nodata = 100.0
                min_val, max_val, mean_val, std_val = (None, None, None, None)
            else:
                frac_nodata = (nd_count / total_pixels) * 100.0
                mean_val = st['sum'] / dt_count
                # Variance = E[x^2] - (E[x])^2
                mean_sq = st['sum_sq'] / dt_count
                var_val = mean_sq - (mean_val * mean_val)
                var_val = max(var_val, 0.0)  # clip to avoid negative
                std_val = math.sqrt(var_val)
                min_val = float(st['min'])
                max_val = float(st['max'])

            # scale results by 65535
            if min_val is not None:
                min_scaled = min_val / 65535.0
                max_scaled = max_val / 65535.0
                mean_scaled = mean_val / 65535.0
                std_scaled  = std_val / 65535.0
            else:
                min_scaled = max_scaled = mean_scaled = std_scaled = None

            # Store band stats in a dictionary
            band_stats = {
                "band": b_i+1,
                "name": band_names[b_i] if b_i < len(band_names) else f"band_{b_i+1}",
                "no_data_pct": round(frac_nodata, 3),
                "min": min_val,
                "max": max_val,
                "mean": round(mean_val, 2) if mean_val is not None else None,
                "std": round(std_val, 2) if std_val is not None else None,
                "min_scaled": round(min_scaled, 4) if min_scaled is not None else None,
                "max_scaled": round(max_scaled, 4) if max_scaled is not None else None,
                "mean_scaled": round(mean_scaled, 4) if mean_scaled is not None else None,
                "std_scaled": round(std_scaled, 4) if std_scaled is not None else None
            }
            
            file_stats["bands"].append(band_stats)
            
    if results_dict is not None:
        results_dict[tif_path.name] = file_stats
        
    return file_stats

def main():
    parser = argparse.ArgumentParser(description="Compute QA stats for features_*.tif in a directory.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the folder containing features_*.tif.")
    parser.add_argument("--output-json", type=str, default="qa_stats.json",
                        help="Path to the output JSON file (default=qa_stats.json).")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_json = Path(args.output_json)
    tif_list = sorted(in_dir.glob("features_*.tif"))

    if len(tif_list) == 0:
        print(f"No 'features_*.tif' found in {in_dir}")
        return

    # Store results in a dictionary
    results = {}
    
    print(f"Processing {len(tif_list)} files in {in_dir}")
    for i, tif_path in enumerate(tif_list, start=1):
        print(f"[{i}/{len(tif_list)}] Analyzing {tif_path.name} ...")
        qa_for_one_file(tif_path, results_dict=results)

    # Write results to JSON file
    with out_json.open('w') as f:
        json.dump(results, f, indent=2)

    print(f"Done. Stats saved to {out_json}")

if __name__ == "__main__":
    main()
