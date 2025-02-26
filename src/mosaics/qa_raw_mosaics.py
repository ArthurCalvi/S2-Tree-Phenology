#!/usr/bin/env python3
"""
Compute basic QA (no-data%, min, max, mean, std) for raw mosaic TIFs:
 - 6 bands: [B2, B4, B8, B11, B12, MSK_CLDPRB]
 - Each band is UInt16
We scan an input directory (e.g. /lustre/fsn1/projects/rech/ego/uyr48jk/mosaic2023/)
for subfolders named "2023xxxx", then look into 's2' subfolder for TIFs.
We store stats per date -> filename -> list of band stats.

All stats except no-data% are also scaled by dividing by 65535.
"""

import os
import argparse
from pathlib import Path
import rasterio
from rasterio.windows import Window
import math
import json

def qa_for_one_file(tif_path: Path, band_names=None) -> dict:
    """
    Read the TIF in blocks, compute no-data%, min, max, mean, std for each band.
    We assume the file is UInt16. If nodata is set, we skip it.
    
    :param tif_path: Path to the input .tif
    :param band_names: List of band name strings, e.g. ["B2","B4","B8","B11","B12","MSK_CLDPRB"]
    :return: Dictionary with stats {"bands": [ {...}, {...}, ... ]}
    """
    file_stats = {"bands": []}
    
    with rasterio.open(tif_path, 'r') as src:
        # If nodata is not set, we can default to 0 or None.
        nodata_val = src.nodata if src.nodata is not None else 0
        profile = src.profile
        
        n_bands = profile["count"]
        width   = profile["width"]
        height  = profile["height"]
        block_w = profile.get("blockxsize", 1024)
        block_h = profile.get("blockysize", 1024)
        
        # Prepare stats accumulators
        # Each band => track min, max, sum, sum_sq, count_data, count_nodata
        stats_per_band = []
        for _ in range(n_bands):
            stats_per_band.append({
                "min":    65535,
                "max":    0,
                "sum":    0.0,
                "sum_sq": 0.0,
                "count_data": 0,
                "count_nodata": 0
            })
        
        # Read block by block to avoid large memory usage
        for y in range(0, height, block_h):
            for x in range(0, width, block_w):
                w = min(block_w, width - x)
                h = min(block_h, height - y)
                window = Window(x, y, w, h)
                
                # shape => (n_bands, h, w)
                arr = src.read(window=window)  # still UInt16
                
                for b_i in range(n_bands):
                    band_data = arr[b_i]
                    if nodata_val is not None:
                        valid_mask = (band_data != nodata_val)
                    else:
                        valid_mask = None
                    
                    if valid_mask is not None:
                        count_nodata = (~valid_mask).sum()
                        stats_per_band[b_i]["count_nodata"] += count_nodata
                        valid_values = band_data[valid_mask]
                    else:
                        valid_values = band_data
                    
                    n_valid = valid_values.size
                    stats_per_band[b_i]["count_data"] += n_valid
                    
                    if n_valid > 0:
                        bmin = valid_values.min()
                        bmax = valid_values.max()
                        stats_per_band[b_i]["min"] = min(stats_per_band[b_i]["min"], bmin)
                        stats_per_band[b_i]["max"] = max(stats_per_band[b_i]["max"], bmax)
                        s  = valid_values.sum(dtype="float64")
                        s2 = (valid_values**2).sum(dtype="float64")
                        stats_per_band[b_i]["sum"]    += s
                        stats_per_band[b_i]["sum_sq"] += s2
        
        total_pixels = width * height
        for b_i in range(n_bands):
            st = stats_per_band[b_i]
            nd_count = st["count_nodata"]
            dt_count = st["count_data"]
            
            if dt_count == 0:
                # No valid data
                frac_nodata = 100.0
                min_val = max_val = mean_val = std_val = None
            else:
                frac_nodata = (nd_count / total_pixels) * 100.0
                mean_val = st["sum"] / dt_count
                # variance = E[x^2] - (E[x])^2
                mean_sq = st["sum_sq"] / dt_count
                var_val = max(mean_sq - mean_val**2, 0.0)
                std_val = math.sqrt(var_val)
                min_val = float(st["min"])
                max_val = float(st["max"])
            
            # scale by 65535
            if min_val is not None:
                min_scaled  = min_val / 65535.0
                max_scaled  = max_val / 65535.0
                mean_scaled = mean_val / 65535.0
                std_scaled  = std_val / 65535.0
            else:
                min_scaled = max_scaled = mean_scaled = std_scaled = None
            
            band_dict = {
                "band": b_i + 1,
                "name": band_names[b_i] if (band_names and b_i < len(band_names)) else f"band_{b_i+1}",
                "no_data_pct": round(frac_nodata, 3),
                "min":   min_val,
                "max":   max_val,
                "mean":  round(mean_val, 2) if mean_val is not None else None,
                "std":   round(std_val, 2)  if std_val  is not None else None,
                "min_scaled":  round(min_scaled, 4) if min_scaled is not None else None,
                "max_scaled":  round(max_scaled, 4) if max_scaled is not None else None,
                "mean_scaled": round(mean_scaled,4) if mean_scaled is not None else None,
                "std_scaled":  round(std_scaled, 4) if std_scaled  is not None else None
            }
            
            file_stats["bands"].append(band_dict)
    
    return file_stats

def main():
    parser = argparse.ArgumentParser(description="QA for raw mosaic TIFs in date-based subfolders.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the mosaic2023 folder containing subdirs like 20230115/s2/*.tif")
    parser.add_argument("--output-json", type=str, default="qa_raw_mosaic.json",
                        help="Path to the output JSON file.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    out_json  = Path(args.output_json).resolve()

    # 6 band names in the TIF
    band_names = ["B2", "B4", "B8", "B11", "B12", "MSK_CLDPRB"]

    # We will store results in a structure: { "YYYYMMDD": { "filename.tif": {...} } }
    results = {}

    # For each subdirectory that starts with '2023' (like '20230115', '20230215', etc.)
    # you can adapt the pattern if needed
    for date_dir in sorted(input_dir.glob("2023*")):
        if not date_dir.is_dir():
            continue
        
        date_str = date_dir.name  # e.g. '20230115'
        s2_dir   = date_dir / "s2"
        if not s2_dir.is_dir():
            continue
        
        # Prepare results for this date
        results[date_str] = {}
        
        tif_files = sorted(s2_dir.glob("*.tif"))
        if not tif_files:
            # no TIF found in s2 subfolder
            continue
        
        print(f"Processing date {date_str} => found {len(tif_files)} TIF(s).")
        for tif_path in tif_files:
            print(f"  Analyzing {tif_path.name} ...")
            file_stats = qa_for_one_file(tif_path, band_names=band_names)
            # Store result
            results[date_str][tif_path.name] = file_stats

    # Write out as JSON
    with out_json.open('w') as f:
        json.dump(results, f, indent=2)

    print(f"Done. QA stats stored in {out_json}")

if __name__ == "__main__":
    main()
