#!/usr/bin/env python3
"""
compare_qa.py

Reads two JSON files:
  1) Mosaic QA stats from qa_mosaics.py, ignoring the last band (MSK_CLDPRB) when
     computing average no-data but storing its mean to compute a "cloud cover %".
  2) Features QA stats from qa_features.py.

We then:
  - Compute average no-data % per tile (only for the first 5 mosaic bands).
  - Compute average mean for MSK_CLDPRB to get cloud cover % = (mean / 2185) * 100.
  - For each feature band, compare no-data % to the tile's mosaic no-data average.
  - Output two JSON files:
      (a) Detailed differences for each tile/band, including the tile's cloud cover.
      (b) Aggregated differences by (index, feature_type).
"""

import json
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple, List, Any


def parse_band_name(band_name: str) -> Tuple[str, str]:
    """
    Parse the feature band name to identify (index, feature_type).
    Examples:
      "ndvi_amplitude_h1" -> ("ndvi", "amplitude_h1")
      "evi_phase_h2"      -> ("evi", "phase_h2")
      "nbr_offset"        -> ("nbr", "offset")
      "crswir_variance"   -> ("crswir", "variance")
    """
    parts = band_name.split("_")
    index = parts[0]  # e.g. 'ndvi', 'evi', 'nbr', 'crswir', ...

    if len(parts) == 3 and parts[1] in ["amplitude", "phase"]:
        # e.g. ndvi_amplitude_h1 => feature_type = amplitude_h1
        feature_type = f"{parts[1]}_{parts[2]}"
    elif len(parts) > 1 and parts[1] in ["offset", "variance"]:
        # e.g. ndvi_offset => feature_type = offset
        feature_type = parts[1]
    else:
        feature_type = "unknown"

    return index, feature_type


def extract_tile_name(tif_name: str) -> str:
    """
    Extract a meaningful tile identifier from the filename.
    For mosaic files like "s2_EPSG2154_1024000_6246400.tif", use the coordinates.
    For feature files like "features_000.tif", extract the numeric index.
    """
    parts = tif_name.split("_")
    
    # Handle mosaic files (s2_EPSG2154_X_Y.tif)
    if len(parts) >= 4 and parts[0] == "s2" and parts[1] == "EPSG2154":
        # Use the coordinates as the tile identifier
        return f"{parts[2]}_{parts[3].split('.')[0]}"
    
    # Handle feature files (features_NNN.tif)
    if len(parts) >= 2 and parts[0] == "features":
        # Extract the numeric index
        return f"features_{parts[1].split('.')[0]}"
    
    # Fallback to the original method
    return parts[0]


def parse_mosaic_json(mosaic_json_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse the mosaic QA JSON to produce, for each tile:
      - 'avg_nodata': average no-data% (only the first 5 bands)
      - 'cloud_pct':  average (MSK_CLDPRB mean / 2184) * 100

    :param mosaic_json_path: Path to qa_raw_mosaic.json
    :return: { tile_name: {"avg_nodata": float, "cloud_pct": float} }
    """
    with mosaic_json_path.open('r') as f:
        mosaic_data = json.load(f)

    # We'll accumulate sums for each tile
    # 'nodata_sum' -> sum of TIF-level no-data averages
    # 'nodata_count' -> how many TIFs contributed
    # 'cloud_mean_sum' -> sum of TIF-level means for the last band
    # 'cloud_mean_count' -> how many TIFs had a valid mean
    tile_stats = {}

    for date_str, tif_dict in mosaic_data.items():
        for tif_name, stats_dict in tif_dict.items():
            tile_name = extract_tile_name(tif_name)
            if tile_name not in tile_stats:
                tile_stats[tile_name] = {
                    "nodata_sum": 0.0,
                    "nodata_count": 0,
                    "cloud_mean_sum": 0.0,
                    "cloud_mean_count": 0
                }
            band_list = stats_dict.get("bands", [])

            # Gather no-data for first 5 bands
            nodata_vals = []
            cloud_mean = None
            for i, binfo in enumerate(band_list):
                # i=0..4 => B2,B4,B8,B11,B12, i=5 => MSK_CLDPRB
                if i < 5:
                    # incorporate no_data_pct for average
                    nd = binfo["no_data_pct"]
                    if nd is not None:
                        nodata_vals.append(nd)
                elif i == 5:
                    # store the raw mean for MSK_CLDPRB
                    # binfo["mean"] is the raw mean (0..65535)
                    if binfo["mean"] is not None:
                        cloud_mean = binfo["mean"]

            # Average no-data% across the first 5 bands
            if len(nodata_vals) > 0:
                avg_nd_5 = sum(nodata_vals) / len(nodata_vals)
                tile_stats[tile_name]["nodata_sum"] += avg_nd_5
                tile_stats[tile_name]["nodata_count"] += 1

            # Summation for cloud probability means
            if cloud_mean is not None:
                tile_stats[tile_name]["cloud_mean_sum"] += cloud_mean
                tile_stats[tile_name]["cloud_mean_count"] += 1

    # Compute final dictionary
    result = {}
    for tile_name, vals in tile_stats.items():
        nodata_count = vals["nodata_count"]
        cloud_count = vals["cloud_mean_count"]

        if nodata_count > 0:
            avg_nodata = vals["nodata_sum"] / nodata_count
        else:
            avg_nodata = 0.0

        if cloud_count > 0:
            avg_cloud_val = vals["cloud_mean_sum"] / cloud_count
            # Convert to %: (raw mean / 2185) * 100

            cloud_pct = (avg_cloud_val / 2185.0) * 100.0
        else:
            cloud_pct = 0.0

        result[tile_name] = {
            "avg_nodata": avg_nodata,
            "cloud_pct": cloud_pct
        }

    return result


def compare_nodata(
    mosaic_json_path: Path,
    features_json_path: Path,
    out_json_path: Path,
    out_agg_json_path: Path
) -> None:
    """
    Compare mosaic no-data stats vs. features no-data stats:
      - mosaic_json_path => from qa_mosaics.py
      - features_json_path => from qa_features.py
      - out_json_path => detailed JSON, with data organized by tile
      - out_agg_json_path => aggregated JSON, data by (index, feature)

    JSON structure includes tile information, band details, and cloud coverage stats.
    """
    # 1) Build dictionary tile -> {"avg_nodata": float, "cloud_pct": float}
    mosaic_nodata_map = parse_mosaic_json(mosaic_json_path)

    # 2) Read features JSON
    with features_json_path.open("r") as f:
        features_data = json.load(f)

    # Create a data structure for detailed results
    detailed_data = {"tiles": {}}
    # For aggregation: group differences by (index, feature_type)
    agg_dict = {}

    # Get ordered lists of tiles from both datasets
    mosaic_tiles = list(mosaic_nodata_map.keys())
    feature_files = list(features_data.keys())
    
    # Ensure we have data to work with
    if not mosaic_tiles or not feature_files:
        print("Warning: One or both datasets are empty")
        return
    
    # Map feature files to mosaic tiles based on order
    # If lengths don't match, we'll use as many as we can
    tile_mapping = {}
    for i, feature_file in enumerate(feature_files):
        if i < len(mosaic_tiles):
            tile_mapping[feature_file] = mosaic_tiles[i]
        else:
            # More feature files than mosaic tiles
            print(f"Warning: No matching mosaic tile for feature file {feature_file}")
            # Use the first tile as fallback (or could skip this file)
            tile_mapping[feature_file] = mosaic_tiles[0]
    
    # Print the mapping for verification
    print("Mapping feature files to mosaic tiles:")
    for feature_file, mosaic_tile in tile_mapping.items():
        print(f"  {feature_file} -> {mosaic_tile}")

    for tif_name, file_stats in features_data.items():
        # Use the mapping to get the corresponding mosaic tile
        mosaic_tile = tile_mapping.get(tif_name)
        if not mosaic_tile:
            print(f"Warning: No mapping found for {tif_name}, skipping")
            continue
            
        tile_info = mosaic_nodata_map.get(mosaic_tile, {"avg_nodata": 0.0, "cloud_pct": 0.0})
        mosaic_avg_nd = tile_info["avg_nodata"]
        cloud_pct = tile_info["cloud_pct"]

        # Initialize tile entry if it doesn't exist
        if tif_name not in detailed_data["tiles"]:
            detailed_data["tiles"][tif_name] = {
                "filename": tif_name,
                "mosaic_tile": mosaic_tile,  # Add the mapped mosaic tile for reference
                "avg_nodata": mosaic_avg_nd,
                "cloud_pct": cloud_pct,
                "bands": []
            }
        
        # For each band in features
        for band_stat in file_stats.get("bands", []):
            band_name = band_stat["name"]
            feat_no_data = band_stat["no_data_pct"]
            if feat_no_data is None:
                continue

            diff = feat_no_data - mosaic_avg_nd
            
            # Add band entry to the tile
            detailed_data["tiles"][tif_name]["bands"].append({
                "name": band_name,
                "mosaic_no_data": round(mosaic_avg_nd, 3),
                "feature_no_data": round(feat_no_data, 3),
                "difference": round(diff, 3)
            })

            # For aggregated stats
            index, feature_type = parse_band_name(band_name)
            key = (index, feature_type)
            if key not in agg_dict:
                agg_dict[key] = [0.0, 0]  # sum of differences, count
            agg_dict[key][0] += diff
            agg_dict[key][1] += 1

    # Build aggregated JSON => average difference by (index, feature_type)
    agg_data = {"by_index_feature": []}
    for (idx, feat), (sum_diff, cnt) in agg_dict.items():
        avg_diff = sum_diff / cnt if cnt > 0 else 0.0
        agg_data["by_index_feature"].append({
            "index": idx,
            "feature_type": feat,
            "avg_difference": round(avg_diff, 3),
            "count": cnt
        })

    # Sort aggregated data for convenience
    agg_data["by_index_feature"].sort(key=lambda x: (x["index"], x["feature_type"]))

    # Write detailed JSON
    with out_json_path.open("w") as f:
        json.dump(detailed_data, f, indent=2)

    # Write aggregated JSON
    with out_agg_json_path.open("w") as f:
        json.dump(agg_data, f, indent=2)


def main() -> None:
    """
    Example usage:
      python compare_nodata.py \
        --mosaic-json qa_raw_mosaic.json \
        --features-json qa_stats.json \
        --out-json diff_details.json \
        --out-agg-json diff_agg.json
    """
    parser = argparse.ArgumentParser(description="Compare mosaic no-data stats vs. features no-data stats.")
    parser.add_argument("--mosaic-json", type=str, required=True,
                        help="Path to the qa_raw_mosaic.json from qa_mosaics.py")
    parser.add_argument("--features-json", type=str, required=True,
                        help="Path to the qa_stats.json from qa_features.py")
    parser.add_argument("--out-json", type=str, default="nodata_differences.json",
                        help="Path to output JSON with detailed differences.")
    parser.add_argument("--out-agg-json", type=str, default="nodata_aggregated.json",
                        help="Path to output JSON with aggregated differences by index & feature type.")
    args = parser.parse_args()

    compare_nodata(
        mosaic_json_path=Path(args.mosaic_json),
        features_json_path=Path(args.features_json),
        out_json_path=Path(args.out_json),
        out_agg_json_path=Path(args.out_agg_json)
    )

    print(f"Done. Wrote detailed JSON to: {args.out_json}")
    print(f"Done. Wrote aggregated JSON to: {args.out_agg_json}")


if __name__ == "__main__":
    main()
