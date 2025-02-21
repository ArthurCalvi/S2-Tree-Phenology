#!/usr/bin/env python3
"""
prepare_inference_feature.py
-----------------------------
Scan a mosaic directory structure (e.g. mosaic2023/20230115/s2/*.tif, etc.)
and prepare a JSON config for each 'tile' or each region we want to process.

We assume each subfolder is named like: 
   20230115/s2/
   20230215/s2/
... 
and contains .tif files with the required 6 bands (B2, B4, B8, B11, B12, MSK_CLDPRB).
We store for each spatial location a list of paths/dates across time. 
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def setup_logger():
    logger = logging.getLogger("prepare_inference_feature")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

def extract_spatial_id(tif_path: Path) -> str:
    """
    Extract a spatial identifier from the TIF filename.
    Assumes filenames like: s2_EPSG2154_512000_6860800.tif
    Returns: "512000_6860800"
    """
    stem = tif_path.stem  # removes .tif
    parts = stem.split('_')
    if len(parts) >= 4:
        return f"{parts[-2]}_{parts[-1]}"
    return stem  # fallback to full stem if pattern doesn't match

def main():
    parser = argparse.ArgumentParser(description="Prepare feature extraction configs.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to the mosaic directory, e.g. mosaic2023/")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to store the generated config files.")
    parser.add_argument("--year", type=str, default="2023",
                        help="Year to scan in mosaic dir (default=2023).")
    parser.add_argument("--max-concurrent-jobs", type=int, default=20,
                        help="Max array concurrency for HPC.")
    parser.add_argument("--min-dates", type=int, default=3,
                        help="Minimum number of dates required per tile (default=3).")
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Starting feature config preparation...")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store TIFs and dates for each spatial location
    spatial_groups = defaultdict(lambda: {"tif_paths": [], "dates": []})

    # We suppose each subfolder is named e.g. 20230115, 20230215, etc.
    subfolders = sorted(input_dir.glob(f"{args.year}*"))
    
    # First, group TIFs by their spatial location
    for sf in subfolders:
        if not sf.is_dir():
            continue
        s2_dir = sf / "s2"
        if not s2_dir.exists():
            continue
        
        # parse date from folder name, e.g. '20230115'
        try:
            dt = datetime.strptime(sf.name, "%Y%m%d")
            date_str = dt.strftime("%Y-%m-%d")
        except:
            logger.warning(f"Cannot parse date from {sf.name}, skip.")
            continue
        
        # gather .tif files
        tifs = sorted(s2_dir.glob("*.tif"))
        for tif in tifs:
            spatial_id = extract_spatial_id(tif)
            spatial_groups[spatial_id]["tif_paths"].append(str(tif))
            spatial_groups[spatial_id]["dates"].append(date_str)

    # Convert groups to tile configs
    tile_info_list = []
    tile_idx = 0
    
    for spatial_id, data in spatial_groups.items():
        # Skip locations with too few dates
        if len(data["dates"]) < args.min_dates:
            logger.warning(
                f"Spatial tile {spatial_id} has only {len(data['dates'])} dates "
                f"(minimum {args.min_dates} required). Skipping."
            )
            continue
            
        # Sort both lists by date
        sorted_pairs = sorted(zip(data["dates"], data["tif_paths"]))
        dates, tif_paths = zip(*sorted_pairs)
        
        tile_info = {
            "tile_idx": tile_idx,
            "spatial_id": spatial_id,
            "dates": list(dates),
            "tif_paths": list(tif_paths)
        }
        tile_info_list.append(tile_info)
        tile_idx += 1

    if not tile_info_list:
        logger.error(
            f"No valid tiles found with minimum {args.min_dates} dates. "
            "Check your input directory structure and min-dates parameter."
        )
        sys.exit(1)

    # Write out a main metadata file
    metadata = {
        "num_tiles": len(tile_info_list),
        "output_dir": str(output_dir),
        "max_concurrent_jobs": args.max_concurrent_jobs,
        "min_dates_per_tile": args.min_dates
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Write each tile config
    for info in tile_info_list:
        cfg_path = output_dir / f"tile_config_{info['tile_idx']:03d}.json"
        with open(cfg_path, "w") as f:
            json.dump(info, f, indent=2)

    # Additionally, write a short summary
    summary_path = output_dir / "job_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Feature Extraction Configuration Summary\n")
        f.write("----------------------------------------\n")
        f.write(f"Found {len(tile_info_list)} unique spatial locations.\n")
        f.write(f"Each tile has at least {args.min_dates} dates.\n")
        f.write(f"Metadata file: metadata.json\n")
        f.write(f"#SBATCH --array=0-{len(tile_info_list)-1}%{args.max_concurrent_jobs}\n")

    logger.info(f"Prepared configs for {len(tile_info_list)} spatial locations.")
    logger.info(f"Each tile has at least {args.min_dates} dates.")
    logger.info(f"Configs saved in: {output_dir}")

if __name__ == "__main__":
    main()
