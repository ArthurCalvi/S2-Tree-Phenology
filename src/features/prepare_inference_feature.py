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
We store for each tile a list of paths/dates. 
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger():
    logger = logging.getLogger("prepare_inference_feature")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

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
    args = parser.parse_args()

    logger = setup_logger()
    logger.info("Starting feature config preparation...")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # We suppose each subfolder is named e.g. 20230115, 20230215, etc.
    # Inside that subfolder there is s2/ with .tif files
    subfolders = sorted(input_dir.glob(f"{args.year}*"))
    tile_info_list = []
    tile_idx = 0

    for sf in subfolders:
        # e.g. sf = mosaic2023/20230115
        if not sf.is_dir():
            continue
        s2_dir = sf / "s2"
        if not s2_dir.exists():
            continue
        
        # parse date from folder name, e.g. '20230115'
        folder_name = sf.name
        # Attempt to parse yyyymmdd
        try:
            dt = datetime.strptime(folder_name, "%Y%m%d")
        except:
            logger.warning(f"Cannot parse date from {folder_name}, skip.")
            continue
        
        # gather .tif files
        tifs = sorted(s2_dir.glob("*.tif"))
        for tif in tifs:
            # For the user’s pipeline, we might define "one tile => one .tif" 
            # Or we might have multiple .tif with different footprints. 
            # Let's do simplest approach: each .tif is one "tile" config
            tile_info = {
                "tile_idx": tile_idx,
                "date": dt.strftime("%Y-%m-%d"),
                "tif_path": str(tif)
            }
            tile_info_list.append(tile_info)
            tile_idx += 1

    # Write out a main metadata file
    metadata = {
        "num_tiles": len(tile_info_list),
        "output_dir": str(output_dir),
        "max_concurrent_jobs": args.max_concurrent_jobs
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Write each tile config
    for i, info in enumerate(tile_info_list):
        cfg_path = output_dir / f"tile_config_{i:03d}.json"
        with open(cfg_path, "w") as f:
            json.dump(info, f, indent=2)

    # Additionally, write a short summary
    summary_path = output_dir / "job_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Feature Extraction Configuration Summary\n")
        f.write("----------------------------------------\n")
        f.write(f"Found {len(tile_info_list)} TIF files.\n")
        f.write(f"Metadata file: metadata.json\n")
        f.write(f"#SBATCH --array=0-{len(tile_info_list)-1}%{args.max_concurrent_jobs}\n")

    logger.info(f"Prepared configs for {len(tile_info_list)} tiles/tifs.")
    logger.info(f"Configs saved in: {output_dir}")

if __name__ == "__main__":
    main()
