#!/usr/bin/env python3
"""
Convert UInt16 Sentinel-2 feature TIFs to UInt8 TIFs (block by block) with LZW compression.
We assume that each input TIF has 24 bands, block size 1024, large dimension 10240 x 10240.
We do a naive scaling = value // 256, so each UInt16 [0..65535] becomes [0..255].
You can tweak the scaling method (percentile-based, min-max, etc.) if desired.
"""

import os
import argparse
from pathlib import Path
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

def convert_tif_uint16_to_uint8(input_tif: Path, output_tif: Path) -> None:
    """
    Read a UInt16 TIF in blocks, scale down to UInt8 by dividing by 256,
    and write it out with LZW compression.
    
    Args:
        input_tif: Path to the input .tif (UInt16)
        output_tif: Path to the output .tif (UInt8)
    """
    with rasterio.open(input_tif, 'r') as src:
        profile = src.profile.copy()
        
        # Get nodata value if it exists
        nodata_value = src.nodata
        
        # Update profile for UInt8
        profile.update({
            'dtype': 'uint8',
            'compress': 'lzw',
            'predictor': 2,  # often better compression for image data
            'BIGTIFF': 'YES' # safer for large outputs
        })
        
        # If there was a nodata value, update it for uint8
        if nodata_value is not None:
            if nodata_value == 0:
                # Keep 0 as nodata
                profile.update({'nodata': 0})
            elif nodata_value == 65535:
                # Map 65535 to 255 for uint8
                profile.update({'nodata': 255})
            else:
                # Scale other nodata values
                profile.update({'nodata': min(255, int(nodata_value / 256))})

        # Create output folder if needed
        output_tif.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_tif, 'w', **profile) as dst:
            block_height = profile['blockysize']
            block_width  = profile['blockxsize']

            # Loop over windows in a row-wise and column-wise manner
            for y in range(0, src.height, block_height):
                for x in range(0, src.width, block_width):
                    h = min(block_height, src.height - y)
                    w = min(block_width, src.width - x)

                    # Fixed parameter names for Window constructor
                    window = Window(col_off=x, row_off=y, width=w, height=h)
                    # Read all bands at once => shape (bands, h, w)
                    arr16 = src.read(window=window)

                    # Create a mask for nodata values if they exist
                    if nodata_value is not None:
                        nodata_mask = (arr16 == nodata_value)
                    
                    # Scale from UInt16 => UInt8 (naive: // 256)
                    arr8 = (arr16.astype('float32') / 256.0).clip(0, 255).astype('uint8')
                    
                    # Preserve nodata values
                    if nodata_value is not None:
                        if nodata_value == 0:
                            arr8[nodata_mask] = 0
                        elif nodata_value == 65535:
                            arr8[nodata_mask] = 255
                        else:
                            arr8[nodata_mask] = min(255, int(nodata_value / 256))

                    dst.write(arr8, window=window)

def main():
    parser = argparse.ArgumentParser(description="Convert all 'features_XXX.tif' from UInt16 to UInt8.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Folder containing features_XXX.tif (UInt16).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Destination folder for the UInt8 TIFs.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tifs = sorted([p for p in input_dir.glob("features_*.tif") if p.is_file()])

    if not tifs:
        print(f"No matching files found in {input_dir}")
        return

    print(f"Found {len(tifs)} .tif files in {input_dir}. Converting...")
    for tif in tqdm(tifs, desc="Converting TIFs", unit="file"):
        out_name = tif.name  # e.g. "features_000.tif"
        out_tif = output_dir / out_name
        print(f"Converting {tif.name} => {out_tif.name}")
        convert_tif_uint16_to_uint8(tif, out_tif)

    print("All conversions done.")

if __name__ == "__main__":
    main()
