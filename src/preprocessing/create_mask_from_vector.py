#!/usr/bin/env python3
"""
create_mask_from_vector.py
--------------------------
Creates a binary mask raster (GeoTIFF) from a vector file (e.g., GeoPackage, Shapefile)
by burning vector features onto a grid defined by a reference raster.

This is useful for creating masks that perfectly align with analysis rasters.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("create_mask_from_vector")

# --- Main Function ---
def create_mask(vector_path: Path, ref_raster_path: Path, output_path: Path, 
                  burn_value: int = 1, nodata_value: int = 0, 
                  output_dtype: str = 'uint8', all_touched: bool = False):
    """
    Rasterizes vector features onto a reference grid to create a mask.

    Args:
        vector_path: Path to the input vector file (readable by GeoPandas).
        ref_raster_path: Path to the reference raster (GeoTIFF, VRT, etc.).
        output_path: Path to save the output GeoTIFF mask.
        burn_value: Value to assign to pixels covered by vector features.
        nodata_value: Value for background pixels (will be set as raster nodata).
        output_dtype: NumPy dtype for the output raster (e.g., 'uint8', 'uint16').
        all_touched: If True, burn all pixels touched by geometries, not just centroids.
    """
    start_time = time.time()
    logger.info(f"Starting mask creation.")
    logger.info(f" Input vector: {vector_path}")
    logger.info(f" Reference raster: {ref_raster_path}")
    logger.info(f" Output mask: {output_path}")

    try:
        # 1. Read reference raster metadata
        logger.info("Reading reference raster metadata...")
        with rasterio.open(ref_raster_path) as ref_src:
            ref_profile = ref_src.profile
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            ref_width = ref_src.width
            ref_height = ref_src.height
            logger.info(f" Reference grid: CRS={ref_crs}, Size=({ref_width}x{ref_height}), Transform={ref_transform}")
            if ref_profile['count'] > 1:
                logger.warning(f"Reference raster has {ref_profile['count']} bands. Using metadata from the first band.")

        # 2. Read vector data
        logger.info("Reading vector data...")
        gdf = gpd.read_file(vector_path)
        logger.info(f" Read {len(gdf)} features from {vector_path.name}.")

        # 3. Check and potentially reproject vector CRS
        if gdf.crs != ref_crs:
            logger.warning(f"Vector CRS ({gdf.crs}) differs from reference raster CRS ({ref_crs}). Reprojecting vector...")
            try:
                gdf = gdf.to_crs(ref_crs)
                logger.info(f"Vector reprojected successfully to {ref_crs}.")
            except Exception as e:
                 logger.error(f"Failed to reproject vector data: {e}")
                 raise ValueError(f"CRS mismatch and reprojection failed. Vector: {gdf.crs}, Reference: {ref_crs}")
        else:
            logger.info("Vector CRS matches reference raster CRS.")

        # Ensure geometries are valid (optional but good practice)
        # gdf = gdf[gdf.is_valid]
        # logger.info(f" Kept {len(gdf)} valid geometries.")

        # Prepare shapes iterator for rasterize
        shapes = ((geom, burn_value) for geom in gdf.geometry)

        # 4. Prepare output raster profile
        output_profile = ref_profile.copy()
        output_profile.update({
            'driver': 'GTiff',
            'height': ref_height,
            'width': ref_width,
            'count': 1,
            'dtype': output_dtype,
            'crs': ref_crs,
            'transform': ref_transform,
            'nodata': nodata_value,
            'compress': 'DEFLATE',
            'predictor': 1, # Suitable for thematic masks
            'zlevel': 9,
            'tiled': True,
            # Keep blocksize from reference or set a default?
            # Let's keep reference blocksize if available, else default
            'blockxsize': ref_profile.get('blockxsize', 256),
            'blockysize': ref_profile.get('blockysize', 256),
            'BIGTIFF': 'YES' # Good practice for potentially large masks
        })
        # Remove keys not applicable to output
        output_profile.pop('photometric', None)
        output_profile.pop('band', None)
        output_profile.pop('interleave', None)
        
        logger.debug(f"Output profile configured: {output_profile}")

        # 5. Rasterize features
        logger.info(f"Rasterizing {len(gdf)} features (burn_value={burn_value}, nodata={nodata_value}, all_touched={all_touched})...")
        rasterized_mask = rasterize(
            shapes=shapes,
            out_shape=(ref_height, ref_width),
            transform=ref_transform,
            fill=nodata_value, # Background value
            default_value=burn_value,
            dtype=output_dtype,
            all_touched=all_touched
        )
        logger.info("Rasterization complete.")

        # 6. Write output raster
        logger.info(f"Writing output mask to {output_path} using nbits=1 optimization...")
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
        # Use nbits=1 creation option for optimized binary mask storage
        creation_opts = {'nbits': 1}
        with rasterio.open(output_path, 'w', **output_profile, **creation_opts) as dst:
            dst.write(rasterized_mask, 1)

        end_time = time.time()
        logger.info(f"Successfully created mask: {output_path}")
        logger.info(f"Processing finished in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
        logger.error(f"Error: Input file not found: {e}")
        sys.exit(1)
    except ImportError as e:
         logger.error(f"Error: Missing dependency. Please install geopandas and rasterio. Details: {e}")
         sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Create a binary mask raster from a vector file based on a reference raster grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vector-file", type=str, required=True,
                        help="Path to the input vector file (e.g., GeoPackage, Shapefile).")
    parser.add_argument("--reference-raster", type=str, required=True,
                        help="Path to the reference raster file (e.g., GeoTIFF, VRT) defining the output grid.")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path for the output GeoTIFF mask file.")
    parser.add_argument("--burn-value", type=int, default=1,
                        help="Value to burn for pixels covered by vector features.")
    parser.add_argument("--nodata-value", type=int, default=0,
                        help="Value for background pixels (will be set as raster nodata value).")
    parser.add_argument("--dtype", type=str, default='uint8',
                        help="Data type for the output raster (e.g., uint8, uint16).")
    parser.add_argument("--all-touched", action='store_true',
                        help="Rasterize all pixels touched by geometry, not just those whose center is covered.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")
    
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    vector_path = Path(args.vector_file)
    ref_raster_path = Path(args.reference_raster)
    output_path = Path(args.output_file)

    # Basic validation
    if not vector_path.exists():
         logger.error(f"Input vector file not found: {vector_path}")
         sys.exit(1)
    if not ref_raster_path.exists():
         logger.error(f"Reference raster file not found: {ref_raster_path}")
         sys.exit(1)
    if output_path.suffix.lower() not in ['.tif', '.tiff']:
        logger.warning(f"Output file extension is not .tif or .tiff ({output_path.suffix}). GeoTIFF format will be used.")
        # Ensure correct suffix?
        # output_path = output_path.with_suffix('.tif')

    create_mask(
        vector_path=vector_path,
        ref_raster_path=ref_raster_path,
        output_path=output_path,
        burn_value=args.burn_value,
        nodata_value=args.nodata_value,
        output_dtype=args.dtype,
        all_touched=args.all_touched
    )

if __name__ == "__main__":
    main() 