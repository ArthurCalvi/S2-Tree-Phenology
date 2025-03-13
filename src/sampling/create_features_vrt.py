#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a VRT file from the feature TIF files in the data/features directory.

A VRT file (Virtual Dataset) is an XML file that references one or more raster datasets
and describes how they should be combined. Even with a single TIF file, creating a VRT
allows for consistency with the workflow that would be used with multiple files.

This script also adds appropriate band names as metadata to the VRT file.
"""

import os
import argparse
import subprocess
import logging
from pathlib import Path

def setup_logging(loglevel="INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, loglevel.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def create_vrt(features_dir, output_vrt=None, verbose=False):
    """
    Create a VRT file from the TIF files in the features directory.
    
    Args:
        features_dir (str): Path to the directory containing feature TIF files
        output_vrt (str, optional): Path to the output VRT file. If None, it will be
                                   [features_dir]/features.vrt
        verbose (bool): Whether to print verbose output
    
    Returns:
        str: Path to the created VRT file
    """
    features_dir = Path(features_dir)
    
    # Find all TIF files in the features directory
    tif_files = list(features_dir.glob("*.tif"))
    if not tif_files:
        raise ValueError(f"No TIF files found in {features_dir}")
    
    logging.info(f"Found {len(tif_files)} TIF files in {features_dir}")
    
    # Define output VRT path
    if output_vrt is None:
        output_vrt = features_dir / "features.vrt"
    else:
        output_vrt = Path(output_vrt)
    
    # Build the VRT using gdalbuildvrt
    logging.info(f"Creating VRT file at {output_vrt}")
    cmd = ["gdalbuildvrt", str(output_vrt)] + [str(f) for f in tif_files]
    
    if verbose:
        logging.info(f"Running command: {' '.join(cmd)}")
    
    subprocess.run(cmd, check=True)
    
    # Now add band names as metadata
    add_band_names(output_vrt)
    
    logging.info(f"Successfully created VRT file at {output_vrt}")
    return str(output_vrt)

def add_band_names(vrt_path):
    """
    Add band names as metadata to the VRT file.
    
    Args:
        vrt_path (str or Path): Path to the VRT file
    """
    logging.info("Adding band names as metadata")
    
    # Define the band names
    indices = ["ndvi", "evi", "nbr", "crswir"]
    measures = ["amplitude_h1", "amplitude_h2", "phase_h1", "phase_h2", "offset", "var_residual"]
    
    band_names = []
    for idx in indices:
        for measure in measures:
            band_names.append(f"{idx}_{measure}")
    
    # Read the VRT file
    with open(vrt_path, 'r') as f:
        vrt_content = f.read()
    
    # Check number of bands in the VRT
    import xml.etree.ElementTree as ET
    root = ET.fromstring(vrt_content)
    bands = root.findall(".//VRTRasterBand")
    
    if len(bands) != len(band_names):
        logging.warning(f"Number of bands in VRT ({len(bands)}) does not match number of band names ({len(band_names)})")
        if len(bands) < len(band_names):
            band_names = band_names[:len(bands)]
        else:
            # Add generic names for extra bands
            for i in range(len(band_names), len(bands)):
                band_names.append(f"band_{i+1}")
    
    # Add band descriptions to the VRT file
    for i, (band, name) in enumerate(zip(bands, band_names), 1):
        # Check if description is already there
        desc = band.find("Description")
        if desc is not None:
            logging.debug(f"Updating description for band {i} to '{name}'")
            desc.text = name
        else:
            logging.debug(f"Adding description for band {i} as '{name}'")
            desc = ET.SubElement(band, "Description")
            desc.text = name
    
    # Write the updated VRT file
    tree = ET.ElementTree(root)
    tree.write(vrt_path, encoding='utf-8', xml_declaration=True)
    logging.info(f"Added band names to {vrt_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create a VRT file from the TIF files in the features directory."
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        default="data/features",
        help="Path to the directory containing feature TIF files"
    )
    parser.add_argument(
        "--output_vrt",
        type=str,
        default=None,
        help="Path to the output VRT file. If not specified, [features_dir]/features.vrt will be used."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.loglevel)
    
    # Create the VRT file
    create_vrt(args.features_dir, args.output_vrt, args.verbose)

if __name__ == "__main__":
    main() 