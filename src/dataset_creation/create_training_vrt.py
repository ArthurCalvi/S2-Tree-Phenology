#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a VRT file from the TIF files in the data/training/training_tiles2023 directory.

This script simply uses gdalbuildvrt to create a virtual dataset from all .tif files
in the specified directory. It also adds band names as metadata to the VRT file,
including the feature bands from create_features_vrt.py plus 5 additional bands:
phenology, genus, species, source, and year.
"""

import argparse
import subprocess
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

def setup_logging(loglevel="INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, loglevel.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def add_band_names(vrt_path):
    """
    Add band names as metadata to the VRT file.
    
    Args:
        vrt_path (str or Path): Path to the VRT file
    """
    logging.info("Adding band names as metadata")
    
    # Define the band names from create_features_vrt.py
    indices = ["ndvi", "evi", "nbr", "crswir"]
    measures = ["amplitude_h1", "amplitude_h2", "phase_h1", "phase_h2", "offset", "var_residual"]
    
    band_names = []
    for idx in indices:
        for measure in measures:
            band_names.append(f"{idx}_{measure}")
    
    # Add the 5 additional bands
    additional_bands = ["phenology", "genus", "species", "source", "year"]
    band_names.extend(additional_bands)
    
    # Read the VRT file
    with open(vrt_path, 'r') as f:
        vrt_content = f.read()
    
    # Check number of bands in the VRT
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

def create_vrt(input_dir, output_vrt=None, verbose=False):
    """
    Create a VRT file from the TIF files in the input directory.
    
    Args:
        input_dir (str): Path to the directory containing TIF files
        output_vrt (str, optional): Path to the output VRT file. If None, it will be
                                   [input_dir]/training_tiles.vrt
        verbose (bool): Whether to print verbose output
    
    Returns:
        str: Path to the created VRT file
    """
    input_dir = Path(input_dir)
    
    # Find all TIF files in the input directory
    tif_files = list(input_dir.glob("*.tif"))
    if not tif_files:
        raise ValueError(f"No TIF files found in {input_dir}")
    
    logging.info(f"Found {len(tif_files)} TIF files in {input_dir}")
    
    # Define output VRT path
    if output_vrt is None:
        output_vrt = input_dir / "training_tiles.vrt"
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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create a VRT file from the TIF files in the specified directory."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/training/training_tiles2023",
        help="Path to the directory containing TIF files"
    )
    parser.add_argument(
        "--output_vrt",
        type=str,
        default=None,
        help="Path to the output VRT file. If not specified, [input_dir]/training_tiles.vrt will be used."
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
    create_vrt(args.input_dir, args.output_vrt, args.verbose)

if __name__ == "__main__":
    main() 