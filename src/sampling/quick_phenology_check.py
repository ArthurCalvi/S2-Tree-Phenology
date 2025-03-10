#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight script to quickly analyze species to phenology mappings in the datasets.
Focuses only on relevant columns and outputs a more compact report.
"""

import argparse
import logging
import pandas as pd
import geopandas as gpd
from collections import defaultdict
import sys
from pathlib import Path


def setup_logging(loglevel):
    """Configure logging based on the provided level."""
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {loglevel}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_dataset(path, source_name):
    """
    Loads a parquet dataset.
    
    Args:
        path: Path to the parquet file
        source_name: Name of the dataset for logging
        
    Returns:
        DataFrame with the dataset
    """
    logging.info(f"Loading {source_name} dataset...")
    try:
        # Load the whole dataset since we need the geometry column for GeoDataFrame
        df = gpd.read_parquet(path)
        logging.info(f"Loaded {source_name} dataset with {len(df)} rows")
        logging.debug(f"Available columns: {', '.join(df.columns)}")
        return df
    except Exception as e:
        logging.error(f"Failed to load {source_name} dataset: {e}")
        return None


def get_species_phenology_mapping(df, source_name):
    """
    Gets the mapping between species and phenology in a dataset.
    
    Args:
        df: DataFrame containing species and phenology columns
        source_name: Name of the data source
        
    Returns:
        Dictionary mapping species to list of phenologies
    """
    # Determine column names
    if "specie_en" in df.columns and "phen_en" in df.columns:
        species_col = "specie_en"
        phenology_col = "phen_en"
    elif "species" in df.columns and "phenology" in df.columns:
        species_col = "species"
        phenology_col = "phenology"
    else:
        logging.error(f"Required columns not found in {source_name} dataset.")
        logging.info(f"Available columns: {df.columns}")
        return {}
    
    logging.info(f"Using columns {species_col} and {phenology_col} for {source_name} dataset")
    
    # Create mapping dictionary
    species_to_phenology = defaultdict(set)
    
    # Group by species and collect unique phenologies
    species_groups = df.groupby(species_col)[phenology_col].unique()
    for species, phenologies in species_groups.items():
        if pd.notna(species):
            for phen in phenologies:
                if pd.notna(phen):
                    species_to_phenology[species].add(phen)
    
    logging.info(f"Found {len(species_to_phenology)} unique species in {source_name} dataset")
    return species_to_phenology


def main():
    parser = argparse.ArgumentParser(
        description="Quick check of species to phenology mappings in the datasets."
    )
    parser.add_argument(
        "--insitu",
        default="data/species/processed/france_species_with_ecoregions.parquet",
        help="Path to the in-situ dataset."
    )
    parser.add_argument(
        "--bdforet",
        default="data/species/processed/bdforet_with_ecoregions.parquet",
        help="Path to BD Forêt file."
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the report as a CSV file."
    )
    parser.add_argument(
        "--conflicts-only",
        action="store_true",
        help="Show only species with conflicting phenology mappings."
    )
    
    args = parser.parse_args()
    setup_logging(args.loglevel)
    
    # Check if files exist
    insitu_path = Path(args.insitu)
    bdforet_path = Path(args.bdforet)
    
    if not insitu_path.exists():
        logging.error(f"In-situ dataset not found: {insitu_path}")
        sys.exit(1)
    
    if not bdforet_path.exists():
        logging.error(f"BD Forêt dataset not found: {bdforet_path}")
        sys.exit(1)
    
    # Load datasets
    insitu_data = load_dataset(insitu_path, "In-situ")
    bdforet_data = load_dataset(bdforet_path, "BD Forêt")
    
    if insitu_data is None or bdforet_data is None:
        logging.error("Failed to load one or both datasets. Exiting.")
        sys.exit(1)
    
    # Get mappings
    insitu_mapping = get_species_phenology_mapping(insitu_data, "In-situ")
    bdforet_mapping = get_species_phenology_mapping(bdforet_data, "BD Forêt")
    
    # Prepare report data
    report_data = []
    
    # Process all species from both datasets
    all_species = sorted(set(insitu_mapping.keys()) | set(bdforet_mapping.keys()))
    logging.info(f"Processing {len(all_species)} unique species across both datasets")
    
    for species in all_species:
        insitu_phens = sorted(insitu_mapping.get(species, set()))
        bdforet_phens = sorted(bdforet_mapping.get(species, set()))
        
        # Determine if there's a conflict
        in_insitu = species in insitu_mapping
        in_bdforet = species in bdforet_mapping
        has_conflict = in_insitu and in_bdforet and insitu_phens != bdforet_phens
        
        # Skip if we only want conflicts and this isn't one
        if args.conflicts_only and not has_conflict:
            continue
        
        # Add to report data
        report_data.append({
            "species": species,
            "in_insitu": in_insitu,
            "in_bdforet": in_bdforet,
            "insitu_phenology": ", ".join(insitu_phens) if insitu_phens else "",
            "bdforet_phenology": ", ".join(bdforet_phens) if bdforet_phens else "",
            "has_conflict": has_conflict
        })
    
    # Create DataFrame from report data
    report_df = pd.DataFrame(report_data)
    
    # Print summary
    total_species = len(all_species)
    shared_species = len(set(insitu_mapping.keys()) & set(bdforet_mapping.keys()))
    conflicts = sum(1 for row in report_data if row["has_conflict"])
    
    print(f"\n{'=' * 80}")
    print(f"Species to Phenology Mapping Analysis")
    print(f"{'=' * 80}")
    print(f"Total unique species across both datasets: {total_species}")
    print(f"Species in in-situ dataset only: {len(insitu_mapping) - shared_species}")
    print(f"Species in BD Forêt dataset only: {len(bdforet_mapping) - shared_species}")
    print(f"Species in both datasets: {shared_species}")
    print(f"Species with conflicting phenology mappings: {conflicts}")
    
    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)
        print(f"\nReport saved to {output_path}")
    
    # Print conflicts in console
    if conflicts > 0:
        print(f"\n{'=' * 80}")
        print(f"Conflicting Species to Phenology Mappings")
        print(f"{'=' * 80}")
        
        conflict_rows = [row for row in report_data if row["has_conflict"]]
        for i, row in enumerate(conflict_rows, 1):
            print(f"{i}. {row['species']}:")
            print(f"   - In-situ: {row['insitu_phenology']}")
            print(f"   - BD Forêt: {row['bdforet_phenology']}")
            print()
    
    logging.info("Analysis complete.")


if __name__ == "__main__":
    main() 