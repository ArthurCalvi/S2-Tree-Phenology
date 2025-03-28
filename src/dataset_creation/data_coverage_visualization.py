#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to visualize data coverage of phenology, genus, and species in the training dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import argparse
import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_categorical_mappings(mapping_file):
    """
    Load categorical mappings from JSON file
    
    Args:
        mapping_file: Path to the JSON file containing mappings
        
    Returns:
        Dictionary containing mappings for phenology, genus, species, and source
    """
    try:
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
        
        # Invert the mappings to go from ID to name
        inverted_mappings = {}
        for category, mapping in mappings.items():
            inverted_mappings[category] = {v: k for k, v in mapping.items()}
        
        logger.info(f"Loaded mappings for: {', '.join(mappings.keys())}")
        return mappings, inverted_mappings
    except Exception as e:
        logger.error(f"Error loading mappings from {mapping_file}: {e}")
        # Provide default phenology mapping as fallback
        default_mappings = {
            "phenology": {"deciduous": 1, "evergreen": 2}
        }
        default_inverted = {
            "phenology": {1: "deciduous", 2: "evergreen"}
        }
        return default_mappings, default_inverted

def create_visualizations(df, mappings, inverted_mappings, output_dir, save_plots=True):
    """
    Create visualizations for the data coverage of phenology, genus, and species.
    
    Args:
        df: DataFrame containing the training dataset
        mappings: Dictionary of category to name->id mappings
        inverted_mappings: Dictionary of category to id->name mappings
        output_dir: Directory to save the visualizations
        save_plots: Whether to save the plots or show them
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Map integer values to phenology, genus, and species names
    if "phenology" in inverted_mappings:
        df['phenology_name'] = df['phenology'].map(inverted_mappings["phenology"])
    
    if "genus" in inverted_mappings:
        df['genus_name'] = df['genus'].map(inverted_mappings["genus"])
    
    if "species" in inverted_mappings:
        df['species_name'] = df['species'].map(inverted_mappings["species"])
    
    # Count number of unique genus and species
    num_genus = df['genus'].nunique()
    num_species = df['species'].nunique()
    
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Number of unique genus values: {num_genus}")
    logger.info(f"Number of unique species values: {num_species}")
    
    # Create a figure for all visualizations
    plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=plt.gcf())
    
    # 1. Phenology distribution (pie chart)
    ax1 = plt.subplot(gs[0, 0])
    phenology_counts = df['phenology_name'].value_counts()
    phenology_counts.plot.pie(
        autopct='%1.1f%%', 
        ax=ax1, 
        shadow=True, 
        startangle=90,
        colors=sns.color_palette("Set2"),
        explode=[0.05] * len(phenology_counts),
        textprops={'fontsize': 12}
    )
    ax1.set_title('Phenology Distribution', fontsize=14)
    ax1.set_ylabel('')
    
    # 2. Top 10 genus distribution (bar chart)
    ax2 = plt.subplot(gs[0, 1])
    if 'genus_name' in df.columns:
        genus_counts = df['genus_name'].value_counts().nlargest(10)
        genus_counts.plot.bar(ax=ax2, color=sns.color_palette("Set3"))
        ax2.set_title('Top 10 Genera by Count', fontsize=14)
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Genus')
        plt.xticks(rotation=45, ha='right')
    else:
        genus_counts = df['genus'].value_counts().nlargest(10)
        genus_counts.plot.bar(ax=ax2, color=sns.color_palette("Set3"))
        ax2.set_title('Top 10 Genera by Count', fontsize=14)
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Genus ID')
        plt.xticks(rotation=45)
    
    # 3. Genus-phenology relationship (heatmap)
    ax3 = plt.subplot(gs[1, :])
    # Create a cross-tabulation of genus and phenology
    if 'genus_name' in df.columns and 'phenology_name' in df.columns:
        genus_pheno_cross = pd.crosstab(df['genus_name'], df['phenology_name'])
    else:
        genus_pheno_cross = pd.crosstab(df['genus'], df['phenology_name'])
    
    # Calculate proportions for each genus
    genus_pheno_norm = genus_pheno_cross.div(genus_pheno_cross.sum(axis=1), axis=0)
    sns.heatmap(genus_pheno_norm, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax3)
    ax3.set_title('Proportion of Phenology Types by Genus', fontsize=14)
    ax3.set_xlabel('Phenology')
    ax3.set_ylabel('Genus')
    plt.yticks(rotation=0)
    
    # 4. Top 10 species distribution (bar chart)
    ax4 = plt.subplot(gs[2, 0])
    if 'species_name' in df.columns:
        species_counts = df['species_name'].value_counts().nlargest(10)
        species_counts.plot.bar(ax=ax4, color=sns.color_palette("Set1"))
        ax4.set_title('Top 10 Species by Count', fontsize=14)
        ax4.set_ylabel('Count')
        ax4.set_xlabel('Species')
        plt.xticks(rotation=45, ha='right')
    else:
        species_counts = df['species'].value_counts().nlargest(10)
        species_counts.plot.bar(ax=ax4, color=sns.color_palette("Set1"))
        ax4.set_title('Top 10 Species by Count', fontsize=14)
        ax4.set_ylabel('Count')
        ax4.set_xlabel('Species ID')
        plt.xticks(rotation=45)
    
    # 5. Species-phenology relationship (stacked bar chart for top 10 species)
    ax5 = plt.subplot(gs[2, 1])
    # Get the top 10 species by count
    if 'species_name' in df.columns:
        top_species = df['species_name'].value_counts().nlargest(10).index
        top_species_df = df[df['species_name'].isin(top_species)]
        species_pheno_cross = pd.crosstab(top_species_df['species_name'], top_species_df['phenology_name'])
    else:
        top_species = df['species'].value_counts().nlargest(10).index
        top_species_df = df[df['species'].isin(top_species)]
        species_pheno_cross = pd.crosstab(top_species_df['species'], top_species_df['phenology_name'])
    
    species_pheno_cross.plot.bar(ax=ax5, stacked=True, color=sns.color_palette("Set2"))
    ax5.set_title('Phenology Distribution for Top 10 Species', fontsize=14)
    ax5.set_ylabel('Count')
    ax5.set_xlabel('Species')
    plt.xticks(rotation=45, ha='right')
    ax5.legend(title='Phenology')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Save or show the figure
    if save_plots:
        plot_path = os.path.join(output_dir, 'data_coverage.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {plot_path}")
    else:
        plt.show()
        
    # Generate a detailed report text file
    report_path = os.path.join(output_dir, 'data_coverage_report.txt')
    with open(report_path, 'w') as f:
        f.write("DATA COVERAGE REPORT\n")
        f.write("===================\n\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Number of unique genus values: {num_genus}\n")
        f.write(f"Number of unique species values: {num_species}\n\n")
        
        # Phenology distribution
        f.write("Phenology Distribution:\n")
        f.write("-----------------------\n")
        phenology_mapping = inverted_mappings.get("phenology", {})
        for phen_id, count in df['phenology'].value_counts().items():
            phen_name = phenology_mapping.get(phen_id, f"Unknown ({phen_id})")
            percentage = (count / len(df)) * 100
            f.write(f"{phen_name} (ID: {phen_id}): {count:,} rows ({percentage:.2f}%)\n")
        f.write("\n")
        
        # Genus distribution
        f.write("Top 10 Genus Distribution:\n")
        f.write("-------------------------\n")
        genus_mapping = inverted_mappings.get("genus", {})
        for genus_id, count in df['genus'].value_counts().nlargest(10).items():
            genus_name = genus_mapping.get(genus_id, f"Unknown")
            percentage = (count / len(df)) * 100
            f.write(f"{genus_name} (ID: {genus_id}): {count:,} rows ({percentage:.2f}%)\n")
        f.write("\n")
        
        # Species distribution
        f.write("Top 10 Species Distribution:\n")
        f.write("---------------------------\n")
        species_mapping = inverted_mappings.get("species", {})
        for species_id, count in df['species'].value_counts().nlargest(10).items():
            species_name = species_mapping.get(species_id, f"Unknown")
            percentage = (count / len(df)) * 100
            f.write(f"{species_name} (ID: {species_id}): {count:,} rows ({percentage:.2f}%)\n")
        f.write("\n")
        
        # Species-phenology relationship for top 10 species
        f.write("Phenology Distribution for Top 10 Species:\n")
        f.write("----------------------------------------\n")
        for species_id in df['species'].value_counts().nlargest(10).index:
            species_df = df[df['species'] == species_id]
            species_name = species_mapping.get(species_id, f"Unknown")
            f.write(f"{species_name} (ID: {species_id}):\n")
            
            for phen_id, count in species_df['phenology'].value_counts().items():
                phen_name = phenology_mapping.get(phen_id, f"Unknown")
                percentage = (count / len(species_df)) * 100
                f.write(f"  - {phen_name}: {count:,} rows ({percentage:.2f}%)\n")
            f.write("\n")
        
    logger.info(f"Report saved to {report_path}")
    
    # Return the path of the saved files for reference
    return {
        'plot': os.path.join(output_dir, 'data_coverage.png') if save_plots else None,
        'report': report_path
    }

def main():
    parser = argparse.ArgumentParser(
        description='Visualize data coverage of phenology, genus, and species in the training dataset.'
    )
    parser.add_argument(
        '--input_file', 
        type=str, 
        default='results/datasets/training_datasets_pixels.parquet',
        help='Path to the input parquet file'
    )
    parser.add_argument(
        '--mapping_file', 
        type=str, 
        default='data/training/training_tiles2023_w_corsica/training_tiles2023/categorical_mappings.json',
        help='Path to the JSON file containing categorical mappings'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='results/data_coverage',
        help='Directory to save the visualizations'
    )
    parser.add_argument(
        '--no_save', 
        action='store_true',
        help='Show plots instead of saving them'
    )
    parser.add_argument(
        '--sample', 
        type=int, 
        default=0,
        help='Number of rows to sample from the dataset (0 for all rows)'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute if needed
    root_dir = Path(__file__).parent.parent.parent
    
    input_file = args.input_file
    if not os.path.isabs(input_file):
        input_file = os.path.join(root_dir, input_file)
    
    mapping_file = args.mapping_file
    if not os.path.isabs(mapping_file):
        mapping_file = os.path.join(root_dir, mapping_file)
    
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(root_dir, output_dir)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        return
    
    # Check if mapping file exists
    if not os.path.exists(mapping_file):
        logger.warning(f"Mapping file {mapping_file} does not exist. Using default mappings.")
    
    # Load categorical mappings
    mappings, inverted_mappings = load_categorical_mappings(mapping_file)
    
    # Load the dataset
    logger.info(f"Loading dataset from {input_file}")
    df = pd.read_parquet(input_file)
    
    # Sample the dataset if requested
    if args.sample > 0:
        logger.info(f"Sampling {args.sample} rows from the dataset")
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
    
    # Create the visualizations
    create_visualizations(df, mappings, inverted_mappings, output_dir, save_plots=not args.no_save)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 