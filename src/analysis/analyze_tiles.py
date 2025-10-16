#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze tile counts and polygons per ecoregion from the dataset parquet files.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

from src.utils import apply_science_style

apply_science_style()

# Add the src directory to the path so we can import from src modules
sys.path.append(str(Path(__file__).parent.parent.parent))

def analyze_tiles():
    """Analyze the selected tiles and polygons per ecoregion."""
    print("Analyzing tile and polygon counts per ecoregion...\n")
    
    # Paths to data files
    original_tiles_file = "data/species/val_train_tiles_2_5_km.parquet"
    tiles_file = "results/datasets/tiles_2_5_km_final.parquet"
    final_dataset_file = "results/datasets/final_dataset.parquet"
    intermediate_dir = "results/datasets/intermediate"
    
    # Ensure results directory exists for plot saving
    os.makedirs("results", exist_ok=True)
    
    # Load the original tiles dataset
    try:
        original_tiles_gdf = gpd.read_parquet(original_tiles_file)
        print(f"Loaded {len(original_tiles_gdf)} tiles from original dataset {original_tiles_file}")
        print(f"Original tiles columns: {original_tiles_gdf.columns.tolist()}")
        
        if 'NomSER' in original_tiles_gdf.columns:
            # Count original tiles per ecoregion
            orig_tiles_per_region = original_tiles_gdf['NomSER'].value_counts()
            print("\nOriginal tiles per ecoregion:")
            for region, count in orig_tiles_per_region.items():
                print(f"  {region}: {count} tiles")
    except Exception as e:
        print(f"Error loading original tiles file: {e}")
    
    # Load the selected tiles dataset (if it exists)
    try:
        tiles_gdf = gpd.read_parquet(tiles_file)
        print(f"\nLoaded {len(tiles_gdf)} tiles from final selection {tiles_file}")
        print(f"Selected tiles columns: {tiles_gdf.columns.tolist()}")
        
        if 'NomSER' in tiles_gdf.columns:
            # Count tiles per ecoregion
            tiles_per_region = tiles_gdf['NomSER'].value_counts()
            print("\nSelected tiles per ecoregion:")
            for region, count in tiles_per_region.items():
                print(f"  {region}: {count} tiles")
            
            # Calculate the selection percentage
            total_selected = len(tiles_gdf)
            total_original = len(original_tiles_gdf) if 'original_tiles_gdf' in locals() else 0
            if total_original > 0:
                selection_percentage = (total_selected / total_original) * 100
                print(f"\nOnly {selection_percentage:.2f}% of original tiles were selected ({total_selected} out of {total_original})")
            
            # Plot the distribution
            plt.figure(figsize=(12, 6))
            sns.barplot(x=tiles_per_region.index, y=tiles_per_region.values)
            plt.title('Number of Selected Tiles per Ecoregion')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('results/selected_tiles_per_ecoregion.png')
            print(f"\nPlot saved to results/selected_tiles_per_ecoregion.png")
            
            # Compare original vs selected if we have the original data
            if 'orig_tiles_per_region' in locals():
                # Create a DataFrame for comparison
                comparison_df = pd.DataFrame({
                    'Original': orig_tiles_per_region,
                    'Selected': tiles_per_region
                }).fillna(0)
                
                # Calculate selection percentage
                comparison_df['Selection %'] = (comparison_df['Selected'] / comparison_df['Original'] * 100).round(2)
                
                # Plot comparison
                plt.figure(figsize=(14, 8))
                comparison_df[['Original', 'Selected']].plot(kind='bar', figsize=(14, 8))
                plt.title('Comparison of Original vs Selected Tiles per Ecoregion')
                plt.ylabel('Number of Tiles')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('results/tiles_comparison.png')
                print(f"Comparison plot saved to results/tiles_comparison.png")
                
                print("\nSelection percentage per ecoregion:")
                for region, row in comparison_df.iterrows():
                    print(f"  {region}: {row['Selection %']}% ({int(row['Selected'])} out of {int(row['Original'])})")
        else:
            print("Warning: No 'NomSER' column found in tiles dataset")
    except Exception as e:
        print(f"Error loading tiles file: {e}")
        
    # Load the final dataset (if it exists)
    try:
        final_gdf = gpd.read_parquet(final_dataset_file)
        print(f"\nLoaded {len(final_gdf)} polygons from final dataset")
        print(f"Final dataset columns: {final_gdf.columns.tolist()}")
        
        if 'NomSER' in final_gdf.columns:
            # Count polygons per ecoregion
            polygons_per_region = final_gdf['NomSER'].value_counts()
            print("\nPolygons per ecoregion:")
            for region, count in polygons_per_region.items():
                print(f"  {region}: {count} polygons")
        else:
            print("Warning: No 'NomSER' column found in final dataset")
            
        # Count tile_ids if that column exists
        if 'tile_id' in final_gdf.columns:
            unique_tiles = final_gdf['tile_id'].nunique()
            print(f"\nNumber of unique tile IDs in final dataset: {unique_tiles}")
            
            # If we also have NomSER, count tiles per region
            if 'NomSER' in final_gdf.columns:
                tiles_per_region = final_gdf.groupby('NomSER')['tile_id'].nunique()
                print("\nUnique tiles per ecoregion in final dataset:")
                for region, count in tiles_per_region.items():
                    print(f"  {region}: {count} unique tiles")
    except Exception as e:
        print(f"Error loading final dataset: {e}")
    
    # Analyze intermediate files
    print("\nAnalyzing intermediate region files:")
    total_polys = 0
    for region_file in os.listdir(intermediate_dir):
        if region_file.endswith('_results.parquet'):
            region_name = region_file.replace('_results.parquet', '')
            file_path = os.path.join(intermediate_dir, region_file)
            
            try:
                region_gdf = gpd.read_parquet(file_path)
                num_polys = len(region_gdf)
                total_polys += num_polys
                
                # Count unique tiles if tile_id exists
                if 'tile_id' in region_gdf.columns:
                    unique_tiles = region_gdf['tile_id'].nunique()
                    print(f"  {region_name}: {num_polys} polygons from {unique_tiles} unique tiles")
                else:
                    print(f"  {region_name}: {num_polys} polygons")
            except Exception as e:
                print(f"  Error loading {region_file}: {e}")
    
    print(f"\nTotal polygons in intermediate files: {total_polys}")
    
    # Compare with the numbers reported in the logs
    print("\nExpected counts from logs:")
    expected_counts = {
        "Alps": 20,
        "Central Massif": 15,
        "Corsica": 4,
        "Greater Semi-Continental East": 1,
        "Mediterranean": 37,
        "Oceanic Southwest": 28,
        "Semi-Oceanic North Center": 1
    }
    for region, count in expected_counts.items():
        print(f"  {region}: {count} tiles")
    print(f"Total expected tiles: {sum(expected_counts.values())}")
    
if __name__ == "__main__":
    analyze_tiles() 
