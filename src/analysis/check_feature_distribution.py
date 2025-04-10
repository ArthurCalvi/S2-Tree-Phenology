#!/usr/bin/env python3
"""
check_feature_distribution.py
-----------------------------
Loads the scaled feature dataset, applies unscaling to convert features to 
physical ranges, applies circular transformations to phase, and then analyzes 
the distribution and range of the final features using descriptive statistics 
and box plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from pathlib import Path
import logging
from tqdm import tqdm
import math

# --- Path Setup ---
try:
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
except NameError:
    # Fallback if __file__ is not defined (e.g., in interactive environment)
    src_path = Path('./src').resolve()
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
        print(f"Fallback: Added {src_path} to sys.path")

# --- Imports from project ---
try:
    from utils import unscale_feature, transform_circular_features
    from constants import AVAILABLE_INDICES as INDICES # Use indices from constants
except ImportError as e:
    print(f"Error importing from src.utils or src.constants: {e}", file=sys.stderr)
    print(f"Attempted to add {src_path} to sys.path.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'
DEFAULT_OUTPUT_DIR = 'results/analysis/feature_distribution' # Specific subdirectory
FEATURE_TYPES_TO_UNSCALE = {
    'amplitude_h1': 'amplitude',
    'amplitude_h2': 'amplitude',
    'phase_h1': 'phase',
    'phase_h2': 'phase',
    'offset': 'offset',
    'var_residual': 'variance' # Assuming column name is 'var_residual'
}

# Helper to get final feature names after transformations
def get_final_feature_names(index):
    """Generate final feature names for an index after unscaling and circular transform."""
    return [
        f'{index}_amplitude_h1', f'{index}_amplitude_h2',
        f'{index}_phase_h1_cos', f'{index}_phase_h1_sin', # Note: cos/sin replace phase
        f'{index}_phase_h2_cos', f'{index}_phase_h2_sin',
        f'{index}_offset', f'{index}_var_residual'
    ]

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Check distribution and range of unscaled features.')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH,
                        help=f'Path to the input dataset parquet file (default: {DEFAULT_DATASET_PATH}).')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save the output stats and plots (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with a small subset of data (first 5000 rows).')
    parser.add_argument('--skip_plots', action='store_true',
                        help='Skip generating box plots (only compute stats).')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {args.output_dir}")

    # --- Load Data ---
    logger.info(f"Loading dataset from {args.dataset_path}...")
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset file not found: {args.dataset_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(args.dataset_path)
        logger.info(f"Dataset loaded: {len(df):,} samples, columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # --- Test Mode ---
    if args.test:
        n_test_samples = 5000
        logger.info(f"Running in TEST MODE: using first {n_test_samples:,} samples.")
        if len(df) > n_test_samples:
            df = df.head(n_test_samples).copy()
        else:
            logger.warning(f"Dataset has less than {n_test_samples:,} samples, using all available data.")
            df = df.copy() # Ensure it's a copy even if not slicing
    else:
        # Ensure we work with a copy to avoid modifying the original DataFrame in memory elsewhere
        df = df.copy() 

    # --- Unscale Features ---
    logger.info("Unscaling features to physical ranges...")
    unscaled_count = 0
    skipped_cols = []
    for index in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index}_{ftype_suffix}"
            if col_name in df.columns:
                try:
                    # Overwrite column with unscaled version
                    df[col_name] = unscale_feature(
                        df[col_name],
                        feature_type=feature_type,
                        index_name=index # Required for amplitude/offset
                    )
                    unscaled_count += 1
                    # Optional: Log range after unscaling
                    # logger.debug(f"Unscaled {col_name}: min={df[col_name].min():.2f}, max={df[col_name].max():.2f}")
                except Exception as e:
                    logger.error(f"Error unscaling column {col_name}: {e}")
                    skipped_cols.append(col_name)
            else:
                # logger.warning(f"Column {col_name} not found for unscaling, skipping.")
                skipped_cols.append(col_name)
    
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    if skipped_cols:
         # Only log unique skipped columns for brevity if many are missing
         unique_skipped = sorted(list(set(skipped_cols)))
         logger.warning(f"Skipped/Not Found {len(unique_skipped)} columns during unscaling: {unique_skipped[:10]}...") # Log first 10

    # --- Transform Circular Features ---
    # This operates on the phase columns which should now be in radians [0, 2*pi]
    logger.info("Applying circular transformation to unscaled phase features...")
    try:
        # transform_circular_features replaces phase_h1/h2 with phase_h1_cos/sin, phase_h2_cos/sin
        df = transform_circular_features(df, INDICES)
        logger.info("Circular transformation complete.")
    except Exception as e:
        logger.error(f"Error during circular transformation: {e}")
        # Decide if script should exit or continue without transformation
        logger.warning("Continuing analysis without circular transformation applied.")


    # --- Analyze Distributions ---
    logger.info("Calculating descriptive statistics for final features...")
    all_final_feature_cols = []
    for index in INDICES:
        all_final_feature_cols.extend(get_final_feature_names(index))
    
    # Keep only features that actually exist in the dataframe after transformations
    existing_feature_cols = [col for col in all_final_feature_cols if col in df.columns]
    
    if not existing_feature_cols:
        logger.error("No final feature columns found in the DataFrame after transformations. Exiting.")
        sys.exit(1)
        
    logger.debug(f"Columns for analysis: {existing_feature_cols}")
    
    # Calculate statistics
    stats = df[existing_feature_cols].describe().transpose()
    
    # Save statistics
    stats_filename = os.path.join(args.output_dir, 'feature_statistics.csv')
    try:
        stats.to_csv(stats_filename)
        logger.info(f"Descriptive statistics saved to: {stats_filename}")
        print("\nFeature Statistics Summary:")
        print(stats)
    except Exception as e:
        logger.error(f"Failed to save statistics: {e}")

    # --- Generate Box Plots (Optional) ---
    if not args.skip_plots:
        logger.info("Generating box plots for feature distributions...")
        plot_errors = 0
        
        # Determine grid size for subplots (aim for roughly square)
        num_indices = len(INDICES)
        ncols = math.ceil(math.sqrt(num_indices))
        nrows = math.ceil(num_indices / ncols)
        
        fig_all, axes_all = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False)
        axes_flat_all = axes_all.flatten()
        
        for i, index in enumerate(tqdm(INDICES, desc="Generating Box Plots")):
            index_final_features = [f for f in get_final_feature_names(index) if f in df.columns]
            
            if not index_final_features:
                logger.warning(f"No final features found for index '{index}' to plot.")
                if i < len(axes_flat_all): axes_flat_all[i].set_visible(False) # Hide unused subplot
                continue

            # Prepare data for plotting (select columns for this index)
            plot_data = df[index_final_features]
            
            # Create individual plot per index
            fig_index, ax_index = plt.subplots(figsize=(12, 6))
            try:
                sns.boxplot(data=plot_data, ax=ax_index, orient='h') # Horizontal looks good for many features
                ax_index.set_title(f'Feature Distribution for {index.upper()}')
                ax_index.tick_params(axis='x', rotation=45)
                fig_index.tight_layout()
                
                # Save individual plot
                plot_filename_index = os.path.join(args.output_dir, f'boxplot_{index}.png')
                fig_index.savefig(plot_filename_index, dpi=100)
                plt.close(fig_index) # Close figure to free memory

                # Add to combined plot
                if i < len(axes_flat_all):
                    sns.boxplot(data=plot_data, ax=axes_flat_all[i], orient='h')
                    axes_flat_all[i].set_title(f'{index.upper()}')
                    axes_flat_all[i].tick_params(axis='x', labelsize=8, rotation=30) # Adjust label size/rotation
                    axes_flat_all[i].tick_params(axis='y', labelsize=8)
                
            except Exception as e:
                logger.error(f"Error generating box plot for index '{index}': {e}")
                plot_errors += 1
                plt.close(fig_index) # Ensure figure is closed on error
                if i < len(axes_flat_all): axes_flat_all[i].set_visible(False) # Hide subplot on error

        # Hide any remaining unused subplots in the combined figure
        for j in range(i + 1, len(axes_flat_all)):
            axes_flat_all[j].set_visible(False)

        # Save the combined plot
        fig_all.suptitle('Box Plots of Feature Distributions by Index', fontsize=16)
        fig_all.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
        plot_filename_all = os.path.join(args.output_dir, 'boxplots_all_indices.png')
        try:
            fig_all.savefig(plot_filename_all, dpi=150)
            logger.info(f"Combined box plot saved to: {plot_filename_all}")
        except Exception as e:
            logger.error(f"Failed to save combined box plot: {e}")
            plot_errors += 1
        plt.close(fig_all)

        if plot_errors > 0:
            logger.warning(f"Encountered {plot_errors} errors during plot generation.")
        else:
            logger.info("Box plot generation complete.")
    else:
        logger.info("Skipping box plot generation as requested.")

    logger.info("Feature distribution check script finished.")

if __name__ == "__main__":
    main()