import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils import apply_science_style
apply_science_style()
import seaborn as sns
import argparse
import os
import sys
from pathlib import Path
import logging
from tqdm import tqdm
import math # Needed if unscaling involves math constants, good practice
import json # To load mappings

# Add the *project root* directory to sys.path
try:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        print(f"Added project root {project_root} to sys.path") # Debug print
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive)
    project_root = Path('.').resolve() # Assume running from project root
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
        print(f"Added current dir {project_root} as project root to sys.path") # Debug print

# Set up logging *before* the try-except block that uses it
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Use explicit import relative to the project root
    from src.utils import unscale_feature, transform_circular_features
    logger.info("Successfully imported from src.utils") # Debug print
except ImportError as e:
    print(f"Error importing from src.utils: {e}", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

# Define constants
DEFAULT_DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}
DEFAULT_MAPPING_PATH = 'data/training/training_tiles2023_w_corsica/training_tiles2023/categorical_mappings.json' # Default path to mapping file
DEFAULT_OUTPUT_DIR = 'results/analysis/pairplots' # Specific subdirectory for pairplots
DEFAULT_PLOT_SAMPLE_SIZE = 140000 # Smaller default sample size for pairplots due to complexity
# Maximum unique categories to plot for genus/species to avoid overly complex legends/palettes
MAX_CATEGORIES_FOR_HUE = 50

# Map feature suffixes to unscaling types
FEATURE_TYPES_TO_UNSCALE = {
    'amplitude_h1': 'amplitude',
    'amplitude_h2': 'amplitude',
    'phase_h1': 'phase', # Will be unscaled to radians [0, 2pi]
    'phase_h2': 'phase',
    'offset': 'offset',
    'var_residual': 'variance'
}

def get_index_features(index):
    """Generate feature names for a specific index after circular transformation."""
    return [
        f'{index}_amplitude_h1',
        f'{index}_amplitude_h2',
        f'{index}_phase_h1_cos',
        f'{index}_phase_h1_sin',
        f'{index}_phase_h2_cos',
        f'{index}_phase_h2_sin',
        f'{index}_offset',
        f'{index}_var_residual'
    ]

def plot_pairplot(df_subset, index_features, index_name, hue_col, output_dir, plot_sample_size):
    """Generate and save a scatter plot matrix (pairplot) for a given index and hue."""

    if df_subset.empty:
        logger.warning(f"Received empty DataFrame for index '{index_name}', hue '{hue_col}'. Cannot plot pairplot.")
        return
    
    if hue_col not in df_subset.columns:
        logger.warning(f"Hue column '{hue_col}' not found in DataFrame for index '{index_name}'. Skipping pairplot.")
        return

    n_points = len(df_subset)
    if n_points > plot_sample_size:
        logger.info(f"Subsampling {plot_sample_size:,} points from {n_points:,} for plotting {index_name} pairplot (hue: {hue_col}).")
        plot_df = df_subset.sample(n=plot_sample_size, random_state=42)
    else:
        logger.info(f"Plotting all {n_points:,} points for {index_name} pairplot (hue: {hue_col}) (<= sample size).")
        plot_df = df_subset

    # Ensure all expected features are present before plotting
    plot_features = [f for f in index_features if f in plot_df.columns]
    if not plot_features:
        logger.warning(f"No features available for plotting pairplot for index '{index_name}'. Skipping.")
        return
    if len(plot_features) < len(index_features):
         missing_p = sorted(list(set(index_features) - set(plot_features)))
         logger.warning(f"Missing some features for pairplot '{index_name}': {missing_p}. Plotting with available features.")
         
    logger.info(f"Generating pairplot for {index_name.upper()} features, colored by {hue_col}...")
    
    # Determine appropriate palette
    if hue_col == 'phenology_label':
        palette = {"Deciduous": "tab:blue", "Evergreen": "tab:orange"}
    elif hue_col == 'eco_region':
        unique_eco_regions = sorted(plot_df['eco_region'].unique())
        # Use a diverse palette suitable for many categories
        palette = sns.color_palette("husl", len(unique_eco_regions))
    elif hue_col in ['genus_name', 'species_name']:
        unique_values = sorted(plot_df[hue_col].unique())
        n_unique = len(unique_values)
        if n_unique > MAX_CATEGORIES_FOR_HUE:
             logger.warning(f"Number of unique values for hue '{hue_col}' ({n_unique}) exceeds maximum ({MAX_CATEGORIES_FOR_HUE}). Plot might be cluttered. Consider filtering.")
             # Still plot, but use a large palette (e.g., tab20 repeated or husl)
             palette = sns.color_palette("husl", n_unique)
        elif n_unique > 20: # Use husl for 21-50 categories
            palette = sns.color_palette("husl", n_unique)
        elif n_unique > 10: # Use tab20 for 11-20 categories
            palette = sns.color_palette("tab20", n_unique)
        else: # Use tab10 for <= 10 categories
            palette = sns.color_palette("tab10", n_unique)
    else:
        palette = None # Default seaborn palette

    try:
        # Use specific variables for the pairplot
        pair_plot = sns.pairplot(
            plot_df,
            vars=plot_features, # Only plot the index-specific features
            hue=hue_col,
            palette=palette,
            plot_kws={'alpha': 0.6, 's': 5, 'rasterized': True}, # Smaller points, rasterized
            diag_kind='kde' # Use KDE for diagonal plots
        )
        pair_plot.fig.suptitle(f'Pairplot of {index_name.upper()} Features (Hue: {hue_col}) - Sampled {len(plot_df):,} points', y=1.02)

        # Set limits for cos/sin features
        cos_sin_suffixes = ('_cos', '_sin')
        for i, row_var in enumerate(pair_plot.y_vars):
            for j, col_var in enumerate(pair_plot.x_vars):
                ax = pair_plot.axes[i, j]
                if i == j: # Skip diagonal plots
                    continue
                # Set x-limit if the column variable is cos/sin
                if col_var.endswith(cos_sin_suffixes):
                    ax.set_xlim(-1.1, 1.1) # Use slightly wider limits for visual clarity
                # Set y-limit if the row variable is cos/sin
                if row_var.endswith(cos_sin_suffixes):
                    ax.set_ylim(-1.1, 1.1) # Use slightly wider limits for visual clarity

        # Save the figure
        output_filename = os.path.join(output_dir, f'pairplot_{index_name}_hue_{hue_col}_sampled.png')
        os.makedirs(output_dir, exist_ok=True)
        pair_plot.savefig(output_filename, dpi=150) # Lower DPI might be better for complex pairplots
        logger.info(f"Saved pairplot for {index_name} (hue: {hue_col}) to {output_filename}")
        plt.close(pair_plot.fig)
    except Exception as e:
        logger.error(f"Error generating pairplot for index '{index_name}', hue '{hue_col}': {e}")
        plt.close('all') # Close any potentially lingering figures


def main():
    parser = argparse.ArgumentParser(description='Generate scatter plot matrices for each spectral index.')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH,
                        help=f'Path to the input dataset parquet file (default: {DEFAULT_DATASET_PATH}).')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save the output plots (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--mapping_path', type=str, default=DEFAULT_MAPPING_PATH,
                        help=f'Path to the JSON file containing categorical mappings (default: {DEFAULT_MAPPING_PATH}).')
    parser.add_argument('--plot_sample_size', type=int, default=DEFAULT_PLOT_SAMPLE_SIZE,
                        help=f'Number of points to sample for plotting (default: {DEFAULT_PLOT_SAMPLE_SIZE:,}).')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with a small subset of data (first 5000 rows).')

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset file not found: {args.dataset_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(args.dataset_path)
        logger.info(f"Dataset loaded: {len(df):,} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Use subset if in test mode
    if args.test:
        n_test_samples = 5000
        logger.info(f"Running in TEST MODE: using first {n_test_samples:,} samples.")
        if len(df) > n_test_samples:
            df = df.head(n_test_samples).copy()
        else:
            logger.warning(f"Dataset has less than {n_test_samples:,} samples, using all available data.")
            df = df.copy()
    else:
        df = df.copy() # Use .copy() to avoid SettingWithCopyWarning

    # Load categorical mappings
    logger.info(f"Loading categorical mappings from {args.mapping_path}...")
    if not os.path.exists(args.mapping_path):
        logger.error(f"Mapping file not found: {args.mapping_path}")
        sys.exit(1)
    try:
        with open(args.mapping_path, 'r') as f:
            mappings = json.load(f)
        # Create reversed mappings (ID string -> Name) for genus and species
        # Ensure keys are strings as parquet stores numerical IDs as strings based on previous check
        genus_id_to_name = {str(v): k for k, v in mappings.get('genus', {}).items()}
        species_id_to_name = {str(v): k for k, v in mappings.get('species', {}).items()}
        logger.info(f"Loaded {len(genus_id_to_name)} genus and {len(species_id_to_name)} species mappings.")
    except Exception as e:
        logger.error(f"Failed to load or process mapping file: {e}")
        sys.exit(1)

    # Check required columns
    required_base_cols = ['phenology', 'eco_region', 'genus', 'species'] # Added genus and species
    if not all(col in df.columns for col in required_base_cols):
        missing_req = [col for col in required_base_cols if col not in df.columns]
        logger.error(f"Dataset missing required base columns: {missing_req}. Found: {list(df.columns)}")
        sys.exit(1)

    # --- Unscale Features --- 
    logger.info("Unscaling features to physical ranges...")
    unscaled_count = 0
    skipped_cols = []
    for index in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index}_{ftype_suffix}"
            if col_name in df.columns:
                try:
                    df[col_name] = unscale_feature(
                        df[col_name],
                        feature_type=feature_type,
                        index_name=index # Required for amplitude/offset
                    )
                    unscaled_count += 1
                except Exception as e:
                    logger.error(f"Error unscaling column {col_name}: {e}")
                    skipped_cols.append(col_name)
            else:
                # This is expected if not all features were generated/saved
                skipped_cols.append(col_name)

    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    if skipped_cols:
        unique_skipped = sorted(list(set(skipped_cols)))
        # Log only if columns were expected but failed, or for debugging
        # logger.debug(f"Skipped/Not Found {len(unique_skipped)} columns during unscaling: {unique_skipped[:5]}...") 

    # Transform circular features and create phenology label
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df = transform_circular_features(df, INDICES)
    df['phenology_label'] = df['phenology'].map(PHENOLOGY_MAPPING)
    logger.info("Circular transformation complete.")

    # Create name columns using mappings
    logger.info("Creating genus_name and species_name columns from mappings...")
    df['genus_name'] = df['genus'].astype(str).map(genus_id_to_name).fillna('Unknown')
    df['species_name'] = df['species'].astype(str).map(species_id_to_name).fillna('Unknown')
    logger.info("Name columns created.")

    # Process each index with tqdm progress bar
    logger.info(f"Starting pairplot generation for indices: {INDICES}")
    for index in tqdm(INDICES, desc="Processing Indices"):
        logger.info(f"--- Processing index: {index.upper()} ---")

        # Get features for this index
        index_features = get_index_features(index)
        logger.debug(f"Expected features for {index}: {index_features}")

        # Check if features exist in the DataFrame
        available_features = [f for f in index_features if f in df.columns]
        logger.debug(f"Available features for {index}: {available_features}")
        if not available_features:
            logger.warning(f"No features found for index '{index}'. Skipping pairplots.")
            continue
        if len(available_features) < len(index_features):
            missing = sorted(list(set(index_features) - set(available_features)))
            logger.warning(f"Missing some expected features for index '{index}': {missing}")
        
        # Include label columns needed for hue and filtering
        cols_to_select = available_features + ['phenology_label', 'eco_region', 'genus', 'species', 'genus_name', 'species_name']
        # Ensure unique columns
        cols_to_select = sorted(list(set(cols_to_select))) 
        
        # Prepare data: select features + labels and drop rows with NaNs *in the features*
        logger.debug(f"Selecting data for {index} (columns: {cols_to_select})...")
        index_data = df[cols_to_select].copy()
        original_count = len(index_data)
        
        # Drop rows with NaNs ONLY in the feature columns for this index
        index_data.dropna(subset=available_features, inplace=True)
        nan_dropped = original_count - len(index_data)
        logger.debug(f"Original sample count for {index}: {original_count:,}, After NaN drop in features: {len(index_data):,}")
        if nan_dropped > 0:
            logger.warning(f"Dropped {nan_dropped:,} rows with NaN values in '{index}' features.")

        if len(index_data) < 2: # Need at least 2 points for pairplot
            logger.warning(f"Insufficient samples ({len(index_data)}) for pairplot on index '{index}' after NaN removal. Skipping.")
            continue

        # Plot pairplots for phenology and eco-region hues
        plot_pairplot(index_data, available_features, index, 'phenology_label', args.output_dir, args.plot_sample_size)
        plot_pairplot(index_data, available_features, index, 'eco_region', args.output_dir, args.plot_sample_size)
        # Add plots for genus and species hues
        plot_pairplot(index_data, available_features, index, 'genus_name', args.output_dir, args.plot_sample_size)
        plot_pairplot(index_data, available_features, index, 'species_name', args.output_dir, args.plot_sample_size)
        logger.debug(f"Pairplot generation complete for {index}.")

    logger.info("Pairplot generation script finished successfully.")

if __name__ == "__main__":
    main() 