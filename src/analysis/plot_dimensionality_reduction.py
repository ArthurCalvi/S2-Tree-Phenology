import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from pathlib import Path
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math

# Add src directory to sys.path to allow importing from src.utils
try:
    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
except NameError:
    # Fallback if __file__ is not defined
    src_path = Path('./src').resolve()
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

try:
    from utils import unscale_feature, transform_circular_features
except ImportError as e:
    print(f"Error importing utils: {e}", file=sys.stderr)
    print(f"Attempted to add {src_path} to sys.path.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}
DEFAULT_OUTPUT_DIR = 'results/analysis'
N_COMPONENTS = 2
DEFAULT_PLOT_SAMPLE_SIZE = 50000 # New constant for default sample size

# Map feature suffixes to unscaling types (mirrors plot_pairplot_matrix.py)
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

def plot_pca_results(pca_df, index, output_dir, plot_sample_size):
    """Plot PCA results colored by phenology and eco-region, using sampling if needed."""

    if pca_df.empty:
        logger.warning(f"Received empty DataFrame for index '{index}'. Cannot plot PCA results.")
        return
        
    n_points = len(pca_df)
    if n_points > plot_sample_size:
        logger.info(f"Subsampling {plot_sample_size:,} points from {n_points:,} for plotting {index} PCA.")
        plot_df = pca_df.sample(n=plot_sample_size, random_state=42)
    else:
        logger.info(f"Plotting all {n_points:,} points for {index} PCA (<= sample size)." )
        plot_df = pca_df

    unique_eco_regions = sorted(plot_df['eco_region'].unique()) # Use plot_df here
    eco_palette = sns.color_palette("husl", len(unique_eco_regions))
    pheno_palette = {"Deciduous": "tab:blue", "Evergreen": "tab:orange"}
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    fig.suptitle(f'PCA of {index.upper()} Features ({N_COMPONENTS} Components) - Sampled {len(plot_df):,} points', fontsize=16, y=1.02)
    
    # Plot 1: Colored by Phenology
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='phenology_label', 
        data=plot_df, 
        ax=axes[0], 
        palette=pheno_palette,
        alpha=0.7, s=10, # Slightly smaller points
        rasterized=True # Rasterize points for faster rendering/saving
    )
    axes[0].set_title('Colored by Phenology')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend(title='Phenology')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Colored by Eco-region
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='eco_region', 
        data=plot_df, 
        ax=axes[1], 
        palette=eco_palette,
        alpha=0.7, s=10, # Slightly smaller points
        rasterized=True # Rasterize points
    )
    axes[1].set_title('Colored by Eco-region')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('') # Y-axis label is shared
    axes[1].legend(title='Eco-region', bbox_to_anchor=(1.03, 1), loc='upper left', fontsize=8, title_fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 0.93, 0.98]) 

    # Save the figure
    output_filename = os.path.join(output_dir, f'pca_distribution_{index}_sampled.png') # Indicate sampling in filename
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    logger.info(f"Saved PCA plot for {index} to {output_filename}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Perform PCA and plot results for each spectral index.')
    parser.add_argument('--dataset_path', type=str, default=DEFAULT_DATASET_PATH,
                        help=f'Path to the input dataset parquet file (default: {DEFAULT_DATASET_PATH}).')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save the output plots (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--plot_sample_size', type=int, default=DEFAULT_PLOT_SAMPLE_SIZE, # New argument
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
            df = df.head(n_test_samples).copy() # Use .copy() to avoid SettingWithCopyWarning later
        else:
            logger.warning(f"Dataset has less than {n_test_samples:,} samples, using all available data.")
            df = df.copy() # Still use .copy()
    else:
        df = df.copy() # Use .copy() even if not testing

    # Check required columns
    required_base_cols = ['phenology', 'eco_region']
    if not all(col in df.columns for col in required_base_cols):
        missing_req = [col for col in required_base_cols if col not in df.columns]
        logger.error(f"Dataset missing required columns: {missing_req}. Found: {list(df.columns)}")
        sys.exit(1)

    # --- Unscale Features (mirroring plot_pairplot_matrix.py) ---
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
                skipped_cols.append(col_name)

    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    if skipped_cols:
        unique_skipped = sorted(list(set(skipped_cols)))
        # logger.debug(f"Skipped/Not Found {len(unique_skipped)} columns during unscaling: {unique_skipped[:5]}...")

    # Transform circular features
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df = transform_circular_features(df, INDICES)
    df['phenology_label'] = df['phenology'].map(PHENOLOGY_MAPPING)
    logger.info("Circular feature transformation complete.")

    # Process each index with tqdm progress bar
    logger.info(f"Starting PCA analysis for indices: {INDICES}")
    for index in tqdm(INDICES, desc="Processing Indices"):
        logger.info(f"--- Processing index: {index.upper()} ---")
        
        # Get features for this index
        index_features = get_index_features(index)
        logger.debug(f"Expected features for {index}: {index_features}")
        
        # Check if features exist in the DataFrame
        available_features = [f for f in index_features if f in df.columns]
        logger.debug(f"Available features for {index}: {available_features}")
        if not available_features:
            logger.warning(f"No features found for index '{index}'. Skipping.")
            continue
        if len(available_features) < len(index_features):
            missing = sorted(list(set(index_features) - set(available_features)))
            logger.warning(f"Missing some expected features for index '{index}': {missing}")

        # Prepare data: select features and drop rows with NaNs in these features
        logger.debug(f"Selecting feature data for {index}...")
        feature_data = df[available_features].copy()
        original_count = len(feature_data)
        feature_data.dropna(inplace=True)
        nan_dropped = original_count - len(feature_data)
        logger.debug(f"Original sample count for {index}: {original_count:,}, After NaN drop: {len(feature_data):,}")
        if nan_dropped > 0:
             logger.warning(f"Dropped {nan_dropped:,} rows with NaN values in '{index}' features.")
        
        if len(feature_data) < N_COMPONENTS:
            logger.warning(f"Insufficient samples ({len(feature_data)}) for PCA on index '{index}' after NaN removal. Skipping.")
            continue
            
        # Get corresponding labels for the non-NaN rows
        logger.debug(f"Extracting labels for {index}...")
        labels = df.loc[feature_data.index, ['phenology_label', 'eco_region']]
        
        # Scale features
        logger.info(f"Scaling {len(available_features)} features for {index}...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        logger.debug(f"Scaling complete for {index}.")
        
        # Apply PCA
        logger.info(f"Applying PCA to {len(scaled_features):,} samples for {index}...")
        pca = PCA(n_components=N_COMPONENTS)
        principal_components = pca.fit_transform(scaled_features)
        logger.debug(f"PCA fitting complete for {index}.")
        
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"Explained variance by component: {explained_variance}")
        logger.info(f"Total explained variance ({N_COMPONENTS} components): {explained_variance.sum():.4f}")
        
        # Create DataFrame for plotting
        logger.debug(f"Creating PCA results DataFrame for {index}...")
        pca_df = pd.DataFrame(
            data=principal_components, 
            columns=[f'PC{i+1}' for i in range(N_COMPONENTS)],
            index=feature_data.index # Keep original index to merge labels
        )
        pca_df = pd.concat([pca_df, labels], axis=1)
        logger.debug(f"PCA DataFrame created for {index}.")

        # Plot results
        logger.info(f"Plotting PCA results for {index}...")
        plot_pca_results(pca_df, index, args.output_dir, args.plot_sample_size)
        logger.debug(f"Plotting complete for {index}.")

    logger.info("Dimensionality reduction script finished successfully.")

if __name__ == "__main__":
    main() 