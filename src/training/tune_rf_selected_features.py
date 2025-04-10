import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
# Use HalvingGridSearchCV for efficient hyperparameter tuning
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import time
import os
import math
import logging
from tqdm import tqdm
import datetime
import argparse
import json
import joblib
import sys
from pathlib import Path
from tabulate import tabulate

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import utility functions
from src.utils import (
    unscale_feature,
    transform_circular_features,
    create_eco_balanced_folds_df,
    # Removed unused imports: compute_metrics, display_fold_distribution, format_confusion_matrix
)

# --- Configuration ---

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/selected_features_tuning.log'), # Changed log file name
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the path to the dataset (can be overridden by command line)
DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'

# Define the phenology mapping (optional, for potential future use in analysis)
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}

# Define the base indices needed for feature transformation
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']

# Map feature suffixes to unscaling types
FEATURE_TYPES_TO_UNSCALE = {
    'amplitude_h1': 'amplitude',
    'amplitude_h2': 'amplitude',
    'phase_h1': 'phase', # Will be unscaled to radians [0, 2pi]
    'phase_h2': 'phase',
    'offset': 'offset',
    'var_residual': 'variance'
}

# --- Helper Functions ---

def count_rf_parameters(rf_model):
    """
    Count the number of parameters in a RandomForest model.
    (Copied from train_rf_selected_features.py)
    """
    n_estimators = len(rf_model.estimators_)
    total_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
    n_classes = rf_model.n_classes_
    params_per_node = 2 + n_classes
    total_parameters = total_nodes * params_per_node
    return {
        'n_estimators': n_estimators,
        'n_nodes': total_nodes,
        'params_per_node': params_per_node,
        'total_parameters': total_parameters
    }

# --- Main Execution ---

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tune RandomForest hyperparameters using HalvingGridSearchCV with selected features and eco-balanced CV.')
    parser.add_argument('--features', '-f', type=str, required=True,
                        help='Comma-separated list of feature names to use for tuning.')
    parser.add_argument('--output_dir', '-o', type=str, default='results/tuning',
                        help='Directory to save the tuning results and best model (default: results/tuning).')
    parser.add_argument('--results_name', '-r', type=str, default='tuning_results.json',
                        help='Filename for the tuning results JSON output (default: tuning_results.json).')
    parser.add_argument('--best_model_name', '-m', type=str, default='best_phenology_model.joblib',
                        help='Filename for the saved best model (default: best_phenology_model.joblib).')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run in test mode with a small subset of data.')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of samples to use in test mode (default: 10000).')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation splits for eco-balancing (default: 5).')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help=f'Path to the dataset parquet file (default: {DATASET_PATH}).')
    parser.add_argument('--factor', type=int, default=3,
                        help='Factor for HalvingGridSearchCV (default: 3).')
    parser.add_argument('--min_resources', type=int, default=180000,
                        help="Minimum resources (samples) for the first iteration of HalvingGridSearchCV. "
                             "Set large enough so the final iteration uses most of the data. (default: 180000)"
                             " In test mode, this will be adjusted automatically.")


    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting hyperparameter tuning script")
    logger.info(f"Output directory: {args.output_dir}")
    if args.test:
        logger.info(f"Running in TEST MODE with {args.test_size} samples")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    try:
        df = pd.read_parquet(args.dataset_path)
        logger.info(f"Dataset loaded: {len(df)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        return

    # If in test mode, use only a subset of the data
    if args.test:
        if len(df) > args.test_size:
            logger.info(f"Sampling {args.test_size} records for test mode...")
            if 'phenology' in df.columns and len(df['phenology'].unique()) > 1:
                 df = df.groupby('phenology', group_keys=False).apply(lambda x: x.sample(min(len(x), args.test_size // len(df['phenology'].unique())), random_state=42))
            else:
                 df = df.sample(args.test_size, random_state=42)
            logger.info(f"Using subset of data: {len(df)} samples")
        else:
            logger.warning(f"Test size ({args.test_size}) is larger than dataset size ({len(df)}). Using full dataset.")


    # --- Unscale Features --- 
    logger.info("Unscaling features to physical ranges...")
    unscaled_count = 0
    skipped_cols = []
    df_copy = df.copy() # Work on a copy
    for index in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index}_{ftype_suffix}"
            if col_name in df_copy.columns:
                try:
                    df_copy[col_name] = unscale_feature(
                        df_copy[col_name],
                        feature_type=feature_type,
                        index_name=index
                    )
                    unscaled_count += 1
                except Exception as e:
                    logger.error(f"Error unscaling column {col_name}: {e}")
                    skipped_cols.append(col_name)
            else:
                skipped_cols.append(col_name)
    df = df_copy # Assign back the modified DataFrame
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    if skipped_cols:
        unique_skipped = sorted(list(set(skipped_cols)))
        # logger.debug(f"Skipped/Not Found {len(unique_skipped)} columns during unscaling: {unique_skipped[:5]}...")

    # Apply cos/sin transformation (needed if phase features are included)
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df = transform_circular_features(df, INDICES)
    logger.info("Circular transformation complete.")

    # Parse and validate features
    selected_features = [f.strip() for f in args.features.split(',')]
    logger.info(f"Using features: {selected_features}")

    # Check if all requested features exist in the dataframe after transformation
    available_features = df.columns.tolist()
    missing_features = [f for f in selected_features if f not in available_features]
    if missing_features:
        logger.error(f"Error: The following requested features are not found in the dataset: {missing_features}")
        return

    logger.info(f"Preparing data and CV splits for {len(selected_features)} features.")

    # Prepare data
    X = df[selected_features]
    y = df['phenology']
    groups = df['eco_region'] # Needed for StratifiedGroupKFold logic in create_eco_balanced_folds_df

    # Create eco-region balanced folds
    # create_eco_balanced_folds_df returns a list of (train_idx, test_idx) tuples
    fold_splits = create_eco_balanced_folds_df(df, n_splits=args.n_splits, random_state=42)
    logger.info(f"Generated {len(fold_splits)} eco-balanced cross-validation splits.")

    # Define the parameter grid for RandomForestClassifier
    param_grid = {
        'n_estimators': [25, 50, 100],
        'max_depth': [15, 30, None], # None means nodes expand until pure or min_samples_split
        'min_samples_split': [15, 30, 60],
        'min_samples_leaf': [10, 20, 40]
    }
    logger.info(f"Parameter grid for tuning: {param_grid}")

    # Define the base model
    # class_weight='balanced' is important for potentially imbalanced classes
    # n_jobs=-1 uses all available cores
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)

    # Define the scoring metric (F1 macro average)
    # Using average='macro' calculates metrics for each label, and finds their unweighted mean.
    # This does not take label imbalance into account. Consider 'f1_weighted' if desired.
    f1_scorer = make_scorer(f1_score, average='macro')

    # Handle sample weights if available
    fit_params = {}
    if 'weight' in df.columns:
        logger.info("Sample weights column 'weight' found. Preparing weights for HalvingGridSearchCV.")
        # HalvingGridSearchCV doesn't directly accept sample_weight in `fit`.
        # We need to pass it via `fit_params`. The key should match the estimator's fit parameter name.
        fit_params['sample_weight'] = df['weight'].values
        # Log some stats about the weights being used
        weight_stats = {
            'min': fit_params['sample_weight'].min(), 'max': fit_params['sample_weight'].max(),
            'mean': fit_params['sample_weight'].mean(), 'median': np.median(fit_params['sample_weight']),
            'non_zero': np.sum(fit_params['sample_weight'] > 0), 'total': len(fit_params['sample_weight'])
        }
        logger.info(f"Sample weights stats: {weight_stats}")
    else:
        logger.info("No 'weight' column found in the dataset. Proceeding without sample weights.")

    # Adjust min_resources for test mode
    actual_min_resources = args.min_resources
    if args.test:
        # Use a fraction of the available samples (e.g., 10%) as min_resources
        n_samples = len(X)
        test_min_resources = max(100, int(n_samples * 0.1)) # Ensure at least 100
        if args.min_resources > n_samples:
            logger.warning(f"Default min_resources ({args.min_resources}) is > test samples ({n_samples}). Adjusting min_resources to {test_min_resources}.")
            actual_min_resources = test_min_resources
        else:
            # Keep user-provided min_resources if it's smaller than test samples
            actual_min_resources = min(args.min_resources, n_samples)
            logger.info(f"Using specified min_resources ({actual_min_resources}) as it's <= test samples.")
    logger.info(f"Using min_resources = {actual_min_resources} for HalvingGridSearchCV")

    # Set up HalvingGridSearchCV
    # 'cv=fold_splits' uses the pre-computed eco-balanced splits
    # 'factor' controls the rate at which candidates are discarded
    # 'min_resources' determines the initial number of samples for the first iteration
    # 'verbose=2' provides detailed output during the search
    search = HalvingGridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=fold_splits,
        scoring=f1_scorer,
        factor=args.factor,
        min_resources=actual_min_resources,
        aggressive_elimination=False, # Can set to True for faster search if resources are limited
        n_jobs=-1, # Use available cores for CV folds within each iteration
        random_state=42,
        verbose=2
    )

    logger.info("Starting HalvingGridSearchCV...")
    start_time = time.time()

    try:
        # Fit the search object
        if fit_params:
             logger.info("Fitting HalvingGridSearchCV with sample weights...")
             search.fit(X, y, groups=groups, **fit_params) # groups might be needed if CV requires it
        else:
             logger.info("Fitting HalvingGridSearchCV without sample weights...")
             search.fit(X, y, groups=groups) # groups might be needed if CV requires it

    except ValueError as ve:
        logger.error(f"Error during HalvingGridSearchCV fitting: {ve}")
        if "may not be supported" in str(ve) and "sample_weight" in str(ve):
             logger.error("It seems sample_weight might not be directly supported in the CV split generation or HalvingGridSearchCV in this configuration.")
             logger.error("Consider modifying the CV strategy or removing weights if this persists.")
        elif "groups" in str(ve):
             logger.warning("The 'groups' parameter might not be needed or compatible with the provided CV object. Trying fit without 'groups'.")
             try:
                 if fit_params:
                     search.fit(X, y, **fit_params)
                 else:
                     search.fit(X, y)
             except Exception as e_retry:
                 logger.error(f"Retry fitting without 'groups' also failed: {e_retry}")
                 return # Exit if fitting fails critically
        else:
            logger.error("An unexpected error occurred during fitting.")
            return # Exit if fitting fails critically

    tuning_time = time.time() - start_time
    logger.info(f"HalvingGridSearchCV completed in {tuning_time:.2f} seconds")

    # --- Process and Save Results ---

    logger.info("--- Tuning Results ---")
    logger.info(f"Best Parameters Found: {search.best_params_}")
    logger.info(f"Best F1 Macro Score (CV): {search.best_score_:.4f}")

    # Get the best model
    best_model = search.best_estimator_
    model_params = count_rf_parameters(best_model)
    logger.info(f"Best model parameters: {model_params['total_parameters']} (across {model_params['n_nodes']} nodes in {model_params['n_estimators']} trees)")


    # Prepare results for saving
    results_to_save = {
        'best_params': search.best_params_,
        'best_score_f1_macro': search.best_score_,
        'selected_features': selected_features,
        'cv_n_splits': args.n_splits,
        'halving_factor': args.factor,
        'halving_min_resources': args.min_resources,
        'tuning_time_seconds': round(tuning_time, 2),
        'best_model_details': model_params,
        'cv_results': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in search.cv_results_.items()} # Convert numpy arrays for JSON
    }

    # Modify output names to include date
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Dated results name
    if args.results_name.endswith('.json'):
        base_results_name = args.results_name.replace('.json', '')
    else:
        base_results_name = args.results_name
    dated_results_name = f"{base_results_name}_{current_date}.json"
    results_path = os.path.join(args.output_dir, dated_results_name)

    # Dated model name
    if args.best_model_name.endswith('.joblib'):
        base_model_name = args.best_model_name.replace('.joblib', '')
    else:
        base_model_name = args.best_model_name
    dated_model_name = f"{base_model_name}_{current_date}.joblib"
    model_path = os.path.join(args.output_dir, dated_model_name)


    # Save tuning results to JSON
    try:
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        logger.info(f"Tuning results saved successfully to: {results_path}")
    except TypeError as te:
        logger.error(f"Error saving tuning results to JSON: {te}. Check for non-serializable types.")
        # Attempt to save without full cv_results if serialization fails
        try:
            del results_to_save['cv_results'] # Remove potentially problematic part
            with open(results_path, 'w') as f:
                json.dump(results_to_save, f, indent=4)
            logger.warning(f"Saved partial tuning results (without full cv_results) to: {results_path}")
        except Exception as e_partial:
            logger.error(f"Error saving partial tuning results: {e_partial}")
    except Exception as e:
        logger.error(f"Error saving tuning results to {results_path}: {e}")


    # Save the best model
    try:
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved successfully to: {model_path}")
    except Exception as e:
        logger.error(f"Error saving best model to {model_path}: {e}")

    logger.info("Script completed successfully.")

if __name__ == "__main__":
    main() 