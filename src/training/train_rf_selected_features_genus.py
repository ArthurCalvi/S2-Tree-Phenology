import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix # Only confusion matrix from here
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold # StratifiedGroupKFold might be used by create_eco_balanced_folds_df
import matplotlib.pyplot as plt
from src.utils import apply_science_style
apply_science_style()
import seaborn as sns
from collections import defaultdict
import time
import os
import math
import logging
from tqdm import tqdm
import datetime
from sklearn.utils import shuffle
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

# Import utility functions and constants
from src.utils import (
    unscale_feature,
    transform_circular_features,
    compute_multiclass_metrics, # Changed from compute_metrics
    create_eco_balanced_folds_df,
    display_fold_distribution, # Kept for potential debugging
    format_multiclass_confusion_matrix,
    count_rf_parameters # Ensure this is available
)
from src.constants import (
    GENUS_MAPPING # Changed from PHENOLOGY_MAPPING
)

# --- Configuration ---

# Set up logging (will be configured in main based on target)
logger = logging.getLogger(__name__)

# Define the path to the dataset (can be overridden by command line)
DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'

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

# Default RF Hyperparameters (can be tuned and overridden if needed)
MAX_DEPTH = 30
N_ESTIMATORS = 100 # Example, adjust as needed
MIN_SAMPLES_SPLIT = 30
MIN_SAMPLES_LEAF = 15

# --- Core Training Functions ---

def evaluate_features_cv(df, features, target_column, target_mapping, n_splits=5):
    """
    Evaluate the given features using cross-validation for the specified target.
    Returns performance metrics averaged over folds and aggregated confusion matrix.
    """
    logger.info(f"Starting cross-validation evaluation for {len(features)} features, target: {target_column}")

    # Prepare data
    X = df[features]
    y = df[target_column]
    labels = sorted(target_mapping.keys())
    target_names = [target_mapping[k] for k in labels]

    # Create eco-region balanced folds
    fold_splits = create_eco_balanced_folds_df(df, n_splits=n_splits, random_state=42) # Ensure target_col is passed if needed by the function

    # Store results
    results_per_fold = []
    all_true = []
    all_pred = []

    # Eco-region specific metrics (can be adapted if needed, more complex for multiclass)
    # For now, focusing on overall metrics for genus.
    # If eco-region breakdown is critical, this part needs careful adaptation for multiclass.
    # logger.info("Eco-region specific metrics are not calculated in this version for multiclass to maintain simplicity. Focus is on overall performance.")
    # eco_results_df = pd.DataFrame() # Placeholder

    results_per_ecoregion = defaultdict(list)
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_splits, desc="Cross-validation folds")):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sample_weights = None
        if 'weight' in df.columns:
            sample_weights = df.iloc[train_idx]['weight'].values

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced', # Good for multi-class
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF
        )
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_val)

        all_true.extend(y_val)
        all_pred.extend(y_pred)

        # Compute metrics for this fold using compute_multiclass_metrics
        fold_metrics = compute_multiclass_metrics(y_val, y_pred, labels=labels, target_names=target_names)
        fold_metrics['fold'] = fold + 1
        results_per_fold.append(fold_metrics)

        logger.info(f"Fold {fold+1}/{n_splits} -> Macro F1: {fold_metrics['f1_macro']:.4f}, Weighted F1: {fold_metrics['f1_weighted']:.4f}, Accuracy: {fold_metrics['accuracy']:.4f}")
        
        # Compute metrics per eco-region for this fold
        eco_regions_in_val = df.iloc[val_idx]['eco_region'].unique()
        for eco_region in eco_regions_in_val:
            eco_mask = df.iloc[val_idx]['eco_region'] == eco_region
            if sum(eco_mask) > 0:
                eco_y_val = y_val[eco_mask]
                eco_y_pred = y_pred[eco_mask]

                if len(eco_y_val) > 0: # Ensure there are samples for the eco-region
                    eco_metrics = compute_multiclass_metrics(eco_y_val, eco_y_pred, labels=labels, target_names=target_names)
                    eco_metrics['fold'] = fold + 1
                    eco_metrics['eco_region'] = eco_region
                    eco_metrics['n_samples_fold'] = len(eco_y_val) # n_samples specific to this fold and eco-region
                    results_per_ecoregion[eco_region].append(eco_metrics)
                    # logger.info(f"  Eco-region {eco_region} in Fold {fold+1} -> Macro F1: {eco_metrics['f1_macro']:.4f} ({len(eco_y_val)} samples)")

    # Aggregate results across folds (overall)
    results_df = pd.DataFrame(results_per_fold)

    # Define metrics to average (adjust based on compute_multiclass_metrics output)
    metrics_keys = [key for key in results_df.columns if key not in ['fold', 'confusion_matrix_df', 'classification_report_dict', 'confusion_matrix']]
    # Specifically list common ones for clarity in logging
    avg_display_metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'precision_weighted', 'recall_macro', 'recall_weighted']
    # Add per-class F1 scores if they exist (e.g., f1_GenusA, f1_GenusB)
    # This uses the target_names (e.g. 'Quercus', 'Fagus') from the target_mapping
    for class_label_name in target_names: # target_names are like ['Quercus', 'Fagus']
        # Construct the key as stored by compute_multiclass_metrics (e.g., 'f1_Quercus')
        # The key in metrics_results from compute_multiclass_metrics is f1_per_class[class_name]
        # The dataframe column will be f'f1_{class_name}' after flattening, if it was done that way.
        # Let's assume compute_multiclass_metrics returns keys like 'f1_classname', this works:
        f1_class_key = f'f1_{class_label_name.replace(" ", "_")}' 
        if f1_class_key in results_df.columns:
            avg_display_metrics.append(f1_class_key)
        # Also add precision and recall for per-class if available and desired for logging
        precision_class_key = f'precision_{class_label_name.replace(" ", "_")}'
        recall_class_key = f'recall_{class_label_name.replace(" ", "_")}'
        if precision_class_key in results_df.columns: avg_display_metrics.append(precision_class_key)
        if recall_class_key in results_df.columns: avg_display_metrics.append(recall_class_key)

    overall_metrics_summary = {}
    for metric in metrics_keys:
        if pd.api.types.is_numeric_dtype(results_df[metric]):
            overall_metrics_summary[f'mean_{metric}'] = results_df[metric].mean()
            overall_metrics_summary[f'std_{metric}'] = results_df[metric].std()

    logger.info(f"=== Overall CV Results for {target_column} (Mean ± Std across folds) ===")
    for metric_base_name in avg_display_metrics:
        mean_key = f'mean_{metric_base_name}'
        std_key = f'std_{metric_base_name}'
        if mean_key in overall_metrics_summary and std_key in overall_metrics_summary:
            logger.info(f"{metric_base_name.replace('_', ' ').title()}: {overall_metrics_summary[mean_key]:.4f} ± {overall_metrics_summary[std_key]:.4f}")

    # Aggregated confusion matrix (using all predictions across folds)
    aggregated_cm_array = confusion_matrix(all_true, all_pred, labels=labels)
    cm_text = format_multiclass_confusion_matrix(aggregated_cm_array, target_names=[target_mapping[l] for l in labels])
    logger.info(f"\nAggregated Confusion Matrix for {target_column} (All Folds):")
    logger.info("" + cm_text)
    
    overall_metrics_summary['aggregated_confusion_matrix'] = aggregated_cm_array.tolist()

    # --- Aggregate Eco-Region Results ---
    eco_results_list = []
    for eco_region, metrics_list_per_eco in results_per_ecoregion.items():
        if not metrics_list_per_eco: continue
        metrics_df_eco = pd.DataFrame(metrics_list_per_eco)
        avg_eco_metrics = {'eco_region': eco_region}
        
        # Calculate mean and std for all relevant numeric metrics per eco-region
        # These are the same keys as in overall_metrics_summary potentially
        for metric_col in metrics_df_eco.columns:
            if pd.api.types.is_numeric_dtype(metrics_df_eco[metric_col]) and metric_col not in ['fold', 'n_samples_fold']:
                avg_eco_metrics[f'{metric_col}_mean'] = metrics_df_eco[metric_col].mean()
                avg_eco_metrics[f'{metric_col}_std'] = metrics_df_eco[metric_col].std()
        
        avg_eco_metrics['n_samples_fold_mean'] = metrics_df_eco['n_samples_fold'].mean() # Avg samples per fold for this eco
        avg_eco_metrics['n_samples_total_cv'] = metrics_df_eco['n_samples_fold'].sum() # Total samples evaluated in CV for this eco
        avg_eco_metrics['n_samples_dataset'] = df[df['eco_region'] == eco_region].shape[0] # Total samples for this eco in the input df subset
        eco_results_list.append(avg_eco_metrics)

    eco_results_df = pd.DataFrame(eco_results_list)
    if not eco_results_df.empty:
        # Sort by a chosen metric, e.g., mean f1_macro, if it exists
        sort_key = 'f1_macro_mean' if 'f1_macro_mean' in eco_results_df.columns else 'accuracy_mean'
        if sort_key in eco_results_df.columns:
            eco_results_df = eco_results_df.sort_values(sort_key, ascending=False)
        
        logger.info(f"\n=== Average Results per Eco-Region for {target_column} (Across Folds) ===")
        # Define columns to show in the eco-region table - adapt as needed
        cols_to_show_eco = ['eco_region', 'n_samples_dataset', 'n_samples_total_cv']
        # Add mean/std for key metrics
        for m_key in ['accuracy', 'f1_macro', 'f1_weighted']:
            if f'{m_key}_mean' in eco_results_df.columns: cols_to_show_eco.append(f'{m_key}_mean')
            if f'{m_key}_std' in eco_results_df.columns: cols_to_show_eco.append(f'{m_key}_std')
        
        # Filter to existing columns to avoid KeyError
        cols_to_show_eco = [col for col in cols_to_show_eco if col in eco_results_df.columns]

        # Rename for display
        col_rename_map_eco = {col: col.replace('_mean', ' Mean').replace('_std', ' Std').replace('_', ' ').title() for col in cols_to_show_eco}
        table_data_eco = eco_results_df[cols_to_show_eco].rename(columns=col_rename_map_eco).round(4)
        logger.info("\n" + tabulate(table_data_eco, headers='keys', tablefmt='psql', showindex=False))
    else:
        logger.info(f"No eco-region specific metrics to display for {target_column}.")

    return overall_metrics_summary, eco_results_df, aggregated_cm_array


def train_final_model(df, features, target_column):
    """
    Train a final RandomForest model on the entire dataset for the specified target.
    """
    logger.info(f"Training final model for {target_column} on {len(df)} samples using {len(features)} features...")

    X = df[features]
    y = df[target_column]

    sample_weights = None
    if 'weight' in df.columns:
        sample_weights = df['weight'].values
        logger.info(f"Using sample weights for final {target_column} model training.")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF
    )
    model.fit(X, y, sample_weight=sample_weights)

    logger.info(f"Final {target_column} model training complete.")
    return model

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Train genus classification model with selected features.')
    parser.add_argument('--features', '-f', type=str, required=False,
                        help='Comma-separated list of feature names. Defaults to predefined genus features.')
    parser.add_argument('--output_dir', '-o', type=str, default='results/final_model_genus', # Changed default
                        help='Directory to save the model and metrics (default: results/final_model_genus).')
    parser.add_argument('--model_name', '-m', type=str, default='genus_model.joblib', # Changed default
                        help='Filename for the saved model (default: genus_model.joblib).')
    parser.add_argument('--metrics_name', '-j', type=str, default='genus_metrics.json', # Changed default
                        help='Filename for the overall metrics JSON output (default: genus_metrics.json).')
    parser.add_argument('--eco_metrics_name', '-e', type=str, default='genus_eco_metrics.csv', # Re-enabled
                        help='Filename for the eco-region metrics CSV output (default: genus_eco_metrics.csv).')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run in test mode with a small subset of data.')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of samples to use in test mode (default: 10000).')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation splits (default: 5).')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help=f'Path to the dataset parquet file (default: {DATASET_PATH}).')

    args = parser.parse_args()

    # --- Configure Logging ---
    log_file_name = f'logs/genus_selected_features_training.log' # Changed log file name
    os.makedirs('logs', exist_ok=True)
    # Remove existing handlers if any were configured at module level
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name),
            logging.StreamHandler()
        ]
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting GENUS classification training script with selected features")
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

    # Define target column and mapping
    target_column = 'genus'
    target_mapping = GENUS_MAPPING
    logger.info(f"Target column set to: {target_column}")
    
    # Ensure target column exists and is not all NaN
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in the DataFrame.")
        return
    if df[target_column].isnull().all():
        logger.error(f"Target column '{target_column}' contains all NaN values. Please check the dataset.")
        return
    df = df.dropna(subset=[target_column]) # Drop rows where target is NaN
    logger.info(f"Removed rows with NaN in '{target_column}'. Remaining samples: {len(df)}")


    # If in test mode, use only a subset of the data, stratified by target
    if args.test:
        if len(df) > args.test_size:
            logger.info(f"Sampling {args.test_size} records for test mode (stratified by {target_column})...")
            n_classes = df[target_column].nunique()
            sample_per_class = args.test_size // n_classes if n_classes > 0 else args.test_size
            if n_classes > 1:
                 df = df.groupby(target_column, group_keys=False).apply(
                     lambda x: x.sample(min(len(x), sample_per_class), random_state=42)
                 )
            else:
                 df = df.sample(min(args.test_size, len(df)), random_state=42) # Ensure not to sample more than available
            logger.info(f"Using subset of data: {len(df)} samples for test mode.")
        else:
            logger.warning(f"Test size ({args.test_size}) is larger than dataset size ({len(df)}). Using full available dataset for test mode.")


    # --- Unscale Features ---
    logger.info("Unscaling features to physical ranges...")
    unscaled_count = 0
    df_copy = df.copy()
    for index_name in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index_name}_{ftype_suffix}"
            if col_name in df_copy.columns:
                try:
                    df_copy[col_name] = unscale_feature(
                        df_copy[col_name],
                        feature_type=feature_type,
                        index_name=index_name
                    )
                    unscaled_count += 1
                except Exception as e:
                    logger.warning(f"Could not unscale column {col_name}: {e}")
    df = df_copy
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")

    # Apply cos/sin transformation
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df = transform_circular_features(df, INDICES)
    logger.info("Circular transformation complete.")

    # Default features from the shell script if not provided
    default_genus_features_str = "ndvi_amplitude_h1,ndvi_amplitude_h2,ndvi_phase_h1_cos,ndvi_phase_h1_sin,ndvi_phase_h2_sin,ndvi_offset,ndvi_var_residual,nbr_amplitude_h1,nbr_phase_h1_sin,nbr_phase_h2_cos,nbr_offset,nbr_var_residual,crswir_amplitude_h1,crswir_phase_h1_cos,crswir_phase_h2_cos,crswir_offset,crswir_var_residual"
    if args.features:
        selected_features = [f.strip() for f in args.features.split(',')]
        logger.info(f"Using features from command line: {selected_features}")
    else:
        selected_features = [f.strip() for f in default_genus_features_str.split(',')]
        logger.info(f"Using default genus features: {selected_features}")


    # Check if all requested features exist
    available_features = df.columns.tolist()
    missing_features = [f for f in selected_features if f not in available_features]
    if missing_features:
        logger.error(f"Error: The following features are not found: {missing_features}")
        logger.info(f"Available features include (first 50): {available_features[:50]}")
        return

    logger.info(f"Using {len(selected_features)} features for training and evaluation of {target_column}.")

    # Modify model name to include date
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    if not args.model_name.endswith('.joblib'):
        args.model_name += '.joblib'
    base_model_name = args.model_name.replace('.joblib', '')
    dated_model_name = f"{base_model_name}_{current_date}.joblib"

    # --- Cross-Validation Evaluation ---
    start_time_cv = time.time()
    overall_metrics_summary, eco_results_df, aggregated_cm_array = evaluate_features_cv(
        df, selected_features, target_column=target_column, target_mapping=target_mapping, n_splits=args.n_splits
    )
    cv_time = time.time() - start_time_cv
    logger.info(f"Cross-validation for {target_column} completed in {cv_time:.2f} seconds")

    # --- Final Model Training ---
    start_time_train = time.time()
    final_model = train_final_model(df, selected_features, target_column=target_column)
    train_time = time.time() - start_time_train
    logger.info(f"Final {target_column} model training completed in {train_time:.2f} seconds")

    # Count model parameters
    model_params = count_rf_parameters(final_model) # Using imported function
    logger.info(f"Model parameters: {model_params['total_parameters']} (across {model_params['n_nodes']} nodes in {model_params['n_estimators']} trees)")

    # Add timings, parameters, and features to metrics
    overall_metrics_summary['cv_evaluation_time_seconds'] = round(cv_time, 2)
    overall_metrics_summary['final_model_training_time_seconds'] = round(train_time, 2)
    overall_metrics_summary['selected_features'] = selected_features
    overall_metrics_summary['model_parameters'] = model_params
    overall_metrics_summary['target_column'] = target_column

    # --- Save Outputs ---
    model_path = os.path.join(args.output_dir, dated_model_name)
    try:
        joblib.dump(final_model, model_path)
        logger.info(f"Final {target_column} model saved to: {model_path}")
    except Exception as e:
        logger.error(f"Error saving {target_column} model to {model_path}: {e}")

    # Create config file for the genus model
    config_name = f"{base_model_name}_{current_date}_config.json"
    config_path = os.path.join(args.output_dir, config_name)
    config_data = {
        'model_name': dated_model_name,
        'creation_date': current_date,
        'target_column': target_column,
        'model_type': 'RandomForestClassifier',
        'n_estimators': model_params['n_estimators'],
        'max_depth': MAX_DEPTH, # Add other relevant RF params
        'min_samples_split': MIN_SAMPLES_SPLIT,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'total_parameters': model_params['total_parameters'],
        'n_nodes': model_params['n_nodes'],
        'selected_features': selected_features,
        'feature_count': len(selected_features),
        'class_weight': 'balanced',
        'random_state': 42,
        'dataset_path_used': args.dataset_path # Record dataset used
    }
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Model config for {target_column} saved to: {config_path}")
    except Exception as e:
        logger.error(f"Error saving {target_column} model config to {config_path}: {e}")

    # Save overall metrics to JSON
    metrics_path = os.path.join(args.output_dir, args.metrics_name)
    try:
        with open(metrics_path, 'w') as f:
            json.dump(overall_metrics_summary, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) # Handle numpy arrays
        logger.info(f"Overall {target_column} metrics saved to: {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving {target_column} metrics to {metrics_path}: {e}")

    # Eco-region metrics saving is currently disabled as per note in evaluate_features_cv
    # eco_metrics_path = os.path.join(args.output_dir, args.eco_metrics_name)
    # try:
    #     # eco_results_df.to_csv(eco_metrics_path, index=False, float_format='%.4f') # If eco_results_df gets populated
    #     logger.info(f"Eco-region metrics for {target_column} would be saved to: {eco_metrics_path} (currently disabled)")
    # except Exception as e:
    #     logger.error(f"Error saving {target_column} eco-region metrics to {eco_metrics_path}: {e}")

    # Save eco-region metrics to CSV
    if not eco_results_df.empty:
        eco_metrics_path = os.path.join(args.output_dir, args.eco_metrics_name)
        try:
            eco_results_df.to_csv(eco_metrics_path, index=False, float_format='%.4f')
            logger.info(f"Eco-region metrics for {target_column} saved to: {eco_metrics_path}")
        except Exception as e:
            logger.error(f"Error saving {target_column} eco-region metrics to {eco_metrics_path}: {e}")
    else:
        logger.info(f"No eco-region metrics data to save for {target_column}.")

    logger.info(f"Genus training script for {target_column} completed successfully.")

if __name__ == "__main__":
    main() 