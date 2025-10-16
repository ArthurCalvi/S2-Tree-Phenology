import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
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
    compute_metrics,
    format_confusion_matrix
)
from src.constants import EFFECTIVE_FOREST_AREA_BY_REGION

# --- Configuration ---

# Set up logging
log_dir = 'logs/domain_adaptation'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'north_south_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the path to the dataset
DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'

# Define the phenology mapping
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

# Define Northern and Southern Eco-regions based on user request and constants
NORTHERN_REGIONS = [
    "Vosges",
    "Greater Crystalline and Oceanic West",
    "Greater Semi-Continental East",
    "Jura",
    "Semi-Oceanic North Center"
]
ALL_REGIONS = list(EFFECTIVE_FOREST_AREA_BY_REGION.keys())
SOUTHERN_REGIONS = [region for region in ALL_REGIONS if region not in NORTHERN_REGIONS]

# Model Hyperparameters (can be tuned or passed as args)
MAX_DEPTH = 30
N_ESTIMATORS = 50
MIN_SAMPLES_SPLIT = 30
MIN_SAMPLES_LEAF = MIN_SAMPLES_SPLIT // 2

# --- Core Functions ---

def train_on_north_data(df_north, features, target='phenology'):
    """
    Train a RandomForest model only on the Northern region data.
    """
    logger.info(f"Training model on Northern regions ({len(df_north)} samples) using {len(features)} features...")

    X_train = df_north[features]
    y_train = df_north[target]

    sample_weights = None
    if 'weight' in df_north.columns:
        sample_weights = df_north['weight'].values
        logger.info("Using sample weights for training.")
        weight_stats = {
            'min': sample_weights.min(), 'max': sample_weights.max(),
            'mean': sample_weights.mean(), 'median': np.median(sample_weights),
            'non_zero': np.sum(sample_weights > 0), 'total': len(sample_weights)
        }
        logger.info(f"Training sample weights stats: {weight_stats}")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        verbose=1
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    logger.info("Model training on Northern data complete.")
    return model

def evaluate_on_south_data(model, df_south, features, target='phenology'):
    """
    Evaluate the trained model on the Southern region data.
    Computes overall metrics and metrics per Southern eco-region.
    """
    logger.info(f"Evaluating model on Southern regions ({len(df_south)} samples)...")

    if len(df_south) == 0:
        logger.warning("Southern region dataset is empty. Skipping evaluation.")
        return {}, pd.DataFrame(), np.zeros((2, 2)) # Return empty results

    X_val = df_south[features]
    y_val = df_south[target]

    # Make predictions
    y_pred = model.predict(X_val)

    # --- Overall Southern Metrics ---
    overall_metrics = compute_metrics(y_val, y_pred)
    logger.info("=== Overall Southern Validation Results ===")
    logger.info(f"Accuracy:           {overall_metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro:           {overall_metrics['f1_macro']:.4f}")
    logger.info(f"F1 Weighted:        {overall_metrics['f1_weighted']:.4f}")
    logger.info(f"Precision Macro:    {overall_metrics['precision_macro']:.4f}")
    logger.info(f"Recall Macro:       {overall_metrics['recall_macro']:.4f}")
    logger.info(f"Precision Weighted: {overall_metrics['precision_weighted']:.4f}")
    logger.info(f"Recall Weighted:    {overall_metrics['recall_weighted']:.4f}")
    logger.info(f"F1 Deciduous:       {overall_metrics['f1_deciduous']:.4f}")
    logger.info(f"F1 Evergreen:       {overall_metrics['f1_evergreen']:.4f}")

    # Aggregated confusion matrix for Southern data
    aggregated_cm_array = confusion_matrix(y_val, y_pred, labels=[1, 2])
    cm_text = format_confusion_matrix(aggregated_cm_array, labels=[f'{PHENOLOGY_MAPPING[1]} (1)', f'{PHENOLOGY_MAPPING[2]} (2)'])
    logger.info("Southern Validation Confusion Matrix:")
    logger.info("" + cm_text)

    # Add aggregated CM values to the overall summary
    tn, fp, fn, tp = aggregated_cm_array.ravel()
    overall_metrics['aggregated_tn'] = int(tn)
    overall_metrics['aggregated_fp'] = int(fp)
    overall_metrics['aggregated_fn'] = int(fn)
    overall_metrics['aggregated_tp'] = int(tp)

    # --- Metrics per Southern Eco-Region ---
    results_per_ecoregion = []
    southern_eco_regions_present = df_south['eco_region'].unique()

    for eco_region in southern_eco_regions_present:
        eco_mask = df_south['eco_region'] == eco_region
        if sum(eco_mask) > 0:
            eco_y_val = y_val[eco_mask]
            eco_y_pred = y_pred[eco_mask]

            eco_metrics = compute_metrics(eco_y_val, eco_y_pred)
            eco_metrics['eco_region'] = eco_region
            eco_metrics['n_samples'] = len(eco_y_val)
            results_per_ecoregion.append(eco_metrics)
            # logger.info(f"  {eco_region} Macro F1: {eco_metrics['f1_macro']:.4f} (on {len(eco_y_val)} samples)") # Can be verbose

    eco_results_df = pd.DataFrame(results_per_ecoregion).sort_values('f1_macro', ascending=False)
    logger.info("=== Results per Southern Eco-Region ===")
    # Select and format columns for tabulation
    metrics_to_show = [
        'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted',
        'f1_deciduous', 'f1_evergreen'
    ]
    cols_to_display = ['eco_region', 'n_samples'] + metrics_to_show
    # Rename columns for better table display (optional, can keep original)
    col_rename_map = {
        'f1_macro': 'F1 Macro', 'f1_weighted': 'F1 Wgt',
        'accuracy': 'Acc', 'n_samples': 'N Samples',
        'precision_macro': 'Prec Macro', 'recall_macro': 'Rec Macro',
        'f1_deciduous': 'F1 Decid', 'f1_evergreen': 'F1 Evrgn'
    }
    table_data = eco_results_df[cols_to_display].rename(columns=col_rename_map).round(4)
    logger.info("" + tabulate(table_data, headers='keys', tablefmt='psql', showindex=False))


    return overall_metrics, eco_results_df, aggregated_cm_array

def count_rf_parameters(rf_model):
    """Counts the number of parameters in a RandomForest model."""
    n_estimators = len(rf_model.estimators_)
    total_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
    n_classes = rf_model.n_classes_
    params_per_node = 2 + n_classes # feature index + threshold + values array
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
    parser = argparse.ArgumentParser(description='Train RF on North, Validate on South for Domain Adaptation.')
    parser.add_argument('--features', '-f', type=str, required=True,
                        help='Comma-separated list of feature names to use.')
    parser.add_argument('--output_dir', '-o', type=str, default='results/domain_adaptation/north_south_rf',
                        help='Directory to save the model and metrics.')
    parser.add_argument('--model_name', '-m', type=str, default='north_south_rf_model.joblib',
                        help='Filename for the saved model.')
    parser.add_argument('--metrics_name', '-j', type=str, default='north_south_validation_metrics.json',
                        help='Filename for the overall southern validation metrics JSON.')
    parser.add_argument('--eco_metrics_name', '-e', type=str, default='north_south_validation_eco_metrics.csv',
                        help='Filename for the southern eco-region validation metrics CSV.')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run in test mode with a small subset of data.')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of samples to use in test mode (from full dataset).')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help=f'Path to the dataset parquet file (default: {DATASET_PATH}).')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting Domain Adaptation (North -> South) Training Script")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Northern regions (Training): {NORTHERN_REGIONS}")
    logger.info(f"Southern regions (Validation): {SOUTHERN_REGIONS}")
    if args.test:
        logger.info(f"Running in TEST MODE with approximately {args.test_size} total samples")

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    try:
        df_full = pd.read_parquet(args.dataset_path)
        logger.info(f"Full dataset loaded: {len(df_full)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset from {args.dataset_path}: {e}")
        return

    # If in test mode, sample before splitting
    if args.test:
        if len(df_full) > args.test_size:
            logger.info(f"Sampling {args.test_size} records for test mode...")
            # Ensure balanced sampling if possible across eco-regions/phenology
            if 'phenology' in df_full.columns and len(df_full['phenology'].unique()) > 1:
                df_full = df_full.groupby(['eco_region', 'phenology'], group_keys=False).apply(
                    lambda x: x.sample(min(len(x), args.test_size // (len(df_full['eco_region'].unique()) * 2) + 1), random_state=42)
                )
                # Limit total size if stratified sampling resulted in more
                if len(df_full) > args.test_size * 1.1: # Allow slight overshoot
                    df_full = df_full.sample(args.test_size, random_state=42)
            else:
                 df_full = df_full.sample(args.test_size, random_state=42)
            logger.info(f"Using subset of data: {len(df_full)} samples")
        else:
            logger.warning(f"Test size ({args.test_size}) is larger than dataset size ({len(df_full)}). Using full dataset.")

    # --- Split Data into North and South ---
    logger.info("Splitting data into Northern and Southern regions...")
    df_north = df_full[df_full['eco_region'].isin(NORTHERN_REGIONS)].copy()
    df_south = df_full[df_full['eco_region'].isin(SOUTHERN_REGIONS)].copy()
    logger.info(f"Northern dataset size (Training): {len(df_north)} samples")
    logger.info(f"Southern dataset size (Validation): {len(df_south)} samples")

    if len(df_north) == 0:
        logger.error("Northern dataset is empty. Cannot train model.")
        return
    if len(df_south) == 0:
        logger.warning("Southern dataset is empty. Evaluation will be skipped.")

    # --- Feature Preprocessing (Unscaling & Transformation) ---
    # Apply to the full dataframe copy before splitting to ensure consistency?
    # Or apply separately? Let's apply to the full copy first.
    logger.info("Preprocessing features (Unscaling and Circular Transformation)...")
    df_processed = df_full.copy() # Work on a copy to avoid modifying original df

    # Unscale Features
    unscaled_count = 0
    skipped_cols = []
    for index in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index}_{ftype_suffix}"
            if col_name in df_processed.columns:
                try:
                    df_processed[col_name] = unscale_feature(
                        df_processed[col_name],
                        feature_type=feature_type,
                        index_name=index
                    )
                    unscaled_count += 1
                except Exception as e:
                    logger.error(f"Error unscaling column {col_name}: {e}")
                    skipped_cols.append(col_name)
            # else: # Don't log every missing column, could be verbose
            #     skipped_cols.append(col_name)
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    # if skipped_cols: logger.debug(f"Skipped/Not Found {len(set(skipped_cols))} columns during unscaling.")

    # Apply cos/sin transformation
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df_processed = transform_circular_features(df_processed, INDICES)
    logger.info("Circular transformation complete.")

    # --- Parse and Validate Selected Features ---
    selected_features = [f.strip() for f in args.features.split(',')]
    logger.info(f"Requested features for training/validation: {selected_features}")

    # Check if all requested features exist in the processed dataframe
    available_features = df_processed.columns.tolist()
    missing_features = [f for f in selected_features if f not in available_features]
    if missing_features:
        logger.error(f"Error: The following requested features are not found in the dataset (after transformation): {missing_features}")
        logger.error(f"Available features include (showing first 100): {available_features[:100]}")
        return
    logger.info(f"Using {len(selected_features)} features.")

    # --- Re-split Processed Data ---
    # Split the *processed* dataframe now
    df_north_processed = df_processed[df_processed['eco_region'].isin(NORTHERN_REGIONS)].copy()
    df_south_processed = df_processed[df_processed['eco_region'].isin(SOUTHERN_REGIONS)].copy()
    logger.info(f"Using {len(df_north_processed)} processed samples for North training.")
    logger.info(f"Using {len(df_south_processed)} processed samples for South validation.")

    # Modify model name to include date
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    if not args.model_name.endswith('.joblib'):
        args.model_name += '.joblib'
    base_model_name = args.model_name.replace('.joblib', '')
    dated_model_name = f"{base_model_name}_{current_date}.joblib"

    # --- Train Model on North Data ---
    start_time_train = time.time()
    model = train_on_north_data(df_north_processed, selected_features, target='phenology')
    train_time = time.time() - start_time_train
    logger.info(f"Model training completed in {train_time:.2f} seconds")

    # Count model parameters
    model_params = count_rf_parameters(model)
    logger.info(f"Model parameters: {model_params['total_parameters']} (across {model_params['n_nodes']} nodes in {model_params['n_estimators']} trees)")

    # --- Evaluate Model on South Data ---
    start_time_eval = time.time()
    # Pass the *processed* south data for evaluation
    south_overall_metrics, south_eco_results_df, _ = evaluate_on_south_data(
        model, df_south_processed, selected_features, target='phenology'
    )
    eval_time = time.time() - start_time_eval
    logger.info(f"Evaluation on Southern data completed in {eval_time:.2f} seconds")

    # Add timings and parameters to metrics summary (use south_overall_metrics as base)
    if south_overall_metrics: # Check if evaluation happened
        south_overall_metrics['training_time_seconds'] = round(train_time, 2)
        south_overall_metrics['evaluation_time_seconds'] = round(eval_time, 2)
        south_overall_metrics['selected_features'] = selected_features # Record features used
        south_overall_metrics['model_parameters'] = model_params
        south_overall_metrics['training_regions'] = NORTHERN_REGIONS
        south_overall_metrics['validation_regions'] = SOUTHERN_REGIONS
        south_overall_metrics['training_samples'] = len(df_north_processed)
        south_overall_metrics['validation_samples'] = len(df_south_processed)

    # --- Save Outputs ---

    # Save trained model
    model_path = os.path.join(args.output_dir, dated_model_name)
    try:
        joblib.dump(model, model_path)
        logger.info(f"Trained model saved successfully to: {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")

    # Save model config
    config_name = f"{base_model_name}_{current_date}_config.json"
    config_path = os.path.join(args.output_dir, config_name)
    config_data = {
        'model_name': dated_model_name,
        'creation_date': current_date,
        'model_type': 'RandomForestClassifier',
        'n_estimators': model_params['n_estimators'],
        'max_depth': MAX_DEPTH,
        'min_samples_split': MIN_SAMPLES_SPLIT,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'total_parameters': model_params['total_parameters'],
        'selected_features': selected_features,
        'feature_count': len(selected_features),
        'class_weight': 'balanced',
        'random_state': 42,
        'training_setup': 'Domain Adaptation: Train North, Validate South',
        'training_regions': NORTHERN_REGIONS,
        'validation_regions': SOUTHERN_REGIONS,
        'training_samples': len(df_north_processed),
        'validation_samples': len(df_south_processed)
    }
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Model config saved successfully to: {config_path}")
    except Exception as e:
        logger.error(f"Error saving model config to {config_path}: {e}")

    # Save overall Southern validation metrics to JSON
    metrics_path = os.path.join(args.output_dir, args.metrics_name)
    if south_overall_metrics:
        try:
            with open(metrics_path, 'w') as f:
                # Convert numpy types if any exist (though compute_metrics should return standard types)
                serializable_metrics = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in south_overall_metrics.items()}
                json.dump(serializable_metrics, f, indent=4)
            logger.info(f"Overall Southern validation metrics saved successfully to: {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics to {metrics_path}: {e}")
    else:
         logger.warning(f"Skipping saving overall Southern metrics as evaluation did not run or produced no results.")

    # Save Southern eco-region metrics to CSV
    eco_metrics_path = os.path.join(args.output_dir, args.eco_metrics_name)
    if not south_eco_results_df.empty:
        try:
            south_eco_results_df.to_csv(eco_metrics_path, index=False, float_format='%.4f')
            logger.info(f"Southern eco-region metrics saved successfully to: {eco_metrics_path}")
        except Exception as e:
            logger.error(f"Error saving eco-region metrics to {eco_metrics_path}: {e}")
    else:
        logger.warning(f"Skipping saving Southern eco-region metrics as evaluation did not run or produced no results.")


    logger.info("Script completed.")

if __name__ == "__main__":
    main() 