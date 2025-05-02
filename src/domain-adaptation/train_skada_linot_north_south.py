import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
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

# Try importing skada components
try:
    from skada import make_da_pipeline
    from skada import LinearOTMappingAdapter # Using LinearOTMappingAdapter for LinOT
except ImportError:
    logging.error("SKADA library not found. Please install it: pip install git+https://github.com/scikit-adaptation/skada")
    sys.exit(1)

# --- Configuration ---

# Set up logging
log_dir = 'logs/domain_adaptation'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'skada_linot_training.log')),
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

# Define Northern and Southern Eco-regions
NORTHERN_REGIONS = [
    "Vosges",
    "Greater Crystalline and Oceanic West",
    "Greater Semi-Continental East",
    "Jura",
    "Semi-Oceanic North Center"
]
ALL_REGIONS = list(EFFECTIVE_FOREST_AREA_BY_REGION.keys())
SOUTHERN_REGIONS = [region for region in ALL_REGIONS if region not in NORTHERN_REGIONS]

# Base Estimator Hyperparameters (RandomForest)
MAX_DEPTH = 30
N_ESTIMATORS = 50
MIN_SAMPLES_SPLIT = 30
MIN_SAMPLES_LEAF = MIN_SAMPLES_SPLIT // 2

# --- Core Functions ---

def train_skada_pipeline(X, y, sample_domain, features):
    """
    Train a SKADA pipeline using LinearOTMappingAdapter and RandomForestClassifier.
    """
    logger.info(f"Training SKADA pipeline (LinearOT + RF) on {len(features)} features...")
    logger.info(f"Total samples: {len(X)}, Source samples: {np.sum(sample_domain < 0)}, Target samples: {np.sum(sample_domain > 0)}")

    # Define the base estimator
    base_estimator = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        verbose=0 # Keep verbose off for pipeline to avoid excessive logs
    )

    # Define the SKADA adapter
    # adapter = LinearOTMappingAdapter(metric='sqeuclidean') # Default metric
    adapter = LinearOTMappingAdapter()

    # Create the DA pipeline
    pipeline = make_da_pipeline(adapter, base_estimator)

    # Fit the pipeline
    # SKADA's fit expects features as X, labels as y, and domain labels
    pipeline.fit(X, y, sample_domain=sample_domain)

    logger.info("SKADA pipeline training complete.")
    return pipeline

def evaluate_skada_pipeline_on_south(pipeline, df_south_processed, features, target='phenology'):
    """
    Evaluate the trained SKADA pipeline on the Southern (target) region data.
    Computes overall metrics and metrics per Southern eco-region.
    """
    logger.info(f"Evaluating SKADA pipeline on Southern regions ({len(df_south_processed)} samples)...")

    if len(df_south_processed) == 0:
        logger.warning("Southern region dataset is empty. Skipping evaluation.")
        return {}, pd.DataFrame(), np.zeros((2, 2))

    X_val_south = df_south_processed[features]
    y_val_south = df_south_processed[target]

    # Make predictions using the pipeline on the target data
    # The pipeline automatically uses the adapted model for prediction
    # Pass sample_domain as a NumPy array with the target label (-1)
    target_domain_labels = np.full(len(X_val_south), -1, dtype=int)
    y_pred_south = pipeline.predict(X_val_south.to_numpy(), sample_domain=target_domain_labels) # Ensure X is also numpy

    # --- Overall Southern Metrics ---
    overall_metrics = compute_metrics(y_val_south, y_pred_south)
    logger.info("=== Overall Southern Validation Results (SKADA LinOT+RF) ===")
    # Log key metrics
    logger.info(f"Accuracy:           {overall_metrics['accuracy']:.4f}")
    logger.info(f"F1 Macro:           {overall_metrics['f1_macro']:.4f}")
    logger.info(f"F1 Weighted:        {overall_metrics['f1_weighted']:.4f}")
    # logger.info(f"Precision Macro:    {overall_metrics['precision_macro']:.4f}") # Example of other metrics
    # logger.info(f"Recall Macro:       {overall_metrics['recall_macro']:.4f}")
    logger.info(f"F1 Deciduous:       {overall_metrics['f1_deciduous']:.4f}")
    logger.info(f"F1 Evergreen:       {overall_metrics['f1_evergreen']:.4f}")

    # Aggregated confusion matrix for Southern data
    aggregated_cm_array = confusion_matrix(y_val_south, y_pred_south, labels=[1, 2])
    cm_text = format_confusion_matrix(aggregated_cm_array, labels=[f'{PHENOLOGY_MAPPING[1]} (1)', f'{PHENOLOGY_MAPPING[2]} (2)'])
    logger.info("Southern Validation Confusion Matrix (SKADA LinOT+RF):")
    logger.info("" + cm_text)

    # Add aggregated CM values to the overall summary
    tn, fp, fn, tp = aggregated_cm_array.ravel() if aggregated_cm_array.size == 4 else (0,0,0,0) # Handle potential empty CM
    overall_metrics['aggregated_tn'] = int(tn)
    overall_metrics['aggregated_fp'] = int(fp)
    overall_metrics['aggregated_fn'] = int(fn)
    overall_metrics['aggregated_tp'] = int(tp)

    # --- Metrics per Southern Eco-Region ---
    results_per_ecoregion = []
    southern_eco_regions_present = df_south_processed['eco_region'].unique()

    for eco_region in southern_eco_regions_present:
        eco_mask = df_south_processed['eco_region'] == eco_region
        if sum(eco_mask) > 0:
            eco_X_val = X_val_south[eco_mask]
            eco_y_val = y_val_south[eco_mask]
            # Pass sample_domain as a NumPy array with the target label (-1) for the subset
            eco_target_domain_labels = np.full(len(eco_X_val), -1, dtype=int)
            eco_y_pred = pipeline.predict(eco_X_val.to_numpy(), sample_domain=eco_target_domain_labels) # Ensure X is also numpy

            eco_metrics = compute_metrics(eco_y_val, eco_y_pred)
            eco_metrics['eco_region'] = eco_region
            eco_metrics['n_samples'] = len(eco_y_val)
            results_per_ecoregion.append(eco_metrics)

    eco_results_df = pd.DataFrame(results_per_ecoregion)
    if not eco_results_df.empty:
        eco_results_df = eco_results_df.sort_values('f1_macro', ascending=False)
        logger.info("=== Results per Southern Eco-Region (SKADA LinOT+RF) ===")
        # Select and format columns for tabulation (similar to previous script)
        metrics_to_show = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'f1_deciduous', 'f1_evergreen'
        ]
        cols_to_display = ['eco_region', 'n_samples'] + metrics_to_show
        col_rename_map = {
            'f1_macro': 'F1 Macro', 'accuracy': 'Acc', 'n_samples': 'N Samples',
            'f1_deciduous': 'F1 Decid', 'f1_evergreen': 'F1 Evrgn'
        }
        table_data = eco_results_df[cols_to_display].rename(columns=col_rename_map).round(4)
        logger.info("\n" + tabulate(table_data, headers='keys', tablefmt='psql', showindex=False))
    else:
        logger.info("No per-eco-region results generated for Southern data.")

    return overall_metrics, eco_results_df, aggregated_cm_array

# Removed count_rf_parameters as it's harder to access the internal RF model easily from pipeline
# Could potentially add if needed by accessing pipeline.estimator_

# --- Main Execution ---

def main():
    # Parse command line arguments (similar to previous script)
    parser = argparse.ArgumentParser(description='Train SKADA (LinOT+RF) on North, Validate on South, or load pre-trained.')
    parser.add_argument('--features', '-f', type=str, required=True,
                        help='Comma-separated list of feature names to use.')
    parser.add_argument('--output_dir', '-o', type=str, default='results/domain_adaptation/skada_linot_rf',
                        help='Directory to save the pipeline and metrics.')
    parser.add_argument('--model_name', '-m', type=str, default='skada_linot_rf_pipeline.joblib',
                        help='Filename for saving the SKADA pipeline (used if training).')
    parser.add_argument('--metrics_name', '-j', type=str, default='skada_linot_validation_metrics.json',
                        help='Filename for the overall southern validation metrics JSON.')
    parser.add_argument('--eco_metrics_name', '-e', type=str, default='skada_linot_validation_eco_metrics.csv',
                        help='Filename for the southern eco-region validation metrics CSV.')
    parser.add_argument('--load_pipeline', type=str, default=None,
                        help='Path to a pre-trained .joblib pipeline file to load and evaluate (skips training).')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run in test mode with a small subset of data.')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of samples to use in test mode (from full dataset).')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help=f'Path to the dataset parquet file (default: {DATASET_PATH}).')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.load_pipeline:
        logger.info(f"Loading pre-trained pipeline from: {args.load_pipeline}")
    else:
        logger.info("Starting Domain Adaptation (SKADA LinOT + RF North -> South) Training Script")
    logger.info(f"Output directory for metrics: {args.output_dir}")
    logger.info(f"Northern regions (Source): {NORTHERN_REGIONS}")
    logger.info(f"Southern regions (Target): {SOUTHERN_REGIONS}")
    if args.test:
        logger.info(f"Running in TEST MODE with approximately {args.test_size} total samples")

    # Load dataset (needed for preprocessing and evaluation regardless of training)
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
            # Simplified sampling for test mode
            df_full = df_full.sample(n=args.test_size, random_state=42)
            logger.info(f"Using subset of data: {len(df_full)} samples")
        else:
            logger.warning(f"Test size ({args.test_size}) is larger than dataset size ({len(df_full)}). Using full dataset.")

    # --- Feature Preprocessing (Unscaling & Transformation) ---
    # Needs to happen even if loading model, to prepare evaluation data
    logger.info("Preprocessing features (Unscaling and Circular Transformation)...")
    df_processed = df_full.copy()
    # Unscale Features (same as before)
    unscaled_count = 0
    for index in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index}_{ftype_suffix}"
            if col_name in df_processed.columns:
                try:
                    df_processed[col_name] = unscale_feature(df_processed[col_name], feature_type, index)
                    unscaled_count += 1
                except Exception as e: logger.error(f"Error unscaling {col_name}: {e}")
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    # Apply cos/sin transformation (same as before)
    logger.info("Applying circular transformation...")
    df_processed = transform_circular_features(df_processed, INDICES)
    logger.info("Circular transformation complete.")

    # --- Parse and Validate Selected Features ---
    selected_features = [f.strip() for f in args.features.split(',')]
    logger.info(f"Requested features for training/validation: {selected_features}")
    available_features = df_processed.columns.tolist()
    missing_features = [f for f in selected_features if f not in available_features]
    if missing_features:
        logger.error(f"Error: Missing requested features: {missing_features}")
        return
    logger.info(f"Using {len(selected_features)} features.")

    # --- Prepare Data (Split North/South - needed for evaluation) ---
    # Note: Combined data is only needed for training
    logger.info("Splitting processed data into North and South regions...")
    df_north_processed = df_processed[df_processed['eco_region'].isin(NORTHERN_REGIONS)].copy()
    df_south_processed = df_processed[df_processed['eco_region'].isin(SOUTHERN_REGIONS)].copy()
    logger.info(f"North (Source) samples: {len(df_north_processed)}, South (Target) samples: {len(df_south_processed)}")

    # Initialize pipeline variable
    pipeline = None
    train_time = 0.0

    if args.load_pipeline:
        # --- Load Pre-trained Pipeline ---
        if not os.path.exists(args.load_pipeline):
            logger.error(f"Pipeline file not found: {args.load_pipeline}")
            return
        try:
            start_time_load = time.time()
            pipeline = joblib.load(args.load_pipeline)
            load_time = time.time() - start_time_load
            logger.info(f"Successfully loaded pipeline from {args.load_pipeline} in {load_time:.2f} seconds.")
            # Basic check if it looks like a pipeline
            if not hasattr(pipeline, 'predict') or not hasattr(pipeline, 'steps'):
                 logger.warning("Loaded object does not appear to be a scikit-learn/skada pipeline.")
                 # You might want to add more robust checks here
        except Exception as e:
            logger.error(f"Error loading pipeline from {args.load_pipeline}: {e}")
            return
    else:
        # --- Train SKADA Pipeline ---
        logger.info("Preparing data for SKADA pipeline training...")
        if len(df_north_processed) == 0:
            logger.error("Northern (Source) dataset is empty. Cannot train SKADA pipeline.")
            return
        if len(df_south_processed) == 0:
            logger.warning("Southern (Target) dataset is empty. Adaptation might be suboptimal.")

        # Combine dataframes only needed for training
        df_combined = pd.concat([df_north_processed, df_south_processed], ignore_index=True)

        # Create X (features) and y (target) for combined data
        X_combined = df_combined[selected_features]
        y_combined = df_combined['phenology']

        # Convert to NumPy arrays for skada/POT compatibility
        X_combined_np = X_combined.to_numpy()
        y_combined_np = y_combined.to_numpy()

        # Create sample_domain: 1 for North (source), -1 for South (target)
        sample_domain = np.concatenate([
            np.full(len(df_north_processed), 1, dtype=int), # Source
            np.full(len(df_south_processed), -1, dtype=int)   # Target
        ])

        logger.info(f"Combined data shapes -> X: {X_combined_np.shape}, y: {y_combined_np.shape}, sample_domain: {sample_domain.shape}")

        start_time_train = time.time()
        pipeline = train_skada_pipeline(X_combined_np, y_combined_np, sample_domain, selected_features)
        train_time = time.time() - start_time_train
        logger.info(f"SKADA pipeline training completed in {train_time:.2f} seconds")

    # --- Evaluate SKADA Pipeline on South Data ---
    if pipeline is None:
        logger.error("Pipeline is not defined. Cannot proceed with evaluation.")
        return

    if len(df_south_processed) == 0:
         logger.warning("Southern (Target) dataset is empty. Skipping evaluation.")
         south_overall_metrics = {}
         south_eco_results_df = pd.DataFrame()
         eval_time = 0.0
    else:
        start_time_eval = time.time()
        south_overall_metrics, south_eco_results_df, _ = evaluate_skada_pipeline_on_south(
            pipeline, df_south_processed, selected_features, target='phenology'
        )
        eval_time = time.time() - start_time_eval
        logger.info(f"Evaluation on Southern data completed in {eval_time:.2f} seconds")

    # Add timings to metrics summary
    if south_overall_metrics: # Check if evaluation happened and produced metrics
        south_overall_metrics['training_time_seconds'] = round(train_time, 2)
        south_overall_metrics['evaluation_time_seconds'] = round(eval_time, 2)
        south_overall_metrics['selected_features'] = selected_features
        south_overall_metrics['adapter'] = 'LinearOTMappingAdapter' # Assuming this structure
        south_overall_metrics['estimator'] = 'RandomForestClassifier' # Assuming this structure
        south_overall_metrics['source_regions'] = NORTHERN_REGIONS
        south_overall_metrics['target_regions'] = SOUTHERN_REGIONS
        south_overall_metrics['source_samples'] = len(df_north_processed)
        south_overall_metrics['target_samples'] = len(df_south_processed)
        south_overall_metrics['pipeline_source'] = args.load_pipeline if args.load_pipeline else 'trained_in_script'

    # --- Save Outputs ---
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    # Save trained pipeline and config ONLY if training was performed
    if not args.load_pipeline and pipeline is not None:
        if not args.model_name.endswith('.joblib'):
            args.model_name += '.joblib'
        base_model_name = args.model_name.replace('.joblib', '')
        dated_model_name = f"{base_model_name}_{current_date}.joblib"
        pipeline_path = os.path.join(args.output_dir, dated_model_name)
        try:
            joblib.dump(pipeline, pipeline_path)
            logger.info(f"Trained SKADA pipeline saved successfully to: {pipeline_path}")
        except Exception as e:
            logger.error(f"Error saving pipeline to {pipeline_path}: {e}")

        # Save pipeline config (simplified)
        config_name = f"{base_model_name}_{current_date}_config.json"
        config_path = os.path.join(args.output_dir, config_name)
        config_data = {
            'pipeline_name': dated_model_name,
            'creation_date': current_date,
            'adapter': 'LinearOTMappingAdapter',
            'estimator': 'RandomForestClassifier',
            'estimator_params': {
                'n_estimators': N_ESTIMATORS,
                'max_depth': MAX_DEPTH,
                'min_samples_split': MIN_SAMPLES_SPLIT,
                'min_samples_leaf': MIN_SAMPLES_LEAF,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'selected_features': selected_features,
            'feature_count': len(selected_features),
            'training_setup': 'Domain Adaptation: SKADA (LinOT+RF) Train North(src)/South(tgt)',
            'source_regions': NORTHERN_REGIONS,
            'target_regions': SOUTHERN_REGIONS,
            'source_samples': len(df_north_processed),
            'target_samples': len(df_south_processed)
        }
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            logger.info(f"Pipeline config saved successfully to: {config_path}")
        except Exception as e:
            logger.error(f"Error saving pipeline config to {config_path}: {e}")

    # Save evaluation metrics regardless of training/loading
    metrics_path = os.path.join(args.output_dir, args.metrics_name)
    if south_overall_metrics:
        try:
            with open(metrics_path, 'w') as f:
                # Ensure metrics dict is serializable
                serializable_metrics = {}
                for k, v in south_overall_metrics.items():
                     if isinstance(v, np.integer): serializable_metrics[k] = int(v)
                     elif isinstance(v, np.floating): serializable_metrics[k] = float(v)
                     else: serializable_metrics[k] = v
                json.dump(serializable_metrics, f, indent=4)
            logger.info(f"Overall Southern validation metrics saved successfully to: {metrics_path}")
        except Exception as e: logger.error(f"Error saving metrics to {metrics_path}: {e}")
    else: logger.warning("Skipping saving overall Southern metrics as none were generated.")

    # Save Southern eco-region metrics to CSV
    eco_metrics_path = os.path.join(args.output_dir, args.eco_metrics_name)
    if not south_eco_results_df.empty:
        try:
            south_eco_results_df.to_csv(eco_metrics_path, index=False, float_format='%.4f')
            logger.info(f"Southern eco-region metrics saved successfully to: {eco_metrics_path}")
        except Exception as e: logger.error(f"Error saving eco-region metrics: {e}")
    else: logger.warning("Skipping saving Southern eco-region metrics as none were generated.")

    logger.info("Script completed.")

if __name__ == "__main__":
    main() 