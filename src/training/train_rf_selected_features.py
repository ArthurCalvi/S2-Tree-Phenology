import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
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
from typing import List, Optional

# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import utility functions
from src.utils import (
    unscale_feature,
    transform_circular_features,
    compute_metrics,
    create_eco_balanced_folds_df,
    display_fold_distribution,
    format_confusion_matrix
)

# --- Configuration ---

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/selected_features_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the path to the dataset
DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'

# Define the phenology mapping
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}

# Define the base indices needed for feature transformation
# Even if not all features from an index are used, the transformation needs the base index name
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

MAX_DEPTH = 30
N_ESTIMATORS = 50
MIN_SAMPLES_SPLIT = 30 # Example starting value
MIN_SAMPLES_LEAF = MIN_SAMPLES_SPLIT // 2
# --- Core Training Functions ---

def evaluate_features_cv(df, features, target='phenology', n_splits=5, id_columns: Optional[List[str]] = None):
    """
    Evaluate the given features using cross-validation.
    Returns performance metrics averaged over folds and aggregated confusion matrix.
    """
    id_columns = id_columns or []
    logger.info(f"Starting cross-validation evaluation for {len(features)} features")

    # Prepare data
    X = df[features]
    y = df[target]

    # Create eco-region balanced folds
    fold_splits = create_eco_balanced_folds_df(df, n_splits=n_splits, random_state=42)

    # Store results
    results_per_fold = []
    results_per_ecoregion = defaultdict(list)
    all_true = []
    all_pred = []

    predictions_chunks: List[pd.DataFrame] = []
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_splits, desc="Cross-validation folds")):
        # logger.info(f"=== Fold {fold+1}/{n_splits} ===") # Less verbose logging

        # Display fold distribution (optional, can be verbose)
        # display_fold_distribution(train_idx, val_idx, df, fold)

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Apply sample weights if available
        sample_weights = None
        if 'weight' in df.columns:
            sample_weights = df.iloc[train_idx]['weight'].values
            # logger.info(f"Using sample weights for training in fold {fold+1}") # Can be verbose

        # Train model
        # logger.info(f"Fold {fold+1}: Training RandomForest on {len(X_train)} samples...") # Less verbose logging
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        # Make predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None

        # Store predictions for overall confusion matrix
        all_true.extend(y_val)
        all_pred.extend(y_pred)

        # Collect per-sample validation predictions for downstream analysis
        val_indices = df.index[val_idx]
        if id_columns:
            val_meta = df.loc[val_indices, id_columns].copy()
        else:
            val_meta = pd.DataFrame(index=val_indices)
        val_meta = val_meta.reset_index().rename(columns={'index': 'sample_index'})
        val_meta['fold'] = fold + 1
        val_meta['y_true'] = y_val.values
        val_meta['y_pred'] = y_pred
        if y_proba is not None:
            class_labels = model.classes_
            for prob_idx, cls in enumerate(class_labels):
                val_meta[f'prob_class_{cls}'] = y_proba[:, prob_idx]
        predictions_chunks.append(val_meta)

        # Compute overall metrics for this fold using the updated compute_metrics
        overall_metrics = compute_metrics(y_val, y_pred)
        overall_metrics['fold'] = fold + 1
        results_per_fold.append(overall_metrics)

        # Log key metrics for the fold (e.g., macro F1)
        logger.info(f"Fold {fold+1}/{n_splits} -> Macro F1: {overall_metrics['f1_macro']:.4f}, Weighted F1: {overall_metrics['f1_weighted']:.4f}, Accuracy: {overall_metrics['accuracy']:.4f}")

        # Compute metrics per eco-region for this fold
        eco_regions_in_val = df.iloc[val_idx]['eco_region'].unique()
        for eco_region in eco_regions_in_val:
            eco_mask = df.iloc[val_idx]['eco_region'] == eco_region
            if sum(eco_mask) > 0:
                eco_y_val = y_val[eco_mask]
                eco_y_pred = y_pred[eco_mask]

                # Use the updated compute_metrics here as well
                eco_metrics = compute_metrics(eco_y_val, eco_y_pred)
                eco_metrics['fold'] = fold + 1
                eco_metrics['eco_region'] = eco_region
                eco_metrics['n_samples'] = len(eco_y_val)
                results_per_ecoregion[eco_region].append(eco_metrics)

                # logger.info(f"  {eco_region} Macro F1: {eco_metrics['f1_macro']:.4f} (on {len(eco_y_val)} samples)") # Can be verbose

    # Aggregate results across folds
    results_df = pd.DataFrame(results_per_fold)
    quantiles = {
        'q25': results_df.quantile(0.25, numeric_only=True).to_dict(),
        'median': results_df.quantile(0.50, numeric_only=True).to_dict(),
        'q75': results_df.quantile(0.75, numeric_only=True).to_dict(),
    }
    
    # Define metrics to average
    metrics_to_average = [
        'accuracy',
        'precision_deciduous', 'recall_deciduous', 'f1_deciduous',
        'precision_evergreen', 'recall_evergreen', 'f1_evergreen',
        'precision_macro', 'recall_macro', 'f1_macro',
        'precision_weighted', 'recall_weighted', 'f1_weighted'
    ]
    
    overall_metrics_summary = {}
    for metric in metrics_to_average:
        overall_metrics_summary[f'mean_{metric}'] = results_df[metric].mean()
        overall_metrics_summary[f'std_{metric}'] = results_df[metric].std()
    overall_metrics_summary['quantiles'] = quantiles

    logger.info("=== Overall CV Results (Mean ± Std across folds) ===")
    # Display key averaged metrics
    logger.info(f"Accuracy:           {overall_metrics_summary['mean_accuracy']:.4f} ± {overall_metrics_summary['std_accuracy']:.4f}")
    logger.info(f"F1 Macro:           {overall_metrics_summary['mean_f1_macro']:.4f} ± {overall_metrics_summary['std_f1_macro']:.4f}")
    logger.info(f"F1 Weighted:        {overall_metrics_summary['mean_f1_weighted']:.4f} ± {overall_metrics_summary['std_f1_weighted']:.4f}")
    logger.info(f"Precision Macro:    {overall_metrics_summary['mean_precision_macro']:.4f} ± {overall_metrics_summary['std_precision_macro']:.4f}")
    logger.info(f"Recall Macro:       {overall_metrics_summary['mean_recall_macro']:.4f} ± {overall_metrics_summary['std_recall_macro']:.4f}")
    logger.info(f"Precision Weighted: {overall_metrics_summary['mean_precision_weighted']:.4f} ± {overall_metrics_summary['std_precision_weighted']:.4f}")
    logger.info(f"Recall Weighted:    {overall_metrics_summary['mean_recall_weighted']:.4f} ± {overall_metrics_summary['std_recall_weighted']:.4f}")
    logger.info(f"F1 Deciduous:       {overall_metrics_summary['mean_f1_deciduous']:.4f} ± {overall_metrics_summary['std_f1_deciduous']:.4f}")
    logger.info(f"F1 Evergreen:       {overall_metrics_summary['mean_f1_evergreen']:.4f} ± {overall_metrics_summary['std_f1_evergreen']:.4f}")

    # Aggregate eco-region results
    eco_results_list = []
    for eco_region, metrics_list in results_per_ecoregion.items():
        metrics_df_eco = pd.DataFrame(metrics_list)
        avg_metrics = {'eco_region': eco_region}
        # Calculate mean and std for all relevant metrics per eco-region
        for metric in metrics_to_average:
             avg_metrics[f'{metric}_mean'] = metrics_df_eco[metric].mean()
             avg_metrics[f'{metric}_std'] = metrics_df_eco[metric].std()
        avg_metrics['n_samples_mean'] = metrics_df_eco['n_samples'].mean() # Avg samples per fold
        avg_metrics['n_samples_total'] = df[df['eco_region'] == eco_region].shape[0] # Total samples in dataset
        eco_results_list.append(avg_metrics)

    eco_results_df = pd.DataFrame(eco_results_list).sort_values('f1_macro_mean', ascending=False) # Sort by mean macro F1
    logger.info("=== Average Results per Eco-Region (Across Folds) ===")
    # Select and format columns for tabulation - showing macro and weighted F1
    cols_to_show = ['eco_region', 'n_samples_total', 'f1_macro_mean', 'f1_macro_std', 'f1_weighted_mean', 'f1_weighted_std', 'accuracy_mean']
    # Rename columns for better table display
    col_rename_map = {
        'f1_macro_mean': 'F1 Macro Mean', 'f1_macro_std': 'F1 Macro Std',
        'f1_weighted_mean': 'F1 Wgt Mean', 'f1_weighted_std': 'F1 Wgt Std',
        'accuracy_mean': 'Acc Mean', 'n_samples_total': 'N Samples'
    }
    table_data = eco_results_df[cols_to_show].rename(columns=col_rename_map).round(4)
    # Use tabulate for formatted output
    logger.info("\n" + tabulate(table_data, headers='keys', tablefmt='psql', showindex=False))

    # Generate aggregated confusion matrix (using all predictions across folds)
    aggregated_cm_array = confusion_matrix(all_true, all_pred, labels=[1, 2])
    cm_text = format_confusion_matrix(aggregated_cm_array, labels=[f'{PHENOLOGY_MAPPING[1]} (1)', f'{PHENOLOGY_MAPPING[2]} (2)'])
    logger.info("\nAggregated Confusion Matrix (All Folds):")
    logger.info("" + cm_text)

    # Add aggregated CM values to the overall summary
    tn, fp, fn, tp = aggregated_cm_array.ravel()
    overall_metrics_summary['aggregated_tn'] = int(tn)
    overall_metrics_summary['aggregated_fp'] = int(fp)
    overall_metrics_summary['aggregated_fn'] = int(fn)
    overall_metrics_summary['aggregated_tp'] = int(tp)

    fold_metrics_df = results_df.reset_index(drop=True)
    predictions_df = pd.concat(predictions_chunks, ignore_index=True) if predictions_chunks else pd.DataFrame()

    # Return the detailed summary dictionary and the eco-region dataframe
    return overall_metrics_summary, eco_results_df, aggregated_cm_array, fold_metrics_df, predictions_df


def train_final_model(df, features, target='phenology'):
    """
    Train a final RandomForest model on the entire dataset using the specified features.
    """
    logger.info(f"Training final model on {len(df)} samples using {len(features)} features...")

    X = df[features]
    y = df[target]

    sample_weights = None
    if 'weight' in df.columns:
        sample_weights = df['weight'].values
        logger.info("Using sample weights for final model training.")
        weight_stats = {
            'min': sample_weights.min(), 'max': sample_weights.max(),
            'mean': sample_weights.mean(), 'median': np.median(sample_weights),
            'non_zero': np.sum(sample_weights > 0), 'total': len(sample_weights)
        }
        logger.info(f"Sample weights stats: {weight_stats}")

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF)
    model.fit(X, y, sample_weight=sample_weights)

    logger.info("Final model training complete.")
    return model

def count_rf_parameters(rf_model):
    """
    Count the number of parameters in a RandomForest model.
    """
    n_estimators = len(rf_model.estimators_)
    total_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
    
    # Parameters per node include:
    # - Feature index (1 param)
    # - Threshold value (1 param) 
    # - Values array (with class probabilities for each node)
    n_classes = rf_model.n_classes_
    params_per_node = 2 + n_classes  # feature index + threshold + values
    
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
    parser = argparse.ArgumentParser(description='Train phenology classification model with selected features.')
    parser.add_argument('--features', '-f', type=str, required=True,
                        help='Comma-separated list of feature names to use for training.')
    parser.add_argument('--output_dir', '-o', type=str, default='results/final_model',
                        help='Directory to save the model and metrics (default: results/final_model).')
    parser.add_argument('--model_name', '-m', type=str, default='phenology_model.joblib',
                        help='Filename for the saved model (default: phenology_model.joblib).')
    parser.add_argument('--metrics_name', '-j', type=str, default='phenology_metrics.json',
                        help='Filename for the overall metrics JSON output (default: phenology_metrics.json).')
    parser.add_argument('--eco_metrics_name', '-e', type=str, default='phenology_eco_metrics.csv',
                        help='Filename for the eco-region metrics CSV output (default: phenology_eco_metrics.csv).')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Run in test mode with a small subset of data.')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of samples to use in test mode (default: 10000).')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation splits (default: 5).')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help=f'Path to the dataset parquet file (default: {DATASET_PATH}).')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting selected features training script")
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
            # Ensure balanced sampling if possible
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
    logger.info(f"Requested features: {selected_features}")

    # Check if all requested features exist in the dataframe after transformation
    available_features = df.columns.tolist()
    missing_features = [f for f in selected_features if f not in available_features]
    if missing_features:
        logger.error(f"Error: The following requested features are not found in the dataset (after transformation): {missing_features}")
        logger.error(f"Available features include: {available_features}")
        return

    logger.info(f"Using {len(selected_features)} features for training and evaluation.")

    # Modify model name to include date
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    if not args.model_name.endswith('.joblib'):
        args.model_name += '.joblib'
    base_model_name = args.model_name.replace('.joblib', '')
    dated_model_name = f"{base_model_name}_{current_date}.joblib"
    
    id_columns = [col for col in ['tile_id', 'row', 'col', 'x', 'y', 'utm_x', 'utm_y', 'eco_region', 'NomSER', 'weight']
                  if col in df.columns]

    # --- Cross-Validation Evaluation ---
    start_time_cv = time.time()
    overall_metrics_summary, eco_results_df, _, fold_metrics_df, predictions_df = evaluate_features_cv(
        df, selected_features, target='phenology', n_splits=args.n_splits, id_columns=id_columns
    )
    cv_time = time.time() - start_time_cv
    logger.info(f"Cross-validation evaluation completed in {cv_time:.2f} seconds")

    # --- Final Model Training ---
    start_time_train = time.time()
    final_model = train_final_model(df, selected_features, target='phenology')
    train_time = time.time() - start_time_train
    logger.info(f"Final model training completed in {train_time:.2f} seconds")

    # Count model parameters
    model_params = count_rf_parameters(final_model)
    logger.info(f"Model parameters: {model_params['total_parameters']} (across {model_params['n_nodes']} nodes in {model_params['n_estimators']} trees)")

    # Add timings and parameters to metrics
    overall_metrics_summary['cv_evaluation_time_seconds'] = round(cv_time, 2)
    overall_metrics_summary['final_model_training_time_seconds'] = round(train_time, 2)
    overall_metrics_summary['selected_features'] = selected_features # Record features used
    overall_metrics_summary['model_parameters'] = model_params
    timestamp_utc = datetime.datetime.utcnow().isoformat() + 'Z'
    overall_metrics_summary['timestamp'] = timestamp_utc
    overall_metrics_summary['dataset_path'] = args.dataset_path
    overall_metrics_summary['n_splits'] = args.n_splits
    overall_metrics_summary['features'] = selected_features
    overall_metrics_summary['model_file'] = dated_model_name
    overall_metrics_summary['config_file'] = f"{base_model_name}_{current_date}_config.json"
    overall_metrics_summary['metrics_file'] = args.metrics_name
    overall_metrics_summary['eco_metrics_file'] = args.eco_metrics_name
    fold_metrics_records = json.loads(fold_metrics_df.to_json(orient='records'))
    overall_metrics_summary['fold_metrics'] = fold_metrics_records

    # --- Save Outputs ---

    # Save final model
    model_path = os.path.join(args.output_dir, dated_model_name)
    try:
        joblib.dump(final_model, model_path)
        logger.info(f"Final model saved successfully to: {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")

    # Create config file with model parameters
    config_name = f"{base_model_name}_{current_date}_config.json"
    config_path = os.path.join(args.output_dir, config_name)
    
    config_data = {
        'model_name': dated_model_name,
        'creation_date': current_date,
        'model_type': 'RandomForestClassifier',
        'n_estimators': model_params['n_estimators'],
        'total_parameters': model_params['total_parameters'],
        'n_nodes': model_params['n_nodes'],
        'selected_features': selected_features,
        'feature_count': len(selected_features),
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        logger.info(f"Model config saved successfully to: {config_path}")
    except Exception as e:
        logger.error(f"Error saving model config to {config_path}: {e}")

    # Persist per-fold metrics table
    fold_metrics_filename = f"{base_model_name}_{current_date}_fold_metrics.csv"
    fold_metrics_path = os.path.join(args.output_dir, fold_metrics_filename)
    try:
        fold_metrics_df.to_csv(fold_metrics_path, index=False, float_format='%.6f')
        logger.info(f"Fold metrics saved successfully to: {fold_metrics_path}")
        overall_metrics_summary['fold_metrics_file'] = fold_metrics_filename
    except Exception as e:
        logger.error(f"Error saving fold metrics to {fold_metrics_path}: {e}")

    # Persist CV predictions per sample for downstream analysis
    cv_predictions_filename = f"{base_model_name}_{current_date}_cv_predictions.parquet"
    cv_predictions_path = os.path.join(args.output_dir, cv_predictions_filename)
    if not predictions_df.empty:
        try:
            predictions_df.to_parquet(cv_predictions_path, index=False)
            logger.info(f"CV predictions saved successfully to: {cv_predictions_path}")
            overall_metrics_summary['cv_predictions_file'] = cv_predictions_filename
        except Exception as e:
            logger.error(f"Error saving CV predictions to {cv_predictions_path}: {e}")
    else:
        logger.warning("No CV predictions were captured; skipping parquet export.")

    # Save overall metrics to JSON
    metrics_path = os.path.join(args.output_dir, args.metrics_name)
    try:
        with open(metrics_path, 'w') as f:
            json.dump(overall_metrics_summary, f, indent=4)
        logger.info(f"Overall metrics saved successfully to: {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {metrics_path}: {e}")

    # Save eco-region metrics to CSV
    eco_metrics_path = os.path.join(args.output_dir, args.eco_metrics_name)
    try:
        eco_results_df.to_csv(eco_metrics_path, index=False, float_format='%.4f')
        logger.info(f"Eco-region metrics saved successfully to: {eco_metrics_path}")
    except Exception as e:
        logger.error(f"Error saving eco-region metrics to {eco_metrics_path}: {e}")

    logger.info("Script completed successfully.")

if __name__ == "__main__":
    main() 
