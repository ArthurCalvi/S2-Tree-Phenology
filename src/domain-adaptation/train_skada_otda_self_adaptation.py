import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
import os
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

# Import utility functions and constants
from src.utils import (
    unscale_feature,
    transform_circular_features,
    compute_metrics,
    format_confusion_matrix,
    create_eco_balanced_folds_df
)
from src.constants import EFFECTIVE_FOREST_AREA_BY_REGION, ALL_ECO_REGIONS # Assuming ALL_ECO_REGIONS is defined

# Try importing skada components
try:
    from skada import make_da_pipeline, MultiLinearMongeAlignmentAdapter
except ImportError:
    logging.error("SKADA library or MultiLinearMongeAlignmentAdapter not found. Please install/update skada: pip install git+https://github.com/scikit-adaptation/skada")
    sys.exit(1)

# --- Configuration ---
log_dir = 'logs/domain_adaptation'
os.makedirs(log_dir, exist_ok=True)
log_file_name = f'skada_monge_self_adapt_training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, log_file_name)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'
DEFAULT_FEATURES = "ndvi_amplitude_h1,ndvi_phase_h1_cos,ndvi_phase_h1_sin,ndvi_phase_h2_sin,ndvi_offset,nbr_amplitude_h1,nbr_phase_h1_cos,nbr_phase_h2_cos,nbr_offset,nbr_var_residual,crswir_phase_h1_cos,crswir_phase_h2_cos,crswir_offset,crswir_var_residual"
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']
FEATURE_TYPES_TO_UNSCALE = {
    'amplitude_h1': 'amplitude', 'amplitude_h2': 'amplitude',
    'phase_h1': 'phase', 'phase_h2': 'phase',
    'offset': 'offset', 'var_residual': 'variance'
}
MAX_DEPTH = 30
N_ESTIMATORS = 50
MIN_SAMPLES_SPLIT = 30
MIN_SAMPLES_LEAF = MIN_SAMPLES_SPLIT // 2

# --- Core Functions ---

def create_eco_region_id_mapping(eco_regions_list):
    """Creates a mapping from eco-region name to a unique positive integer ID."""
    return {region: i + 1 for i, region in enumerate(sorted(list(set(eco_regions_list))))}

def generate_sample_domain_labels(df, eco_region_col, region_to_id_map, use_negative_ids=False):
    """Generates domain labels for the dataframe based on the eco_region_to_id_map."""
    multiplier = -1 if use_negative_ids else 1
    sample_domain = df[eco_region_col].apply(lambda region: region_to_id_map.get(region, 0) * multiplier).to_numpy()
    if np.all(sample_domain == 0):
        logger.warning(f"All generated domain labels are 0. Check {eco_region_col} and mapping. Used negative: {use_negative_ids}")
    return sample_domain

def train_skada_pipeline_monge(X, y, sample_domain, features):
    logger.info(f"Training SKADA pipeline (MultiLinearMongeAlignment + RF) on {len(features)} features.")
    logger.info(f"Total samples for training: {len(X)}")
    unique_domains, counts = np.unique(sample_domain, return_counts=True)
    for domain_id, count in zip(unique_domains, counts):
        logger.info(f"  Training Domain ID {domain_id}: {count} samples")

    base_estimator = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1, class_weight='balanced',
        max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF, verbose=0
    )
    adapter = MultiLinearMongeAlignmentAdapter()
    pipeline = make_da_pipeline(adapter, base_estimator)
    pipeline.fit(X, y, sample_domain=sample_domain)
    logger.info("SKADA pipeline (MultiLinearMongeAlignment+RF) training complete.")
    return pipeline

def evaluate_skada_on_validation_set(pipeline, df_val, selected_features, region_to_id_map, target_col='phenology', eco_region_col='eco_region'):
    logger.info(f"Evaluating SKADA pipeline on validation set across various eco-regions.")
    results_per_eco_region = []
    aggregated_true_y = []
    aggregated_pred_y = []
    
    unique_val_regions = sorted(df_val[eco_region_col].unique())

    for eco_region_name in unique_val_regions:
        logger.info(f"--- Evaluating on Eco-Region: {eco_region_name} (Validation Set) ---")
        df_eco_val = df_val[df_val[eco_region_col] == eco_region_name].copy()

        if len(df_eco_val) == 0: # Should not happen if unique_val_regions is from df_val
            logger.warning(f"No validation samples for eco-region '{eco_region_name}'. Skipping.")
            continue

        X_val_eco = df_eco_val[selected_features].to_numpy()
        y_val_eco = df_eco_val[target_col].to_numpy()
        
        # For prediction, use the original positive region's ID
        prediction_domain_id = region_to_id_map.get(eco_region_name, 0) # Use positive ID
        if prediction_domain_id == 0:
            logger.error(f"Could not find ID for eco-region {eco_region_name} in mapping for prediction. Skipping.")
            continue
            
        predict_sample_domain = np.full(len(X_val_eco), prediction_domain_id, dtype=int)
        
        logger.info(f"Predicting on {len(X_val_eco)} samples from '{eco_region_name}' (using original Domain ID: {prediction_domain_id})")
        y_pred_eco = pipeline.predict(X_val_eco, sample_domain=predict_sample_domain)

        eco_metrics = compute_metrics(y_val_eco, y_pred_eco)
        eco_metrics['eco_region'] = eco_region_name
        eco_metrics['n_samples_val'] = len(y_val_eco)
        eco_metrics['domain_id_pred'] = int(prediction_domain_id)
        results_per_eco_region.append(eco_metrics)

        aggregated_true_y.extend(y_val_eco)
        aggregated_pred_y.extend(y_pred_eco)

        logger.info(f"Metrics for '{eco_region_name}' (Validation): Acc: {eco_metrics['accuracy']:.4f}, F1 Macro: {eco_metrics['f1_macro']:.4f}")
        cm_array_eco = confusion_matrix(y_val_eco, y_pred_eco, labels=[1, 2])
        logger.info(f"CM for '{eco_region_name}':\n{format_confusion_matrix(cm_array_eco, [PHENOLOGY_MAPPING[1], PHENOLOGY_MAPPING[2]])}")

    eco_results_df = pd.DataFrame(results_per_eco_region)
    if not eco_results_df.empty:
        logger.info("=== Summary Results per Eco-Region (SKADA MongeAlignment+RF Validation Set) ===")
        metrics_to_show = ['accuracy', 'f1_macro', 'f1_deciduous', 'f1_evergreen']
        cols_to_display = ['eco_region', 'n_samples_val'] + metrics_to_show
        col_rename_map = {'f1_macro': 'F1 Macro', 'accuracy': 'Acc', 'n_samples_val': 'N Val Samples'}
        table_data = eco_results_df[cols_to_display].rename(columns=col_rename_map).round(4)
        logger.info("\n" + tabulate(table_data, headers='keys', tablefmt='psql', showindex=False))

    overall_val_metrics = {}
    if aggregated_true_y and aggregated_pred_y:
        logger.info("=== Overall Aggregated Validation Metrics (All Eco-Regions) ===")
        overall_val_metrics = compute_metrics(np.array(aggregated_true_y), np.array(aggregated_pred_y))
        logger.info(f"Acc: {overall_val_metrics['accuracy']:.4f}, F1 Macro: {overall_val_metrics['f1_macro']:.4f}")
        agg_cm_array = confusion_matrix(np.array(aggregated_true_y), np.array(aggregated_pred_y), labels=[1, 2])
        tn, fp, fn, tp = agg_cm_array.ravel() if agg_cm_array.size == 4 else (0,0,0,0)
        overall_val_metrics.update({'agg_tn': int(tn), 'agg_fp': int(fp), 'agg_fn': int(fn), 'agg_tp': int(tp)})
        logger.info(f"Aggregated CM:\n{format_confusion_matrix(agg_cm_array, [PHENOLOGY_MAPPING[1], PHENOLOGY_MAPPING[2]])}")

    return overall_val_metrics, eco_results_df

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Train SKADA (MultiLinearMongeAlignment+RF) with self-adaptation setup and evaluate.')
    parser.add_argument('--features', '-f', type=str, default=DEFAULT_FEATURES, help=f'Features. Default: "{DEFAULT_FEATURES}"')
    parser.add_argument('--eco_regions_to_use', type=str, default=None, help='Comma-separated eco-regions. Default: All in constants.ALL_ECO_REGIONS')
    parser.add_argument('--output_dir', '-o', type=str, default='results/domain_adaptation/skada_monge_self_adapt')
    parser.add_argument('--model_name', '-m', type=str, default='skada_monge_self_adapt_pipeline.joblib')
    parser.add_argument('--metrics_name', '-j', type=str, default='skada_monge_self_adapt_overall_val_metrics.json')
    parser.add_argument('--eco_metrics_name', '-e', type=str, default='skada_monge_self_adapt_val_eco_metrics.csv')
    parser.add_argument('--load_pipeline', type=str, default=None, help='Load pre-trained pipeline.')
    parser.add_argument('--run_test_mode', action='store_true', help='Small subset of data for test.')
    parser.add_argument('--test_size_per_region', type=int, default=1000, help='Samples per region in test mode.')
    parser.add_argument('--train_split_ratio', type=float, default=0.8, help='Train set ratio (0.0-1.0). Ignored if n_splits > 1.')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for train/test split and fold generation.')
    parser.add_argument('--n_splits', type=int, default=1, help='Number of CV splits. If 1, uses train_split_ratio for a single split.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Starting SKADA MultiLinearMongeAlignment Self-Adaptation Script. Args: {args}")

    regions_for_experiment = [r.strip() for r in args.eco_regions_to_use.split(',')] if args.eco_regions_to_use else ALL_ECO_REGIONS
    if not regions_for_experiment:
        logger.error("No eco-regions specified or found in constants. Exiting."); return
    logger.info(f"Eco-regions for experiment: {regions_for_experiment}")

    logger.info(f"Loading dataset from {args.dataset_path}...")
    try:
        df_full = pd.read_parquet(args.dataset_path)
    except Exception as e: logger.error(f"Failed to load dataset: {e}"); return
    logger.info(f"Full dataset loaded: {len(df_full)} samples")

    df_filtered = df_full[df_full['eco_region'].isin(regions_for_experiment)].copy()
    if args.run_test_mode:
        logger.info(f"TEST MODE: Sampling approx {args.test_size_per_region} per region.")
        df_filtered = df_filtered.groupby('eco_region').sample(n=args.test_size_per_region, random_state=args.random_seed, replace=True).reset_index(drop=True)
    logger.info(f"Data for experiment after filtering/sampling: {len(df_filtered)} samples from {df_filtered['eco_region'].nunique()} regions.")

    if df_filtered.empty:
        logger.error("No data after filtering for specified regions. Exiting."); return

    logger.info("Preprocessing features (Unscaling and Circular Transformation)...")
    df_processed = df_filtered.copy()
    unscaled_count = 0
    for index_name in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index_name}_{ftype_suffix}"
            if col_name in df_processed.columns:
                try:
                    df_processed[col_name] = unscale_feature(df_processed[col_name], feature_type, index_name)
                    unscaled_count += 1
                except Exception as e: logger.error(f"Error unscaling {col_name}: {e}")
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    df_processed = transform_circular_features(df_processed, INDICES)
    logger.info("Circular transformation complete.")

    selected_features = [f.strip() for f in args.features.split(',')]
    missing_features = [f for f in selected_features if f not in df_processed.columns]
    if missing_features:
        logger.error(f"Missing features after preprocessing: {missing_features}. Available: {df_processed.columns.tolist()}"); return
    logger.info(f"Using {len(selected_features)} features: {selected_features}")

    # Create eco-region to ID mapping
    eco_region_to_id_map = create_eco_region_id_mapping(df_processed['eco_region'].unique())
    logger.info(f"Eco-region to ID map: {eco_region_to_id_map}")

    # Conditional CV or single split
    if not args.load_pipeline and args.n_splits > 1:
        logger.info(f"Starting {args.n_splits}-fold Cross-Validation...")
        fold_splits = create_eco_balanced_folds_df(df_processed, n_splits=args.n_splits, random_state=args.random_seed)
        
        all_fold_overall_metrics = []
        all_fold_eco_metrics_dfs = []
        total_train_time = 0
        total_eval_time = 0

        for i, (train_idx, val_idx) in enumerate(fold_splits):
            logger.info(f"--- Fold {i+1}/{args.n_splits} ---")
            df_train_fold = df_processed.iloc[train_idx]
            df_val_fold = df_processed.iloc[val_idx]

            logger.info(f"Fold {i+1}: Train set: {len(df_train_fold)} samples, Validation set: {len(df_val_fold)} samples")
            if df_train_fold.empty or df_val_fold.empty:
                logger.error(f"Fold {i+1}: Train or validation set is empty. Skipping fold."); continue

            logger.info(f"Fold {i+1}: Preparing data for SKADA (MongeAlignment Self-Adapt) pipeline training...")
            X_train_da_fold = df_train_fold[selected_features].to_numpy()
            y_train_da_fold = df_train_fold['phenology'].to_numpy()
            sample_domain_train_fold = generate_sample_domain_labels(df_train_fold, 'eco_region', eco_region_to_id_map, use_negative_ids=False)
            
            if np.all(sample_domain_train_fold == 0):
                logger.error(f"Fold {i+1}: All training domain labels are 0. Skipping fold."); continue
            
            logger.info(f"Fold {i+1}: Training data shapes -> X: {X_train_da_fold.shape}, y: {y_train_da_fold.shape}, sample_domain: {sample_domain_train_fold.shape}")
            start_time_train_fold = time.time()
            pipeline_fold = train_skada_pipeline_monge(X_train_da_fold, y_train_da_fold, sample_domain_train_fold, selected_features)
            train_time_fold = time.time() - start_time_train_fold
            total_train_time += train_time_fold
            logger.info(f"Fold {i+1}: SKADA (MongeAlignment Self-Adapt) training completed in {train_time_fold:.2f}s")

            if pipeline_fold is None: 
                logger.error(f"Fold {i+1}: Pipeline training failed. Skipping evaluation for this fold."); continue

            logger.info(f"Fold {i+1}: Evaluating pipeline on validation set...")
            start_time_eval_fold = time.time()
            overall_val_metrics_fold, eco_results_df_fold = evaluate_skada_on_validation_set(
                pipeline_fold, df_val_fold, selected_features, eco_region_to_id_map, target_col='phenology'
            )
            eval_time_fold = time.time() - start_time_eval_fold
            total_eval_time += eval_time_fold
            logger.info(f"Fold {i+1}: Evaluation completed in {eval_time_fold:.2f}s")

            if overall_val_metrics_fold:
                overall_val_metrics_fold['fold'] = i + 1
                all_fold_overall_metrics.append(overall_val_metrics_fold)
            if not eco_results_df_fold.empty:
                eco_results_df_fold['fold'] = i + 1
                all_fold_eco_metrics_dfs.append(eco_results_df_fold)
        
        # Aggregate metrics from all folds
        if not all_fold_overall_metrics:
            logger.error("No metrics collected from any fold. Cannot aggregate.")
            overall_val_metrics = {}
            eco_results_df = pd.DataFrame()
        else:
            logger.info("=== Aggregating Cross-Validation Results ===")
            # Overall metrics aggregation
            overall_metrics_df = pd.DataFrame(all_fold_overall_metrics)
            mean_overall_metrics = overall_metrics_df.drop(columns=['fold']).mean().to_dict()
            std_overall_metrics = overall_metrics_df.drop(columns=['fold']).std().to_dict()
            overall_val_metrics = {f"mean_{k}": v for k, v in mean_overall_metrics.items()}
            overall_val_metrics.update({f"std_{k}": v for k, v in std_overall_metrics.items()})
            overall_val_metrics['n_folds_completed'] = len(all_fold_overall_metrics)
            logger.info("Mean Overall Validation Metrics (across folds):")
            for k, v in mean_overall_metrics.items():
                if isinstance(v, float): logger.info(f"  {k}: {v:.4f} (std: {std_overall_metrics.get(k, 0):.4f})")

            # Eco-region metrics aggregation
            if not all_fold_eco_metrics_dfs:
                eco_results_df = pd.DataFrame()
            else:
                concatenated_eco_df = pd.concat(all_fold_eco_metrics_dfs, ignore_index=True)
                eco_results_df = concatenated_eco_df.drop(columns=['fold']).groupby('eco_region').agg(['mean', 'std'])
                eco_results_df.columns = ['_'.join(col).strip() for col in eco_results_df.columns.values] # Flatten MultiIndex
                eco_results_df = eco_results_df.reset_index()
                logger.info("Mean Per-Eco-Region Validation Metrics (across folds):")
                logger.info("\n" + tabulate(eco_results_df.round(4), headers='keys', tablefmt='psql', showindex=False))
        
        # Use average times for summary
        train_time = total_train_time / args.n_splits if args.n_splits > 0 else 0
        eval_time = total_eval_time / args.n_splits if args.n_splits > 0 else 0
        pipeline = None # No single pipeline to save in CV mode

    # Single split / Load pipeline logic (remains largely the same)
    else: 
        if args.load_pipeline:
            logger.info(f"Loading pre-trained pipeline from: {args.load_pipeline}")
            if not os.path.exists(args.load_pipeline): 
                logger.error(f"Pipeline file not found: {args.load_pipeline}"); return
            try: 
                pipeline = joblib.load(args.load_pipeline)
                train_time = 0 # No training if loading
            except Exception as e: 
                logger.error(f"Error loading pipeline: {e}"); return
            
            # For evaluating a loaded model, we still need some validation data.
            # We'll use the full df_processed as validation, or a split if preferred.
            # For simplicity, let's use all df_processed for evaluation here, assuming it's a test set in this context.
            # Or, perform a split if a hold-out is desired for the loaded model's eval.
            logger.info(f"Splitting data for evaluating loaded pipeline ({args.train_split_ratio*100:.0f}% train-like / {(1-args.train_split_ratio)*100:.0f}% val-like)..._)")
            # We don't actually train, but split to get a df_val for consistency in evaluation call
            _, df_val_for_loaded_model = train_test_split(
                df_processed, test_size=(1-args.train_split_ratio), random_state=args.random_seed,
                stratify=df_processed[['eco_region', 'phenology']] if 'phenology' in df_processed.columns and 'eco_region' in df_processed.columns and len(df_processed['phenology'].unique()) > 1 and len(df_processed['eco_region'].unique()) > 1 else None
            )
            if df_val_for_loaded_model.empty:
                logger.warning("Validation set for loaded model is empty, using full processed data for evaluation.")
                df_val_for_loaded_model = df_processed
            
            start_time_eval = time.time()
            overall_val_metrics, eco_results_df = evaluate_skada_on_validation_set(
                pipeline, df_val_for_loaded_model, selected_features, eco_region_to_id_map, target_col='phenology'
            )
            eval_time = time.time() - start_time_eval

        else: # Single split (n_splits=1 or default)
            logger.info(f"Splitting data into train/validation ({args.train_split_ratio*100:.0f}%/{ (1-args.train_split_ratio)*100:.0f}%)..._)")
            try:
                df_train, df_val = train_test_split(
                    df_processed, test_size=(1 - args.train_split_ratio), random_state=args.random_seed,
                    stratify=df_processed[['eco_region', 'phenology']] # Stratify by both
                )
            except ValueError as e:
                logger.warning(f"Stratified split failed ({e}). Trying to stratify by 'eco_region' only.")
                try:
                    df_train, df_val = train_test_split(
                        df_processed, test_size=(1 - args.train_split_ratio), random_state=args.random_seed,
                        stratify=df_processed['eco_region']
                    )
                except ValueError as e2:
                    logger.error(f"Stratified split by eco_region also failed ({e2}). Using non-stratified split. Results may be skewed.")
                    df_train, df_val = train_test_split(df_processed, test_size=(1 - args.train_split_ratio), random_state=args.random_seed)

            logger.info(f"Train set: {len(df_train)} samples, Validation set: {len(df_val)} samples")
            if df_train.empty or df_val.empty:
                logger.error("Train or validation set is empty after split. Exiting."); return

            pipeline = None
            train_time = 0.0
            # ... (pipeline training as before for single split) ...
            logger.info("Preparing data for SKADA (MongeAlignment Self-Adapt) pipeline training...")
            X_train_da = df_train[selected_features].to_numpy()
            y_train_da = df_train['phenology'].to_numpy()
            sample_domain_train = generate_sample_domain_labels(df_train, 'eco_region', eco_region_to_id_map, use_negative_ids=False)
            
            if np.all(sample_domain_train == 0):
                logger.error("All training domain labels are 0. Cannot train. Check eco_region_id_mapping or data."); return
                
            logger.info(f"Training data shapes -> X: {X_train_da.shape}, y: {y_train_da.shape}, sample_domain: {sample_domain_train.shape}")
            start_time_train = time.time()
            pipeline = train_skada_pipeline_monge(X_train_da, y_train_da, sample_domain_train, selected_features)
            train_time = time.time() - start_time_train
            logger.info(f"SKADA (MongeAlignment Self-Adapt) training completed in {train_time:.2f}s")

            if pipeline is None: logger.error("Pipeline is not defined. Cannot proceed."); return

            logger.info("Evaluating SKADA (MongeAlignment Self-Adapt) pipeline on validation set...")
            start_time_eval = time.time()
            overall_val_metrics, eco_results_df = evaluate_skada_on_validation_set(
                pipeline, df_val, selected_features, eco_region_to_id_map, target_col='phenology'
            )
            eval_time = time.time() - start_time_eval
            logger.info(f"Evaluation on validation set completed in {eval_time:.2f}s")

    # --- Save Outputs (metrics, model, config) ---
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    if overall_val_metrics:
        overall_val_metrics['training_time_seconds'] = round(train_time, 2)
        overall_val_metrics['evaluation_time_seconds'] = round(eval_time, 2)
        # ... add other relevant info like in previous script ...
        metrics_path = os.path.join(args.output_dir, args.metrics_name)
        try:
            serializable_metrics = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in overall_val_metrics.items()}
            with open(metrics_path, 'w') as f: json.dump(serializable_metrics, f, indent=4)
            logger.info(f"Overall validation metrics saved to: {metrics_path}")
        except Exception as e: logger.error(f"Error saving overall val metrics: {e}")

    if not eco_results_df.empty:
        eco_metrics_path = os.path.join(args.output_dir, args.eco_metrics_name)
        try:
            eco_results_df.to_csv(eco_metrics_path, index=False, float_format='%.4f')
            logger.info(f"Per-eco-region validation metrics saved to: {eco_metrics_path}")
        except Exception as e: logger.error(f"Error saving eco val metrics: {e}")

    if not args.load_pipeline and pipeline is not None: # Only save if we trained a single pipeline
        dated_model_name = args.model_name.replace('.joblib', f'_{current_date}.joblib')
        pipeline_path = os.path.join(args.output_dir, dated_model_name)
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"Trained SKADA (MongeAlignment Self-Adapt) pipeline saved to: {pipeline_path}")
        
        config_path = os.path.join(args.output_dir, dated_model_name.replace('.joblib', '_config.json'))
        config_data = {
            'pipeline_name': dated_model_name, 'creation_date': current_date,
            'adapter': 'MultiLinearMongeAlignmentAdapter', 'adapter_params': {},
            'estimator': 'RandomForestClassifier', # ... other params ...
            'selected_features': selected_features, 'feature_count': len(selected_features),
            'training_setup': f'Domain Adaptation: SKADA (MultiLinearMongeAlignment+RF) Self-Adaptation on All Eco-Regions ({args.n_splits}-fold CV if >1 else "single split")',
            'eco_regions_used_in_experiment': list(eco_region_to_id_map.keys()),
            'train_split_ratio_if_single_split': args.train_split_ratio if args.n_splits <=1 else None,
            'n_cv_folds': args.n_splits if args.n_splits > 1 else None,
            'random_seed': args.random_seed,
            'n_samples_processed_total': len(df_processed)
        }
        with open(config_path, 'w') as f: json.dump(config_data, f, indent=4)
        logger.info(f"Pipeline config saved to: {config_path}")

    logger.info("Script completed.")

if __name__ == "__main__":
    main() 