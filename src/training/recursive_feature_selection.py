"""
Recursive Feature Selection for Tree Phenology Classification

This script performs recursive feature selection to find the optimal subset 
of features for tree phenology classification (deciduous vs. evergreen).

Key features:
- Uses Random Forest for feature importance assessment
- Implements recursive feature elimination with cross-validation (RFECV)
- Maintains eco-region balanced sampling in cross-validation folds
- Evaluates performance metrics for different feature counts
- Selects a minimum of 6 features
- Generates visualizations of feature importance and performance metrics

Usage:
    python recursive_feature_selection.py [--test] [--output OUTPUT_DIR]

Author: Generated based on existing codebase
Date: April 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.utils import shuffle
from tqdm import tqdm
import os
import logging
import time
import argparse
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.base import clone
import sys
from pathlib import Path
import math

# Add the project root directory to the Python path (matching other scripts)
try:
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environment)
    project_root = str(Path('.').resolve())
    if project_root not in sys.path:
        sys.path.append(project_root)

# --- Imports from src package ---
# Assuming the script is run as a module from the project root (python -m src.training...)
from src.utils import unscale_feature, transform_circular_features
# Import the necessary utility functions for folds and metrics
from src.utils import create_eco_balanced_folds_df, compute_metrics
# We might need constants directly later, add if needed:
# from src.constants import INDICES, PHENOLOGY_MAPPING, DATASET_PATH etc.

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/recursive_feature_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory if it doesn't exist
os.makedirs('results/feature_selection', exist_ok=True)

# Define the indices to test
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']

# Define the dataset path 
DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'

# Define the phenology mapping
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}

# Map feature suffixes to unscaling types
FEATURE_TYPES_TO_UNSCALE = {
    'amplitude_h1': 'amplitude',
    'amplitude_h2': 'amplitude',
    'phase_h1': 'phase', # Will be unscaled to radians [0, 2pi]
    'phase_h2': 'phase',
    'offset': 'offset',
    'var_residual': 'variance'
}

def get_all_features():
    """
    Get all features from all indices *after* circular transformation.
    """
    features = []
    for index in INDICES:
        features.extend([
            f'{index}_amplitude_h1',
            f'{index}_amplitude_h2',
            f'{index}_phase_h1_cos',
            f'{index}_phase_h1_sin',
            f'{index}_phase_h2_cos',
            f'{index}_phase_h2_sin',
            f'{index}_offset',
            f'{index}_var_residual'
        ])
    return features

class CustomRFECV:
    """
    Custom Recursive Feature Elimination with Cross-Validation 
    that preserves eco-region balanced folds.
    """
    def __init__(self, estimator, min_features_to_select=6, step=1, cv=5, scoring='f1_score'):
        self.estimator = estimator
        self.min_features_to_select = min_features_to_select
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.feature_importances_ = None
        self.ranking_ = None
        self.support_ = None
        self.n_features_ = None
        self.cv_results_ = None # Stores metrics per feature count
        self.fold_splits = None    # Store fold splits for later use
        self.best_model_eco_metrics = None # Store eco-region metrics for the best model

    def fit(self, X, y, df):
        """Fit the RFE model with cross-validation."""
        logger.info(f"Starting recursive feature elimination with min {self.min_features_to_select} features...")
        
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        
        # Create eco-region balanced folds using imported function
        logger.info("Creating eco-region balanced folds using src.utils...")
        self.fold_splits = create_eco_balanced_folds_df(df, n_splits=self.cv, random_state=42) # Pass random_state for consistency
        logger.info(f"Created {self.cv} folds successfully")
        
        # Initialize variables to store results
        self.cv_results_ = {
            'mean_test_score': [],
            'std_test_score': [],
            'n_features': [],
            'features': [],
            'metrics_per_fold': [],
            'feature_importances': []
        }
        
        # Start with all features
        support = np.ones(n_features, dtype=bool)
        current_n_features = n_features
        
        while current_n_features >= self.min_features_to_select: # Adjusted loop condition
            # Current feature set
            current_features = [f for i, f in enumerate(feature_names) if support[i]]
            
            logger.info(f"Evaluating with {current_n_features} features")
            # logger.debug(f"Current features: {current_features}") # Reduced verbosity
            
            # Cross-validation scores for this feature set
            fold_metrics = []
            fold_importances = []
            
            # For each fold
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(self.fold_splits, desc=f"Cross-validation with {current_n_features} features")):
                # logger.debug(f"Processing fold {fold_idx+1}/{self.cv} with {len(train_idx)} train, {len(val_idx)} val samples")
                
                # Split data
                X_train = X.iloc[train_idx][current_features]
                X_val = X.iloc[val_idx][current_features]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
                
                # logger.debug(f"Fold {fold_idx+1} - Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
                
                # Apply sample weights if available
                sample_weights = None
                if 'weight' in df.columns:
                    sample_weights = df.iloc[train_idx]['weight'].values
                
                # Train model
                # logger.debug(f"Fold {fold_idx+1} - Training Random Forest model...")
                model = clone(self.estimator) # Clone estimator for each fold
                model.fit(X_train, y_train, sample_weight=sample_weights)
                # logger.debug(f"Fold {fold_idx+1} - Model training completed")
                
                # Store feature importance
                fold_importances.append(pd.Series(model.feature_importances_, index=current_features))
                
                # Make predictions
                # logger.debug(f"Fold {fold_idx+1} - Making predictions on validation set...")
                y_pred = model.predict(X_val)
                # logger.debug(f"Fold {fold_idx+1} - Predictions completed")
                
                # Compute metrics immediately using imported function
                metrics = compute_metrics(y_val, y_pred)
                metrics['fold'] = fold_idx + 1
                fold_metrics.append(metrics)
                # logger.debug(f"Fold {fold_idx+1} - F1 Score: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
            # Calculate average scores across folds
            fold_metrics_df = pd.DataFrame(fold_metrics)
            mean_score = fold_metrics_df[self.scoring].mean()
            std_score = fold_metrics_df[self.scoring].std()
            
            # Average feature importances across folds
            avg_importance = pd.concat(fold_importances, axis=1).mean(axis=1)
            
            # Store results
            self.cv_results_['mean_test_score'].append(mean_score)
            self.cv_results_['std_test_score'].append(std_score)
            self.cv_results_['n_features'].append(current_n_features)
            self.cv_results_['features'].append(current_features)
            self.cv_results_['metrics_per_fold'].append(fold_metrics_df)
            self.cv_results_['feature_importances'].append(avg_importance)
            
            logger.info(f"  {current_n_features} Features -> F1 Score: {mean_score:.4f} ± {std_score:.4f}")
            
            # If we're at the minimum number of features, break the loop *after* evaluating it
            if current_n_features == self.min_features_to_select:
                break
            
            # Calculate number of features to eliminate for the *next* iteration
            if isinstance(self.step, list):
                # Get appropriate step size based on how many features remain
                completed_evaluations = len(self.cv_results_['n_features']) # How many loops done
                # Use the step size corresponding to the *next* loop (index = completed_evaluations)
                # Clamp index to avoid going out of bounds if more loops run than steps defined
                step_idx = min(completed_evaluations, len(self.step) - 1) 
                n_to_eliminate = self.step[step_idx]
                # logger.debug(f"Using step size {n_to_eliminate} (step index {step_idx} for next iteration)")
            elif isinstance(self.step, int):
                n_to_eliminate = self.step
            else: # Assuming fraction
                n_to_eliminate = max(1, int(current_n_features * self.step))

            # Ensure we don't eliminate more features than allowed to reach min_features
            n_to_eliminate = min(n_to_eliminate, current_n_features - self.min_features_to_select)
            # Ensure we don't try to eliminate 0 features if step calculation resulted in 0
            n_to_eliminate = max(1, n_to_eliminate) 

            # Get feature importances for current feature set
            supported_indices = np.where(support)[0]
            current_feature_names = [feature_names[i] for i in supported_indices]
            
            # Map average importances to the supported indices
            current_importances = np.array([avg_importance.get(name, 0) for name in current_feature_names]) # Use .get for safety

            # Find the indices of the features to eliminate
            indices_to_eliminate_relative = np.argsort(current_importances)[:n_to_eliminate]
            
            # Map these relative indices back to the original feature indices
            original_indices_to_eliminate = supported_indices[indices_to_eliminate_relative]

            # Update support mask
            support[original_indices_to_eliminate] = False
            current_n_features = np.sum(support) # Update the count for the next loop iteration
            
            eliminated_names = [feature_names[i] for i in original_indices_to_eliminate]
            # logger.debug(f"Eliminated {len(original_indices_to_eliminate)} features: {eliminated_names}. Remaining: {current_n_features}")
        
        # Store final results based on best score
        self.ranking_ = np.zeros(n_features, dtype=int)
        # Rank features based on when they were eliminated (lower rank = eliminated later/more important)
        elimination_order = {}
        rank_counter = 1
        # Iterate backwards through the elimination process
        for i in range(len(self.cv_results_['n_features']) - 1, -1, -1):
             current_eval_features = set(self.cv_results_['features'][i])
             prev_eval_features = set(self.cv_results_['features'][i-1]) if i > 0 else set(feature_names)
             eliminated_in_this_step = prev_eval_features - current_eval_features
             for feature in eliminated_in_this_step:
                 if feature not in elimination_order:
                     elimination_order[feature] = rank_counter
             rank_counter += len(eliminated_in_this_step)
        # Assign rank 1 to features never eliminated
        final_selected_features = set(self.cv_results_['features'][-1])
        for feature in final_selected_features:
             if feature not in elimination_order:
                  elimination_order[feature] = rank_counter
                  
        # Fill self.ranking_ based on computed order (lower value is better rank)
        max_rank = rank_counter # Total number of ranks assigned
        for i, feature in enumerate(feature_names):
            # Rank based on elimination order (higher value = eliminated earlier = worse rank)
            # We want higher rank number for less important features
            self.ranking_[i] = elimination_order.get(feature, max_rank) 


        # Find best feature count based on mean F1 score
        best_idx = np.argmax(self.cv_results_['mean_test_score'])
        self.n_features_ = self.cv_results_['n_features'][best_idx]
        best_features = self.cv_results_['features'][best_idx]
        
        # Set support for best feature set
        self.support_ = np.zeros(n_features, dtype=bool)
        for feature in best_features:
            idx = feature_names.index(feature)
            self.support_[idx] = True
        
        # Store feature importances from the best model's CV runs
        self.feature_importances_ = np.zeros(n_features)
        best_importances_series = self.cv_results_['feature_importances'][best_idx]
        for feature, importance in best_importances_series.items():
            idx = feature_names.index(feature)
            self.feature_importances_[idx] = importance
        
        logger.info(f"Best number of features: {self.n_features_} (based on F1 score)")
        logger.info(f"Selected features: {best_features}")

        # For the best model, calculate metrics per eco-region
        logger.info(f"Calculating eco-region metrics for the best model ({self.n_features_} features)...")
        self.best_model_eco_metrics = self.calculate_eco_region_metrics(X, y, df, best_features)

        return self
    
    def calculate_eco_region_metrics(self, X, y, df, best_features):
        """Calculate metrics per eco-region using the best features across CV folds."""
        eco_metrics_accumulator = defaultdict(list) # Store metrics list per region
        eco_regions = df['eco_region'].unique()
        
        # Iterate through each fold's results
        for fold_idx, (train_idx, val_idx) in enumerate(self.fold_splits):
            # Train a model *once* on this fold's training data
            X_train_fold = X.iloc[train_idx][best_features]
            y_train_fold = y.iloc[train_idx]
            model_fold = clone(self.estimator)
            
            # Apply sample weights if available for this fold's training
            sample_weights_fold = None
            if 'weight' in df.columns:
                sample_weights_fold = df.iloc[train_idx]['weight'].values
            model_fold.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)

            # Evaluate on validation subsets for each eco-region present in this fold's val set
            df_val_fold = df.iloc[val_idx]
            X_val_fold = X.iloc[val_idx][best_features]
            y_val_fold = y.iloc[val_idx]
            y_pred_fold = model_fold.predict(X_val_fold)

            for eco_region in df_val_fold['eco_region'].unique():
                # Get mask for this eco-region *within this fold's validation set*
                eco_mask_val_fold = (df_val_fold['eco_region'] == eco_region)
                
                if np.sum(eco_mask_val_fold) == 0:
                    continue # Should not happen based on loop logic, but safe check

                y_val_eco = y_val_fold[eco_mask_val_fold]
                y_pred_eco = y_pred_fold[eco_mask_val_fold]
                
                # Calculate metrics for this eco-region in this fold
                if len(np.unique(y_val_eco)) > 0: # Need at least one label
                    metrics = compute_metrics(y_val_eco, y_pred_eco)
                    metrics['n_samples_fold'] = len(y_val_eco) # Samples for this region IN THIS FOLD
                    eco_metrics_accumulator[eco_region].append(metrics)

        # Average metrics across folds for each eco-region
        final_eco_metrics = {}
        for eco_region, metrics_list in eco_metrics_accumulator.items():
            if metrics_list:
                # Calculate total samples for this eco-region across all validation folds
                total_samples_eco = df[df['eco_region'] == eco_region].shape[0]
                
                # Use np.mean, handling potential NaNs if a metric couldn't be calculated in a fold
                avg_f1 = np.nanmean([m['f1_score'] for m in metrics_list])
                avg_precision = np.nanmean([m['precision'] for m in metrics_list])
                avg_recall = np.nanmean([m['recall'] for m in metrics_list])
                avg_accuracy = np.nanmean([m['accuracy'] for m in metrics_list])
                
                # Sum TP, FP, TN, FN across folds for aggregate CM values per eco-region
                total_tp = np.sum([m['tp'] for m in metrics_list])
                total_fp = np.sum([m['fp'] for m in metrics_list])
                total_tn = np.sum([m['tn'] for m in metrics_list])
                total_fn = np.sum([m['fn'] for m in metrics_list])

                final_eco_metrics[eco_region] = {
                    'f1_score': avg_f1,
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'accuracy': avg_accuracy, # Add average accuracy
                    'tp': total_tp, # Use summed CM values
                    'fp': total_fp,
                    'tn': total_tn,
                    'fn': total_fn,
                    'n_samples': total_samples_eco # Total samples for this region in the dataset
                }
        
        return final_eco_metrics

def plot_feature_selection_results(rfecv, features, output_dir='results/feature_selection'):
    """
    Plot the results of feature selection.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Plot number of features vs. cross-validation score
    plt.figure(figsize=(12, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel(f"Cross validation score ({rfecv.scoring})") # Use actual scoring metric
    plt.errorbar(
        rfecv.cv_results_['n_features'],
        rfecv.cv_results_['mean_test_score'],
        yerr=rfecv.cv_results_['std_test_score'],
        fmt='o-'
    )
    # Highlight the best score
    best_idx = np.argmax(rfecv.cv_results_['mean_test_score'])
    best_n = rfecv.cv_results_['n_features'][best_idx]
    best_score = rfecv.cv_results_['mean_test_score'][best_idx]
    plt.plot(best_n, best_score, 'r*', markersize=15, label=f'Best ({best_n} features)')
    plt.legend()
    plt.grid(True)
    plt.title("Feature Selection Cross-Validation Scores")
    plt.savefig(output_dir_path / "feature_selection_cv_scores.png")
    plt.close()
    
    # Plot feature importance of selected features (from the best model)
    if rfecv.support_ is not None and rfecv.feature_importances_ is not None:
        selected_indices = np.where(rfecv.support_)[0]
        # Ensure features list matches the original feature order
        all_feature_names = features # Assuming 'features' passed is the original full list
        selected_features = [all_feature_names[i] for i in selected_indices]
        # Select importances corresponding to the support mask
        selected_importances = rfecv.feature_importances_[selected_indices]
        
        if len(selected_features) > 0:
            # Sort by importance
            sorted_idx = np.argsort(selected_importances)[::-1]
            sorted_features = [selected_features[i] for i in sorted_idx]
            sorted_importances = selected_importances[sorted_idx]
            
            plt.figure(figsize=(12, max(6, len(sorted_features) * 0.4))) # Adjust height
            plt.barh(range(len(sorted_features)), sorted_importances, align='center')
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Relative Importance (Gini)')
            plt.title(f'Feature Importance of Selected {len(sorted_features)} Features (Best Model)')
            plt.gca().invert_yaxis()  # Show highest values at the top
            plt.tight_layout()
            plt.savefig(output_dir_path / "selected_features_importance.png")
            plt.close()
        else:
            logger.warning("No features were selected, cannot plot importance.")
    else:
         logger.warning("Support or feature_importances_ not available in RFECV object, cannot plot importance.")

    
    # Extract and plot metrics per feature count
    metrics_data = []
    for i, n_feat in enumerate(rfecv.cv_results_['n_features']):
        fold_metrics_df = rfecv.cv_results_['metrics_per_fold'][i]
        # Add n_features column to each fold's metrics
        fold_metrics_df['n_features'] = n_feat
        metrics_data.append(fold_metrics_df)
    
    # Concatenate all fold metrics into a single DataFrame
    if metrics_data:
         metrics_df = pd.concat(metrics_data, ignore_index=True)
    else:
         metrics_df = pd.DataFrame() # Handle case with no results

    if not metrics_df.empty:
        # Plot f1, precision, recall by feature count (with error bars)
        plt.figure(figsize=(14, 8))
        
        # Group by number of features and calculate mean and std for each metric
        grouped = metrics_df.groupby('n_features')
        means = grouped.mean()
        stds = grouped.std()
        
        x = means.index.astype(int) # Ensure x-axis is integer
        
        plt.errorbar(x, means['f1_score'], yerr=stds['f1_score'], fmt='o-', label='F1 Score')
        plt.errorbar(x, means['precision'], yerr=stds['precision'], fmt='s-', label='Precision')
        plt.errorbar(x, means['recall'], yerr=stds['recall'], fmt='^-', label='Recall')
        
        plt.xlabel('Number of Features')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Feature Count (Mean ± Std over CV Folds)')
        plt.xticks(x) # Ensure ticks match number of features
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir_path / "performance_metrics_by_feature_count.png")
        plt.close()
        
        # Plot confusion matrix metrics (tp, tn, fp, fn) summed over folds by feature count
        # Calculate sums for CM components per feature count
        cm_sums = grouped[['tp', 'fp', 'tn', 'fn']].sum()

        fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
        metrics_to_plot = [('tp', 'Total True Positives'), ('fp', 'Total False Positives'), 
                           ('tn', 'Total True Negatives'), ('fn', 'Total False Negatives')]
        
        for i, (metric, title) in enumerate(metrics_to_plot):
            row, col = i // 2, i % 2
            axs[row, col].plot(cm_sums.index.astype(int), cm_sums[metric], 'o-')
            axs[row, col].set_title(title)
            axs[row, col].grid(True)
            axs[row, col].set_ylabel('Total Count across Folds')
            if row == 1: # Add x-label only to bottom plots
                 axs[row, col].set_xlabel('Number of Features')
                 axs[row, col].set_xticks(cm_sums.index.astype(int)) # Ensure ticks match feature counts

        plt.suptitle('Aggregated Confusion Matrix Components by Feature Count', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir_path / "confusion_matrix_metrics_by_feature_count.png")
        plt.close()
        
        # Create a summary table of mean metrics
        summary_table = means[['f1_score', 'precision', 'recall', 'accuracy']].copy()
        
        # Add standard deviations for key scores
        for col in ['f1_score', 'precision', 'recall', 'accuracy']:
            summary_table[f'{col}_std'] = stds[col]
            
        # Add summed CM components
        summary_table = summary_table.join(cm_sums)

        # Reorder columns for clarity
        summary_cols = ['f1_score', 'f1_score_std', 'precision', 'precision_std', 
                        'recall', 'recall_std', 'accuracy', 'accuracy_std',
                        'tp', 'fp', 'tn', 'fn']
        summary_table = summary_table[summary_cols].sort_index(ascending=False) # Show highest features first
        summary_table.to_csv(output_dir_path / "feature_selection_metrics_summary.csv", float_format='%.4f')
        
        # Save the raw metrics per fold per feature count
        metrics_df.to_csv(output_dir_path / "feature_selection_metrics_raw.csv", index=False, float_format='%.4f')

    else:
        logger.warning("Metrics DataFrame is empty, cannot generate plots or summary tables.")
        summary_table = pd.DataFrame() # Return empty dataframe

    # Return the per-fold metrics df and the summary table
    return metrics_df, summary_table

def create_feature_selection_report(rfecv, features, metrics_df, summary_table, eco_region_metrics, output_path='results/feature_selection/feature_selection_report.pdf'):
    """Create a comprehensive PDF report of feature selection results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    with PdfPages(output_path) as pdf:
        # --- Title Page ---
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.6, 'Tree Phenology Classification', ha='center', va='center', fontsize=26, fontweight='bold')
        plt.text(0.5, 0.5, 'Recursive Feature Elimination (RFECV) Report', ha='center', va='center', fontsize=22)
        plt.text(0.5, 0.35, f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}', ha='center', va='center', fontsize=14)
        plt.text(0.5, 0.3, f'Scoring Metric: {rfecv.scoring}', ha='center', va='center', fontsize=14)
        plt.text(0.5, 0.25, f'Minimum Features: {rfecv.min_features_to_select}', ha='center', va='center', fontsize=14)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # --- CV Scores Plot ---
        fig = plt.figure(figsize=(12, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel(f"Cross validation score ({rfecv.scoring})")
        plt.errorbar(
            rfecv.cv_results_['n_features'],
            rfecv.cv_results_['mean_test_score'],
            yerr=rfecv.cv_results_['std_test_score'],
            fmt='o-', label='Mean Score ± Std Dev'
        )
        # Highlight the best score
        best_idx = np.argmax(rfecv.cv_results_['mean_test_score'])
        best_n = rfecv.cv_results_['n_features'][best_idx]
        best_score = rfecv.cv_results_['mean_test_score'][best_idx]
        plt.plot(best_n, best_score, 'r*', markersize=15, label=f'Best Score ({best_n} features)')
        plt.xticks(rfecv.cv_results_['n_features'])
        plt.legend()
        plt.grid(True)
        plt.title("Feature Selection Cross-Validation Scores")
        pdf.savefig(fig)
        plt.close(fig)
        
        # --- Selected Features Importance ---
        if rfecv.support_ is not None and rfecv.feature_importances_ is not None:
            selected_indices = np.where(rfecv.support_)[0]
            all_feature_names = features
            selected_features = [all_feature_names[i] for i in selected_indices]
            selected_importances = rfecv.feature_importances_[selected_indices]
            
            if len(selected_features) > 0:
                sorted_idx = np.argsort(selected_importances)[::-1]
                sorted_features = [selected_features[i] for i in sorted_idx]
                sorted_importances = selected_importances[sorted_idx]
                
                fig = plt.figure(figsize=(12, max(6, len(sorted_features) * 0.4)))
                plt.barh(range(len(sorted_features)), sorted_importances, align='center')
                plt.yticks(range(len(sorted_features)), sorted_features)
                plt.xlabel('Relative Importance (Gini)')
                plt.title(f'Feature Importance of Selected {len(sorted_features)} Features (Best Model)')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        
        # --- Performance Metrics Plot ---
        if not metrics_df.empty:
            grouped = metrics_df.groupby('n_features')
            means = grouped.mean()
            stds = grouped.std()
            x = means.index.astype(int)
            
            fig = plt.figure(figsize=(14, 8))
            plt.errorbar(x, means['f1_score'], yerr=stds['f1_score'], fmt='o-', label='F1 Score')
            plt.errorbar(x, means['precision'], yerr=stds['precision'], fmt='s-', label='Precision')
            plt.errorbar(x, means['recall'], yerr=stds['recall'], fmt='^-', label='Recall')
            plt.xlabel('Number of Features')
            plt.ylabel('Score')
            plt.title('Performance Metrics by Feature Count (Mean ± Std over CV Folds)')
            plt.xticks(x)
            plt.legend()
            plt.grid(True)
            pdf.savefig(fig)
            plt.close(fig)
        
        # --- Confusion Matrix Metrics Plot ---
        if not metrics_df.empty:
            # Ensure 'grouped' is defined here if not carried over
            if 'grouped' not in locals():
                grouped = metrics_df.groupby('n_features')
            cm_sums = grouped[['tp', 'fp', 'tn', 'fn']].sum()
            fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
            metrics_to_plot = [('tp', 'Total True Positives'), ('fp', 'Total False Positives'), 
                               ('tn', 'Total True Negatives'), ('fn', 'Total False Negatives')]
            
            plot_x_axis = cm_sums.index.astype(int)
            for i, (metric, title) in enumerate(metrics_to_plot):
                row, col = i // 2, i % 2
                axs[row, col].plot(plot_x_axis, cm_sums[metric], 'o-')
                axs[row, col].set_title(title)
                axs[row, col].grid(True)
                axs[row, col].set_ylabel('Total Count across Folds')
                if row == 1: 
                     axs[row, col].set_xlabel('Number of Features')
                     axs[row, col].set_xticks(plot_x_axis)
            
            plt.suptitle('Aggregated Confusion Matrix Components by Feature Count', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
        
        # --- Summary Table of Metrics ---
        if not summary_table.empty:
            fig, ax = plt.subplots(figsize=(12, max(4, len(summary_table) * 0.5))) # Adjust height
            ax.axis('tight')
            ax.axis('off')
            
            # Format table data for display
            table_data_display = summary_table.copy().reset_index()
            # Select and reorder columns for the table
            display_cols = ['n_features', 'f1_score', 'f1_score_std', 'precision', 'precision_std', 
                            'recall', 'recall_std', 'accuracy', 'accuracy_std', 'tp', 'fp', 'tn', 'fn']
            # Ensure all display columns exist in the dataframe, add NaN if not
            for col in display_cols:
                if col not in table_data_display.columns:
                    table_data_display[col] = np.nan
            table_data_display = table_data_display[display_cols]
            
            # Format numbers
            float_cols = ['f1_score', 'f1_score_std', 'precision', 'precision_std', 'recall', 'recall_std', 'accuracy', 'accuracy_std']
            int_cols = ['tp', 'fp', 'tn', 'fn']
            for col in float_cols:
                 table_data_display[col] = table_data_display[col].map('{:.4f}'.format)
            for col in int_cols:
                 # Handle potential NaN before converting to int
                 table_data_display[col] = table_data_display[col].fillna(0).astype(int)

            the_table = ax.table(cellText=table_data_display.values,
                                colLabels=table_data_display.columns,
                                loc='center',
                                cellLoc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(9)
            the_table.scale(1.1, 1.3)
            
            plt.title('Summary of Metrics by Feature Count (Mean/Std over Folds)', fontsize=16)
            pdf.savefig(fig)
            plt.close(fig)
        
        # --- Page Showing Selected Features ---
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        plt.text(0.5, 0.95, 'Selected Features (Best Model)', ha='center', fontsize=18, fontweight='bold')
        
        # Use rfecv.support_ to get selected features from the original list
        best_features = [f for i, f in enumerate(features) if rfecv.support_[i]]
        
        # Re-calculate best_idx based on the actual CV results stored
        best_idx = np.argmax(rfecv.cv_results_['mean_test_score'])
        best_f1_score = rfecv.cv_results_['mean_test_score'][best_idx]
        
        features_text = f"Best number of features: {rfecv.n_features_}\n"
        features_text += f"Achieved Mean F1 Score: {best_f1_score:.4f}\n\n"
        features_text += "Selected features (Importance Ranked where available):\n"
        
        # Get importance if available and sort
        ranked_features = []
        if hasattr(rfecv, 'feature_importances_') and rfecv.feature_importances_ is not None:
             # Ensure importances correspond to selected features
             selected_importances = rfecv.feature_importances_[rfecv.support_]
             if len(selected_importances) == len(best_features):
                 sorted_indices = np.argsort(selected_importances)[::-1]
                 ranked_features = [best_features[i] for i in sorted_indices]
             else:
                 logger.warning("Mismatch between number of selected features and importances. Sorting alphabetically.")
                 ranked_features = sorted(best_features) # Fallback
        else:
             ranked_features = sorted(best_features) # Sort alphabetically if no importance

        for i, feature in enumerate(ranked_features, 1):
            features_text += f"{i}. {feature}\n"
        
        plt.text(0.1, 0.8, features_text, fontsize=11, va='top', linespacing=1.5)
        
        pdf.savefig(fig)
        plt.close(fig)

        # --- Eco-region Metrics Table --- 
        if eco_region_metrics:
            eco_metrics_df = pd.DataFrame.from_dict(eco_region_metrics, orient='index')
            eco_metrics_df = eco_metrics_df.reset_index().rename(columns={'index': 'eco_region'})
            
            # Sort by F1-score (descending)
            eco_metrics_df = eco_metrics_df.sort_values('f1_score', ascending=False)

            fig, ax = plt.subplots(figsize=(12, max(4, len(eco_metrics_df) * 0.5))) 
            ax.axis('tight')
            ax.axis('off')
            
            # Select and reorder columns for the table
            eco_display_cols = ['eco_region', 'n_samples', 'f1_score', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'tn', 'fn']
            # Ensure columns exist
            for col in eco_display_cols:
                if col not in eco_metrics_df.columns:
                     eco_metrics_df[col] = np.nan
            eco_table_data = eco_metrics_df[eco_display_cols]
            
            # Format the data
            formatted_data = []
            for _, row in eco_table_data.iterrows():
                formatted_row = [
                    row['eco_region'],
                    str(int(row['n_samples'])) if pd.notna(row['n_samples']) else 'N/A',
                    f"{row['f1_score']:.4f}" if pd.notna(row['f1_score']) else 'N/A',
                    f"{row['precision']:.4f}" if pd.notna(row['precision']) else 'N/A',
                    f"{row['recall']:.4f}" if pd.notna(row['recall']) else 'N/A',
                    f"{row['accuracy']:.4f}" if pd.notna(row['accuracy']) else 'N/A',
                    str(int(row['tp'])) if pd.notna(row['tp']) else 'N/A',
                    str(int(row['fp'])) if pd.notna(row['fp']) else 'N/A',
                    str(int(row['tn'])) if pd.notna(row['tn']) else 'N/A',
                    str(int(row['fn'])) if pd.notna(row['fn']) else 'N/A'
                ]
                formatted_data.append(formatted_row)
            
            the_table = ax.table(cellText=formatted_data,
                                colLabels=eco_display_cols,
                                loc='center',
                                cellLoc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(9)
            the_table.scale(1.1, 1.3)
            
            plt.title('Performance Metrics per Eco-Region (Best Model - Aggregated over CV Folds)', fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            # Add a placeholder page if metrics are empty
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, 'Eco-region metrics could not be calculated or were empty.',
                    ha='center', va='center', fontsize=16)
            plt.axis('off')
            pdf.savefig()
            plt.close()

        # --- Add eco-region stability plot if available (requires calculating std across eco-regions per N features) ---
        # This requires recalculating eco-region metrics for *each* feature count evaluated,
        # which is computationally expensive and not currently done in the main loop.
        # If this plot is needed, the CustomRFECV fit method needs significant modification.
        # Placeholder for now:
        # if eco_stability_df is not None and not eco_stability_df.empty:
        #    ... (plotting code) ...

    logger.info(f"Feature selection report saved to {output_path}")


def main():
    """
    Main function to run the recursive feature selection.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Recursive feature selection for tree phenology classification.')
    parser.add_argument('--output', type=str, default='results/feature_selection',
                        help='Output directory for results.')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode with a small subset of data.')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of samples to use in test mode (default: 10000).')
    parser.add_argument('--min_features', type=int, default=6,
                        help='Minimum number of features to select (default: 6).')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help='Path to the dataset parquet file.')
    parser.add_argument('--step_sizes', type=str, default='1',
                        help='Comma-separated list of step sizes (features to remove at each iteration), or a single float (e.g., 0.1 for 10%%).')
    parser.add_argument('--n_estimators', type=int, default=30, help='Number of trees in RandomForest.') # RF Param
    parser.add_argument('--max_depth', type=int, default=None, help='Max depth of trees in RandomForest.') # RF Param
    parser.add_argument('--min_samples_split', type=int, default=2, help='Min samples split in RandomForest.') # RF Param
    parser.add_argument('--min_samples_leaf', type=int, default=1, help='Min samples leaf in RandomForest.') # RF Param
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of cross-validation folds.')
    parser.add_argument('--scoring', type=str, default='f1_score', help='Scoring metric for feature selection (e.g., f1_score, accuracy).')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    try:
        df = pd.read_parquet(args.dataset_path)
        logger.info(f"Dataset loaded: {len(df)} samples")
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    
    # If in test mode, use only a subset of the data
    if args.test:
        logger.info(f"Running in TEST MODE with approximately {args.test_size} samples")
        if len(df) > args.test_size:
            # Ensure we keep a balanced subset of phenology classes
            n_classes = df['phenology'].nunique()
            sample_per_class = args.test_size // n_classes if n_classes > 0 else args.test_size
            
            df = df.groupby('phenology', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_per_class), random_state=42)
            )
            logger.info(f"Using subset of data: {len(df)} samples")
        else:
            logger.warning(f"Test size ({args.test_size}) >= dataset size ({len(df)}). Using full dataset.")

    
    # --- Unscale Features --- 
    logger.info("Unscaling features to physical ranges...")
    unscaled_count = 0
    skipped_cols = []
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    for index in tqdm(INDICES, desc="Unscaling Indices"):
        for ftype_suffix, feature_type in FEATURE_TYPES_TO_UNSCALE.items():
            col_name = f"{index}_{ftype_suffix}"
            if col_name in df_copy.columns:
                try:
                    # Apply unscale_feature to the column
                    df_copy[col_name] = unscale_feature(
                        df_copy[col_name],
                        feature_type=feature_type,
                        index_name=index # Required for amplitude/offset
                    )
                    unscaled_count += 1
                except Exception as e:
                    logger.warning(f"Could not unscale column {col_name}: {e}") # Use warning, not error
                    skipped_cols.append(col_name)
            # else: # Don't log every missing feature, can be too verbose if only some indices used
            #     skipped_cols.append(col_name) 
    df = df_copy # Assign the modified copy back to df
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    # if skipped_cols: # Reduce verbosity
    #     unique_skipped = sorted(list(set(skipped_cols)))
    #     logger.debug(f"Skipped/Not Found {len(unique_skipped)} columns during unscaling: {unique_skipped[:5]}...") 

    # --- Apply Circular Transformation (using imported function) ---
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df = transform_circular_features(df, INDICES) # Use imported function
    logger.info("Circular transformation complete.")

    # Print dataset info
    logger.info("\n=== Dataset Info ===")
    logger.info(f"Phenology distribution:\n{df['phenology'].value_counts(normalize=True).round(3).to_string()}")
    logger.info(f"\nEco-region distribution (Top 10):\n{df['eco_region'].value_counts(normalize=True).round(3).head(10).to_string()}")
    logger.info(f"\nNumber of unique tiles: {df['tile_id'].nunique()}")
    
    # Get all potential features after transformation
    all_possible_features = get_all_features()
    # Filter features to only those present in the final DataFrame
    features = [f for f in all_possible_features if f in df.columns]
    logger.info(f"Using {len(features)} available features for selection: {features}")
    
    # Prepare data
    X = df[features]
    y = df['phenology']
    
    # Initialize model with parameters from args
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        random_state=42, 
        n_jobs=-1, 
        class_weight='balanced',
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )
    logger.info(f"RandomForestClassifier configured with: n_estimators={args.n_estimators}, max_depth={args.max_depth}, min_samples_split={args.min_samples_split}, min_samples_leaf={args.min_samples_leaf}")

    
    # Initialize step sizes
    try:
        # Try interpreting as comma-separated integers
        step_input = [int(x.strip()) for x in args.step_sizes.split(',')]
        logger.info(f"Using integer step sizes: {step_input}")
    except ValueError:
        try:
            # Try interpreting as a single float
            step_input = float(args.step_sizes)
            if 0 < step_input < 1:
                logger.info(f"Using fractional step size: {step_input}")
            else:
                raise ValueError("Fractional step size must be between 0 and 1")
        except ValueError:
             logger.error(f"Invalid step_sizes argument: '{args.step_sizes}'. Must be comma-separated integers or a single float between 0 and 1.")
             sys.exit(1)

    # Initialize RFE
    rfecv = CustomRFECV(
        estimator=rf,
        min_features_to_select=args.min_features,
        step=step_input,
        cv=args.cv_folds,
        scoring=args.scoring
    )
    logger.info(f"CustomRFECV initialized with: min_features={args.min_features}, step={step_input}, cv={args.cv_folds}, scoring='{args.scoring}'")

    
    # Fit the model
    rfecv.fit(X, y, df) # df is passed to CustomRFECV for fold creation
    
    # Plot results and get metrics tables
    metrics_df, summary_table = plot_feature_selection_results(rfecv, features, output_dir)
    
    # Save selected features list
    # Use rfecv.support_ to get the boolean mask for selected features
    selected_features = [features[i] for i, supported in enumerate(rfecv.support_) if supported]
    selected_features_path = output_dir / "selected_features.txt"
    with open(selected_features_path, 'w') as f:
        f.write(f"# Feature Selection Results ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"# Best number of features: {rfecv.n_features_} (based on {args.scoring})\n\n")
        f.write("# Selected features:\n")
        for feature in selected_features:
            f.write(f"{feature}\n")
    logger.info(f"Selected features list saved to {selected_features_path}")

    
    # Save eco-region metrics if available
    if hasattr(rfecv, 'best_model_eco_metrics') and rfecv.best_model_eco_metrics:
        # Convert dict to DataFrame
        eco_metrics_df = pd.DataFrame.from_dict(rfecv.best_model_eco_metrics, orient='index')
        eco_metrics_df = eco_metrics_df.reset_index().rename(columns={'index': 'eco_region'})
        # Reorder columns
        eco_cols_order = ['eco_region', 'n_samples', 'f1_score', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'tn', 'fn']
        # Ensure all expected columns exist, add if missing (e.g., if some metric failed)
        for col in eco_cols_order:
             if col not in eco_metrics_df.columns:
                  eco_metrics_df[col] = np.nan 
        eco_metrics_df = eco_metrics_df[eco_cols_order]
        eco_metrics_path = output_dir / "eco_region_metrics_best_model.csv"
        eco_metrics_df.to_csv(eco_metrics_path, index=False, float_format='%.4f')
        logger.info(f"Eco-region metrics for best model saved to {eco_metrics_path}")
    else:
        logger.warning("No eco-region metrics were calculated or available to save.")

    # Create PDF report
    report_path = output_dir / "feature_selection_report.pdf"
    try:
        create_feature_selection_report(
            rfecv,
            features, # Pass the full list of features used
            metrics_df, # Per-fold metrics
            summary_table, # Aggregated metrics per n_features
            rfecv.best_model_eco_metrics, # Eco-region metrics dict
            output_path=report_path
        )
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}", exc_info=True)


    # Report execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Feature selection completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best number of features found: {rfecv.n_features_}")
    logger.info(f"Results saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main() 