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
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
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

def create_eco_balanced_folds_df(df, n_splits=5, random_state=42):
    """
    Create folds that balance eco-region distribution while preserving tile integrity.
    This ensures that:
    1. Tiles are never split between training and validation sets
    2. The distribution of eco-regions in each fold is similar to the overall distribution
    """
    logger.info("Creating eco-region balanced folds...")
    
    # Reset the index to ensure we work with the current DataFrame indices
    df = df.reset_index(drop=True)
    
    # Get unique eco-regions and their overall distribution
    eco_regions = df['eco_region'].unique()
    overall_eco_dist = df['eco_region'].value_counts(normalize=True)
    
    # Initialize lists to store folds
    all_folds = []
    for _ in range(n_splits):
        all_folds.append({'train_idx': [], 'val_idx': []})
    
    # Process each eco-region separately
    for eco_region in tqdm(eco_regions, desc="Processing eco-regions for balanced folds"):
        # Filter data for this eco-region
        eco_df = df[df['eco_region'] == eco_region]
        
        # Get unique tiles for this eco-region
        eco_tiles = eco_df['tile_id'].unique()
        
        # Shuffle tiles to randomize
        eco_tiles = shuffle(eco_tiles, random_state=random_state)
        
        # Split tiles into n_splits groups
        tile_groups = np.array_split(eco_tiles, n_splits)
        
        # Create folds for this eco-region
        for fold_idx in range(n_splits):
            # Validation tiles for this fold
            val_tiles = tile_groups[fold_idx]
            
            # Get indices for validation
            val_mask = eco_df['tile_id'].isin(val_tiles)
            val_indices = eco_df.index[val_mask].tolist()
            
            # Get indices for training (all other tiles)
            train_tiles = np.concatenate([tile_groups[i] for i in range(n_splits) if i != fold_idx])
            train_mask = eco_df['tile_id'].isin(train_tiles)
            train_indices = eco_df.index[train_mask].tolist()
            
            # Add to fold
            all_folds[fold_idx]['train_idx'].extend(train_indices)
            all_folds[fold_idx]['val_idx'].extend(val_indices)
    
    # Convert to array format and validate
    fold_splits = []
    for fold in all_folds:
        train_idx = np.array(fold['train_idx'])
        val_idx = np.array(fold['val_idx'])
        
        # Check for overlap
        assert len(np.intersect1d(train_idx, val_idx)) == 0, "Overlap detected between train and validation indices"
        
        # Check that all indices are accounted for
        assert len(train_idx) + len(val_idx) == len(df), "Some indices are missing in the fold split"
        
        fold_splits.append((train_idx, val_idx))
    
    # Calculate and display eco-region distribution per fold
    logger.info("Eco-region distribution per fold:")
    fold_eco_dists = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_eco_dist = df.iloc[train_idx]['eco_region'].value_counts(normalize=True)
        val_eco_dist = df.iloc[val_idx]['eco_region'].value_counts(normalize=True)
        
        fold_eco_dists.append({
            'fold': fold_idx + 1,
            'train_dist': train_eco_dist,
            'val_dist': val_eco_dist
        })
        
        logger.info(f"Fold {fold_idx + 1}:")
        logger.info(f"Training distribution: \n{train_eco_dist.to_string()}")
        logger.info(f"Validation distribution: \n{val_eco_dist.to_string()}")
        logger.info("---")
    
    return fold_splits

def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    return metrics

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
        
    def fit(self, X, y, df):
        """Fit the RFE model with cross-validation."""
        logger.info(f"Starting recursive feature elimination with min {self.min_features_to_select} features...")
        
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        
        # Create eco-region balanced folds
        logger.info("Creating eco-region balanced folds...")
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
        
        while current_n_features > self.min_features_to_select:
            # Current feature set
            current_features = [f for i, f in enumerate(feature_names) if support[i]]
            
            logger.info(f"Evaluating with {current_n_features} features")
            logger.info(f"Current features: {current_features}")
            
            # Cross-validation scores for this feature set
            fold_metrics = []
            fold_importances = []
            
            # For each fold
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(self.fold_splits, desc=f"Cross-validation with {current_n_features} features")):
                logger.info(f"Processing fold {fold_idx+1}/{self.cv} with {len(train_idx)} training samples and {len(val_idx)} validation samples")
                
                # Split data
                X_train = X.iloc[train_idx][current_features]
                X_val = X.iloc[val_idx][current_features]
                y_train = y.iloc[train_idx]
                y_val = y.iloc[val_idx]
                
                logger.info(f"Fold {fold_idx+1} - Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
                
                # Apply sample weights if available
                sample_weights = None
                if 'weight' in df.columns:
                    sample_weights = df.iloc[train_idx]['weight'].values
                
                # Train model
                logger.info(f"Fold {fold_idx+1} - Training Random Forest model...")
                model = self.estimator
                model.fit(X_train, y_train, sample_weight=sample_weights)
                logger.info(f"Fold {fold_idx+1} - Model training completed")
                
                # Store feature importance
                fold_importances.append(pd.Series(model.feature_importances_, index=current_features))
                
                # Make predictions
                logger.info(f"Fold {fold_idx+1} - Making predictions on validation set...")
                y_pred = model.predict(X_val)
                logger.info(f"Fold {fold_idx+1} - Predictions completed")
                
                # Compute metrics immediately
                metrics = compute_metrics(y_val, y_pred)
                metrics['fold'] = fold_idx + 1
                fold_metrics.append(metrics)
                logger.info(f"Fold {fold_idx+1} - F1 Score: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            
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
            
            logger.info(f"  F1 Score: {mean_score:.4f} ± {std_score:.4f}")
            
            # If we're at the minimum number of features, break
            if current_n_features <= self.min_features_to_select:
                break
            
            # Calculate number of features to eliminate
            if isinstance(self.step, list):
                # Get appropriate step size based on how many features remain
                remaining_steps = len(self.step) - len(self.cv_results_['n_features'])
                step_idx = min(max(0, len(self.step) - remaining_steps - 1), len(self.step) - 1)
                n_to_eliminate = self.step[step_idx]
                logger.info(f"Using step size {n_to_eliminate} (step {len(self.cv_results_['n_features'])+1})")
            elif isinstance(self.step, int):
                n_to_eliminate = min(self.step, current_n_features - self.min_features_to_select)
            else:
                n_to_eliminate = max(1, int(current_n_features * self.step))
                n_to_eliminate = min(n_to_eliminate, current_n_features - self.min_features_to_select)
            
            # Get feature importances for current feature set
            supported_indices = np.where(support)[0]
            current_feature_names = [feature_names[i] for i in supported_indices]
            
            # Map average importances to the supported indices
            current_importances = np.array([avg_importance[name] for name in current_feature_names])

            # Find the indices of the features to eliminate
            actual_n_to_eliminate = min(n_to_eliminate, len(current_importances))
            indices_to_eliminate_relative = np.argsort(current_importances)[:actual_n_to_eliminate]
            
            # Map these relative indices back to the original feature indices
            original_indices_to_eliminate = supported_indices[indices_to_eliminate_relative]

            # Update support mask
            support[original_indices_to_eliminate] = False
            current_n_features = np.sum(support) # Update the count
            
            logger.info(f"Eliminated {len(original_indices_to_eliminate)} features. Remaining: {current_n_features}")
        
        # Store final results
        self.ranking_ = np.zeros(n_features, dtype=int)
        for i, selected in enumerate(self.cv_results_['features']):
            for feature in selected:
                idx = feature_names.index(feature)
                self.ranking_[idx] = min(self.ranking_[idx] if self.ranking_[idx] != 0 else n_features, n_features - i)
        
        # Find best feature count based on scores
        best_idx = np.argmax(self.cv_results_['mean_test_score'])
        self.n_features_ = self.cv_results_['n_features'][best_idx]
        best_features = self.cv_results_['features'][best_idx]
        
        # Set support for best feature set
        self.support_ = np.zeros(n_features, dtype=bool)
        for feature in self.cv_results_['features'][best_idx]:
            idx = feature_names.index(feature)
            self.support_[idx] = True
        
        # Store feature importances from the best model
        self.feature_importances_ = np.zeros(n_features)
        best_importances = self.cv_results_['feature_importances'][best_idx]
        for feature, importance in best_importances.items():
            idx = feature_names.index(feature)
            self.feature_importances_[idx] = importance
        
        logger.info(f"Best number of features: {self.n_features_}")
        logger.info(f"Selected features: {best_features}")

        # For the best model, calculate metrics per eco-region
        logger.info(f"Calculating eco-region metrics for the best model ({self.n_features_} features)...")
        self.best_model_eco_metrics = self.calculate_eco_region_metrics(X, y, df, best_features)

        return self
    
    def calculate_eco_region_metrics(self, X, y, df, best_features):
        """Calculate metrics per eco-region using the best features"""
        eco_metrics = {}
        eco_regions = df['eco_region'].unique()
        
        for eco_region in eco_regions:
            # Get eco-region mask
            eco_mask = df['eco_region'] == eco_region
            
            # Calculate metrics for each fold
            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(self.fold_splits):
                # Get validation indices for this eco-region
                val_eco_idx = np.array([i for i in val_idx if eco_mask.iloc[i]])
                
                if len(val_eco_idx) == 0:
                    continue  # Skip if no validation samples for this eco-region in this fold
                
                # Train on all training data
                X_train = X.iloc[train_idx][best_features]
                y_train = y.iloc[train_idx]
                
                # Test only on this eco-region's validation data
                X_val_eco = X.iloc[val_eco_idx][best_features]
                y_val_eco = y.iloc[val_eco_idx]
                
                # Train and predict
                model = clone(self.estimator)  # Clone to ensure fresh model
                model.fit(X_train, y_train)
                y_pred_eco = model.predict(X_val_eco)
                
                # Calculate metrics
                if len(np.unique(y_val_eco)) > 1:  # Ensure at least two classes
                    metrics = compute_metrics(y_val_eco, y_pred_eco)
                    fold_scores.append(metrics)
            
            if fold_scores:
                # Average metrics across folds
                eco_metrics[eco_region] = {
                    'f1_score': np.mean([m['f1_score'] for m in fold_scores]),
                    'precision': np.mean([m['precision'] for m in fold_scores]),
                    'recall': np.mean([m['recall'] for m in fold_scores]),
                    'n_samples': sum(eco_mask)
                }
        
        return eco_metrics

def calculate_eco_region_metrics(predictions_list):
    """Calculate performance metrics grouped by eco-region."""
    if not predictions_list:
        logger.warning("No predictions found to calculate eco-region metrics.")
        return pd.DataFrame()

    logger.info("Calculating metrics per eco-region...")
    pred_df = pd.DataFrame(predictions_list)
    eco_metrics = []

    for region, group in pred_df.groupby('eco_region'):
        metrics = compute_metrics(group['y_true'], group['y_pred'])
        metrics['eco_region'] = region
        metrics['n_samples'] = len(group)
        eco_metrics.append(metrics)
        logger.info(f"  Eco-region {region} (N={len(group)}): F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    eco_metrics_df = pd.DataFrame(eco_metrics)
    # Reorder columns for clarity
    cols = ['eco_region', 'n_samples', 'f1_score', 'precision', 'recall', 'accuracy', 'tp', 'fp', 'tn', 'fn']
    eco_metrics_df = eco_metrics_df[cols]
    return eco_metrics_df

def plot_feature_selection_results(rfecv, features, output_dir='results/feature_selection'):
    """
    Plot the results of feature selection.
    """
    # Plot number of features vs. cross-validation score
    plt.figure(figsize=(12, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (F1)")
    plt.errorbar(
        rfecv.cv_results_['n_features'],
        rfecv.cv_results_['mean_test_score'],
        yerr=rfecv.cv_results_['std_test_score'],
        fmt='o-'
    )
    plt.grid(True)
    plt.title("Feature Selection Cross-Validation Scores")
    plt.savefig(f"{output_dir}/feature_selection_cv_scores.png")
    plt.close()
    
    # Plot feature importance of selected features
    selected_indices = np.where(rfecv.support_)[0]
    selected_features = [features[i] for i in selected_indices]
    selected_importances = rfecv.feature_importances_[selected_indices]
    
    # Sort by importance
    sorted_idx = np.argsort(selected_importances)[::-1]
    sorted_features = [selected_features[i] for i in sorted_idx]
    sorted_importances = selected_importances[sorted_idx]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_features)), sorted_importances, align='center')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance of Selected Features')
    plt.gca().invert_yaxis()  # Show highest values at the top
    plt.tight_layout()
    plt.savefig(f"{output_dir}/selected_features_importance.png")
    plt.close()
    
    # Extract and plot metrics per feature count
    metrics_data = []
    for i, n_feat in enumerate(rfecv.cv_results_['n_features']):
        # Get all predictions across all folds for this feature count
        all_true = []
        all_pred = []
        fold_metrics = rfecv.cv_results_['metrics_per_fold'][i]
        
        for _, row in fold_metrics.iterrows():
            metrics_data.append({
                'n_features': n_feat,
                'f1_score': row['f1_score'],
                'precision': row['precision'],
                'recall': row['recall'],
                'tp': row['tp'],
                'fp': row['fp'],
                'tn': row['tn'],
                'fn': row['fn'],
                'fold': row['fold']
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Plot f1, precision, recall by feature count (with error bars)
    plt.figure(figsize=(14, 8))
    
    # Group by number of features and calculate mean and std for each metric
    grouped = metrics_df.groupby('n_features')
    means = grouped.mean()
    stds = grouped.std()
    
    x = means.index
    
    plt.errorbar(x, means['f1_score'], yerr=stds['f1_score'], fmt='o-', label='F1 Score')
    plt.errorbar(x, means['precision'], yerr=stds['precision'], fmt='s-', label='Precision')
    plt.errorbar(x, means['recall'], yerr=stds['recall'], fmt='^-', label='Recall')
    
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Feature Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/performance_metrics_by_feature_count.png")
    plt.close()
    
    # Plot confusion matrix metrics (tp, tn, fp, fn) by feature count
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    
    metrics_to_plot = [('tp', 'True Positives'), ('fp', 'False Positives'), 
                       ('tn', 'True Negatives'), ('fn', 'False Negatives')]
    
    for i, (metric, title) in enumerate(metrics_to_plot):
        row, col = i // 2, i % 2
        axs[row, col].errorbar(x, means[metric], yerr=stds[metric], fmt='o-')
        axs[row, col].set_title(title)
        axs[row, col].grid(True)
    
    # Set common labels
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Number of Features', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    plt.suptitle('Confusion Matrix Metrics by Feature Count', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/confusion_matrix_metrics_by_feature_count.png")
    plt.close()
    
    # Create a table of all metrics
    summary_table = metrics_df.groupby('n_features').mean()[
        ['f1_score', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn']
    ].round(4)
    
    # Add standard deviations
    for col in ['f1_score', 'precision', 'recall']:
        summary_table[f'{col}_std'] = metrics_df.groupby('n_features')[col].std().round(4)
    
    summary_table.to_csv(f"{output_dir}/feature_selection_metrics.csv")
    
    # Return the metrics data for further analysis
    return metrics_df, summary_table

def create_feature_selection_report(rfecv, features, metrics_df, summary_table, eco_region_metrics_df, eco_stability_df=None, output_path='results/feature_selection/feature_selection_report.pdf'):
    """Create a comprehensive PDF report of feature selection results."""
    with PdfPages(output_path) as pdf:
        # Title page
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Tree Phenology Classification\nFeature Selection Report',
                ha='center', va='center', fontsize=24)
        plt.text(0.5, 0.4, f'Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Cross-validation scores plot
        plt.figure(figsize=(12, 6))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (F1)")
        plt.errorbar(
            rfecv.cv_results_['n_features'],
            rfecv.cv_results_['mean_test_score'],
            yerr=rfecv.cv_results_['std_test_score'],
            fmt='o-'
        )
        plt.grid(True)
        plt.title("Feature Selection Cross-Validation Scores")
        pdf.savefig()
        plt.close()
        
        # Selected features importance
        selected_indices = np.where(rfecv.support_)[0]
        selected_features = [features[i] for i in selected_indices]
        selected_importances = rfecv.feature_importances_[selected_indices]
        
        # Sort by importance
        sorted_idx = np.argsort(selected_importances)[::-1]
        sorted_features = [selected_features[i] for i in sorted_idx]
        sorted_importances = selected_importances[sorted_idx]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_features)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance of Selected Features')
        plt.gca().invert_yaxis()  # Show highest values at the top
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Performance metrics plot
        plt.figure(figsize=(14, 8))
        
        # Group by number of features and calculate mean and std for each metric
        grouped = metrics_df.groupby('n_features')
        means = grouped.mean()
        stds = grouped.std()
        
        x = means.index
        
        plt.errorbar(x, means['f1_score'], yerr=stds['f1_score'], fmt='o-', label='F1 Score')
        plt.errorbar(x, means['precision'], yerr=stds['precision'], fmt='s-', label='Precision')
        plt.errorbar(x, means['recall'], yerr=stds['recall'], fmt='^-', label='Recall')
        
        plt.xlabel('Number of Features')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Feature Count')
        plt.legend()
        plt.grid(True)
        pdf.savefig()
        plt.close()
        
        # Confusion matrix metrics plots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        
        metrics_to_plot = [('tp', 'True Positives'), ('fp', 'False Positives'), 
                           ('tn', 'True Negatives'), ('fn', 'False Negatives')]
        
        for i, (metric, title) in enumerate(metrics_to_plot):
            row, col = i // 2, i % 2
            axs[row, col].errorbar(x, means[metric], yerr=stds[metric], fmt='o-')
            axs[row, col].set_title(title)
            axs[row, col].grid(True)
        
        # Set common labels
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Number of Features', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        
        plt.suptitle('Confusion Matrix Metrics by Feature Count', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # Summary table of metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table with selected columns
        table_data = summary_table.reset_index()
        table_data = table_data[['n_features', 'f1_score', 'f1_score_std', 
                                'precision', 'precision_std', 'recall', 'recall_std',
                                'tp', 'fp', 'tn', 'fn']]
        
        the_table = ax.table(cellText=table_data.values.round(4),
                            colLabels=table_data.columns,
                            loc='center',
                            cellLoc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.2, 1.5)
        
        plt.title('Summary of Metrics by Feature Count', fontsize=16)
        pdf.savefig()
        plt.close()
        
        # Page showing selected features
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        plt.text(0.5, 0.9, 'Selected Features', ha='center', fontsize=18, fontweight='bold')
        
        best_idx = np.argmax(rfecv.cv_results_['mean_test_score'])
        best_n_features = rfecv.cv_results_['n_features'][best_idx]
        best_features = rfecv.cv_results_['features'][best_idx]
        
        features_text = f"Best number of features: {best_n_features}\n\n"
        features_text += "Selected features:\n"
        
        for i, feature in enumerate(best_features, 1):
            features_text += f"{i}. {feature}\n"
        
        plt.text(0.1, 0.7, features_text, fontsize=12, va='top')
        
        pdf.savefig()
        plt.close()

        # --- Add this section for Eco-region Metrics --- 
        if not eco_region_metrics_df.empty:
            fig, ax = plt.subplots(figsize=(12, max(4, len(eco_region_metrics_df) * 0.5))) # Adjust height based on rows
            ax.axis('tight')
            ax.axis('off')
            
            # Select and reorder columns for the table
            eco_table_data = eco_region_metrics_df[['eco_region', 'n_samples', 'f1_score', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn']]
            
            # Format the data - convert to strings with appropriate formatting
            formatted_data = []
            for _, row in eco_table_data.iterrows():
                formatted_row = [
                    row['eco_region'],  # Keep eco_region as is (string)
                    str(row['n_samples']),  # Convert n_samples to string
                    f"{row['f1_score']:.4f}",  # Format floating point values with 4 decimal places
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    str(int(row['tp'])),  # Format integer values
                    str(int(row['fp'])),
                    str(int(row['tn'])),
                    str(int(row['fn']))
                ]
                formatted_data.append(formatted_row)
            
            the_table = ax.table(cellText=formatted_data,
                                colLabels=eco_table_data.columns,
                                loc='center',
                                cellLoc='center')
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1.2, 1.5)
            
            plt.title('Performance Metrics per Eco-Region (Best Model)', fontsize=16)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        else:
            # Add a placeholder page if metrics are empty
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, 'Eco-region metrics could not be calculated.',
                    ha='center', va='center', fontsize=16)
            plt.axis('off')
            pdf.savefig()
            plt.close()
        # --- End of added section ---

        # Add eco-region stability plot if available
        if eco_stability_df is not None and not eco_stability_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(eco_stability_df['n_features'], eco_stability_df['f1_std_across_eco'], 'o-', color='red')
            plt.xlabel('Number of Features')
            plt.ylabel('Standard Deviation of F1 Scores Across Eco-regions')
            plt.title('Eco-region F1-Score Stability by Feature Count')
            plt.grid(True)
            pdf.savefig()
            plt.close()
            
            # Plot both overall F1 and eco-region stability
            plt.figure(figsize=(12, 6))
            
            # Create two y-axes
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax2 = ax1.twinx()
            
            # Plot mean F1 score on left axis
            ax1.plot(rfecv.cv_results_['n_features'], rfecv.cv_results_['mean_test_score'], 'o-', color='blue', label='Mean F1 Score')
            ax1.set_xlabel('Number of Features')
            ax1.set_ylabel('Mean F1 Score', color='blue')
            ax1.tick_params(axis='y', colors='blue')
            
            # Plot eco-region F1 std on right axis
            ax2.plot(eco_stability_df['n_features'], eco_stability_df['f1_std_across_eco'], 's-', color='red', label='Eco-region F1 Std')
            ax2.set_ylabel('Std Dev of F1 Across Eco-regions', color='red')
            ax2.tick_params(axis='y', colors='red')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            plt.title('F1 Score and Eco-region Stability by Feature Count')
            plt.grid(True)
            pdf.savefig(fig)
            plt.close()

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
                        help='Comma-separated list of step sizes (features to remove at each iteration)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    df = pd.read_parquet(args.dataset_path)
    logger.info(f"Dataset loaded: {len(df)} samples")
    
    # If in test mode, use only a subset of the data
    if args.test:
        logger.info(f"Running in TEST MODE with {args.test_size} samples")
        # Ensure we keep a balanced subset of phenology classes
        sample_per_class = args.test_size // 2
        df_subset = pd.DataFrame()
        for pheno_class in df['phenology'].unique():
            class_df = df[df['phenology'] == pheno_class]
            if len(class_df) > sample_per_class:
                class_sample = class_df.sample(sample_per_class, random_state=42)
            else:
                class_sample = class_df  # Use all if we have fewer than needed
            df_subset = pd.concat([df_subset, class_sample])
        
        df = df_subset
        logger.info(f"Using subset of data: {len(df)} samples")
    
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
                    logger.error(f"Error unscaling column {col_name}: {e}")
                    skipped_cols.append(col_name)
            else:
                skipped_cols.append(col_name)
    df = df_copy # Assign the modified copy back to df
    logger.info(f"Unscaling complete. Processed {unscaled_count} columns.")
    if skipped_cols:
        unique_skipped = sorted(list(set(skipped_cols)))
        # logger.debug(f"Skipped/Not Found {len(unique_skipped)} columns during unscaling: {unique_skipped[:5]}...") 

    # --- Apply Circular Transformation (using imported function) ---
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df = transform_circular_features(df, INDICES) # Use imported function
    logger.info("Circular transformation complete.")

    # Print dataset info
    logger.info("\n=== Dataset Info ===")
    logger.info(f"Phenology distribution:\n{df['phenology'].value_counts().to_string()}")
    logger.info(f"\nEco-region distribution:\n{df['eco_region'].value_counts().to_string()}")
    logger.info(f"\nNumber of unique tiles: {df['tile_id'].nunique()}")
    
    # Get all features
    features = get_all_features()
    logger.info(f"Total number of features: {len(features)}")
    
    # Prepare data
    X = df[features]
    y = df['phenology']
    
    # Initialize model
    rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Initialize step sizes
    step_sizes = [int(x) for x in args.step_sizes.split(',')]

    # Initialize RFE with adaptive step size
    rfecv = CustomRFECV(
        estimator=rf,
        min_features_to_select=args.min_features,
        step=step_sizes,  # Pass the list of step sizes
        cv=5,
        scoring='f1_score'
    )
    
    # Fit the model
    rfecv.fit(X, y, df)
    
    # Plot results
    metrics_df, summary_table = plot_feature_selection_results(rfecv, features, args.output)
    
    # Save selected features
    selected_indices = np.where(rfecv.support_)[0]
    selected_features = [features[i] for i in selected_indices]
    
    with open(f"{args.output}/selected_features.txt", 'w') as f:
        f.write(f"Best number of features: {rfecv.n_features_}\n\n")
        f.write("Selected features:\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")
    
    # Save eco-region metrics if available
    if hasattr(rfecv, 'best_model_eco_metrics') and rfecv.best_model_eco_metrics:
        eco_metrics_df = pd.DataFrame([
            {
                'eco_region': region,
                **metrics
            }
            for region, metrics in rfecv.best_model_eco_metrics.items()
        ])
        eco_metrics_df.to_csv(f"{args.output}/eco_region_metrics.csv", index=False)
        logger.info(f"Eco-region metrics saved to {args.output}/eco_region_metrics.csv")
    
    # Report execution time
    elapsed_time = time.time() - start_time
    logger.info(f"Feature selection completed in {elapsed_time:.2f} seconds")
    logger.info(f"Best number of features: {rfecv.n_features_}")
    logger.info(f"Selected features: {selected_features}")

if __name__ == "__main__":
    main() 