import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report
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
from matplotlib.backends.backend_pdf import PdfPages
import sys
from pathlib import Path

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

# Import utility functions
from src.utils import (
    unscale_feature,
    transform_circular_features,
    compute_metrics,
    create_eco_balanced_folds_df,
    display_fold_distribution,
    format_confusion_matrix
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phenology_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory if it doesn't exist
os.makedirs('results/models', exist_ok=True)

# Define the indices to test
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']

# Define the path to the dataset
DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'

# Map feature suffixes to unscaling types
FEATURE_TYPES_TO_UNSCALE = {
    'amplitude_h1': 'amplitude',
    'amplitude_h2': 'amplitude',
    'phase_h1': 'phase', # Will be unscaled to radians [0, 2pi]
    'phase_h2': 'phase',
    'offset': 'offset',
    'var_residual': 'variance'
}

# Define the phenology mapping
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}

# Define the features for each index
def get_features(index):
    """
    Get the features list for a specific index.
    For phase features, we'll use both cos and sin transformations.
    """
    return [
        f'{index}_amplitude_h1',
        f'{index}_amplitude_h2',
        f'{index}_phase_h1_cos',  # Transformed feature
        f'{index}_phase_h1_sin',  # Transformed feature
        f'{index}_phase_h2_cos',  # Transformed feature
        f'{index}_phase_h2_sin',  # Transformed feature
        f'{index}_offset',
        f'{index}_var_residual'
    ]

def plot_feature_importance(importances, features, index, output_dir='results/models'):
    """Plot and save feature importance."""
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title(f'Feature Importance - {index.upper()}')
    plt.bar(range(len(importances)), importances[indices], align='center')
    
    # Format feature names for better readability
    formatted_features = []
    for i in indices:
        feature = features[i]
        if '_phase_h' in feature:
            if '_cos' in feature:
                formatted_features.append(feature.replace('_cos', ' (cos)'))
            elif '_sin' in feature:
                formatted_features.append(feature.replace('_sin', ' (sin)'))
        else:
            formatted_features.append(feature)
    
    plt.xticks(range(len(importances)), formatted_features, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/phenology_{index}_feature_importance.png")
    plt.close()
    
    return formatted_features, importances[indices]

def create_pdf_report(summary_df, all_results, output_path='results/models/phenology_report.pdf'):
    """Create a comprehensive PDF report of all results using matplotlib PdfPages."""
    logger.info("Generating PDF report...")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create PDF using matplotlib's PdfPages
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.text(0.5, 0.6, 'Tree Phenology Classification Report', 
                 horizontalalignment='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.5, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 horizontalalignment='center', fontsize=16)
        pdf.savefig(fig)
        plt.close()
        
        # Summary page with performance comparison
        fig = plt.figure(figsize=(12, 10))
        
        # Performance comparison across indices as bar chart
        plt.subplot(2, 1, 1)
        indices = summary_df['index']
        f1_scores = summary_df['f1_score']
        f1_stds = summary_df['f1_std']
        
        x = np.arange(len(indices))
        width = 0.35
        
        plt.bar(x, f1_scores, width, yerr=f1_stds, label='F1 Score')
        plt.xlabel('Vegetation Index')
        plt.ylabel('F1 Score')
        plt.title('Performance Comparison Across Indices')
        plt.xticks(x, indices)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Summary metrics table
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        # Prepare data for table
        table_data = summary_df[['index', 'f1_score', 'f1_std', 'precision', 'recall', 'training_time']].values
        rounded_data = [[row[0]] + [f"{val:.4f}" for val in row[1:4]] + [f"{row[4]:.4f}", f"{row[5]:.2f}"] for row in table_data]
        
        # Create table
        table = plt.table(
            cellText=rounded_data,
            colLabels=['Index', 'F1 Score', 'F1 Std', 'Precision', 'Recall', 'Training Time (s)'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title("Performance Metrics by Index", fontsize=14, pad=20)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Result pages for each index
        for index, (results_df, eco_results_df, cm, feat_importance) in all_results.items():
            # Page for each index
            fig = plt.figure(figsize=(12, 12))
            plt.suptitle(f'Results for {index.upper()}', fontsize=18, y=0.98)
            
            # Confusion matrix
            plt.subplot(2, 2, 1)
            tn, fp, fn, tp = cm.ravel()
            cm_display = np.array([[tn, fp], [fn, tp]])
            
            im = plt.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            
            # Add labels and values
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, [f'{PHENOLOGY_MAPPING[1]} (1)', f'{PHENOLOGY_MAPPING[2]} (2)'])
            plt.yticks(tick_marks, [f'{PHENOLOGY_MAPPING[1]} (1)', f'{PHENOLOGY_MAPPING[2]} (2)'])
            
            # Add text values inside cells
            thresh = cm_display.max() / 2
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, format(cm_display[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm_display[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            # Feature importance
            plt.subplot(2, 2, 2)
            formatted_features, importances = feat_importance
            
            # Display top N features for readability
            max_features = min(8, len(formatted_features))
            plt.barh(range(max_features), importances[:max_features], align='center')
            plt.yticks(range(max_features), formatted_features[:max_features])
            plt.title('Top Feature Importance')
            plt.ylabel('Feature')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()  # Display highest values at the top
            
            # Eco-region results
            plt.subplot(2, 1, 2)
            plt.axis('off')
            
            # Prepare data for eco-region table
            eco_data = eco_results_df[['eco_region', 'f1_score', 'f1_std', 'precision', 'recall']].values
            rounded_eco_data = [[row[0]] + [f"{val:.4f}" for val in row[1:]] for row in eco_data]
            
            # Show only top N eco-regions for readability
            max_regions = min(10, len(rounded_eco_data))
            eco_table = plt.table(
                cellText=rounded_eco_data[:max_regions],
                colLabels=['Eco-Region', 'F1 Score', 'F1 Std', 'Precision', 'Recall'],
                loc='center',
                cellLoc='center'
            )
            eco_table.auto_set_font_size(False)
            eco_table.set_fontsize(9)
            eco_table.scale(1.2, 1.4)
            
            plt.title("Eco-Region Results", fontsize=14, pad=20)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
            pdf.savefig(fig)
            plt.close()
        
        # Add PDF metadata
        d = pdf.infodict()
        d['Title'] = 'Tree Phenology Classification Report'
        d['Author'] = 'Automatic Reporting System'
        d['Subject'] = 'Model performance analysis'
        d['Keywords'] = 'phenology, classification, machine learning'
        d['CreationDate'] = datetime.datetime.now()
        d['ModDate'] = datetime.datetime.now()
    
    logger.info(f"PDF report generated at: {output_path}")

def train_and_evaluate(df, features, index_name, target='phenology', n_splits=5):
    """
    Train RandomForest with 5-fold cross-validation based on tile_id
    and evaluate performance per eco-region.
    """
    logger.info(f"Starting training for {index_name.upper()} with {len(features)} features")
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Create eco-region balanced folds
    fold_splits = create_eco_balanced_folds_df(df, n_splits=n_splits, random_state=42)
    
    # Store results
    results_per_fold = []
    results_per_ecoregion = defaultdict(list)
    
    # Store all predictions for confusion matrix
    all_true = []
    all_pred = []
    
    # Store feature importances
    feature_importances = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_splits, desc=f"Cross-validation folds for {index_name}")):
        logger.info(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        # Display fold distribution
        distribution = display_fold_distribution(train_idx, val_idx, df, fold)
        logger.info(f"\n--- Fold {fold+1} Distribution ---")
        logger.info(f"Training set: {len(train_idx)} samples")
        logger.info(f"Validation set: {len(val_idx)} samples")
        logger.info(f"Train/Val ratio: {len(train_idx)/len(val_idx):.2f}")
        logger.info("\nDistribution per eco-region:")
        logger.info("\n" + distribution.round(2).to_string())
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply sample weights if available
        sample_weights = None
        if 'weight' in df.columns:
            sample_weights = df.iloc[train_idx]['weight'].values
            sample_weights_stats = {
                'min': sample_weights.min(),
                'max': sample_weights.max(),
                'mean': sample_weights.mean(),
                'median': np.median(sample_weights),
                'non_zero': np.sum(sample_weights > 0),
                'total': len(sample_weights)
            }
            logger.info(f"Using sample weights from dataset: {sample_weights_stats}")
        
        # Train model
        logger.info(f"Training RandomForest on {len(X_train)} samples...")
        model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Store feature importance
        feature_importances.append(model.feature_importances_)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Store predictions for overall confusion matrix
        all_true.extend(y_val)
        all_pred.extend(y_pred)
        
        # Compute overall metrics
        overall_metrics = compute_metrics(y_val, y_pred)
        overall_metrics['fold'] = fold + 1
        results_per_fold.append(overall_metrics)
        
        logger.info(f"Overall F1 Score: {overall_metrics['f1_score']:.4f}")
        
        # Compute metrics per eco-region
        eco_regions = df.iloc[val_idx]['eco_region'].unique()
        
        for eco_region in eco_regions:
            eco_mask = df.iloc[val_idx]['eco_region'] == eco_region
            if sum(eco_mask) > 0:
                eco_y_val = y_val[eco_mask]
                eco_y_pred = y_pred[eco_mask]
                
                eco_metrics = compute_metrics(eco_y_val, eco_y_pred)
                eco_metrics['fold'] = fold + 1
                eco_metrics['eco_region'] = eco_region
                results_per_ecoregion[eco_region].append(eco_metrics)
                
                logger.info(f"  {eco_region} F1 Score: {eco_metrics['f1_score']:.4f} (on {len(eco_y_val)} samples)")
    
    # Aggregate results
    results_df = pd.DataFrame(results_per_fold)
    logger.info("\n=== Overall Results ===")
    logger.info(f"Average F1 Score: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    
    # Aggregate eco-region results
    eco_results = []
    for eco_region, metrics_list in results_per_ecoregion.items():
        metrics_df = pd.DataFrame(metrics_list)
        avg_metrics = {
            'eco_region': eco_region,
            'f1_score': metrics_df['f1_score'].mean(),
            'f1_std': metrics_df['f1_score'].std(),
            'precision': metrics_df['precision'].mean(),
            'recall': metrics_df['recall'].mean(),
        }
        eco_results.append(avg_metrics)
    
    eco_results_df = pd.DataFrame(eco_results).sort_values('f1_score', ascending=False)
    logger.info("\n=== Results per Eco-Region ===")
    logger.info("\n" + eco_results_df[['eco_region', 'f1_score', 'f1_std', 'precision', 'recall']].round(4).to_string())
    
    # Generate text confusion matrix
    cm = confusion_matrix(all_true, all_pred)
    cm_text = format_confusion_matrix(cm, labels=[f'{PHENOLOGY_MAPPING[1]} (1)', f'{PHENOLOGY_MAPPING[2]} (2)'])
    logger.info("\n" + cm_text)
    
    # Calculate confusion matrix stats
    tn, fp, fn, tp = cm.ravel()
    logger.info("\n=== Confusion Matrix Stats ===")
    logger.info(f"True Positives (TP): {tp}")
    logger.info(f"False Positives (FP): {fp}")
    logger.info(f"True Negatives (TN): {tn}")
    logger.info(f"False Negatives (FN): {fn}")
    logger.info(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")
    
    # Plot average feature importance
    avg_importance = np.mean(feature_importances, axis=0)
    formatted_features, feature_importances_sorted = plot_feature_importance(avg_importance, features, index_name)
    
    return results_df, eco_results_df, cm, (formatted_features, feature_importances_sorted)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train phenology classification models')
    parser.add_argument('--test', '-t', action='store_true', 
                        help='Run in test mode with a small subset of data')
    parser.add_argument('--test_size', type=int, default=10000,
                        help='Number of samples to use in test mode (default: 10000)')
    args = parser.parse_args()

    logger.info("Starting phenology training script")
    if args.test:
        logger.info(f"Running in TEST MODE with {args.test_size} samples")
    
    logger.info("Loading dataset...")
    # Load the entire dataset as requested
    df = pd.read_parquet(DATASET_PATH)
    logger.info(f"Dataset loaded: {len(df)} samples")
    
    # If in test mode, use only a subset of the data
    if args.test:
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
    
    # Check for weight column
    if 'weight' in df.columns:
        weight_stats = {
            'min': df['weight'].min(),
            'max': df['weight'].max(),
            'mean': df['weight'].mean(),
            'median': df['weight'].median(),
            'non_zero': (df['weight'] > 0).sum(),
            'total': len(df)
        }
        logger.info(f"Weight column found in dataset with stats: {weight_stats}")
    else:
        logger.warning("No weight column found in dataset")
    
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

    # Apply cos/sin transformation to phase features
    logger.info("Applying circular transformation to unscaled (radian) phase features...")
    df = transform_circular_features(df, INDICES)
    logger.info("Circular transformation complete.")
    
    # Print dataset info
    logger.info("\n=== Dataset Info ===")
    logger.info(f"Phenology distribution:\n{df['phenology'].value_counts().to_string()}")
    logger.info(f"\nEco-region distribution:\n{df['eco_region'].value_counts().to_string()}")
    logger.info(f"\nNumber of unique tiles: {df['tile_id'].nunique()}")
    
    # Summary results across indices
    summary_results = []
    all_results = {}
    
    # Process each index
    for index in tqdm(INDICES, desc="Processing indices"):
        logger.info(f"\n\n{'='*50}")
        logger.info(f"Training with {index.upper()} features")
        logger.info(f"{'='*50}")
        
        features = get_features(index)
        logger.info(f"Features: {', '.join(features)}")
        
        # Train and evaluate
        start_time = time.time()
        results_df, eco_results_df, cm, feat_importance = train_and_evaluate(df, features, index)
        elapsed_time = time.time() - start_time
        
        logger.info(f"\nTraining completed in {elapsed_time:.2f} seconds")
        
        # Save results
        results_df.to_csv(f"results/models/phenology_{index}_overall_results.csv", index=False)
        eco_results_df.to_csv(f"results/models/phenology_{index}_ecoregion_results.csv", index=False)
        
        # Add to summary
        summary_results.append({
            'index': index,
            'f1_score': results_df['f1_score'].mean(),
            'f1_std': results_df['f1_score'].std(),
            'precision': results_df['precision'].mean(),
            'recall': results_df['recall'].mean(),
            'training_time': elapsed_time
        })
        
        # Store all results for PDF report
        all_results[index] = (results_df, eco_results_df, cm, feat_importance)
    
    # Create summary comparison
    summary_df = pd.DataFrame(summary_results).sort_values('f1_score', ascending=False)
    summary_df.to_csv("results/models/phenology_indices_comparison.csv", index=False)
    
    logger.info("\n=== Indices Comparison ===")
    logger.info("\n" + summary_df[['index', 'f1_score', 'f1_std', 'precision', 'recall', 'training_time']].round(4).to_string())
    
    # Plot comparison of F1 scores
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['index'], summary_df['f1_score'], yerr=summary_df['f1_std'])
    plt.xlabel('Vegetation Index')
    plt.ylabel('F1 Score')
    plt.title('Performance Comparison Across Indices')
    plt.ylim(0, 1)
    plt.savefig("results/models/phenology_indices_comparison.png")
    plt.close()
    
    # Create PDF report
    create_pdf_report(summary_df, all_results, "results/models/phenology_report.pdf")
    
    logger.info("Training script completed successfully")

if __name__ == "__main__":
    main() 