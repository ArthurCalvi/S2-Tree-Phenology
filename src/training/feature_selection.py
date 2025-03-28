"""
Feature Selection for Tree Phenology Classification

This script performs feature importance analysis and selection to find
the optimal subset of Sentinel-2 time-series harmonic features for
tree phenology classification (deciduous vs. evergreen).

Key features:
- Evaluates features across different vegetation indices (NDVI, EVI, NBR, CRSWIR)
- Compares feature importance across different eco-regions
- Uses various feature selection methods (RFE, SelectFromModel, RFECV)
- Applies cross-validation to avoid overfitting
- Generates visualizations of feature importance by index and eco-region
- Supports incremental feature selection to find optimal feature count

Usage:
    python feature_selection.py [--method METHOD] [--output OUTPUT_DIR]

Author: Generated with Claude AI
Date: April 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_score
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import os
import logging
import time
import argparse
from collections import defaultdict
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_selection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create results directory if it doesn't exist
os.makedirs('results/feature_selection', exist_ok=True)

# Define the indices to test
INDICES = ['ndvi', 'evi', 'nbr', 'crswir']

# Define the phenology mapping
PHENOLOGY_MAPPING = {1: 'Deciduous', 2: 'Evergreen'}

# Define the dataset path 
DATASET_PATH = 'results/datasets/training_datasets_pixels.parquet'

def get_all_features():
    """
    Get all features from all indices.
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

def transform_circular_features(df):
    """
    Apply cos/sin transformation to phase features which are circular in nature.
    This prevents the model from treating the phase as a linear feature.
    """
    logger.info("Applying cos/sin transformation to phase features...")
    transformed_df = df.copy()
    
    # Apply transformation for each index
    for index in tqdm(INDICES, desc="Transforming phase features"):
        # Transform phase_h1
        transformed_df[f'{index}_phase_h1_cos'] = np.cos(transformed_df[f'{index}_phase_h1'])
        transformed_df[f'{index}_phase_h1_sin'] = np.sin(transformed_df[f'{index}_phase_h1'])
        
        # Transform phase_h2
        transformed_df[f'{index}_phase_h2_cos'] = np.cos(transformed_df[f'{index}_phase_h2'])
        transformed_df[f'{index}_phase_h2_sin'] = np.sin(transformed_df[f'{index}_phase_h2'])
    
    # Drop the original phase features
    phase_cols = [col for col in transformed_df.columns if ('phase_h' in col and 
                                                         not ('cos' in col or 'sin' in col))]
    transformed_df = transformed_df.drop(columns=phase_cols)
    
    return transformed_df

def evaluate_all_features(df, target='phenology'):
    """
    Evaluate the importance of all features using a Random Forest model.
    """
    logger.info("Evaluating importance of all features...")
    
    # Get all transformed features
    features = get_all_features()
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Train a Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Create a DataFrame with features and their importances
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Group features by index
    index_importance = defaultdict(float)
    for feature, importance in zip(features, importances):
        for index in INDICES:
            if feature.startswith(index):
                index_importance[index] += importance
                break
    
    # Normalize index importances
    total_importance = sum(index_importance.values())
    index_importance = {k: v / total_importance for k, v in index_importance.items()}
    
    return feature_importance_df, index_importance

def feature_importance_by_ecoregion(df, target='phenology'):
    """
    Compute feature importance for each eco-region.
    """
    logger.info("Computing feature importance by eco-region...")
    
    # Get all eco-regions
    eco_regions = df['eco_region'].unique()
    
    # Get all transformed features
    features = get_all_features()
    
    # Store results
    eco_region_importances = {}
    
    # For each eco-region, train a model and get feature importances
    for eco_region in tqdm(eco_regions, desc="Processing eco-regions"):
        # Filter data for this eco-region
        eco_df = df[df['eco_region'] == eco_region]
        
        # Skip eco-regions with too few samples
        if len(eco_df) < 1000:
            logger.warning(f"Skipping eco-region {eco_region} with only {len(eco_df)} samples.")
            continue
        
        # Prepare data
        X = eco_df[features]
        y = eco_df[target]
        
        # Train a Random Forest model
        rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Store importances
        eco_region_importances[eco_region] = {
            'features': features,
            'importances': importances
        }
    
    return eco_region_importances

def recursive_feature_elimination(df, n_features_to_select=10, target='phenology'):
    """
    Use Recursive Feature Elimination to select the top features.
    """
    logger.info(f"Performing Recursive Feature Elimination to select {n_features_to_select} features...")
    
    # Get all transformed features
    features = get_all_features()
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Use Random Forest as the estimator
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Create RFE object
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)
    
    # Fit RFE
    rfe.fit(X, y)
    
    # Get selected features
    selected_features = [feature for feature, selected in zip(features, rfe.support_) if selected]
    
    # Get feature ranking
    feature_ranking = pd.DataFrame({
        'Feature': features,
        'Rank': rfe.ranking_
    }).sort_values('Rank')
    
    return selected_features, feature_ranking

def select_features_from_model(df, threshold='median', target='phenology'):
    """
    Use SelectFromModel to select important features based on a threshold.
    """
    logger.info(f"Using SelectFromModel with threshold '{threshold}'...")
    
    # Get all transformed features
    features = get_all_features()
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Use Random Forest as the estimator
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Create SelectFromModel object
    sfm = SelectFromModel(rf, threshold=threshold)
    
    # Fit SelectFromModel
    sfm.fit(X, y)
    
    # Get selected features
    selected_features = [feature for feature, selected in zip(features, sfm.get_support()) if selected]
    
    return selected_features

def cross_validated_feature_selection(df, target='phenology', n_splits=5):
    """
    Use RFECV (Recursive Feature Elimination with Cross-Validation) to select features.
    """
    logger.info(f"Performing cross-validated feature selection with {n_splits} splits...")
    
    # Get all transformed features
    features = get_all_features()
    
    # Prepare data
    X = df[features]
    y = df[target]
    
    # Use Random Forest as the estimator
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Create RFECV object
    rfecv = RFECV(estimator=rf, step=1, cv=n_splits, scoring='f1', n_jobs=-1)
    
    # Fit RFECV
    logger.info("Fitting RFECV (this may take some time)...")
    rfecv.fit(X, y)
    
    # Get selected features
    selected_features = [feature for feature, selected in zip(features, rfecv.support_) if selected]
    
    logger.info(f"Optimal number of features: {rfecv.n_features_}")
    
    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (F1)")
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
    plt.savefig('results/feature_selection/cv_feature_selection.png')
    plt.close()
    
    return selected_features, rfecv.cv_results_

def incremental_feature_selection(df, target='phenology', max_features=32, n_splits=5):
    """
    Incrementally add features based on importance and evaluate performance.
    """
    logger.info("Performing incremental feature selection...")
    
    # Get feature importance
    feature_importance_df, _ = evaluate_all_features(df, target)
    
    # Sort features by importance
    sorted_features = feature_importance_df['Feature'].tolist()
    
    # Store results
    results = []
    
    # For each number of features
    for n_features in tqdm(range(1, min(max_features, len(sorted_features)) + 1),
                          desc="Testing feature counts"):
        # Select top n_features
        selected_features = sorted_features[:n_features]
        
        # Prepare data
        X = df[selected_features]
        y = df[target]
        
        # Train and evaluate with cross-validation
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X, y, cv=n_splits, scoring='f1', n_jobs=-1)
        
        # Store results
        results.append({
            'n_features': n_features,
            'mean_f1': scores.mean(),
            'std_f1': scores.std(),
            'selected_features': selected_features
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        results_df['n_features'], 
        results_df['mean_f1'], 
        yerr=results_df['std_f1'], 
        fmt='-o'
    )
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Number of Features')
    plt.grid(True)
    plt.savefig('results/feature_selection/incremental_feature_selection.png')
    plt.close()
    
    # Find optimal number of features
    optimal_result = results_df.loc[results_df['mean_f1'].idxmax()]
    logger.info(f"Optimal number of features: {optimal_result['n_features']}")
    logger.info(f"Selected features: {optimal_result['selected_features']}")
    
    return results_df, optimal_result

def plot_feature_importance(feature_importance_df, output_dir='results/feature_selection'):
    """
    Plot overall feature importance as a horizontal bar chart.
    """
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=True)
    
    # Extract index and feature type
    feature_importance_df['Index'] = feature_importance_df['Feature'].apply(
        lambda x: x.split('_')[0]
    )
    feature_importance_df['Feature_Type'] = feature_importance_df['Feature'].apply(
        lambda x: '_'.join(x.split('_')[1:])
    )
    
    # Plot
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        x='Importance', 
        y='Feature', 
        hue='Index',
        data=feature_importance_df,
        palette='viridis'
    )
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    # Plot by index
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Index', 
        y='Importance', 
        data=feature_importance_df.groupby('Index')['Importance'].sum().reset_index(),
        palette='viridis'
    )
    plt.title('Importance by Vegetation Index')
    plt.xlabel('Vegetation Index')
    plt.ylabel('Summed Importance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/index_importance.png')
    plt.close()
    
    # Plot heatmap of importance by index and feature type
    pivot_df = feature_importance_df.pivot_table(
        index='Feature_Type', 
        columns='Index', 
        values='Importance',
        aggfunc='sum'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Feature Importance Heatmap (Index vs Feature Type)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_heatmap.png')
    plt.close()

def plot_eco_region_importance(eco_region_importances, output_dir='results/feature_selection'):
    """
    Plot feature importance for each eco-region and compare them.
    """
    # Create a directory for eco-region plots
    eco_dir = f'{output_dir}/eco_regions'
    os.makedirs(eco_dir, exist_ok=True)
    
    # Create a DataFrame to store top features for each eco-region
    top_features_df = pd.DataFrame()
    
    # Plot for each eco-region
    for eco_region, data in eco_region_importances.items():
        features = data['features']
        importances = data['importances']
        
        # Create a DataFrame for this eco-region
        eco_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances,
            'Eco_Region': eco_region
        }).sort_values('Importance', ascending=False)
        
        # Store top 10 features
        top_features = eco_df.head(10)
        top_features_df = pd.concat([top_features_df, top_features])
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Importance', 
            y='Feature', 
            data=eco_df.head(20),
            palette='viridis'
        )
        plt.title(f'Top 20 Features for {eco_region}')
        plt.tight_layout()
        plt.savefig(f'{eco_dir}/{eco_region}_top_features.png')
        plt.close()
    
    # Create a heatmap of top features across eco-regions
    pivot_df = top_features_df.pivot_table(
        index='Feature', 
        columns='Eco_Region', 
        values='Importance',
        fill_value=0
    )
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Top Features Importance Across Eco-Regions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eco_region_feature_comparison.png')
    plt.close()
    
    # Calculate correlation matrix between eco-regions
    eco_importance_matrix = defaultdict(dict)
    all_features = list(set([f for data in eco_region_importances.values() for f in data['features']]))
    
    for eco1 in eco_region_importances:
        feat1 = eco_region_importances[eco1]['features']
        imp1 = eco_region_importances[eco1]['importances']
        feat_imp1 = dict(zip(feat1, imp1))
        
        for eco2 in eco_region_importances:
            if eco1 != eco2:
                feat2 = eco_region_importances[eco2]['features']
                imp2 = eco_region_importances[eco2]['importances']
                feat_imp2 = dict(zip(feat2, imp2))
                
                # Calculate correlation between importance vectors
                common_features = set(feat1).intersection(set(feat2))
                if common_features:
                    x = [feat_imp1[f] for f in common_features]
                    y = [feat_imp2[f] for f in common_features]
                    correlation = np.corrcoef(x, y)[0, 1]
                    eco_importance_matrix[eco1][eco2] = correlation
    
    # Convert to DataFrame
    correlation_df = pd.DataFrame(eco_importance_matrix)
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation of Feature Importance Between Eco-Regions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eco_region_correlation.png')
    plt.close()

def create_feature_selection_report(feature_importance_df, index_importance, 
                                    eco_region_importances, rfe_features,
                                    sfm_features, rfecv_features, 
                                    incremental_results, output_path):
    """
    Create a comprehensive PDF report of feature selection results.
    """
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
        
        # Overall feature importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        ax.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
        ax.set_title('Top 20 Features by Importance')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Index importance
        plt.figure(figsize=(10, 6))
        plt.bar(index_importance.keys(), index_importance.values())
        plt.title('Importance by Vegetation Index')
        plt.ylabel('Normalized Importance')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Feature importance heatmap
        feature_importance_df['Index'] = feature_importance_df['Feature'].apply(
            lambda x: x.split('_')[0]
        )
        feature_importance_df['Feature_Type'] = feature_importance_df['Feature'].apply(
            lambda x: '_'.join(x.split('_')[1:])
        )
        
        pivot_df = feature_importance_df.pivot_table(
            index='Feature_Type', 
            columns='Index', 
            values='Importance',
            aggfunc='sum'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Feature Importance Heatmap (Index vs Feature Type)')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Selected features by different methods
        methods = {
            'RFE': rfe_features,
            'SelectFromModel': sfm_features,
            'RFECV': rfecv_features,
            'Incremental': incremental_results['selected_features']
        }
        
        plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2)
        
        for i, (method, features) in enumerate(methods.items()):
            ax = plt.subplot(gs[i])
            ax.axis('off')
            ax.set_title(f'Selected Features using {method}')
            features_text = "\n".join([f"- {f}" for f in features[:15]])
            if len(features) > 15:
                features_text += f"\n... and {len(features) - 15} more"
            ax.text(0.1, 0.9, features_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Incremental feature selection results
        incremental_df = pd.DataFrame(incremental_results)
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(
            incremental_df['n_features'], 
            incremental_df['mean_f1'], 
            yerr=incremental_df['std_f1'], 
            fmt='-o'
        )
        plt.xlabel('Number of Features')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Number of Features')
        plt.grid(True)
        pdf.savefig()
        plt.close()
        
        # Eco-region feature importance
        for eco_region, data in eco_region_importances.items():
            features = data['features']
            importances = data['importances']
            
            # Create a DataFrame for this eco-region
            eco_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            plt.barh(eco_df['Feature'][:15], eco_df['Importance'][:15])
            plt.title(f'Top 15 Features for {eco_region}')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def calculate_feature_correlation(df, features, target='phenology'):
    """
    Calculate correlation between features and target variable.
    
    For binary classification, we use point-biserial correlation coefficient,
    which is mathematically equivalent to Pearson correlation when one variable is binary.
    
    Returns:
        DataFrame with feature names and their correlation with target
    """
    logger.info("Calculating feature correlation with target variable...")
    
    corr_results = []
    
    for feature in tqdm(features, desc="Computing correlations"):
        # Calculate point-biserial correlation (equivalent to Pearson for binary target)
        correlation = df[feature].corr(df[target])
        corr_results.append({
            'Feature': feature,
            'Correlation': abs(correlation),  # Use absolute value for ranking purposes
            'Raw_Correlation': correlation    # Keep raw value for direction
        })
    
    # Convert to DataFrame and sort
    corr_df = pd.DataFrame(corr_results).sort_values('Correlation', ascending=False)
    
    return corr_df

def analyze_features_by_ecoregion(df, target='phenology'):
    """
    Analyze feature correlation with target across different eco-regions.
    
    For each eco-region, calculate correlation of each feature with target.
    
    Returns:
        Dictionary with eco-region analysis results
    """
    logger.info("Analyzing features by eco-region...")
    
    # Get all eco-regions
    eco_regions = df['eco_region'].unique()
    
    # Get all features
    features = get_all_features()
    
    # Store results
    eco_region_analysis = {}
    
    # For each eco-region, calculate correlation
    for eco_region in tqdm(eco_regions, desc="Processing eco-regions"):
        # Filter data for this eco-region
        eco_df = df[df['eco_region'] == eco_region]
        
        # Skip eco-regions with too few samples
        if len(eco_df) < 100:
            logger.warning(f"Skipping eco-region {eco_region} with only {len(eco_df)} samples.")
            continue
            
        # Check if we have both classes in this eco-region
        if len(eco_df[target].unique()) < 2:
            logger.warning(f"Skipping eco-region {eco_region} with only one class.")
            continue
        
        # Calculate correlation
        corr_df = calculate_feature_correlation(eco_df, features, target)
        
        # Store results
        eco_region_analysis[eco_region] = {
            'correlation': corr_df,
            'sample_size': len(eco_df)
        }
    
    return eco_region_analysis

def plot_correlation_by_feature(corr_df, output_dir='results/feature_selection'):
    """
    Plot correlation of features with target variable.
    """
    # Sort by correlation
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    # Extract index and feature type
    corr_df['Index'] = corr_df['Feature'].apply(
        lambda x: x.split('_')[0]
    )
    corr_df['Feature_Type'] = corr_df['Feature'].apply(
        lambda x: '_'.join(x.split('_')[1:])
    )
    
    # Plot top 20 features
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        x='Correlation', 
        y='Feature', 
        hue='Index',
        data=corr_df.head(20),
        palette='viridis'
    )
    plt.title('Top 20 Features by Correlation with Target')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlation.png')
    plt.close()
    
    # Plot by index
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Index', 
        y='Correlation', 
        data=corr_df.groupby('Index')['Correlation'].mean().reset_index(),
        palette='viridis'
    )
    plt.title('Average Correlation by Vegetation Index')
    plt.xlabel('Vegetation Index')
    plt.ylabel('Mean Correlation')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/index_correlation.png')
    plt.close()
    
    # Plot heatmap of correlation by index and feature type
    pivot_df = corr_df.pivot_table(
        index='Feature_Type', 
        columns='Index', 
        values='Correlation',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Feature Correlation Heatmap (Index vs Feature Type)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlation_heatmap.png')
    plt.close()

def plot_feature_across_ecoregions(eco_region_analysis, output_dir='results/feature_selection'):
    """
    Plot how feature correlations vary across eco-regions.
    """
    # Create directories
    eco_dir = f'{output_dir}/eco_regions'
    os.makedirs(eco_dir, exist_ok=True)
    
    # Collect top features for each eco-region
    top_features_df = pd.DataFrame()
    
    for eco_region, data in eco_region_analysis.items():
        corr_df = data['correlation'].copy()
        corr_df['Eco_Region'] = eco_region
        corr_df['Sample_Size'] = data['sample_size']
        
        # Get top 10 features
        top_features = corr_df.head(10)
        top_features_df = pd.concat([top_features_df, top_features])
        
        # Plot top 20 features for this eco-region
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='Correlation', 
            y='Feature', 
            data=corr_df.head(20),
            palette='viridis'
        )
        plt.title(f'Top 20 Features by Correlation for Eco-Region {eco_region}')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.savefig(f'{eco_dir}/{eco_region}_correlation.png')
        plt.close()
    
    # Create a heatmap for top features across eco-regions
    pivot_df = top_features_df.pivot_table(
        index='Feature', 
        columns='Eco_Region', 
        values='Correlation',
        fill_value=0
    )
    
    # Sort features by average value across eco-regions
    pivot_df['Average'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('Average', ascending=False).drop(columns=['Average'])
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Top Features Correlation Across Eco-Regions')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eco_region_correlation_comparison.png')
    plt.close()
    
    # Create consistency analysis
    feature_counts = top_features_df['Feature'].value_counts()
    top_consistent = feature_counts[feature_counts > 1].index.tolist()
    
    if top_consistent:
        # For each consistent feature, plot its value across eco-regions
        for feature in top_consistent[:min(10, len(top_consistent))]:
            feature_data = []
            for eco_region, data in eco_region_analysis.items():
                df = data['correlation']
                feature_row = df[df['Feature'] == feature]
                if not feature_row.empty:
                    feature_data.append({
                        'Eco_Region': eco_region,
                        'Correlation': feature_row.iloc[0]['Correlation'],
                        'Sample_Size': data['sample_size']
                    })
            
            if feature_data:
                feature_df = pd.DataFrame(feature_data)
                
                plt.figure(figsize=(14, 6))
                # Sort by correlation value
                feature_df = feature_df.sort_values('Correlation', ascending=False)
                
                # Calculate alpha values
                max_sample = feature_df['Sample_Size'].max()
                
                # Create bars individually with their respective alpha values
                bars = []
                for i, (_, row) in enumerate(feature_df.iterrows()):
                    alpha = row['Sample_Size'] / max_sample
                    bar = plt.bar(i, row['Correlation'], alpha=alpha)
                    bars.append(bar[0])  # bar is a container with one Rectangle
                
                # Set x-axis labels
                plt.xticks(range(len(feature_df)), feature_df['Eco_Region'], rotation=90)
                
                # Add sample size as text on bars
                for bar, sample in zip(bars, feature_df['Sample_Size']):
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        0.05,
                        f'n={sample}',
                        ha='center', 
                        va='bottom', 
                        rotation=90
                    )
                
                plt.title(f'{feature} Correlation Across Eco-Regions')
                plt.ylabel('Correlation')
                plt.tight_layout()
                plt.savefig(f'{eco_dir}/{feature}_correlation.png')
                plt.close()

def create_correlation_report(overall_corr_df, eco_region_analysis, output_path):
    """
    Create a comprehensive PDF report of feature correlation analysis.
    """
    with PdfPages(output_path) as pdf:
        # Title page
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Tree Phenology Classification\nFeature Correlation Analysis',
                ha='center', va='center', fontsize=24)
        plt.text(0.5, 0.4, f'Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Overall correlation section
        plt.figure(figsize=(12, 8))
        plt.barh(overall_corr_df['Feature'][:15], overall_corr_df['Correlation'][:15])
        plt.title('Top 15 Features by Correlation with Target')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Correlation by vegetation index
        overall_corr_df['Index'] = overall_corr_df['Feature'].apply(lambda x: x.split('_')[0])
        index_corr = overall_corr_df.groupby('Index')['Correlation'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(index_corr['Index'], index_corr['Correlation'])
        plt.title('Average Correlation by Vegetation Index')
        plt.ylabel('Mean Absolute Correlation')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # For each eco-region, show top correlated features
        for eco_region, data in eco_region_analysis.items():
            corr_df = data['correlation']
            
            plt.figure(figsize=(12, 6))
            plt.barh(corr_df['Feature'][:10], corr_df['Correlation'][:10])
            plt.title(f'Top 10 Features by Correlation for {eco_region}')
            plt.xlabel('Absolute Correlation')
            plt.tight_layout()
            pdf.savefig()
            plt.close()

def main():
    """
    Main function to run the feature correlation analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature correlation analysis for tree phenology classification.')
    parser.add_argument('--output', type=str, default='results/feature_selection',
                        help='Output directory for results.')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Size of random sample to use (if None, use full dataset).')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                        help='Path to the dataset parquet file.')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}...")
    df = pd.read_parquet(args.dataset_path)
    
    # Sample data if needed
    if args.sample_size is not None:
        logger.info(f"Taking a random sample of {args.sample_size} rows...")
        df = df.sample(args.sample_size, random_state=42)
    
    # Apply cos/sin transformation to circular features
    df = transform_circular_features(df)
    
    # Get all features
    features = get_all_features()
    
    # Calculate overall correlation
    logger.info("Calculating overall feature correlation...")
    overall_corr_df = calculate_feature_correlation(df, features)
    
    # Plot overall correlation results
    logger.info("Plotting correlation results...")
    plot_correlation_by_feature(overall_corr_df, args.output)
    
    # Analyze features by eco-region
    logger.info("Analyzing features by eco-region...")
    eco_region_analysis = analyze_features_by_ecoregion(df)
    
    # Plot feature correlation across eco-regions
    logger.info("Plotting feature correlation across eco-regions...")
    plot_feature_across_ecoregions(eco_region_analysis, output_dir=args.output)
    
    # Create comprehensive report
    logger.info("Creating correlation analysis report...")
    create_correlation_report(
        overall_corr_df,
        eco_region_analysis,
        f"{args.output}/feature_correlation_report.pdf"
    )
    
    logger.info("Feature correlation analysis completed successfully!")

if __name__ == "__main__":
    main() 