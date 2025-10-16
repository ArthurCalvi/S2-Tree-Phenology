#!/usr/bin/env python3
"""
QA Report Generator for No-Data Differences Analysis

This script generates a comprehensive report with visualizations to analyze
the differences in no-data percentages between mosaics and features.
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.utils import apply_science_style
apply_science_style()
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import matplotlib.gridspec as gridspec

def load_data(json_path, agg_json_path):
    """Load the detailed and aggregated JSON data."""
    with open(json_path, 'r') as f:
        detailed_data = json.load(f)
    
    with open(agg_json_path, 'r') as f:
        agg_data = json.load(f)
    
    return detailed_data, agg_data

def prepare_dataframes(detailed_data, agg_data):
    """Convert JSON data to pandas DataFrames for easier analysis."""
    # Detailed data
    rows = []
    for tile_name, tile_info in detailed_data['tiles'].items():
        for band in tile_info['bands']:
            rows.append({
                'tile': tile_name,
                'filename': tile_info['filename'],
                'cloud_pct': tile_info['cloud_pct'],
                'band': band['name'],
                'mosaic_no_data': band['mosaic_no_data'],
                'feature_no_data': band['feature_no_data'],
                'difference': band['difference']
            })
    
    df_detailed = pd.DataFrame(rows)
    
    # Parse band names to extract index and feature type
    df_detailed['index'] = df_detailed['band'].apply(lambda x: x.split('_')[0])
    df_detailed['feature_type'] = df_detailed['band'].apply(
        lambda x: '_'.join(x.split('_')[1:]) if len(x.split('_')) > 1 else 'unknown'
    )
    
    # Aggregated data
    df_agg = pd.DataFrame(agg_data['by_index_feature'])
    
    return df_detailed, df_agg

def create_report(df_detailed, df_agg, output_path):
    """Generate a comprehensive PDF report with visualizations."""
    with PdfPages(output_path) as pdf:
        # Title page
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.5, 'No-Data Differences Analysis Report', 
                 fontsize=24, ha='center', va='center')
        plt.text(0.5, 0.45, 'Comparing Mosaic and Feature No-Data Percentages', 
                 fontsize=16, ha='center', va='center')
        plt.text(0.5, 0.4, f'Generated on {pd.Timestamp.now().strftime("%Y-%m-%d")}', 
                 fontsize=12, ha='center', va='center')
        pdf.savefig()
        plt.close()
        
        # 1. Executive Summary
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Executive Summary', fontsize=20, ha='center')
        
        # Calculate key metrics
        total_tiles = df_detailed['tile'].nunique()
        total_bands = df_detailed['band'].nunique()
        avg_diff = df_detailed['difference'].mean()
        max_diff = df_detailed['difference'].max()
        max_diff_band = df_detailed.loc[df_detailed['difference'].idxmax(), 'band']
        max_diff_tile = df_detailed.loc[df_detailed['difference'].idxmax(), 'tile']
        avg_cloud = df_detailed['cloud_pct'].mean()
        
        summary_text = [
            f"Total Tiles Analyzed: {total_tiles}",
            f"Total Bands Analyzed: {total_bands}",
            f"Average No-Data Difference: {avg_diff:.2f}%",
            f"Maximum No-Data Difference: {max_diff:.2f}% (Band: {max_diff_band}, Tile: {max_diff_tile})",
            f"Average Cloud Coverage: {avg_cloud:.2f}%",
            "",
            "Key Findings:",
            "1. " + ("High" if avg_diff > 5 else "Low") + f" average no-data difference ({avg_diff:.2f}%) indicates " + 
            ("potential issues" if avg_diff > 5 else "good alignment") + " between mosaics and features.",
            "2. " + ("High" if max_diff > 20 else "Low") + f" maximum difference ({max_diff:.2f}%) suggests " + 
            ("specific bands or tiles require attention" if max_diff > 20 else "consistent processing across all bands and tiles."),
            "3. " + ("High" if avg_cloud > 10 else "Low") + f" cloud coverage ({avg_cloud:.2f}%) " + 
            ("may impact" if avg_cloud > 10 else "has minimal impact on") + " the quality of the data."
        ]
        
        for i, line in enumerate(summary_text):
            plt.text(0.1, 0.85 - i*0.05, line, fontsize=12)
        
        pdf.savefig()
        plt.close()
        
        # 2. Overall Distribution of Differences
        plt.figure(figsize=(12, 8))
        plt.suptitle('Distribution of No-Data Differences', fontsize=16)
        
        plt.subplot(2, 2, 1)
        sns.histplot(df_detailed['difference'], kde=True)
        plt.title('Histogram of Differences')
        plt.xlabel('Difference (%)')
        plt.ylabel('Count')
        
        plt.subplot(2, 2, 2)
        sns.boxplot(y=df_detailed['difference'])
        plt.title('Boxplot of Differences')
        plt.ylabel('Difference (%)')
        
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df_detailed, x='mosaic_no_data', y='feature_no_data', alpha=0.5)
        plt.plot([0, 100], [0, 100], 'r--')  # Diagonal line for reference
        plt.title('Mosaic vs Feature No-Data')
        plt.xlabel('Mosaic No-Data (%)')
        plt.ylabel('Feature No-Data (%)')
        
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df_detailed, x='cloud_pct', y='difference', alpha=0.5)
        plt.title('Cloud Coverage vs Difference')
        plt.xlabel('Cloud Coverage (%)')
        plt.ylabel('Difference (%)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # 3. Analysis by Index
        plt.figure(figsize=(12, 10))
        plt.suptitle('No-Data Differences by Spectral Index', fontsize=16)
        
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df_detailed, x='index', y='difference')
        plt.title('Distribution of Differences by Index')
        plt.xlabel('Spectral Index')
        plt.ylabel('Difference (%)')
        
        plt.subplot(2, 1, 2)
        index_avg = df_detailed.groupby('index')['difference'].mean().sort_values(ascending=False)
        sns.barplot(x=index_avg.index, y=index_avg.values)
        plt.title('Average Difference by Index')
        plt.xlabel('Spectral Index')
        plt.ylabel('Average Difference (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # 4. Analysis by Feature Type
        plt.figure(figsize=(12, 10))
        plt.suptitle('No-Data Differences by Feature Type', fontsize=16)
        
        plt.subplot(2, 1, 1)
        sns.boxplot(data=df_detailed, x='feature_type', y='difference')
        plt.title('Distribution of Differences by Feature Type')
        plt.xlabel('Feature Type')
        plt.ylabel('Difference (%)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 1, 2)
        feature_avg = df_detailed.groupby('feature_type')['difference'].mean().sort_values(ascending=False)
        sns.barplot(x=feature_avg.index, y=feature_avg.values)
        plt.title('Average Difference by Feature Type')
        plt.xlabel('Feature Type')
        plt.ylabel('Average Difference (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # 5. Heatmap of Index vs Feature Type
        plt.figure(figsize=(14, 10))
        
        # Create a pivot table for the heatmap
        heatmap_data = df_detailed.pivot_table(
            values='difference', 
            index='index', 
            columns='feature_type', 
            aggfunc='mean'
        )
        
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('Average Difference by Index and Feature Type', fontsize=16)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 6. Top 10 Largest Differences
        plt.figure(figsize=(12, 8))
        top_diff = df_detailed.nlargest(10, 'difference')
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=top_diff, y='band', x='difference', palette='viridis')
        plt.title('Top 10 Bands with Largest Differences')
        plt.xlabel('Difference (%)')
        plt.ylabel('Band')
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=top_diff, y='tile', x='difference', palette='viridis')
        plt.title('Tiles with Largest Differences')
        plt.xlabel('Difference (%)')
        plt.ylabel('Tile')
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 7. Cloud Coverage Analysis
        plt.figure(figsize=(12, 8))
        plt.suptitle('Cloud Coverage Analysis', fontsize=16)
        
        plt.subplot(2, 2, 1)
        sns.histplot(df_detailed['cloud_pct'].unique(), kde=True)
        plt.title('Distribution of Cloud Coverage')
        plt.xlabel('Cloud Coverage (%)')
        plt.ylabel('Count')
        
        plt.subplot(2, 2, 2)
        cloud_groups = pd.cut(df_detailed['cloud_pct'], bins=[0, 5, 10, 20, 100], 
                             labels=['0-5%', '5-10%', '10-20%', '>20%'])
        cloud_diff = df_detailed.groupby(cloud_groups)['difference'].mean()
        sns.barplot(x=cloud_diff.index, y=cloud_diff.values)
        plt.title('Average Difference by Cloud Coverage')
        plt.xlabel('Cloud Coverage')
        plt.ylabel('Average Difference (%)')
        
        plt.subplot(2, 2, 3)
        top_cloudy = df_detailed.drop_duplicates('tile').nlargest(10, 'cloud_pct')
        sns.barplot(data=top_cloudy, y='tile', x='cloud_pct')
        plt.title('Top 10 Tiles with Highest Cloud Coverage')
        plt.xlabel('Cloud Coverage (%)')
        plt.ylabel('Tile')
        
        plt.subplot(2, 2, 4)
        sns.regplot(data=df_detailed, x='cloud_pct', y='difference', scatter_kws={'alpha':0.3})
        plt.title('Correlation: Cloud Coverage vs Difference')
        plt.xlabel('Cloud Coverage (%)')
        plt.ylabel('Difference (%)')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # 8. Recommendations
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        plt.text(0.5, 0.95, 'Recommendations', fontsize=20, ha='center')
        
        # Determine recommendations based on the data
        high_diff_indices = df_detailed.groupby('index')['difference'].mean()
        high_diff_indices = high_diff_indices[high_diff_indices > high_diff_indices.mean()]
        
        high_diff_features = df_detailed.groupby('feature_type')['difference'].mean()
        high_diff_features = high_diff_features[high_diff_features > high_diff_features.mean()]
        
        high_cloud_impact = np.corrcoef(df_detailed['cloud_pct'], df_detailed['difference'])[0, 1]
        
        recommendations = [
            "Based on the analysis, we recommend the following actions:",
            "",
            "1. Data Quality Improvements:",
            f"   - {'Review and potentially reprocess' if avg_diff > 5 else 'Continue monitoring'} the overall no-data handling in the processing pipeline.",
            f"   - {'Investigate' if max_diff > 20 else 'Note'} the maximum difference of {max_diff:.2f}% in band {max_diff_band} for tile {max_diff_tile}.",
            "",
            "2. Index-Specific Recommendations:",
        ]
        
        for idx, val in high_diff_indices.items():
            recommendations.append(f"   - {idx}: Review processing parameters (avg diff: {val:.2f}%)")
        
        recommendations.extend([
            "",
            "3. Feature Type Recommendations:",
        ])
        
        for feat, val in high_diff_features.items():
            recommendations.append(f"   - {feat}: Evaluate extraction methodology (avg diff: {val:.2f}%)")
        
        recommendations.extend([
            "",
            "4. Cloud Coverage Handling:",
            f"   - {'Improve' if avg_cloud > 10 else 'Maintain'} cloud masking procedures (avg coverage: {avg_cloud:.2f}%).",
            f"   - Cloud coverage has a {'significant' if abs(high_cloud_impact) > 0.3 else 'minimal'} impact on no-data differences " +
            f"(correlation: {high_cloud_impact:.2f})."
        ])
        
        for i, line in enumerate(recommendations):
            plt.text(0.1, 0.85 - i*0.03, line, fontsize=12)
        
        pdf.savefig()
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate QA report for no-data differences analysis.")
    parser.add_argument("--detailed-json", type=str, required=True,
                        help="Path to the detailed JSON file (nodata_differences.json)")
    parser.add_argument("--agg-json", type=str, required=True,
                        help="Path to the aggregated JSON file (nodata_aggregated.json)")
    parser.add_argument("--output", type=str, default="results/QA/qa_report.pdf",
                        help="Path to the output PDF report")
    args = parser.parse_args()
    
    # Load data
    detailed_data, agg_data = load_data(args.detailed_json, args.agg_json)
    
    # Prepare DataFrames
    df_detailed, df_agg = prepare_dataframes(detailed_data, agg_data)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create report
    create_report(df_detailed, df_agg, args.output)
    
    print(f"Report generated successfully: {args.output}")

if __name__ == "__main__":
    main()