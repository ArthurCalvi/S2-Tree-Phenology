#!/usr/bin/env python3
"""
This test script compares different IRLS iteration parameters (1, 5, and 10) on a
single 1024x1024 window. We evaluate performance by measuring:
1. Inference speed
2. Nodata percentage
3. Statistical differences (min, max, mean, std)

We generate a comprehensive report showing the comparison results.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # So we can write PDF without a display
import matplotlib.pyplot as plt
from src.utils import apply_science_style
apply_science_style()
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from rasterio.windows import Window
import logging
import time
import argparse
from typing import List, Optional

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging to write to file with timestamp
logging.basicConfig(
    level=logging.DEBUG,
    filename=f'logs/test_compare_irls_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Adjust imports to your actual project structure:
sys.path.append(str(Path(__file__).parent.parent))  # add the src directory to path
from features.inference_feature import WindowFeature
from constants import AVAILABLE_INDICES, TARGET_AMP_RANGE, TARGET_OFFSET_RANGE

def calculate_nodata_percentage(arr):
    """Calculate the percentage of NaN values in an array."""
    if isinstance(arr, np.ma.MaskedArray):
        return np.sum(arr.mask) / arr.size * 100
    else:
        return np.sum(np.isnan(arr)) / arr.size * 100

def test_compare_irls_iterations(
    tif_paths: Optional[List[Path]] = None,
    tif_dates: Optional[List[datetime]] = None,
    irls_iterations: List[int] = [1, 5, 10],
    window_size: int = 1024,
    output_dir: Path = Path("results/test")
):
    print("\n=== Starting IRLS iteration comparison test ===")
    
    # 1) Build the list of TIF paths for each month in 2023 if not provided
    if tif_paths is None or tif_dates is None:
        print("\n1. Building list of TIF paths...")
        base_dir = Path("data/mosaics/2023")
        monthly_subfolders = [
            "20230115", "20230215", "20230315", "20230415", "20230515", "20230615",
            "20230715", "20230815", "20230915", "20231015", "20231115", "20231215"
        ]
        tif_paths = []
        tif_dates = []

        for folder in monthly_subfolders:
            tif_path = base_dir / folder / "s2" / "s2_EPSG2154_512000_6860800.tif"
            if not tif_path.exists():
                raise FileNotFoundError(f"Missing TIF: {tif_path}")
            tif_paths.append(tif_path)
            year = 2023
            month = int(folder[4:6])
            day = int(folder[6:8])
            tif_dates.append(datetime(year, month, day))
    
    print(f"Found {len(tif_paths)} TIF files")

    # 2) Define our window from the top-left corner
    print(f"\n2. Defining {window_size}x{window_size} window...")
    w = Window(col_off=0, row_off=0, width=window_size, height=window_size)
    print(f"Window parameters: {w}")

    # 3) Define IRLS iteration values to test
    irls_iterations = [1, 5, 10]
    
    # Create a dictionary to store results for each iteration value
    results = {}
    feature_cubes = {}

    # 4) Run tests for each IRLS iteration value
    for iter_val in irls_iterations:
        print(f"\n3. Running test with max_iter={iter_val}...")
        
        # Create WindowFeature object with current max_iter value
        logger = logging.getLogger(f"max_iter_{iter_val}")
        window_feature = WindowFeature(
            tif_paths=tif_paths,
            dates=tif_dates,
            window=w,
            num_harmonics=2,
            max_iter=iter_val,
            logger=logger
        )
        
        # Compute features and measure time
        start_time = time.time()
        feature_cube = window_feature.compute_features()
        computation_time = time.time() - start_time
        
        print(f"Feature computation complete. Shape: {feature_cube.shape}")
        print(f"Computation time: {computation_time:.2f} seconds")
        
        # Store results
        feature_cubes[iter_val] = feature_cube
        results[iter_val] = {
            'time': computation_time,
            'shape': feature_cube.shape,
            'band_stats': []
        }
        
    # 5) Calculate statistics for each parameter configuration
    print("\n4. Calculating statistics and comparisons...")
    
    # Define feature names for easier reference
    feature_names = ["Amplitude (h1)", "Amplitude (h2)", "Phase (h1)", "Phase (h2)", "Offset", "Residual Variance"]
    
    # Process each band for each iteration value
    for iter_val, feature_cube in feature_cubes.items():
        for band_i in range(feature_cube.shape[0]):
            band_data = feature_cube[band_i].astype(float)
            
            # Determine which index and feature type this band corresponds to
            group_index = band_i // len(feature_names)
            feature_type = feature_names[band_i % len(feature_names)]
            indice_label = AVAILABLE_INDICES[group_index].lower() if group_index < len(AVAILABLE_INDICES) else "unknown"
            
            # Convert to physical values for analysis
            if "Amplitude" in feature_type:
                max_amp = TARGET_AMP_RANGE[indice_label][1]
                band_data = (band_data / 65535.0) * max_amp
            elif "Offset" in feature_type:
                max_offset = TARGET_OFFSET_RANGE[indice_label][1]
                band_data = (band_data / 65535.0) * max_offset
            elif "Phase" in feature_type:
                band_data = (band_data / 65535.0) * (2 * np.pi) - np.pi
            
            # Calculate statistics
            nodata_pct = calculate_nodata_percentage(band_data)
            stats = {
                'index': AVAILABLE_INDICES[group_index],
                'feature': feature_type,
                'nodata_pct': nodata_pct,
                'min': np.nanmin(band_data),
                'max': np.nanmax(band_data),
                'mean': np.nanmean(band_data),
                'std': np.nanstd(band_data)
            }
            
            results[iter_val]['band_stats'].append(stats)
    
    # 6) Generate a comprehensive PDF report
    print("\n5. Generating PDF report...")
    
    today_str = datetime.now().strftime("%Y%m%d")
    results_dir = output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = f"irls_comparison_test_{window_size}x{window_size}_{today_str}.pdf"
    pdf_path = results_dir / pdf_name
    
    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("IRLS Iteration Parameter Comparison", fontsize=16)
        plt.axis('off')
        
        info_text = (
            f"Comparison of IRLS Iteration Parameters (max_iter): {', '.join(map(str, irls_iterations))}\n\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Test Parameters:\n"
            f" - Window size: {window_size}×{window_size}\n"
            f" - Data source: {len(tif_paths)} TIFs\n"
            f" - Number of harmonics: 2\n"
        )
        
        plt.text(0.1, 0.6, info_text, fontsize=12)
        pdf.savefig(fig)
        plt.close(fig)
        
        # Performance comparison page
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        fig.suptitle("Performance Comparison", fontsize=16)
        
        # Plot computation time
        times = [results[iter_val]['time'] for iter_val in irls_iterations]
        axes[0].bar(irls_iterations, times)
        axes[0].set_xlabel('IRLS Iterations')
        axes[0].set_ylabel('Computation Time (seconds)')
        axes[0].set_title('Computation Time by IRLS Iterations')
        
        # Create a table for the time data
        time_table = pd.DataFrame({
            'IRLS Iterations': irls_iterations,
            'Time (seconds)': [f"{t:.2f}" for t in times],
            'Relative Speed': [f"{times[0]/t:.2f}x" for t in times]
        })
        ax_table = axes[1].table(
            cellText=time_table.values,
            colLabels=time_table.columns,
            loc='center',
            cellLoc='center'
        )
        ax_table.auto_set_font_size(False)
        ax_table.set_fontsize(10)
        ax_table.scale(1, 1.5)
        axes[1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
        
        # NoData percentage comparison
        for feature_type in feature_names:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            fig.suptitle(f"NoData Percentage Comparison: {feature_type}", fontsize=16)
            
            # Prepare data for plotting
            nodata_data = []
            for iter_val in irls_iterations:
                iter_stats = results[iter_val]['band_stats']
                for i, stats in enumerate(iter_stats):
                    if stats['feature'] == feature_type:
                        nodata_data.append({
                            'iter_val': iter_val,
                            'index': stats['index'],
                            'nodata_pct': stats['nodata_pct']
                        })
            
            # Convert to DataFrame for easier plotting
            nodata_df = pd.DataFrame(nodata_data)
            indices = nodata_df['index'].unique()
            
            # Plot grouped bar chart
            x = np.arange(len(indices))
            width = 0.2
            offsets = np.linspace(-width, width, len(irls_iterations))
            
            for i, iter_val in enumerate(irls_iterations):
                subset = nodata_df[nodata_df['iter_val'] == iter_val]
                values = [subset[subset['index'] == idx]['nodata_pct'].values[0] if not subset[subset['index'] == idx].empty else 0 for idx in indices]
                ax.bar(x + offsets[i], values, width, label=f'max_iter={iter_val}')
            
            ax.set_xlabel('Spectral Index')
            ax.set_ylabel('NoData Percentage')
            ax.set_xticks(x)
            ax.set_xticklabels(indices)
            ax.legend()
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
        
        # Stats comparison for each index and feature type
        # For each index
        for idx in AVAILABLE_INDICES:
            # For each feature type
            for feature in feature_names:
                # Create stats comparison page
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle(f"Statistics Comparison: {idx} - {feature}", fontsize=16)
                
                # Get data for current index and feature
                stats_data = []
                for iter_val in irls_iterations:
                    for stat in results[iter_val]['band_stats']:
                        if stat['index'] == idx and stat['feature'] == feature:
                            stats_data.append({
                                'iter_val': iter_val,
                                'min': stat['min'],
                                'max': stat['max'],
                                'mean': stat['mean'],
                                'std': stat['std']
                            })
                
                # Convert to DataFrame
                stats_df = pd.DataFrame(stats_data)
                
                # Plot min, max, mean, std
                metrics = ['min', 'max', 'mean', 'std']
                for i, metric in enumerate(metrics):
                    row, col = i // 2, i % 2
                    axes[row, col].bar(stats_df['iter_val'], stats_df[metric])
                    axes[row, col].set_xlabel('IRLS Iterations')
                    axes[row, col].set_ylabel(metric.capitalize())
                    axes[row, col].set_title(f'{metric.capitalize()} by IRLS Iterations')
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
        
        # Summary comparison table
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Summary Comparison", fontsize=16)
        plt.axis('off')
        
        # Create summary table
        summary_data = []
        for iter_val in irls_iterations:
            avg_nodata = np.mean([stat['nodata_pct'] for stat in results[iter_val]['band_stats']])
            summary_data.append({
                'IRLS Iterations': iter_val,
                'Time (s)': f"{results[iter_val]['time']:.2f}",
                'Avg NoData %': f"{avg_nodata:.2f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Create table
        summary_table = plt.table(
            cellText=summary_df.values,
            colLabels=summary_df.columns,
            loc='center',
            cellLoc='center'
        )
        summary_table.auto_set_font_size(False)
        summary_table.set_fontsize(12)
        summary_table.scale(1, 2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)
    
    print(f"\n=== Test completed successfully ===")
    print(f"PDF saved at: {pdf_path}")

def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """Try to extract a date from a filename using common patterns."""
    import re
    
    # Try to match YYYYMMDD pattern
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if date_match:
        year, month, day = map(int, date_match.groups())
        return datetime(year, month, day)
    
    # Try to match YYYY_MM_DD pattern
    date_match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
    if date_match:
        year, month, day = map(int, date_match.groups())
        return datetime(year, month, day)
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Compare IRLS iteration parameters on satellite imagery.')
    parser.add_argument('--tifs', nargs='+', type=str, help='List of TIF files to process')
    parser.add_argument('--dates', nargs='+', type=str, help='List of dates (YYYY-MM-DD) corresponding to TIF files')
    parser.add_argument('--iterations', nargs='+', type=int, default=[1, 5, 10], 
                        help='IRLS iteration values to test')
    parser.add_argument('--window-size', type=int, default=1024, 
                        help='Size of the square window to process (default: 1024)')
    parser.add_argument('--output-dir', type=str, default='results/test',
                        help='Directory to save the output PDF report')
    
    args = parser.parse_args()
    
    # Process TIF paths and dates
    tif_paths = None
    tif_dates = None
    
    if args.tifs:
        tif_paths = [Path(tif) for tif in args.tifs]
        
        # Check if all TIFs exist
        for tif_path in tif_paths:
            if not tif_path.exists():
                raise FileNotFoundError(f"TIF file not found: {tif_path}")
        
        # Process dates
        if args.dates:
            if len(args.dates) != len(args.tifs):
                raise ValueError("Number of dates must match number of TIF files")
            
            tif_dates = []
            for date_str in args.dates:
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    tif_dates.append(date_obj)
                except ValueError:
                    raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
        else:
            # Try to extract dates from filenames
            tif_dates = []
            for tif_path in tif_paths:
                date = parse_date_from_filename(tif_path.name)
                if date is None:
                    raise ValueError(f"Could not extract date from filename: {tif_path.name}. "
                                    "Please provide dates using --dates")
                tif_dates.append(date)
    
    # Run the test with provided parameters
    test_compare_irls_iterations(
        tif_paths=tif_paths,
        tif_dates=tif_dates,
        irls_iterations=args.iterations,
        window_size=args.window_size,
        output_dir=Path(args.output_dir)
    )

if __name__ == "__main__":
    main() 