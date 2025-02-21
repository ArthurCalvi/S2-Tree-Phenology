"""
This test script demonstrates feature extraction at the folder level,
using prepare_inference_feature.py to set up the configuration and
then running FolderFeature with limited tiles and windows.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import subprocess
import matplotlib
matplotlib.use('Agg')  # So we can write PDF without a display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import rasterio
from rasterio.windows import Window
import sys 

import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    filename=f'logs/test_folder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent)) 
from src.constants import AVAILABLE_INDICES, TARGET_AMP_RANGE, TARGET_OFFSET_RANGE
from src.features.inference_feature import FolderFeature

def test_folder_feature():
    print("\n=== Starting folder feature test ===")
    
    # 1) Set up paths
    print("\n1. Setting up test paths...")
    mosaic_dir = Path("data/mosaics/2023")
    config_dir = Path("results/test/configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    # 2) Run prepare_inference_feature.py to create configs
    print("\n2. Preparing inference configs...")
    prepare_script = Path("src/features/prepare_inference_feature.py")
    cmd = [
        "python", str(prepare_script),
        "--input-dir", str(mosaic_dir),
        "--output-dir", str(config_dir),
        "--year", "2023",
        "--max-concurrent-jobs", "4"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Config preparation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error preparing configs: {e}")
        return

    # 3) Create and run FolderFeature test
    print("\n3. Running FolderFeature test...")
    logger = logging.getLogger(__name__)
    
    folder_feature = FolderFeature(
        config_dir=config_dir,
        logger=logger,
        block_size=512,  # Process in 512x512 chunks
        max_workers=4    # Use 4 parallel workers
    )

    # Run test with 1 tile, 8 windows per tile
    start_time = datetime.now()
    folder_feature.run_test(
        max_tiles=1,      # Process only 1 tile
        max_windows=8,    # Process only 8 windows per tile
        num_harmonics=2,  # Use 2 harmonics
        max_iter=5        # Use 5 IRLS iterations
    )
    computation_time = datetime.now() - start_time

    print(f"\n=== Test completed successfully ===")
    print(f"Computation time: {computation_time}")
    print(f"Results saved in: {folder_feature.output_dir}")

    # 4) Generate a PDF report
    print("\n4. Generating PDF report...")
    pdf_name = f"folder_feature_test_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf_path = config_dir.parent / pdf_name

    # Helper function to compute stats for a feature band
    def compute_stats(arr, indice_label, feature_type):
        arrf = arr.astype(float)
        if "Amplitude" in feature_type:
            max_amp = TARGET_AMP_RANGE[indice_label][1]
            arrf = (arrf / 65535.0) * max_amp
        elif "Offset" in feature_type:
            max_offset = TARGET_OFFSET_RANGE[indice_label][1]
            arrf = (arrf / 65535.0) * max_offset
        elif "Phase" in feature_type:
            arrf = (arrf / 65535.0) * (2 * np.pi) - np.pi
        
        return {
            "min": np.nanmin(arrf),
            "max": np.nanmax(arrf),
            "mean": np.nanmean(arrf),
            "std": np.nanstd(arrf)
        }

    feature_names = ["Amplitude (h1)", "Amplitude (h2)", "Phase (h1)", "Phase (h2)", "Offset", "Residual Variance"]

    with PdfPages(pdf_path) as pdf:
        # First page: parameters and overview
        fig1 = plt.figure(figsize=(8.5, 11))
        txt = (
            "Folder Feature Extraction Test\n"
            f"Date: {datetime.now()}\n\n"
            "Parameters:\n"
            f" - Config directory: {config_dir}\n"
            f" - block_size=512, max_workers=4\n"
            f" - num_harmonics=2, max_iter=5\n"
            f" - max_tiles=1, max_windows=8\n"
            f" - Output directory: {folder_feature.output_dir}\n"
            f" - Computation time: {computation_time}\n"
            "\nFollowing pages show sample windows from processed tiles.\n"
        )
        fig1.text(0.1, 0.7, txt, fontsize=12)
        pdf.savefig(fig1)
        plt.close(fig1)

        # Find and process output TIFs
        for tif_path in folder_feature.output_dir.glob("*.tif"):
            print(f"  - Processing output file: {tif_path.name}")
            
            with rasterio.open(tif_path) as src:
                # Sample from first window of the tile
                window = Window(
                    0,
                    0, 
                    512, 
                    512
                )
                data = src.read(window=window)
                
                # Create a page for each spectral index
                for idx_i, index_name in enumerate(AVAILABLE_INDICES):
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle(f"{tif_path.stem}: {index_name} Features")
                    axes = axes.flatten()

                    # Plot the 6 feature bands for this index
                    for feat_i, feat_name in enumerate(feature_names):
                        band_idx = idx_i * len(feature_names) + feat_i
                        band_data = data[band_idx].astype(float)
                        
                        # Convert to physical values
                        if "Amplitude" in feat_name:
                            max_amp = TARGET_AMP_RANGE[index_name.lower()][1]
                            band_data = (band_data / 65535.0) * max_amp
                        elif "Offset" in feat_name:
                            max_offset = TARGET_OFFSET_RANGE[index_name.lower()][1]
                            band_data = (band_data / 65535.0) * max_offset
                        elif "Phase" in feat_name:
                            band_data = (band_data / 65535.0) * (2 * np.pi) - np.pi

                        stats = compute_stats(data[band_idx], index_name.lower(), feat_name)
                        im = axes[feat_i].imshow(band_data, cmap='viridis')
                        fig.colorbar(im, ax=axes[feat_i])
                        axes[feat_i].set_title(f"{feat_name}\nmin={stats['min']:.3f}, max={stats['max']:.3f}")
                        axes[feat_i].axis('off')

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"PDF report saved at: {pdf_path}")

if __name__ == "__main__":
    test_folder_feature() 