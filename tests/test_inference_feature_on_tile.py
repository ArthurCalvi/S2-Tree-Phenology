"""
This test script demonstrates feature extraction on a complete tile
from the 12 monthly TIF files in data/mosaics/2023. We then generate
a PDF report showing sample windows from the tile and their features.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # So we can write PDF without a display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.windows import Window
import logging

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging to write to file with timestamp
logging.basicConfig(
    level=logging.DEBUG,
    filename=f'logs/test_tile_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Adjust imports to your actual project structure:
sys.path.append(str(Path(__file__).parent.parent))  # so we can import from src
from src.features.inference_feature import TileFeature
from src.constants import AVAILABLE_INDICES, TARGET_AMP_RANGE, TARGET_OFFSET_RANGE

def test_inference_feature_on_tile():
    print("\n=== Starting tile feature inference test ===")
    
    # 1) Build the list of TIF paths for each month in 2023
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

    # 2) Set up output path and parameters
    print("\n2. Setting up output parameters...")
    results_dir = Path("results/test")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_tif = results_dir / f"test_tile_features_{datetime.now().strftime('%Y%m%d')}.tif"

    # 3) Create and run the TileFeature
    print("\n3. Creating and running TileFeature...")
    tile_feature = TileFeature(
        tif_paths=tif_paths,
        dates=tif_dates,
        output_path=output_tif,
        num_harmonics=2,
        max_iter=5,
        block_size=512,  # Process in 512x512 chunks
        max_workers=4,   # Use 4 parallel workers
        logger=logging.getLogger(__name__)
    )

    start_time = datetime.now()
    tile_feature.run(max_windows=8)  # Limit to 8 windows
    computation_time = datetime.now() - start_time
    print(f"Tile processing complete. Output saved to: {output_tif}")
    print(f"Computation time: {computation_time}")

    # 4) Generate a PDF with sample results
    print("\n4. Generating PDF report...")
    pdf_name = f"tile_feature_test_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf_path = results_dir / pdf_name

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

    # Sample windows to visualize (e.g., corners and center)
    sample_windows = [
        ("Top-Left", Window(0, 0, 512, 512)),
        ("Top-Right", Window(tile_feature.width-512, 0, 512, 512)),
        ("Center", Window(tile_feature.width//2-256, tile_feature.height//2-256, 512, 512)),
        ("Bottom-Left", Window(0, tile_feature.height-512, 512, 512)),
        ("Bottom-Right", Window(tile_feature.width-512, tile_feature.height-512, 512, 512))
    ]

    feature_names = ["Amplitude (h1)", "Amplitude (h2)", "Phase (h1)", "Phase (h2)", "Offset", "Residual Variance"]

    with PdfPages(pdf_path) as pdf:
        # First page: parameters and overview
        fig1 = plt.figure(figsize=(8.5, 11))
        txt = (
            "Tile Feature Extraction Test\n"
            f"Date: {datetime.now()}\n\n"
            "Parameters:\n"
            f" - TIFs: {len(tif_paths)} (monthly from Jan to Dec 2023)\n"
            f" - Tile dimensions: {tile_feature.width}x{tile_feature.height}\n"
            f" - block_size=512, max_workers=4\n"
            f" - num_harmonics=2, max_iter=5\n"
            f" - Output file: {output_tif}\n"
            f" - Computation time: {computation_time}\n"
            "\nFollowing pages show sample windows from different tile locations.\n"
        )
        fig1.text(0.1, 0.7, txt, fontsize=12)
        pdf.savefig(fig1)
        plt.close(fig1)

        # Read the output TIF and create visualizations
        with rasterio.open(output_tif) as src:
            for window_name, window in sample_windows:
                print(f"  - Processing sample window: {window_name}")
                
                # Read all bands for this window
                data = src.read(window=window)
                
                # Create a page for each spectral index
                for idx_i, index_name in enumerate(AVAILABLE_INDICES):
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle(f"{window_name}: {index_name} Features")
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

    print(f"\n=== Test completed successfully ===")
    print(f"PDF report saved at: {pdf_path}")

if __name__ == "__main__":
    test_inference_feature_on_tile() 