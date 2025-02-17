# test_inference_feature_on_window.py
"""
This test script demonstrates feature extraction on a single 512x512 window
from the 12 monthly TIF files in data/mosaics/2023. We then generate
a PDF report with each feature band visualized, stats computed, and scaling functions noted.
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
from rasterio.windows import Window

# Adjust imports to your actual project structure:
sys.path.append(str(Path(__file__).parent.parent))  # so we can import from src
from src.features.inference_feature import WindowFeature
from src.constants import AVAILABLE_INDICES

def test_inference_feature_on_window():
    print("\n=== Starting feature inference test ===")
    
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

    # 2) Define our 512x512 window from the top-left corner
    print("\n2. Defining 512x512 window...")
    w = Window(col_off=0, row_off=0, width=512, height=512)
    print(f"Window parameters: {w}")

    # 3) Create the WindowFeature
    print("\n3. Creating WindowFeature object...")
    window_feature = WindowFeature(
        tif_paths=tif_paths,
        dates=tif_dates,
        window=w,
        num_harmonics=2,
        max_iter=5,
        logger=None
    )
    print("WindowFeature object created successfully")

    # 4) Compute the features => shape (#features, 512, 512)
    print("\n4. Computing features...")
    start_time = datetime.now()
    feature_cube = window_feature.compute_features()  # uint16 array
    computation_time = datetime.now() - start_time
    print(f"Feature computation complete. Shape: {feature_cube.shape}")
    print(f"Computation time: {computation_time}")

    # 5) Generate a PDF with the results
    print("\n5. Generating PDF report...")
    today_str = datetime.now().strftime("%Y%m%d")
    results_dir = Path("results/test")
    results_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = f"inference_feature_test_{today_str}.pdf"
    pdf_path = results_dir / pdf_name

    # Define feature names corresponding to each sub-band in the order they appear.
    # Order: amplitude for harmonic 1, amplitude for harmonic 2,
    # phase for harmonic 1, phase for harmonic 2, offset, residual variance.
    feature_names = ["Amplitude (h1)", "Amplitude (h2)", "Phase (h1)", "Phase (h2)", "Offset", "Residual Variance"]

    # Define scaling function descriptions for each feature.
    scaling_descriptions = {
        "Amplitude (h1)": "scale_amplitude: scales [0,2.0] -> [0,65535]",
        "Amplitude (h2)": "scale_amplitude: scales [0,2.0] -> [0,65535]",
        "Phase (h1)": "scale_phase: shifts [-π,π] to [0,2π] then scales to [0,65535]",
        "Phase (h2)": "scale_phase: shifts [-π,π] to [0,2π] then scales to [0,65535]",
        "Offset": "scale_offset: scales [0,2.0] -> [0,65535]",
        "Residual Variance": "scale_array_to_uint16: clips [0,2.0] then scales to [0,65535]"
    }

    def compute_stats(arr):
        arrf = arr.astype(float)
        return {
            "min": np.nanmin(arrf),
            "max": np.nanmax(arrf),
            "mean": np.nanmean(arrf),
            "std": np.nanstd(arrf)
        }

    with PdfPages(pdf_path) as pdf:
        # ---- First page: parameters used
        fig1 = plt.figure(figsize=(8.5, 11))
        txt = (
            "Feature Extraction Test\n"
            f"Date: {datetime.now()}\n\n"
            "Parameters:\n"
            f" - TIFs: {len(tif_paths)} (monthly from Jan to Dec 2023)\n"
            f" - Window: (row=0, col=0, width=512, height=512)\n"
            f" - num_harmonics=2, max_iter=5\n"
            f" - Computation time: {computation_time}\n"
            "\nWill show each feature band, stats, and the scaling function used in subsequent pages.\n"
        )
        fig1.text(0.1, 0.7, txt, fontsize=12)
        pdf.savefig(fig1)
        plt.close(fig1)

        # Each feature band gets its own page.
        n_bands = feature_cube.shape[0]
        for band_i in range(n_bands):
            band_data = feature_cube[band_i].astype(float)
            stats = compute_stats(band_data)
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(band_data, cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Determine which index and feature type this band corresponds to.
            # We assume that for each index in AVAILABLE_INDICES, the algorithm produces 6 bands.
            group_index = band_i // len(feature_names)
            feature_type = feature_names[band_i % len(feature_names)]
            # If group_index is within the range of AVAILABLE_INDICES, use it; otherwise mark as Unknown.
            indice_label = AVAILABLE_INDICES[group_index].upper() if group_index < len(AVAILABLE_INDICES) else "Unknown"
            ax.set_title(f"Index: {indice_label} | Feature: {feature_type}")
            
            # Append both stat information and scaling function description to the xlabel.
            scaling_info = scaling_descriptions.get(feature_type, "No scaling info available")
            ax.set_xlabel(
                f"min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, std={stats['std']:.2f}\n{scaling_info}"
            )
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  - Added feature: {indice_label} - {feature_type} to PDF")

    print(f"\n=== Test completed successfully ===")
    print(f"PDF saved at: {pdf_path}")

if __name__ == "__main__":
    test_inference_feature_on_window()
