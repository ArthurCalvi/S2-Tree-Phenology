#!/usr/bin/env python3
"""
inference_feature.py
---------------------
Compute harmonic features for one or more tiles. We use an OOP approach
at three levels: WindowFeature, TileFeature, FolderFeature.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]  # Go up 2 levels from src/features/
sys.path.append(str(project_root))

import argparse
import logging
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime

# Import from your own modules
from src.constants import (
    DEFAULT_NUM_HARMONICS,
    DEFAULT_MAX_ITER,
    AVAILABLE_INDICES,
    BandData        # <--- Our new data class
)
from src.utils import (
    compute_indices, 
    compute_quality_weights,
    robust_harmonic_fitting,
    scale_amplitude,
    scale_phase,
    scale_offset,
    scale_array_to_uint16
)

###############################################################################
# OOP CLASS: WindowFeature
###############################################################################

class WindowFeature:
    """
    Class to handle the feature extraction logic for one window 
    in the final mosaic. Reads multiple TIFs (one per date),
    builds a BandData object, computes spectral indices, fits harmonics, 
    returns multi-band features for that chunk in memory.
    """
    def __init__(
        self,
        tif_paths: list[Path],
        dates: list[datetime],
        window: Window,
        num_harmonics: int = DEFAULT_NUM_HARMONICS,
        max_iter: int = DEFAULT_MAX_ITER,
        logger: logging.Logger = None
    ):
        """
        Args:
            tif_paths: The list of TIF files (one per date).
            dates: List of acquisition datetimes, must match len(tif_paths).
            window: The rasterio Window specifying our chunk of interest.
            num_harmonics: # of harmonics to fit
            max_iter: IRLS iteration
            logger: logger instance
        """
        self.tif_paths = tif_paths
        self.dates = dates
        self.window = window
        self.num_harmonics = num_harmonics
        self.max_iter = max_iter
        self.logger = logger

        self.H = window.height
        self.W = window.width
        self.T = len(tif_paths)

        # We assume the mosaic has 6 bands in this order: B2, B4, B8, B11, B12, MSK_CLDPRB
        self.band_order = {
            "B2": 1,
            "B4": 2,
            "B8": 3,
            "B11":4,
            "B12":5,
            "MSK_CLDPRB":6
        }
    
    def _log_array_stats(self, arr: np.ndarray, label: str):
        """Helper method to log basic statistics of an array at debug level."""
        self.logger.debug(
            f"{label} stats: min={np.nanmin(arr):.3f}, "
            f"mean={np.nanmean(arr):.3f}, "
            f"std={np.nanstd(arr):.3f}, "
            f"max={np.nanmax(arr):.3f}"
        )
    
    def _read_bands_into_BandData(self) -> BandData:
        """
        Reads B2, B4, B8, B11, B12, MSK_CLDPRB from each TIF in the specified Window,
        scales them from uint16 => float, and returns a BandData object.
        (We do not read DEM here, but you can do so if you have a tile-level DEM.)
        """
        # Allocate arrays: shape (T,H,W)
        b2  = np.zeros((self.T, self.H, self.W), dtype=np.float32)
        b4  = np.zeros((self.T, self.H, self.W), dtype=np.float32)
        b8  = np.zeros((self.T, self.H, self.W), dtype=np.float32)
        b11 = np.zeros((self.T, self.H, self.W), dtype=np.float32)
        b12 = np.zeros((self.T, self.H, self.W), dtype=np.float32)
        cld = np.zeros((self.T, self.H, self.W), dtype=np.float32)

        # Example scale factor
        scale_factor = 21.845

        for i, tifp in enumerate(self.tif_paths):
            with rasterio.open(tifp) as src:
                b2_uint16  = src.read(self.band_order["B2"],  window=self.window)
                b4_uint16  = src.read(self.band_order["B4"],  window=self.window)
                b8_uint16  = src.read(self.band_order["B8"],  window=self.window)
                b11_uint16 = src.read(self.band_order["B11"], window=self.window)
                b12_uint16 = src.read(self.band_order["B12"], window=self.window)
                cld_uint16 = src.read(self.band_order["MSK_CLDPRB"], window=self.window)

            # Convert to float, ~[0..3000] for DN, [0..100] for cloud-prob
            b2[i]  = b2_uint16.astype(np.float32)  / scale_factor
            b4[i]  = b4_uint16.astype(np.float32)  / scale_factor
            b8[i]  = b8_uint16.astype(np.float32)  / scale_factor
            b11[i] = b11_uint16.astype(np.float32) / scale_factor
            b12[i] = b12_uint16.astype(np.float32) / scale_factor
            cld[i] = cld_uint16.astype(np.float32) / scale_factor

        band_data = BandData(
            b2=b2, b4=b4, b8=b8, b11=b11, b12=b12,
            msk_cldprb=cld,
            dates=self.dates,
            dem=None
        )
        return band_data

    def compute_features(self) -> np.ndarray:
        """
        Build a BandData object, compute QA weights from the msk_cldprb,
        compute spectral indices, robust harmonic fitting, and return
        a multi-band array (#features, H, W).
        """
        # 1) Read raw data into BandData
        band_data = self._read_bands_into_BandData()
        self.logger.debug("After reading bands into BandData:")
        self._log_array_stats(band_data.b2, "B2")
        self._log_array_stats(band_data.b4, "B4")
        self._log_array_stats(band_data.b8, "B8")
        self._log_array_stats(band_data.b11, "B11")
        self._log_array_stats(band_data.b12, "B12")
        self._log_array_stats(band_data.msk_cldprb, "MSK_CLDPRB")
        
        # 2) Convert raw cloud-prob to QA weights
        qa_weights = compute_quality_weights(band_data.msk_cldprb, logger=self.logger)
        self.logger.debug("Quality weights:")
        self._log_array_stats(qa_weights, "QA Weights")
        
        # 3) Compute spectral indices
        ndvi, evi, nbr, crswir = compute_indices(
            band_data.b2, band_data.b4, band_data.b8,
            band_data.b11, band_data.b12,
            logger=self.logger
        )
        indices_dict = {"ndvi": ndvi, "evi": evi, "nbr": nbr, "crswir": crswir}
        for key, val in indices_dict.items():
            self.logger.debug(f"Spectral index '{key}':")
            self._log_array_stats(val, key)
        
        # 4) For each index, run robust harmonic fitting and scale each sub-band.
        out_bands = []
        for idx_name in AVAILABLE_INDICES:
            data_cube = indices_dict[idx_name]
            self.logger.debug(f"Starting harmonic fitting for index '{idx_name}' with input data:")
            self._log_array_stats(data_cube, f"{idx_name} input")
            
            results = robust_harmonic_fitting(
                data_cube,
                qa_weights,
                band_data.dates,
                num_harmonics=self.num_harmonics,
                max_iter=self.max_iter,
                logger=self.logger
            )
            amp_h1, amp_h2, phs_h1, phs_h2, offset_map, var_map = results
            
            self.logger.debug(f"Raw harmonic fitting results for '{idx_name}':")
            self._log_array_stats(amp_h1, f"{idx_name} amp_h1 raw")
            self._log_array_stats(amp_h2, f"{idx_name} amp_h2 raw")
            self._log_array_stats(phs_h1, f"{idx_name} phs_h1 raw")
            self._log_array_stats(phs_h2, f"{idx_name} phs_h2 raw")
            self._log_array_stats(offset_map, f"{idx_name} offset raw")
            self._log_array_stats(var_map, f"{idx_name} variance raw")
            
            sa1 = scale_amplitude(amp_h1, idx_name)
            sa2 = scale_amplitude(amp_h2, idx_name)
            sp1 = scale_phase(phs_h1)
            sp2 = scale_phase(phs_h2)
            so  = scale_offset(offset_map, idx_name)
            var_clamped = np.clip(var_map, 0, 2)
            sv  = scale_array_to_uint16(var_clamped, 0, 2)
            
            self.logger.debug(f"Scaled harmonic fitting results for '{idx_name}':")
            self._log_array_stats(sa1, f"{idx_name} amplitude h1 scaled")
            self._log_array_stats(sa2, f"{idx_name} amplitude h2 scaled")
            self._log_array_stats(sp1, f"{idx_name} phase h1 scaled")
            self._log_array_stats(sp2, f"{idx_name} phase h2 scaled")
            self._log_array_stats(so, f"{idx_name} offset scaled")
            self._log_array_stats(sv, f"{idx_name} residual variance scaled")
            
            out_bands.extend([sa1, sa2, sp1, sp2, so, sv])
        
        # Final shape => (#features, H, W)
        output_cube = np.stack(out_bands, axis=0)
        self.logger.debug("Final output cube:")
        self._log_array_stats(output_cube, "Output cube")
        return output_cube


###############################################################################
# TILEFEATURE, FOLDERFEATURE, ETC. REMAIN SIMILAR
###############################################################################

class TileFeature:
    """
    Orchestrates window-based feature extraction for a single tile. 
    We do not do DEM reading in this example, but you could add it 
    if the tile has a specific DEM TIF, merging it with BandData if needed.
    """
    def __init__(
        self,
        tif_paths: list[Path],
        dates: list[datetime],
        output_path: Path,
        num_harmonics: int,
        max_iter: int,
        block_size: int,
        max_workers: int,
        logger: logging.Logger
    ):
        self.tif_paths = tif_paths
        self.dates = dates
        self.output_path = output_path
        self.num_harmonics = num_harmonics
        self.max_iter = max_iter
        self.block_size = block_size
        self.max_workers = max_workers
        self.logger = logger

        if not tif_paths:
            raise ValueError("No TIF files provided.")
        if len(tif_paths) != len(dates):
            raise ValueError("Mismatch: len(tif_paths) != len(dates).")

        if not tif_paths[0].exists():
            raise FileNotFoundError(f"First TIF not found: {tif_paths[0]}")

        # Read dimension from first TIF
        with rasterio.open(tif_paths[0]) as src:
            self.profile = src.profile.copy()
            self.width = src.width
            self.height = src.height

        # #features => for 2 harmonics => 6 sub-bands per index => 6 * #indices
        # If you do n harmonics => (2n +2) sub-bands per index
        from src.constants import AVAILABLE_INDICES  # or we already have it
        n = self.num_harmonics
        nb_indices = len(AVAILABLE_INDICES)
        nb_bands = (2*n + 2) * nb_indices

        self.out_profile = self.profile.copy()
        self.out_profile.update({
            "count": nb_bands,
            "dtype": "uint16",
            "compress": "lzw",
            "predictor": 2,
            "BIGTIFF": "YES"
        })

    def _generate_windows(self):
        for y in range(0, self.height, self.block_size):
            for x in range(0, self.width, self.block_size):
                w = min(self.block_size, self.width - x)
                h = min(self.block_size, self.height - y)
                yield Window(x, y, w, h)

    def _process_window(self, window: Window) -> np.ndarray:
        wf = WindowFeature(
            tif_paths=self.tif_paths,
            dates=self.dates,
            window=window,
            num_harmonics=self.num_harmonics,
            max_iter=self.max_iter,
            logger=self.logger
        )
        return wf.compute_features()

    def run(self, max_windows: int = 10):
        self.logger.info(f"Generating features => {self.output_path.name}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        windows_list = list(self._generate_windows())
        # Limit the number of windows if specified
        if max_windows:
            windows_list = windows_list[:max_windows]
            self.logger.info(f"Limited to first {max_windows} windows")
            
        self.logger.info(f"block_size={self.block_size}, #windows={len(windows_list)}")

        with rasterio.open(self.output_path, "w", **self.out_profile) as dst:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {}
                for w in windows_list:
                    fut = executor.submit(self._process_window, w)
                    future_map[fut] = w

                for fut in tqdm(as_completed(future_map), total=len(windows_list), desc="TileFeature"):
                    w = future_map[fut]
                    try:
                        arr = fut.result()  # (band_count, w.height, w.width)
                        band_count = arr.shape[0]
                        for b_i in range(band_count):
                            dst.write(arr[b_i], b_i+1, window=w)
                    except Exception as e:
                        self.logger.error(f"Window {w} failed: {e}")


class FolderFeature:
    """
    Manages multiple tiles from config_dir (metadata.json + tile_config_*.json).
    """
    def __init__(self, config_dir: Path, logger: logging.Logger, block_size=1024, max_workers=4):
        self.config_dir = config_dir
        self.logger = logger
        self.block_size = block_size
        self.max_workers = max_workers

        meta_file = config_dir / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"metadata.json not found in {config_dir}")
        with open(meta_file) as f:
            self.metadata = json.load(f)
        self.num_tiles = self.metadata["num_tiles"]
        self.output_dir = Path(self.metadata["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_tile(self, tile_idx: int, num_harmonics=2, max_iter=10, max_windows=None):
        cfg_file = self.config_dir / f"tile_config_{tile_idx:03d}.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"{cfg_file} not found.")

        with open(cfg_file) as f:
            tile_cfg = json.load(f)

        # Use the new config keys (lists of tif_paths and dates)
        tif_paths = [Path(tp) for tp in tile_cfg["tif_paths"]]
        dates = [datetime.strptime(dt, "%Y-%m-%d") for dt in tile_cfg["dates"]]

        out_name = f"features_{tile_cfg['tile_idx']:03d}.tif"
        out_path = self.output_dir / out_name

        tile_feat = TileFeature(
            tif_paths=tif_paths,
            dates=dates,
            output_path=out_path,
            num_harmonics=num_harmonics,
            max_iter=max_iter,
            block_size=self.block_size,
            max_workers=self.max_workers,
            logger=self.logger
        )
        tile_feat.run(max_windows=max_windows)

    def run_test(self, max_tiles: int = 2, max_windows: int = 8, num_harmonics=2, max_iter=10):
        """Run a test with limited number of tiles and windows per tile.
        
        Args:
            max_tiles: Maximum number of tiles to process
            max_windows: Maximum number of windows to process per tile
            num_harmonics: Number of harmonics to fit
            max_iter: Maximum number of IRLS iterations
        """
        test_tiles = min(max_tiles, self.num_tiles)
        self.logger.info(f"Running test on {test_tiles} tiles with {max_windows} windows each")
        
        for idx in range(test_tiles):
            self.logger.info(f"Processing test tile {idx}/{test_tiles - 1}")
            self.process_tile(
                idx, 
                num_harmonics=num_harmonics, 
                max_iter=max_iter,
                max_windows=max_windows
            )
        
        self.logger.info("Test run completed")

    def run_all(self, num_harmonics=2, max_iter=10):
        for idx in range(self.num_tiles):
            self.logger.info(f"Processing tile index {idx}/{self.num_tiles - 1}")
            self.process_tile(idx, num_harmonics=num_harmonics, max_iter=max_iter)


def setup_logger():
    logger = logging.getLogger("inference_feature")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    logger.addHandler(ch)
    return logger

def main():
    parser = argparse.ArgumentParser(description="OOP Feature Extraction (Window/Tile/Folder).")
    parser.add_argument("--config-dir", type=str, required=True, 
                        help="Path to config dir with metadata.json & tile_config_XXX.json")
    parser.add_argument("--tile-idx", type=int, default=None, 
                        help="Index of tile to process (if not set, process all).")
    parser.add_argument("--block-size", type=int, default=1024, 
                        help="Spatial chunk size.")
    parser.add_argument("--max-workers", type=int, default=4, 
                        help="Parallel workers.")
    parser.add_argument("--num-harmonics", type=int, default=2, 
                        help="Number of harmonics to fit.")
    parser.add_argument("--max-iter", type=int, default=10, 
                        help="IRLS iterations for robust fitting.")
    args = parser.parse_args()

    logger = setup_logger()
    folder_feature = FolderFeature(
        config_dir=Path(args.config_dir),
        logger=logger,
        block_size=args.block_size,
        max_workers=args.max_workers
    )

    if args.tile_idx is not None:
        folder_feature.process_tile(args.tile_idx, num_harmonics=args.num_harmonics, max_iter=args.max_iter)
    else:
        folder_feature.run_all(num_harmonics=args.num_harmonics, max_iter=args.max_iter)

    logger.info("All done.")

if __name__ == "__main__":
    main()
