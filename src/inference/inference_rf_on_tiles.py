#!/usr/bin/env python3
"""
inference_rf_on_tiles.py
-------------------------
Performs inference using a trained RandomForest model on feature tiles.
Reads feature tiles (output from inference_feature.py), applies necessary
transformations (unscaling, phase conversion), selects the features the
model was trained on, and predicts phenology classes pixel by pixel.
Uses a windowed approach for memory efficiency and potential parallelism.
"""

import sys
import argparse
import logging
import json
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import time
from sklearn.ensemble import RandomForestClassifier
import os # Import os to get PID

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import from src modules
try:
    from src.utils import (
        unscale_feature,
        transform_circular_features, # Expects DataFrame, need adaptation or pixel-wise logic
        FEATURE_SUFFIX_TO_TYPE,
        PHASE_FEATURE_SUFFIXES,
        ALL_FEATURE_BAND_INDICES # Map feature names to band indices in the feature tile
    )
    from src.constants import PHENOLOGY_MAPPING # For potential output class mapping
except ImportError as e:
    print(f"Error importing from src: {e}. Make sure PYTHONPATH includes project root.", file=sys.stderr)
    sys.exit(1)

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inference_rf_on_tiles")

DEFAULT_BLOCK_SIZE = 2048 # 2048 pixels
DEFAULT_MAX_WORKERS = 4

# --- Top-level Worker Functions (for pickling) ---
def _process_window_rf(input_tile_path: Path, window: Window, model: RandomForestClassifier, model_features: list[str], output_dtype: np.dtype):
    """Helper function to process a single window for RF inference. Suitable for ProcessPoolExecutor."""
    pid = os.getpid()
    logger.info(f"[PID {pid}] Processing window {window}") # Log PID and window
    try:
        window_inferer = WindowInferenceRF(
            input_tile_path, window, model, model_features,
            output_dtype=output_dtype
        )
        result_prob_map_float = window_inferer.run() # This returns float32 map

        if result_prob_map_float is not None:
            # Scale float probabilities [0, 1] to uint8 [0, 255]
            result_prob_map_uint8 = (result_prob_map_float * 255).round().astype(np.uint8)
            return window, result_prob_map_uint8
        else:
            # If inference failed, return None
            return window, None

    except Exception as e:
        logger.error(f"_process_window_rf failed for window {window} on {input_tile_path.name}: {e}", exc_info=True)
        # Return None or raise exception to signal failure
        return window, None # Indicate failure for this window

def process_single_tile_worker(input_tile_path: Path, output_dir: Path, model: RandomForestClassifier, model_features: list[str], block_size: int, num_workers: int, rf_n_jobs: int):
    """Function executed by each worker process OR for single tile mode."""
    try:
        # Removed per-worker print statement
        output_tile_path = output_dir / input_tile_path.name
        tile_inferer = TileInferenceRF(
            model=model,                    # Pass loaded model
            model_features=model_features,  # Pass loaded features
            input_tile_path=input_tile_path,
            output_tile_path=output_tile_path,
            block_size=block_size,
            num_workers=num_workers,         # Controls window parallelism (now always 1)
            rf_n_jobs=rf_n_jobs             # Controls internal RF parallelism
        )
        tile_inferer.run()
        return f"Successfully processed {input_tile_path.name}"
    except Exception as e:
        # logger.error(f"Worker failed on tile {input_tile_path.name}: {e}\", exc_info=True)
        print(f"ERROR: Worker failed on tile {input_tile_path.name}: {e}") # Use print
        return f"Failed to process {input_tile_path.name}"

# --- Helper Functions ---

def get_feature_metadata(feature_name: str) -> tuple[str, str, bool]:
    """
    Derives the index name, feature type, and phase status from a feature name.
    Example: 'ndvi_amplitude_h1' -> ('ndvi', 'amplitude', False)
             'evi_phase_h2_cos' -> ('evi', 'phase', True) # Note: Handles transformed name
    """
    index_name = feature_name.split('_')[0]
    is_phase = False
    feature_type = None

    # Check for transformed phase features first
    if feature_name.endswith('_cos') or feature_name.endswith('_sin'):
        is_phase = True
        # Determine base feature name (e.g., evi_phase_h2) to find original type
        base_name_parts = feature_name.split('_')[:-1] # Remove _cos or _sin
        base_feature_name = '_'.join(base_name_parts)
        for suffix, ftype in FEATURE_SUFFIX_TO_TYPE.items():
            if base_feature_name.endswith(suffix):
                 feature_type = ftype
                 break
    else:
        # Check for original feature types
        for suffix, ftype in FEATURE_SUFFIX_TO_TYPE.items():
            if feature_name.endswith(suffix):
                feature_type = ftype
                if ftype == 'phase':
                    is_phase = True
                break

    if feature_type is None:
        raise ValueError(f"Could not determine feature type for {feature_name}")

    return index_name, feature_type, is_phase

# --- Core Classes ---

class WindowInferenceRF:
    """Handles RF inference for a single window."""
    def __init__(self,
                 input_tile_path: Path,
                 window: Window,
                 model: RandomForestClassifier,
                 model_features: list[str], # Features the model expects, in order
                 output_dtype: np.dtype = np.uint8,
                 ):
        self.input_tile_path = input_tile_path
        self.window = window
        self.model = model
        self.model_features = model_features
        self.output_dtype = np.float32 # Probabilities are float
        self.H = window.height
        self.W = window.width

        # Determine which bands need to be read based on model_features
        self.bands_to_read = []
        self.feature_indices_in_tile = [] # Indices of model features within the feature tile
        self.read_feature_names = [] # Names corresponding to self.feature_indices_in_tile
        
        # Get the full list of features available in the tile based on constants
        # Assuming the tile was created by inference_feature.py and follows the standard order
        all_tile_features = []
        from src.constants import AVAILABLE_INDICES
        for index in AVAILABLE_INDICES:
             all_tile_features.extend([
                 f"{index}_amplitude_h1", f"{index}_amplitude_h2",
                 f"{index}_phase_h1", f"{index}_phase_h2",
                 f"{index}_offset", f"{index}_var_residual"
             ])
             
        tile_feature_to_band_index = {name: i + 1 for i, name in enumerate(all_tile_features)}

        # Map required model features (raw, before phase transform) to their bands
        raw_features_needed = set()
        for mf in self.model_features:
            if mf.endswith('_cos') or mf.endswith('_sin'):
                raw_feat = '_'.join(mf.split('_')[:-1]) # e.g., ndvi_phase_h1
                raw_features_needed.add(raw_feat)
            else:
                raw_features_needed.add(mf) # Assume non-phase or original phase name

        for raw_feat in sorted(list(raw_features_needed)): # Sort for consistent reading order
            if raw_feat in tile_feature_to_band_index:
                band_idx = tile_feature_to_band_index[raw_feat]
                self.bands_to_read.append(band_idx)
                self.read_feature_names.append(raw_feat)
            else:
                 logger.warning(f"Required raw feature '{raw_feat}' not found in standard tile features. Skipping.")

        if not self.bands_to_read:
             raise ValueError("Could not map any required model features to bands in the input tile.")
        logger.debug(f"Window Inference: Reading bands {self.bands_to_read} for features {self.read_feature_names}")


    def run(self) -> np.ndarray:
        """Reads data, preprocesses, predicts, and returns the classification map."""
        try:
            with rasterio.open(self.input_tile_path) as src:
                # Read only the necessary bands (uint16)
                # Shape: (n_bands_read, H, W)
                input_data_uint16 = src.read(self.bands_to_read, window=self.window)

            # --- Preprocessing ---
            # 1. Unscale features
            # 2. Apply phase transformations (cos/sin)
            # 3. Select and order features for the model
            # We need a structure like a DataFrame for transform_circular_features easily
            # Alternative: process pixel-wise or adapt transform_circular_features

            # Create a DataFrame-like structure for processing
            # Flatten spatial dims: (n_bands_read, H*W)
            n_pixels = self.H * self.W
            flat_data_uint16 = input_data_uint16.reshape(len(self.bands_to_read), n_pixels)
            
            processed_data = {} # Dict to hold processed feature columns

            for i, raw_feat_name in enumerate(self.read_feature_names):
                data_col_uint16 = flat_data_uint16[i, :]
                index_name, feature_type, is_phase = get_feature_metadata(raw_feat_name)

                # Unscale feature
                unscaled_data = unscale_feature(
                    data_col_uint16,
                    feature_type=feature_type,
                    index_name=index_name
                )

                # Apply phase transform if needed
                if is_phase:
                    cos_feat_name = f"{raw_feat_name}_cos"
                    sin_feat_name = f"{raw_feat_name}_sin"
                    # Only compute if required by the model
                    if cos_feat_name in self.model_features:
                        processed_data[cos_feat_name] = np.cos(unscaled_data)
                    if sin_feat_name in self.model_features:
                        processed_data[sin_feat_name] = np.sin(unscaled_data)
                elif raw_feat_name in self.model_features: # Non-phase feature required by model
                     processed_data[raw_feat_name] = unscaled_data

            # Create DataFrame for final selection and ordering (might be memory intensive)
            # Consider a more memory-efficient approach if needed (e.g., numpy array directly)
            df = pd.DataFrame(processed_data)

            # Select and order features exactly as the model expects
            # Handle missing columns gracefully (though ideally, they should be present if needed)
            missing_features = [f for f in self.model_features if f not in df.columns]
            if missing_features:
                 logger.error(f"Window ({self.window}): Missing expected features after processing: {missing_features}. Prediction might fail or be inaccurate.")
                 # Fill missing features with a default value (e.g., 0) or raise error?
                 # For now, let's fill with 0, but this is suboptimal.
                 for mf in missing_features:
                     df[mf] = 0.0
                 
            # Select columns in the exact order required by the model, keep as DataFrame
            X_predict_df = df[self.model_features]

            # --- Prediction ---
            # Ensure input is finite (replace NaNs/infs if any occurred during processing)
            # Check DataFrame for non-finite values
            if not np.all(np.isfinite(X_predict_df.values)):
                logger.warning(f"Window ({self.window}): Non-finite values detected in input features before prediction. Replacing with 0.")
                # Replace non-finite values in the DataFrame
                X_predict_df = X_predict_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

            # # Ensure the model uses only one job internally to avoid nested parallelism
            # if hasattr(self.model, 'n_jobs'):
            #     self.model.n_jobs = 1

            # Predict probabilities: shape (n_pixels, n_classes)
            # Allow model to use internal parallelism (n_jobs > 1) if configured
            probabilities_flat = self.model.predict_proba(X_predict_df)
            n_classes = probabilities_flat.shape[1]

            # --- Reshape Output ---
            # Reshape to (H, W, n_classes) then transpose to (n_classes, H, W) for rasterio
            probabilities_map = probabilities_flat.reshape(self.H, self.W, n_classes).transpose(2, 0, 1).astype(self.output_dtype)

            return probabilities_map

        except Exception as e:
            logger.error(f"Error processing window {self.window}: {e}", exc_info=True)
            # Return an empty/default array on error?
            # Need to know n_classes to return correct shape
            n_classes_fallback = getattr(self.model, 'n_classes_', 2) # Default to 2 classes if unavailable
            return np.zeros((n_classes_fallback, self.H, self.W), dtype=self.output_dtype)


class TileInferenceRF:
    """Orchestrates window-based RF inference for a single feature tile."""
    def __init__(self,
                 model: RandomForestClassifier,
                 model_features: list[str],
                 input_tile_path: Path,
                 output_tile_path: Path,
                 block_size: int = DEFAULT_BLOCK_SIZE,
                 num_workers: int = 1, # Controls window parallelism (now unused)
                 rf_n_jobs: int = 1    # Controls internal RF parallelism
                 ):

        if not input_tile_path.exists():
            raise FileNotFoundError(f"Input tile not found: {input_tile_path}")

        self.input_tile_path = input_tile_path
        self.output_tile_path = output_tile_path
        self.output_path = self.output_tile_path # Assign output_path here
        self.model = model
        self.model_features = model_features
        self.block_size = block_size
        self.num_workers = num_workers # Store worker count (should always be 1 now)
        self.rf_n_jobs = rf_n_jobs # Store desired n_jobs for internal RF

        # Get profile from input tile
        with rasterio.open(self.input_tile_path) as src:
            self.input_profile = src.profile
            self.width = src.width
            self.height = src.height

        # Define output profile (1 band, uint8 for class labels)
        self.output_profile = self.input_profile.copy()
        self.output_profile.update({
            "count": 1,
            "dtype": 'uint8', # Class labels 0, 1, 2...
            "nodata": 0,      # Assuming 0 is NoData or Background if needed
            "compress": "lzw",
            "predictor": 2,
            "BIGTIFF": "YES"   # Or "IF_SAFER"
        })

        # *** Update output profile for probabilities ***
        n_classes = getattr(self.model, 'n_classes_', 2) # Get number of classes
        self.output_profile.update({
            "count": n_classes,
            "dtype": 'uint8', # Change dtype to uint8
            "nodata": None,   # Explicitly set nodata to None for uint8 probabilities (0-255)
            # Retain compression settings
            "compress": "lzw", 
            "predictor": 2, 
            "BIGTIFF": "YES" 
        })
        logger.info(f"Output profile configured for {n_classes} bands (uint8, 0-255 scaled prob).")

    def _generate_windows(self):
        """Generates processing windows."""
        for j in range(0, self.height, self.block_size):
            for i in range(0, self.width, self.block_size):
                width = min(self.block_size, self.width - i)
                height = min(self.block_size, self.height - j)
                yield Window(i, j, width, height)

    def run(self):
        """Runs the inference process for the tile."""
        start_time = time.time()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # --- Set desired n_jobs for the model instance for this tile --- 
        if hasattr(self.model, 'n_jobs'):
            try:
                original_n_jobs = self.model.n_jobs
                if self.rf_n_jobs is not None and self.model.n_jobs != self.rf_n_jobs:
                     logger.info(f"Setting model n_jobs to {self.rf_n_jobs} for tile {self.input_tile_path.name} (original was {original_n_jobs})")
                     self.model.n_jobs = self.rf_n_jobs
                elif self.rf_n_jobs is None:
                     logger.info(f"Using model's default n_jobs ({original_n_jobs}) for tile {self.input_tile_path.name}")
            except Exception as e:
                 logger.warning(f"Could not set n_jobs on model: {e}")
        # -------------------------------------------------------------

        windows = list(self._generate_windows())

        # Use ProcessPoolExecutor if num_workers > 1
        use_parallel = self.num_workers > 1
        msg_suffix = f"with {self.num_workers} workers" if use_parallel else "sequentially"
        logger.info(f"Processing {len(windows)} windows for tile {self.input_tile_path.name} {msg_suffix}.")

        with rasterio.open(self.output_tile_path, 'w', **self.output_profile) as dst:
            if use_parallel:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    # Submit all window processing tasks
                    futures = [executor.submit(_process_window_rf,
                                              self.input_tile_path,
                                              window,
                                              self.model,
                                              self.model_features,
                                              self.output_profile['dtype'])
                               for window in windows]

                    # Collect results as they complete and write to output
                    for future in tqdm(as_completed(futures), total=len(windows), desc=f"Tile {self.input_tile_path.name}"):
                        try:
                            window, result_prob_map = future.result()
                            if result_prob_map is not None:
                                dst.write(result_prob_map, window=window)
                            else:
                                logger.warning(f"Window {window} failed (result was None) for tile {self.input_tile_path.name}")
                        except Exception as e:
                            # future.result() might raise an exception if the worker failed badly
                            logger.error(f"A window future failed with exception for tile {self.input_tile_path.name}: {e}", exc_info=True)
            else:
                # Sequential processing (original logic, slightly adapted)
                for window in tqdm(windows, desc=f"Tile {self.input_tile_path.name} (Seq)"):
                    try:
                        _window, result_prob_map = _process_window_rf(
                            self.input_tile_path, window, self.model, self.model_features,
                            output_dtype=self.output_profile['dtype']
                        )
                        if result_prob_map is not None:
                            dst.write(result_prob_map, window=window)
                        else:
                            logger.warning(f"Window {window} failed sequentially for tile {self.input_tile_path.name}")
                    except Exception as e:
                        logger.error(f"Sequential window {window} failed for tile {self.input_tile_path.name}: {e}", exc_info=True)

        # Removed end time logging per tile to keep output clean
        # end_time = time.time()


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run RandomForest inference on a folder of feature tiles.")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input feature tiles (GeoTIFF).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the output probability map tiles (GeoTIFF).")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained RandomForest model (.joblib file).")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help=f"Processing block size (default: {DEFAULT_BLOCK_SIZE}).")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS}). Set to 1 for sequential.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")
    parser.add_argument("--tile-idx", type=int, default=None,
                        help="Index of the specific tile to process (for job array tasks). If set, ignores --workers and processes only this tile.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        # Set logger level for all handlers as well if needed
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")


    try:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        model_path = Path(args.model)

        if not input_dir.is_dir():
            logger.error(f"Input directory not found: {input_dir}")
            sys.exit(1)
        if not model_path.exists():
             logger.error(f"Model file not found: {model_path}")
             sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

        # Find all input GeoTIFF tiles
        input_files = list(input_dir.glob('*.tif*')) # Use glob to find .tif and .tiff
        input_files.sort() # Ensure consistent order for tile indexing
        if not input_files:
             logger.error(f"No .tif or .tiff files found in {input_dir}")
             sys.exit(1)

        logger.info(f"Found {len(input_files)} input tiles in {input_dir}.")

        # --- Load Model and Config Once ---
        logger.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        if not isinstance(model, RandomForestClassifier):
            logger.warning(f"Loaded object from {model_path} is not a RandomForestClassifier.")

        # Load associated config file to get features
        config_path = model_path.with_suffix('.json').with_name(model_path.stem + '_config.json')
        if not config_path.exists():
            alt_config_path = model_path.parent / (model_path.stem.split('_')[0] + '_config.json')
            if alt_config_path.exists():
                config_path = alt_config_path
            else:
                logger.error(f"Model config file not found at {config_path} or similar patterns.")
                sys.exit(1)
        
        logger.info(f"Loading model config from: {config_path}")
        with open(config_path, 'r') as f:
            model_config = json.load(f)

        model_features = model_config.get('selected_features')
        if not model_features:
            logger.error("Could not find 'selected_features' list in model config.")
            sys.exit(1)
        # Log the features being used
        logger.info(f"Model expects {len(model_features)} features: {model_features}")

        # --- Execution Logic ---
        if args.tile_idx is not None:
            # Process a single tile based on index (Job Array Mode)
            if args.tile_idx < 0 or args.tile_idx >= len(input_files):
                logger.error(f"Tile index {args.tile_idx} is out of range (0-{len(input_files)-1}).")
                sys.exit(1)

            tile_to_process = input_files[args.tile_idx]
            logger.info(f"Processing single tile (index {args.tile_idx}): {tile_to_process.name} sequentially (windows), RF n_jobs=-1.")

            # Run the worker function directly for the single tile
            # Pass num_workers=1 for sequential windows.
            # Pass rf_n_jobs=-1 to use all cores for internal RF prediction.
            result = process_single_tile_worker(
                tile_to_process, output_dir, model, model_features, args.block_size, 
                num_workers=1, 
                rf_n_jobs=-1 
            )
            logger.info(result) # Log the result message

        else:
            # Process all tiles in parallel (Original Mode)
            logger.info(f"Starting parallel processing of {len(input_files)} tiles with {args.workers} workers (sequential windows, RF n_jobs=1 within each tile).")
            # This mode primarily parallelizes over tiles.
            results = []
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tile processing tasks
                # Pass num_workers=1 for sequential windows.
                # Pass rf_n_jobs=1 to ensure single-threaded RF prediction within each tile worker.
                futures = [executor.submit(process_single_tile_worker,
                                          tile_path, output_dir, model, model_features, args.block_size, 
                                          num_workers=1, 
                                          rf_n_jobs=1)
                           for tile_path in input_files]

                # Collect results as they complete, using tqdm for progress bar
                for future in tqdm(as_completed(futures), total=len(input_files), desc="Processing Tiles"):
                     results.append(future.result()) # Store success/failure message

            logger.info("All tiles processed.")
            # Optional: Log summary of results
            success_count = sum(1 for res in results if "Successfully" in res)
            failure_count = len(results) - success_count
            logger.info(f"Processing Summary: Success={success_count}, Failures={failure_count}")

    except FileNotFoundError as e: # Catch errors before parallel execution starts
        logger.error(f"Setup error - File not found: {e}")
        sys.exit(1)
    except ValueError as e: # Catch errors before parallel execution starts
        logger.error(f"Setup error - Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 