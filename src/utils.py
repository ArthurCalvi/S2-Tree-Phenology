"""
utils.py
---------
Helper routines: 
 - scaling to Uint16,
 - computing indices, 
 - reading data, 
 - robust harmonic fitting, etc.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from datetime import datetime
import logging
import math
import math
from typing import Optional, Callable, List, Tuple
import pandas as pd
import os
from collections import defaultdict
from sklearn.utils import shuffle
from pathlib import Path
try:  # Optional PyTorch dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:  # pragma: no cover - optional dependency
    from types import SimpleNamespace

    torch = None
    nn = SimpleNamespace(Module=object)  # type: ignore[attr-defined]
    F = SimpleNamespace()

    class Dataset:  # type: ignore
        pass

    class DataLoader:  # type: ignore
        pass
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from contextlib import contextmanager

try:
    import scienceplots  # noqa: F401
    _SCIENCEPLOTS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _SCIENCEPLOTS_AVAILABLE = False

from src.constants import (
    PHASE_RANGE,
    TARGET_AMP_RANGE, 
    TARGET_OFFSET_RANGE,
    BASE_FEATURE_NAMES_PER_INDEX,
    OUTPUT_FEATURE_NAMES_PER_INDEX,
    ALL_FEATURE_BAND_INDICES,
    FEATURE_SUFFIX_TO_TYPE,
    PHASE_FEATURE_SUFFIXES,
    PHASE_TRANSFORM_SUFFIXES,
    PHENOLOGY_BAND
)


def apply_science_style() -> None:
    """Apply the global SciencePlots style if the package is available."""
    if _SCIENCEPLOTS_AVAILABLE:  # pragma: no cover - styling
        plt.style.use(["science", "no-latex"])


@contextmanager
def science_style():
    """Context manager to temporarily apply the SciencePlots style."""
    if not _SCIENCEPLOTS_AVAILABLE:
        yield
    else:  # pragma: no cover - styling only
        with plt.style.context(["science", "no-latex"]):
            yield

def scale_array_to_uint16(
    data: np.ndarray,
    min_val: float,
    max_val: float
) -> np.ndarray:
    """
    Scale a floating array into [0..65535] (uint16),
    clipping values outside [min_val, max_val].
    """
    clipped = np.clip(data, min_val, max_val)
    scaled = ( (clipped - min_val) / (max_val - min_val) ) * 65535
    return scaled.astype(np.uint16)

def scale_phase(phase: np.ndarray) -> np.ndarray:
    """
    Convert phase from [-pi..pi] to [0..2*pi], then scale to uint16.
    """
    shifted = phase - PHASE_RANGE[0]  # PHASE_RANGE[0] = -pi, so shift by +pi
    range_size = PHASE_RANGE[1] - PHASE_RANGE[0]  # = 2*pi
    scaled = (shifted / range_size) * 65535
    scaled_clipped = np.clip(scaled, 0, 65535).astype(np.uint16)
    return scaled_clipped

def scale_amplitude(amp: np.ndarray, indice: str) -> np.ndarray:
    """Scale amplitude array to [0..65535] using target range defined for the given indice.

    The target ranges are defined in constants.TARGET_AMP_RANGE.
    """
    min_val, max_val = TARGET_AMP_RANGE[indice]
    return scale_array_to_uint16(amp, min_val, max_val)

def scale_offset(offset: np.ndarray, indice: str) -> np.ndarray:
    """Scale offset array to [0..65535] using target range defined for the given indice.

    The target ranges are defined in constants.TARGET_OFFSET_RANGE.
    """
    min_val, max_val = TARGET_OFFSET_RANGE[indice]
    return scale_array_to_uint16(offset, min_val, max_val)

def compute_indices(
    b2: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    b11: np.ndarray,
    b12: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute NDVI, EVI, NBR, CRSWIR from raw band data. 
    b2, b4, etc. might be float or int. 
    Return (ndvi, evi, nbr, crswir).
    """
    if logger:
        logger.info("Computing spectral indices for each time slice...")

    eps = 1e-6
    crswir_coeff = (1610 - 842) / (2190 - 842)

    # NDVI
    ndvi = (b8 - b4) / (b8 + b4 + eps)

    # EVI (some standard constants)
    evi_denom = (b8 + 6.0 * b4 - 7.5 * b2 + 1.0 + eps)
    evi = 2.5 * (b8 - b4) / evi_denom

    # NBR
    nbr = (b8 - b12) / (b8 + b12 + eps)

    # CRSWIR as example
    crswir_denom = ((b12 - b8) * crswir_coeff + b8) + eps
    crswir = b11 / crswir_denom

    # Clipping each index to its valid range:
    ndvi = np.clip(ndvi, -1.0, 1.0)
    evi = np.clip(evi, -1.0, 1.0)
    nbr = np.clip(nbr, -1.0, 1.0)
    crswir = np.clip(crswir, 0.0, 5.0)

    return ndvi, evi, nbr, crswir

def compute_quality_weights(
    cldprb: np.ndarray,
    logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Convert the cloud probability (0..100) into a weight array 
    in [0..1], e.g. weight = 1 - (cldprb/100).
    So if MSK_CLDPRB = 100 => weight=0 (completely cloud), 
       if 0 => weight=1. 
    """
    if logger:
        logger.info("Computing QA weights from MSK_CLDPRB band")
    cldprb_float = cldprb.astype(float)
    w = 1.0 - (cldprb_float / 100.0)
    # Clip to [0..1]
    w = np.clip(w, 0.0, 1.0)
    return w

def solve_params(ATA: np.ndarray, ATb: np.ndarray) -> np.ndarray:
    """
    Solve linear equations with error handling for non-invertible cases.
    """
    try:
        return np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.full(ATb.shape, np.nan)  # Return NaN for non-invertible matrices

def robust_harmonic_fitting(
    data_cube: np.ndarray,        # shape: (T,H,W)  (time series data)
    weight_cube: np.ndarray,      # shape: (T,H,W)  (initial QA weights in [0..1])
    dates: List[datetime],        # length T
    num_harmonics: int = 2,
    max_iter: int = 10,
    tol: float = 5e-2,
    percentile: float = 75.0,
    min_param_threshold: float = 1e-5,
    callback: Optional[Callable[[int, float], None]] = None,
    logger: Optional[logging.Logger] = None,
    debug: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Fits a robust periodic function (Fourier harmonics) to the time series in data_cube
    using IRLS with a Huber-type loss. Closely follows the 'fit_periodic_function_with_harmonics_robust'
    logic, but adapted to the pipeline signature.

    Args:
        data_cube: float array of shape (T,H,W), possibly containing NaNs.
        weight_cube: float array of shape (T,H,W) with initial QA weights in [0..1].
        dates: list of datetime objects (length T).
        num_harmonics: number of harmonics to fit.
        max_iter: maximum IRLS iterations.
        tol: tolerance on relative parameter change (based on percentile).
        percentile: percentile used for the relative change stopping criterion.
        min_param_threshold: threshold to avoid division by zero in relative change.
        callback: optional callback with signature (iteration, percentile_value).
        logger: optional logger for debugging.
        debug: if True, can enable additional plotting or checks.

    Returns:
        A tuple containing:
         - amplitude_h1, amplitude_h2, ..., amplitude_hN  (each shape (H,W))
         - phase_h1, phase_h2, ..., phase_hN              (each shape (H,W))
         - offset_map                                     (shape (H,W))
         - residual_variance_map                          (shape (H,W))
    """
    try:
        if logger:
            logger.info(f"Starting robust harmonic fitting with {num_harmonics} harmonics")

        T, H, W = data_cube.shape
        # Reshape from (T,H,W) to (T, N) where N=H*W
        N = H * W
        data_2d = data_cube.reshape(T, N)
        weights_2d = weight_cube.reshape(T, N)

        # 1) Handle NaNs in data: fill with temporal mean
        nan_mask = np.isnan(data_2d)
        pixel_means = np.nanmean(data_2d, axis=0)  # shape (N,)
        # Where data is NaN, fill with pixel_means
        data_filled = np.where(nan_mask, np.tile(pixel_means, (T,1)), data_2d)
        # Set weights to 0 where data was NaN
        weights_2d = np.where(nan_mask, 0.0, weights_2d)

        # 2) Setup design matrix for harmonics
        times_array = np.array(dates, dtype='datetime64[D]')
        days_since_start = (times_array - times_array[0]).astype(int)
        t_normalized = days_since_start / 365.25

        # build cos/sin for each harmonic plus a constant offset
        harmonics = []
        for k in range(1, num_harmonics + 1):
            freq = 2.0 * math.pi * k
            harmonics.append(np.cos(freq * t_normalized))
            harmonics.append(np.sin(freq * t_normalized))
        A = np.stack(harmonics + [np.ones_like(t_normalized)], axis=-1)  # shape: (T, 2*num_harmonics+1)

        # 3) Initial parameter estimation via ordinary least squares
        A_pinv = np.linalg.pinv(A)          # shape: (2*N+1, T)
        init_params = A_pinv @ data_filled  # shape: (2*num_harmonics+1, N)
        params = init_params.T              # shape: (N, 2*num_harmonics+1)

        # 4) IRLS
        for iteration in range(max_iter):
            params_old = params.copy()

            # Weighted design
            # For each pixel, we solve normal eq. We'll do a batched approach:
            fitted_2d = (A @ params.T).reshape(T, N)  # shape (T,N)
            residuals_2d = data_filled - fitted_2d

            # compute std dev per pixel => shape (N,)
            sigma_res = np.std(residuals_2d, axis=0, ddof=1)
            sigma_res[sigma_res==0] = 1e-9
            # Huber threshold
            delta = 1.345 * sigma_res

            # Evaluate absolute residual
            abs_res = np.abs(residuals_2d)
            # robust weight
            w_robust = np.where(abs_res <= delta, 1.0, delta / (abs_res + 1e-9))

            # combine with existing QA weights
            w_total = weights_2d * w_robust

            # Solve for parameters pixel-by-pixel
            # shape of A is (T,M), w_total is (T,N), data_filled is (T,N)
            # We'll do it with a loop or broadcast:
            M = A.shape[1]
            new_params = np.zeros((N, M), dtype=np.float32)

            # Precompute expansions
            # A_expanded -> shape (T,M,1) so we can multiply by w easily
            # but a simpler approach is: for i in range(N), build diag of w, etc.
            # We'll do a direct approach with np.einsum

            # For performance in real HPC code, you might do advanced vectorization.
            for i in range(N):
                w_col = w_total[:, i]  # shape (T,)
                if w_col.sum() < 1e-9:
                    # no valid info => keep zero
                    continue
                Aw = (A.T * w_col).T  # shape (T,M)
                ATA = Aw.T @ A        # shape (M,M)
                ATy = Aw.T @ data_filled[:, i]  # shape (M,)
                solved = solve_params(ATA, ATy)
                new_params[i] = np.nan_to_num(solved, nan=0.0)

            params = new_params

            # Check convergence: compute relative change
            param_diff = np.abs(params - params_old)
            # to avoid division by zero
            denom = np.maximum(np.abs(params_old), min_param_threshold)
            relative_change = param_diff / denom
            # flatten:
            rel_flat = relative_change.ravel()
            p_value = np.percentile(rel_flat, percentile)

            if callback:
                callback(iteration, p_value)

            if logger:
                logger.debug(f"Iteration {iteration+1}, percentile {percentile} of param change: {p_value:.5f}")

            if p_value < tol:
                if logger:
                    logger.info(f"Converged after {iteration + 1} iterations (relative change: {p_value:.5f})")
                break

        # 5) final residual
        fitted_final = (A @ params.T).reshape(T, N)
        final_residuals = data_filled - fitted_final
        var_resid = np.var(final_residuals, axis=0)

        # 6) Extract parameters as amplitude/phase for each harmonic
        # shape of params => (N, 2*num_harmonics+1). We'll transpose to (2*num_harmonics+1, N)
        params_t = params.T
        amplitude_maps = []
        phase_maps = []
        # each harmonic: (A, B) => amplitude=sqrt(A^2+B^2), phase=atan2(B,A)
        idx = 0
        for k in range(num_harmonics):
            A_k = params_t[idx]   # shape (N,)
            B_k = params_t[idx+1] # shape (N,)
            idx += 2
            amplitude = np.sqrt(A_k**2 + B_k**2)
            raw_phase = np.arctan2(B_k, A_k)  # in [-pi,pi]

            # we replicate the "phase_adjusted" logic if you want, but it's optional
            # The user code used:
            # phase_adjusted = (raw_phase - (2π(k+1)*t_norm[0])) % (2π)
            # then shift if > π
            # This is more of a cosmetic shift for referencing the initial day.
            # We'll do it for parity:
            freq_factor = 2*math.pi*(k+1)*t_normalized[0]
            phase_adj = (raw_phase - freq_factor) % (2*math.pi)
            phase_norm = np.where(phase_adj>math.pi, phase_adj - 2*math.pi, phase_adj)

            amplitude_maps.append(amplitude.reshape(H,W))
            phase_maps.append(phase_norm.reshape(H,W))

        # offset => last row in params_t
        offset_map_1d = params_t[idx]  # shape (N,)
        offset_map = offset_map_1d.reshape(H,W)
        var_map = var_resid.reshape(H,W)

        return (*amplitude_maps, *phase_maps, offset_map, var_map)

    except Exception as e:
        if logger:
            logger.error(f"Error in robust harmonic fitting: {str(e)}")
        raise

# --- Training Utility Functions ---

def unscale_feature(
    scaled_data: np.ndarray, 
    feature_type: str, 
    index_name: Optional[str] = None
) -> np.ndarray:
    """
    Unscale feature data from uint16 [0, 65535] back to its original physical range.

    Args:
        scaled_data: Numpy array of scaled data (uint16 or float).
        feature_type: Type of feature ('amplitude', 'phase', 'offset', 'variance').
        index_name: Name of the spectral index (e.g., 'ndvi'), required for 
                    'amplitude' and 'offset' types to look up the original range.

    Returns:
        Numpy array of unscaled data in its original physical range.
    """
    scaled_data_float = scaled_data.astype(np.float64) # Use float64 for precision
    
    if feature_type == 'phase':
        # Unscales from [0, 65535] to [0, 2*pi] (matching the previous logic)
        # Note: Original physical range from fitting was [-pi, pi]. 
        # To get that range: return ((scaled_data_float / 65535.0) * (2 * math.pi)) - math.pi
        return (scaled_data_float / 65535.0) * (2.0 * math.pi)
        
    elif feature_type == 'amplitude':
        if index_name is None or index_name not in TARGET_AMP_RANGE:
            raise ValueError(f"index_name '{index_name}' is required and must be valid for feature_type 'amplitude'.")
        min_val, max_val = TARGET_AMP_RANGE[index_name]
        range_val = max_val - min_val
        return (scaled_data_float / 65535.0) * range_val + min_val
        
    elif feature_type == 'offset':
        if index_name is None or index_name not in TARGET_OFFSET_RANGE:
            raise ValueError(f"index_name '{index_name}' is required and must be valid for feature_type 'offset'.")
        min_val, max_val = TARGET_OFFSET_RANGE[index_name]
        range_val = max_val - min_val
        return (scaled_data_float / 65535.0) * range_val + min_val
        
    elif feature_type == 'variance':
        # Variance was scaled from a clamped range [0, 2]
        min_val, max_val = 0.0, 2.0
        range_val = max_val - min_val
        return (scaled_data_float / 65535.0) * range_val + min_val
        
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}. Must be one of 'amplitude', 'phase', 'offset', 'variance'.")

def transform_circular_features(df, indices):
    """
    Apply cos/sin transformation to phase features which are circular in nature.
    Assumes the input phase features (e.g., ndvi_phase_h1, evi_phase_h2) 
    are already in **radians** (e.g., previously unscaled from [0, 65535] to [0, 2*pi]).
    This prevents the model from treating the phase as a linear feature.
    Drops the original radian phase features after transformation.
    
    Args:
        df: DataFrame containing phase features in **radians**.
        indices: List of indices to transform (e.g., ['ndvi', 'evi', 'nbr', 'crswir'])
        
    Returns:
        DataFrame with transformed phase features (cosine and sine components).
    """
    transformed_df = df.copy()
    original_phase_cols = []

    # Apply transformation for each index
    for index in indices:
        phase_h1_col = f'{index}_phase_h1'
        phase_h2_col = f'{index}_phase_h2'

        if phase_h1_col in transformed_df.columns:
            # Apply cos/sin directly, assuming input is radians
            transformed_df[f'{index}_phase_h1_cos'] = np.cos(transformed_df[phase_h1_col])
            transformed_df[f'{index}_phase_h1_sin'] = np.sin(transformed_df[phase_h1_col])
            original_phase_cols.append(phase_h1_col)

        if phase_h2_col in transformed_df.columns:
            # Apply cos/sin directly, assuming input is radians
            transformed_df[f'{index}_phase_h2_cos'] = np.cos(transformed_df[phase_h2_col])
            transformed_df[f'{index}_phase_h2_sin'] = np.sin(transformed_df[phase_h2_col])
            original_phase_cols.append(phase_h2_col)

    # Drop the original radian phase features if they existed
    transformed_df = transformed_df.drop(columns=original_phase_cols, errors='ignore')
    
    return transformed_df

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics for **binary** phenology classification.
    
    Args:
        y_true: True labels (assuming 1: Deciduous, 2: Evergreen)
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics including confusion matrix values, accuracy,
        precision/recall (Deciduous, Evergreen, macro, weighted), 
        and various F1 scores.
    """
    from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
    
    metrics_results = {
        'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
        'accuracy': 0.0,
        'precision_deciduous': 0.0,
        'recall_deciduous': 0.0,
        'precision_evergreen': 0.0, 
        'recall_evergreen': 0.0,
        'precision_macro': 0.0,
        'recall_macro': 0.0,
        'precision_weighted': 0.0,
        'recall_weighted': 0.0,
        'f1_evergreen': 0.0,
        'f1_deciduous': 0.0,
        'f1_weighted': 0.0,
        'f1_macro': 0.0
    }

    # Ensure there are predictions to compute confusion matrix
    if len(y_true) == 0 or len(y_pred) == 0:
        return metrics_results # Return default zeroed metrics
        
    # --- Calculate all metrics using sklearn functions for robustness --- 
    labels = [1, 2] # Explicitly define labels for consistency
    pos_label_evergreen = 2   # Define Evergreen as the positive class for its binary metrics
    pos_label_deciduous = 1   # Define Deciduous as the positive class for its binary metrics

    try:
        # Confusion Matrix Components (TN, FP, FN, TP for pos_label=2, i.e. Evergreen)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if cm.size == 4: # Ensure it's a 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
            metrics_results['tn'] = int(tn) # TN for Evergreen means True Deciduous predicted as Deciduous
            metrics_results['fp'] = int(fp) # FP for Evergreen means False Evergreen (Deciduous predicted as Evergreen)
            metrics_results['fn'] = int(fn) # FN for Evergreen means False Negative (Evergreen predicted as Deciduous)
            metrics_results['tp'] = int(tp) # TP for Evergreen means True Evergreen predicted as Evergreen
        else: # Handle unexpected CM shape (e.g., only one class predicted/present)
             # Calculate TP/FP/TN/FN manually based on pos_label=2 (Evergreen)
             metrics_results['tp'] = int(np.sum((y_true == pos_label_evergreen) & (y_pred == pos_label_evergreen)))
             metrics_results['fp'] = int(np.sum((y_true != pos_label_evergreen) & (y_pred == pos_label_evergreen)))
             metrics_results['tn'] = int(np.sum((y_true != pos_label_evergreen) & (y_pred != pos_label_evergreen)))
             metrics_results['fn'] = int(np.sum((y_true == pos_label_evergreen) & (y_pred != pos_label_evergreen)))

        # Accuracy
        total = metrics_results['tp'] + metrics_results['fp'] + metrics_results['tn'] + metrics_results['fn']
        metrics_results['accuracy'] = float((metrics_results['tp'] + metrics_results['tn']) / total) if total > 0 else 0.0

        # Precision (Deciduous, Evergreen, Macro, Weighted)
        metrics_results['precision_deciduous'] = float(precision_score(y_true, y_pred, labels=labels, pos_label=pos_label_deciduous, average='binary', zero_division=0))
        metrics_results['precision_evergreen'] = float(precision_score(y_true, y_pred, labels=labels, pos_label=pos_label_evergreen, average='binary', zero_division=0))
        metrics_results['precision_macro'] = float(precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
        metrics_results['precision_weighted'] = float(precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))

        # Recall (Deciduous, Evergreen, Macro, Weighted)
        metrics_results['recall_deciduous'] = float(recall_score(y_true, y_pred, labels=labels, pos_label=pos_label_deciduous, average='binary', zero_division=0))
        metrics_results['recall_evergreen'] = float(recall_score(y_true, y_pred, labels=labels, pos_label=pos_label_evergreen, average='binary', zero_division=0))
        metrics_results['recall_macro'] = float(recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
        metrics_results['recall_weighted'] = float(recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))
        
        # F1 Scores (Evergreen, Deciduous, Macro, Weighted)
        metrics_results['f1_evergreen'] = float(f1_score(y_true, y_pred, labels=labels, pos_label=pos_label_evergreen, average='binary', zero_division=0))
        metrics_results['f1_deciduous'] = float(f1_score(y_true, y_pred, labels=labels, pos_label=pos_label_deciduous, average='binary', zero_division=0))
        metrics_results['f1_macro'] = float(f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
        metrics_results['f1_weighted'] = float(f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))

    except Exception as e:
        # Log error or handle? For now, return the zeroed dict if any error occurs during calculation
        # Consider adding logging here if needed
        print(f"Warning: Error calculating metrics: {e}. Returning zeroed metrics.") # Simple print for now
        # Re-initialize to default zeroed metrics in case of error after partial calculation
        metrics_results = {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'accuracy': 0.0,
            'precision_deciduous': 0.0,
            'recall_deciduous': 0.0,
            'precision_evergreen': 0.0, 
            'recall_evergreen': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'f1_evergreen': 0.0,
            'f1_deciduous': 0.0,
            'f1_weighted': 0.0,
            'f1_macro': 0.0
        }

    return metrics_results

def compute_multiclass_metrics(y_true, y_pred, labels=None, target_names=None):
    """
    Compute classification metrics for multi-class problems.

    Args:
        y_true: True labels (numpy array).
        y_pred: Predicted labels (numpy array).
        labels: Ordered list of labels to include in the report. If None, uses unique labels present in y_true or y_pred.
        target_names: Optional display names matching the labels (same order).

    Returns:
        Dictionary of metrics including confusion matrix, accuracy,
        precision/recall/f1 (macro, weighted, per-class).
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
    
    # Determine unique labels if not provided
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
    
    # Ensure target names match labels length if provided
    if target_names and len(labels) != len(target_names):
        raise ValueError("Number of labels and target_names must match.")
    elif target_names is None:
        target_names = [str(lbl) for lbl in labels] # Use string representation of labels

    # Map labels to target names for per-class results
    label_to_name = dict(zip(labels, target_names))

    metrics_results = {
        'confusion_matrix': None,
        'accuracy': 0.0,
        'precision_macro': 0.0,
        'recall_macro': 0.0,
        'f1_macro': 0.0,
        'precision_weighted': 0.0,
        'recall_weighted': 0.0,
        'f1_weighted': 0.0,
        'precision_per_class': {}, # Dict mapping class name -> score
        'recall_per_class': {},
        'f1_per_class': {}
    }

    # Ensure there are predictions to compute metrics
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: Empty true or predicted labels passed to compute_multiclass_metrics.")
        metrics_results['confusion_matrix'] = np.zeros((len(labels), len(labels)), dtype=int)
        return metrics_results # Return default zeroed metrics
    
    try:
        # Confusion Matrix
        metrics_results['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=labels)

        # Accuracy
        metrics_results['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-Score (Macro, Weighted, Per-Class)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, 
            y_pred, 
            labels=labels, 
            average=None, # Calculate per-class first
            zero_division=0
        )
        
        # Per-class metrics
        for i, label in enumerate(labels):
            class_name = label_to_name[label]
            metrics_results['precision_per_class'][class_name] = float(precision[i])
            metrics_results['recall_per_class'][class_name] = float(recall[i])
            metrics_results['f1_per_class'][class_name] = float(f1[i])
            
        # Macro averages (unweighted mean)
        metrics_results['precision_macro'] = float(np.mean(precision))
        metrics_results['recall_macro'] = float(np.mean(recall))
        metrics_results['f1_macro'] = float(np.mean(f1))
        
        # Weighted averages (weighted by support)
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, 
            y_pred, 
            labels=labels, 
            average='weighted', 
            zero_division=0
        )
        metrics_results['precision_weighted'] = float(precision_w)
        metrics_results['recall_weighted'] = float(recall_w)
        metrics_results['f1_weighted'] = float(f1_w)

    except Exception as e:
        print(f"Warning: Error calculating multi-class metrics: {e}. Returning partially calculated or zeroed metrics.")
        # Ensure CM is at least initialized if error occurred before calculation
        if metrics_results['confusion_matrix'] is None:
             metrics_results['confusion_matrix'] = np.zeros((len(labels), len(labels)), dtype=int)

    return metrics_results

def create_eco_balanced_folds_df(df, n_splits=5, random_state=42):
    """
    Create folds from DataFrame that balance eco-region distribution 
    while preserving tile integrity.
    
    Args:
        df: DataFrame containing 'eco_region' and 'tile_id' columns
        n_splits: Number of folds to create
        random_state: Random seed for reproducibility
        
    Returns:
        List of tuples (train_idx, val_idx) for each fold
    """
    # Reset the index to ensure we work with the current DataFrame indices
    df = df.reset_index(drop=True)
    
    # Get unique eco-regions
    eco_regions = df['eco_region'].unique()
    
    # Initialize lists to store folds
    all_folds = []
    for _ in range(n_splits):
        all_folds.append({'train_idx': [], 'val_idx': []})
    
    # Process each eco-region separately
    for eco_region in eco_regions:
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
        
        fold_splits.append((train_idx, val_idx))
    
    return fold_splits

def display_fold_distribution(train_idx, val_idx, df, fold):
    """
    Display training and validation distribution per eco-region.
    
    Args:
        train_idx: Indices for training set
        val_idx: Indices for validation set
        df: DataFrame containing data
        fold: Fold number (0-indexed)
        
    Returns:
        DataFrame with distribution statistics
    """
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # Calculate distribution per eco-region
    train_dist = train_df.groupby('eco_region').size()
    val_dist = val_df.groupby('eco_region').size()
    
    # Calculate percentages
    train_pct = train_dist / train_dist.sum() * 100
    val_pct = val_dist / val_dist.sum() * 100
    
    # Calculate train-val ratio
    ratio = pd.Series(index=train_dist.index, dtype=float)
    for region in train_dist.index:
        if region in val_dist and val_dist[region] > 0:
            ratio[region] = train_dist[region] / val_dist[region]
        else:
            ratio[region] = float('inf') if train_dist[region] > 0 else 0
    
    # Combine into a single DataFrame
    distribution = pd.DataFrame({
        'Train Count': train_dist,
        'Train %': train_pct,
        'Val Count': val_dist.reindex(train_dist.index, fill_value=0),
        'Val %': val_pct.reindex(train_dist.index, fill_value=0),
        'Total Count': train_dist + val_dist.reindex(train_dist.index, fill_value=0),
        'Train/Val Ratio': ratio
    })
    
    return distribution

def format_confusion_matrix(cm_array, labels=None):
    """
    Format a **binary** confusion matrix numpy array as text for display.
    
    Args:
        cm_array: Confusion matrix as numpy array (must be 2x2).
        labels: List of two labels for display, e.g. ["Class 0", "Class 1"].
                Order should correspond to the CM: labels[0] for row/col 0, labels[1] for row/col 1.
        
    Returns:
        String containing formatted confusion matrix, or an error message if cm_array is not 2x2.
    """
    from tabulate import tabulate # Import tabulate here

    if cm_array.shape != (2, 2):
        return "Error: format_confusion_matrix is designed for 2x2 matrices only."

    if labels is None or len(labels) != 2:
        # Default labels if not provided or incorrect length
        display_labels = ["Class 0", "Class 1"]
        # Log a warning if labels were provided but incorrect, so user is aware
        if labels is not None:
            # Assuming logger is configured elsewhere or add a basic print warning
            print(f"Warning: Invalid labels provided to format_confusion_matrix. Expected 2 labels, got {len(labels)}. Using default.")
    else:
        display_labels = labels

    # Extract values assuming cm_array is confirmed to be 2x2
    tn, fp, fn, tp = cm_array.ravel()

    # Prepare data for tabulate
    table_data = [
        ["Actual " + display_labels[0], tn, fp],
        ["Actual " + display_labels[1], fn, tp]
    ]
    
    # Define headers
    headers = ["", "Predicted " + display_labels[0], "Predicted " + display_labels[1]]

    # Generate table using tabulate
    cm_text = tabulate(table_data, headers=headers, tablefmt="grid")
    
    return "Confusion Matrix:\n" + cm_text

def format_multiclass_confusion_matrix(cm_array, target_names):
    """
    Format a multi-class confusion matrix numpy array as text for display.

    Args:
        cm_array: Confusion matrix as numpy array (NxN).
        target_names: List of class names corresponding to the rows/columns.

    Returns:
        String containing formatted confusion matrix.
    """
    from tabulate import tabulate # Import tabulate here
    
    n_classes = len(target_names)
    if cm_array.shape != (n_classes, n_classes):
        return f"Error: Confusion matrix shape {cm_array.shape} does not match number of target names {n_classes}."

    # Prepare data for tabulate
    headers = ["Actual \\ Predicted"] + list(target_names)
    table_data = []
    for i, name in enumerate(target_names):
        row = [name] + cm_array[i, :].tolist()
        table_data.append(row)

    # Generate table using tabulate
    cm_text = tabulate(table_data, headers=headers, tablefmt="grid")
    
    return "Confusion Matrix:\n" + cm_text

# --- U-Net Specific Utility Functions ---

def get_tile_eco_region_mapping(mapping_file='results/datasets/training_datasets_pixels.parquet',
                                logger: Optional[logging.Logger] = None):
    """
    Load mapping of tiles to eco-regions from parquet file.
    
    Returns a dictionary mapping tile IDs to eco-region names.
    """
    # Use provided logger or create a default one
    logger = logger or logging.getLogger(__name__)
    
    try:
        # Load the parquet file
        logger.info(f"Loading tile to eco-region mapping from {mapping_file}")
        if not os.path.exists(mapping_file):
             raise FileNotFoundError(f"Eco-region mapping file not found: {mapping_file}")
        df = pd.read_parquet(mapping_file)
        logger.info(f"Mapping file loaded with {len(df)} entries")
        
        # Verify that required columns exist
        if 'tile_id' not in df.columns:
            raise ValueError("'tile_id' column not found in mapping file")
        
        if 'eco_region' not in df.columns:
            raise ValueError("'eco_region' column not found in mapping file")
        
        # Create dictionary mapping tile IDs to eco-regions
        # Convert IDs to strings to ensure matching
        df['tile_id'] = df['tile_id'].astype(str)
        
        # Get unique tile-eco_region combinations
        tile_eco_df = df[['tile_id', 'eco_region']].drop_duplicates()
        
        # Create mapping dictionary
        tile_to_eco = tile_eco_df.set_index('tile_id')['eco_region'].to_dict()
        
        # Log unique eco-regions in the mapping file
        eco_regions = df['eco_region'].unique()
        logger.info(f"Found {len(eco_regions)} unique eco-regions in mapping file")
        
        # Report the results
        eco_counts = {} # Count tiles per eco-region in the final map
        for eco in tile_to_eco.values():
            eco_counts[eco] = eco_counts.get(eco, 0) + 1
            
        logger.info(f"Created mapping for {len(tile_to_eco)} unique tiles across {len(eco_counts)} eco-regions")
        # Sort by count for clarity
        #for eco, count in sorted(eco_counts.items(), key=lambda item: item[1], reverse=True):
        #    logger.debug(f"  {eco}: {count} tiles")
            
        return tile_to_eco
    
    except Exception as e:
        logger.error(f"Error loading eco-region mapping: {e}")
        raise # Re-raise exception to signal failure

def create_eco_balanced_folds_tiles(tiles, n_splits=5, random_state=42,
                                    logger: Optional[logging.Logger] = None):
    """
    Create folds for TIFF tiles that balance eco-region distribution.
    Ensures that each eco-region is properly represented in training and validation.
    
    Args:
        tiles: List of tile paths
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_tiles, val_tiles) tuples for each fold
    """
    # Use provided logger or create a default one
    logger = logger or logging.getLogger(__name__)
    
    logger.info("Creating eco-region balanced folds for tiles...")
    rng = np.random.RandomState(random_state) # Local RNG

    try:
        tile_to_eco = get_tile_eco_region_mapping(logger=logger)
    except Exception as e:
        logger.error(f"Failed to load eco-region mapping: {e}. Falling back to random split.")
        tiles = shuffle(tiles, random_state=random_state)
        fold_indices = np.array_split(np.arange(len(tiles)), n_splits)
        return [([tiles[i] for i in np.concatenate(fold_indices[:k] + fold_indices[k+1:])], 
                 [tiles[i] for i in fold_indices[k]]) for k in range(n_splits)]

    eco_to_tiles = defaultdict(list)
    unknown_eco_tiles = []
    for tile_path in tiles:
        # Extract tile ID assuming format 'tile_ID_...' or similar
        try:
            tile_id = Path(tile_path).stem.split('_')[1]
        except IndexError:
            logger.warning(f"Could not parse tile ID from {tile_path}. Treating as unknown eco-region.")
            unknown_eco_tiles.append(tile_path)
            continue
            
        eco_region = tile_to_eco.get(tile_id)
        if eco_region:
            eco_to_tiles[eco_region].append(tile_path)
        else:
            # logger.debug(f"Tile ID {tile_id} not found in eco-region mapping. Treating as unknown.")
            unknown_eco_tiles.append(tile_path)

    # Log the distribution of tiles by eco-region
    logger.info("Distribution of input tiles by eco-region:")
    for eco, eco_tiles_list in eco_to_tiles.items():
        logger.info(f"  {eco}: {len(eco_tiles_list)} tiles")
    if unknown_eco_tiles:
        logger.info(f"  Unknown: {len(unknown_eco_tiles)} tiles")

    # Initialize fold assignments
    fold_assignments = [{"train": [], "val": []} for _ in range(n_splits)]

    # For each eco-region, distribute tiles evenly across folds
    for eco_region, eco_tiles_list in eco_to_tiles.items():
        # Shuffle tiles within each eco-region
        eco_tiles_list = shuffle(eco_tiles_list, random_state=random_state)
        
        # Create stratified folds for this eco-region using array_split for better distribution
        indices = np.arange(len(eco_tiles_list))
        fold_indices = np.array_split(indices, n_splits)
        
        # Assign to validation for each fold, rest to training
        for fold_idx in range(n_splits):
            val_idxs = fold_indices[fold_idx]
            if len(val_idxs) > 0: # Check if list is not empty
                fold_assignments[fold_idx]["val"].extend([eco_tiles_list[i] for i in val_idxs])
                # Training tiles are all other folds for this eco-region
                train_idxs = np.concatenate([fold_indices[k] for k in range(n_splits) if k != fold_idx])
                if len(train_idxs) > 0: # Check if list is not empty
                    fold_assignments[fold_idx]["train"].extend([eco_tiles_list[i] for i in train_idxs])

    # Distribute unknown tiles (assign randomly to validation folds, add to others' train)
    if unknown_eco_tiles:
        logger.info(f"Distributing {len(unknown_eco_tiles)} tiles with unknown eco-regions...")
        unknown_eco_tiles = shuffle(unknown_eco_tiles, random_state=random_state)
        unknown_fold_indices = np.array_split(np.arange(len(unknown_eco_tiles)), n_splits)
        for fold_idx in range(n_splits):
            val_idxs = unknown_fold_indices[fold_idx]
            if len(val_idxs) > 0:
                val_set = [unknown_eco_tiles[i] for i in val_idxs]
                fold_assignments[fold_idx]["val"].extend(val_set)
                # Add these validation tiles to the training sets of all *other* folds
                for train_fold_idx in range(n_splits):
                    if train_fold_idx != fold_idx:
                        fold_assignments[train_fold_idx]["train"].extend(val_set)

    # Finalize: Convert to list of tuples, remove duplicates, ensure non-empty val
    final_fold_splits = []
    total_assigned_val_tiles = 0
    processed_val_tiles = set()

    for fold_idx, fold_data in enumerate(fold_assignments):
        # Ensure unique tiles in train and val sets for this fold
        train_set = sorted(list(set(fold_data["train"])))
        val_set = sorted(list(set(fold_data["val"])))
        
        # Critical: Ensure no validation tile is also in the training set for THIS fold
        original_val_len = len(val_set)
        val_set = [tile for tile in val_set if tile not in train_set]
        if len(val_set) < original_val_len:
            logger.warning(f"Fold {fold_idx+1}: Removed {original_val_len - len(val_set)} tiles from validation set because they were also in training set.")

        # If validation is empty after removing overlap, try borrowing from training
        if not val_set and train_set:
            logger.warning(f"Fold {fold_idx+1} has empty validation set after overlap removal. Borrowing one tile from training.")
            borrowed_tile = train_set.pop(rng.choice(len(train_set))) # Remove from train
            val_set.append(borrowed_tile) # Add to val
            
        # Only add fold if validation set is non-empty
        if val_set:
            final_fold_splits.append((train_set, val_set))
            logger.info(f"Fold {fold_idx + 1}: {len(train_set)} train, {len(val_set)} val tiles.")
            total_assigned_val_tiles += len(val_set)
            processed_val_tiles.update(val_set)
        else:
             logger.error(f"Fold {fold_idx+1} resulted in an empty validation set. Skipping this fold.")

    # Final sanity check: Ensure all original tiles are covered in validation sets across folds
    if len(processed_val_tiles) != len(tiles):
         logger.warning(f"Validation sets cover {len(processed_val_tiles)} unique tiles, but input had {len(tiles)}. Check fold generation logic.")
         missing_tiles = set(tiles) - processed_val_tiles
         logger.warning(f"Missing tiles (first 5): {list(missing_tiles)[:5]}")

    return final_fold_splits


class SmallUNet(nn.Module):
    """
    Small U-Net architecture for phenology classification
    """
    def __init__(self, in_channels, n_classes=3):
        super(SmallUNet, self).__init__()
        filters = [16, 32, 64]
        self.enc1 = self._conv_block(in_channels, filters[0])
        self.enc2 = self._conv_block(filters[0], filters[1])
        self.enc3 = self._conv_block(filters[1], filters[2])
        self.bottleneck = self._conv_block(filters[2], filters[2])
        self.dec3 = self._conv_block(filters[2] * 2, filters[1])
        self.dec2 = self._conv_block(filters[1] * 2, filters[0])
        self.dec1 = self._conv_block(filters[0] * 2, filters[0])
        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self._initialize_weights()
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.final.weight, mean=0, std=0.01)
        nn.init.constant_(self.final.bias, 0)
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.dec3(torch.cat([self.upsample(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upsample(dec2), enc1], dim=1))
        return self.final(dec1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class SpatialTransforms:
    """Performs conservative data augmentation suitable for satellite imagery"""
    def __init__(self, p_flip=0.5):
        self.p_flip = p_flip
    
    def __call__(self, features, label, weight=None):
        # Random horizontal flip
        if np.random.random() < self.p_flip:
            features = torch.flip(features, dims=[-1])
            label = torch.flip(label, dims=[-1])
            if weight is not None: # Flip weight map if present
                 weight = torch.flip(weight, dims=[-1])
        
        # Random vertical flip
        if np.random.random() < self.p_flip:
            features = torch.flip(features, dims=[-2])
            label = torch.flip(label, dims=[-2])
            if weight is not None: # Flip weight map if present
                 weight = torch.flip(weight, dims=[-2])
            
        # Return based on whether weight map was provided
        # If a scalar weight was passed, it shouldn't be flipped
        if isinstance(weight, torch.Tensor) and weight.ndim > 0: # Check if it's a map
            return features, label, weight 
        else: # Assume scalar weight or None
            return features, label # Return only features and label

class PhenologyTileDataset(Dataset):
    """
    Dataset for loading phenology TIFF tiles, supporting selected features 
    or all features for a specific index. Handles unscaling, phase transformation,
    and normalization.
    """
    def __init__(self, 
                 tile_paths: List[str],
                 index: Optional[str] = None, # e.g., 'ndvi'. If provided, selected_features is ignored.
                 selected_features: Optional[List[str]] = None, # e.g., ['ndvi_amp_h1', 'ndvi_phase_h1_cos']
                 transform: Optional[Callable] = None,
                 patch_size: int = 64,
                 random_state: int = 42,
                 global_stats: Optional[dict] = None,
                 compute_global_stats: bool = False,
                 use_sample_weights: bool = True,
                 logger: Optional[logging.Logger] = None):
        
        self.tile_paths = tile_paths
        self.index = index
        self.original_selected_features = selected_features
        self.transform = transform
        self.patch_size = patch_size
        self.random_state = random_state
        self.global_stats = global_stats
        self.use_sample_weights = use_sample_weights
        self.phenology_band = 25 # Defined in constants?
        self.logger = logger or logging.getLogger(__name__) # Use provided logger or create one

        if index and selected_features:
            self.logger.warning(f"Both index ('{index}') and selected_features provided. Ignoring selected_features.")
            self.original_selected_features = None
        elif not index and not selected_features:
            raise ValueError("Either 'index' or 'selected_features' must be provided.")

        # Determine features to load and process
        self._configure_features()

        # Load eco-region info if needed
        self.eco_weights = None
        self.tile_to_eco = None
        if self.use_sample_weights:
            try:
                self.tile_to_eco = get_tile_eco_region_mapping()
                self._load_eco_region_weights()
            except Exception as e:
                logger.warning(f"Could not load eco-region weights: {e}. Proceeding without.")
                self.use_sample_weights = False

        # Compute global statistics if requested
        if compute_global_stats and self.global_stats is None:
            logger.info(f"Computing global statistics for {len(self.output_feature_names)} output features...")
            self.global_stats = self._compute_global_statistics()
            logger.info("Global statistics computed.")
        elif not self.global_stats:
             logger.warning("No global statistics provided or computed. Normalization will not be applied.")
        
        # Extract patches
        logger.info(f"Preparing dataset: {len(self.raw_feature_names)} raw features -> {self.num_output_features} output features.")
        self.patches = self._extract_patches()
        logger.info(f"Dataset prepared with {len(self.patches)} patches.")

    def _configure_features(self):
        """Determine the raw features to read and the output features based on index or selection."""
        if self.index:
            # Use all features for the specified index
            if self.index not in BASE_FEATURE_NAMES_PER_INDEX:
                raise ValueError(f"Invalid index '{self.index}'. Available: {list(BASE_FEATURE_NAMES_PER_INDEX.keys())}")
            self.raw_feature_names = BASE_FEATURE_NAMES_PER_INDEX[self.index]
            self.output_feature_names = OUTPUT_FEATURE_NAMES_PER_INDEX[self.index]
        else:
            # Use selected features
            self.output_feature_names = []
            temp_raw_features = {} # Use dict to handle duplicates (e.g. phase_h1 for cos and sin)
            
            for feature_name in self.original_selected_features:
                is_transform = False
                transform_type = None
                base_feature_name = feature_name
                for suffix in PHASE_TRANSFORM_SUFFIXES:
                    if feature_name.endswith(suffix):
                        base_feature_name = feature_name[:-len(suffix)]
                        is_transform = True
                        break
                
                if base_feature_name not in ALL_FEATURE_BAND_INDICES:
                    raise ValueError(f"Base feature '{base_feature_name}' (from '{feature_name}') not found.")
                
                # Store raw feature info only once
                if base_feature_name not in temp_raw_features:
                     temp_raw_features[base_feature_name] = {
                         'band': ALL_FEATURE_BAND_INDICES[base_feature_name]
                     }
                     
                # Add to output list if valid
                self.output_feature_names.append(feature_name)
            
            # Finalize raw feature list based on unique base names needed
            self.raw_feature_names = list(temp_raw_features.keys()) 

        # Common setup: Determine bands, types, phase status for raw features
        self.feature_bands = []
        self.raw_feature_types = []
        self.raw_feature_is_phase = []
        self.raw_feature_indices_map = {} # Map raw_name -> index in list

        for i, raw_name in enumerate(self.raw_feature_names):
            self.feature_bands.append(ALL_FEATURE_BAND_INDICES[raw_name])
            is_phase = any(raw_name.endswith(suffix) for suffix in PHASE_FEATURE_SUFFIXES)
            self.raw_feature_is_phase.append(is_phase)
            feature_type = None
            for suffix, ftype in FEATURE_SUFFIX_TO_TYPE.items():
                if raw_name.endswith(suffix):
                    feature_type = ftype
                    break
            if feature_type is None:
                 raise ValueError(f"Could not determine feature type for {raw_name}")
            self.raw_feature_types.append(feature_type)
            self.raw_feature_indices_map[raw_name] = i
        
        self.num_output_features = len(self.output_feature_names)
        self.logger.debug(f"Raw features to read ({len(self.raw_feature_names)}): {self.raw_feature_names}")
        self.logger.debug(f"Output features ({self.num_output_features}): {self.output_feature_names}")

    def _load_eco_region_weights(self):
        # (Identical to the one from train_unet_selected_features.py)
        try:
            weights_file = 'results/datasets/training_datasets_pixels.parquet'
            if not os.path.exists(weights_file):
                self.logger.warning(f"Weights file {weights_file} not found. Disabling eco-weights.")
                self.use_sample_weights = False
                return
            weights_df = pd.read_parquet(weights_file)
            if 'weight' in weights_df.columns and 'eco_region' in weights_df.columns:
                self.eco_weights = weights_df.groupby('eco_region')['weight'].first().to_dict()
                self.logger.info(f"Loaded weights for {len(self.eco_weights)} eco-regions. Eco-weighting ENABLED.")
            else:
                self.logger.warning("Weight or eco_region column not found. Disabling eco-weights.")
                self.use_sample_weights = False
        except Exception as e:
            self.logger.error(f"Error loading eco-region weights: {e}")
            self.use_sample_weights = False

    def _compute_global_statistics(self):
        """Compute global mean/std statistics for the OUTPUT features."""
        feature_samples = {name: [] for name in self.output_feature_names}
        rng = np.random.RandomState(self.random_state)
        max_samples_per_tile = 1000
        
        for tile_path in tqdm(self.tile_paths, desc="Computing global stats", leave=False):
            try:
                with rasterio.open(tile_path) as src:
                    feature_data = src.read(self.feature_bands)
                    valid_mask = np.all(np.isfinite(feature_data), axis=0)
                    valid_indices = np.where(valid_mask)
                    if len(valid_indices[0]) == 0: continue
                    
                    num_samples = min(max_samples_per_tile, len(valid_indices[0]))
                    sample_indices = rng.choice(len(valid_indices[0]), num_samples, replace=False)
                    sampled_coords = (valid_indices[0][sample_indices], valid_indices[1][sample_indices])
                    sampled_raw_data = feature_data[:, sampled_coords[0], sampled_coords[1]]
                    
                    processed_samples = np.zeros((self.num_output_features, num_samples), dtype=np.float64)
                    out_idx = 0
                    for i, raw_feat_name in enumerate(self.raw_feature_names):
                        raw_samples = sampled_raw_data[i]
                        feature_type = self.raw_feature_types[i]
                        index_name = raw_feat_name.split('_')[0] 
                        unscaled_samples = unscale_feature(raw_samples, feature_type, index_name)

                        if self.raw_feature_is_phase[i]:
                            cos_feat_name = f"{raw_feat_name}_cos"
                            sin_feat_name = f"{raw_feat_name}_sin"
                            if cos_feat_name in self.output_feature_names:
                                processed_samples[out_idx] = np.cos(unscaled_samples)
                                out_idx += 1
                            if sin_feat_name in self.output_feature_names:
                                processed_samples[out_idx] = np.sin(unscaled_samples)
                                out_idx += 1
                        else:
                            processed_samples[out_idx] = unscaled_samples
                            out_idx += 1
                            
                    for i, out_name in enumerate(self.output_feature_names):
                        feature_samples[out_name].extend(processed_samples[i].tolist())
            except Exception as e: self.logger.warning(f"Stat sampling error on {tile_path}: {e}")
        
        stats = {}
        for out_name, samples in feature_samples.items():
            if samples:
                samples_array = np.array(samples)
                stats[out_name] = {'mean': float(np.mean(samples_array)), 'std': float(np.std(samples_array))}
            else:
                self.logger.warning(f"No valid samples for {out_name} stats. Using mean=0, std=1.")
                stats[out_name] = {'mean': 0.0, 'std': 1.0}
        return stats

    def _extract_patches(self):
        """Extract and process patches from tiles using a structured grid approach."""
        all_patches = []
        patch_size = self.patch_size
        half_size = patch_size // 2
        stride = patch_size - 8 # 8-pixel overlap, so stride is 56

        for tile_path in tqdm(self.tile_paths, desc="Extracting patches", leave=False):
            try:
                with rasterio.open(tile_path) as src:
                    feature_data = src.read(self.feature_bands) # Shape: (n_raw, H, W)
                    phenology_data = src.read(self.phenology_band) # Shape: (H, W)
                    _, height, width = feature_data.shape

                    if height < patch_size or width < patch_size:
                        self.logger.debug(f"Skipping tile {tile_path}: dimensions ({height}x{width}) smaller than patch size ({patch_size}).")
                        continue # Tile too small for even one patch

                    tile_id = Path(tile_path).stem.split('_')[1]
                    eco_weight = 1.0
                    if self.use_sample_weights and self.tile_to_eco and self.eco_weights:
                        eco_region = self.tile_to_eco.get(tile_id)
                        if eco_region: eco_weight = self.eco_weights.get(eco_region, 1.0)

                    # Calculate grid dimensions based on stride
                    nx = math.floor((width - patch_size) / stride) + 1
                    ny = math.floor((height - patch_size) / stride) + 1

                    patch_centers = set() # Use set to avoid duplicates if stride is small

                    # Generate primary grid centers
                    for i in range(ny):
                        for j in range(nx):
                            center_y = half_size + i * stride
                            center_x = half_size + j * stride
                            # Ensure center allows full patch extraction within bounds
                            if center_y < height - half_size and center_x < width - half_size:
                                patch_centers.add((center_y, center_x))

                    # Generate intersection centers (centers between primary grid points)
                    for i in range(ny - 1):
                        for j in range(nx - 1):
                            center_y = half_size + i * stride + stride / 2
                            center_x = half_size + j * stride + stride / 2
                            int_center_y = int(round(center_y))
                            int_center_x = int(round(center_x))
                            # Ensure center allows full patch extraction within bounds
                            if int_center_y >= half_size and int_center_y < height - half_size and \
                               int_center_x >= half_size and int_center_x < width - half_size:
                                patch_centers.add((int_center_y, int_center_x))

                    patch_count = 0
                    # Iterate through the calculated structured grid centers
                    for y, x in patch_centers:
                        # Coordinates are already checked to allow patch extraction, center is int
                        y_start, y_end = y - half_size, y + half_size
                        x_start, x_end = x - half_size, x + half_size

                        # Ensure indices are integers for slicing (should be, but safety check)
                        y_start, y_end = int(y_start), int(y_end)
                        x_start, x_end = int(x_start), int(x_end)

                        feature_patch_raw = feature_data[:, y_start:y_end, x_start:x_end]
                        label_patch = phenology_data[y_start:y_end, x_start:x_end]

                        # Validation: Check for valid labels and finite features
                        if not (np.any(label_patch > 0) and np.all(np.isfinite(feature_patch_raw))):
                            continue

                        # Process features (unscale, transform, normalize) - This logic remains the same
                        processed_features = np.zeros((self.num_output_features, self.patch_size, self.patch_size), dtype=np.float32)
                        out_idx = 0
                        for i, raw_feat_name in enumerate(self.raw_feature_names):
                            current_feature_data = feature_patch_raw[i]
                            feature_type = self.raw_feature_types[i]
                            index_name = raw_feat_name.split('_')[0]
                            unscaled_patch_data = unscale_feature(current_feature_data, feature_type, index_name)

                            if self.raw_feature_is_phase[i]:
                                cos_feat_name = f"{raw_feat_name}_cos"
                                sin_feat_name = f"{raw_feat_name}_sin"
                                if cos_feat_name in self.output_feature_names:
                                    feat_cos = np.cos(unscaled_patch_data)
                                    if self.global_stats:
                                        mean_cos = self.global_stats[cos_feat_name]['mean']
                                        std_cos = self.global_stats[cos_feat_name]['std']
                                        processed_features[out_idx] = (feat_cos - mean_cos) / (std_cos + 1e-8)
                                    else: processed_features[out_idx] = feat_cos # No normalization
                                    out_idx += 1
                                if sin_feat_name in self.output_feature_names:
                                    feat_sin = np.sin(unscaled_patch_data)
                                    if self.global_stats:
                                        mean_sin = self.global_stats[sin_feat_name]['mean']
                                        std_sin = self.global_stats[sin_feat_name]['std']
                                        processed_features[out_idx] = (feat_sin - mean_sin) / (std_sin + 1e-8)
                                    else: processed_features[out_idx] = feat_sin # No normalization
                                    out_idx += 1
                            else:
                                # Need to determine the correct output feature name
                                # Assumption: Non-phase raw features map directly to an output feature
                                # Find the corresponding output feature name
                                out_feat_name = None
                                if raw_feat_name in self.output_feature_names:
                                     out_feat_name = raw_feat_name
                                else:
                                    # This case should ideally not happen if configure_features is correct
                                    # Try to find the name based on out_idx matching
                                    if out_idx < len(self.output_feature_names):
                                         out_feat_name = self.output_feature_names[out_idx]
                                    else:
                                         self.logger.error(f"Mismatch finding output feature name for raw: {raw_feat_name} at out_idx {out_idx}")
                                         # Fallback or raise error? For now, skip normalization if name not found
                                         processed_features[out_idx] = unscaled_patch_data
                                         out_idx += 1
                                         continue # Skip normalization if name lookup failed

                                if out_feat_name and self.global_stats:
                                    mean_val = self.global_stats[out_feat_name]['mean']
                                    std_val = self.global_stats[out_feat_name]['std']
                                    processed_features[out_idx] = (unscaled_patch_data - mean_val) / (std_val + 1e-8)
                                elif out_feat_name: # If stats not available but name found
                                    processed_features[out_idx] = unscaled_patch_data # No normalization
                                # Increment out_idx only if an output feature was processed
                                if out_feat_name:
                                     out_idx += 1
                                
                        # Append the processed patch
                        all_patches.append({
                            'features': processed_features,
                            'label': label_patch,
                            'tile_id': tile_id,
                            'weight': eco_weight
                        })
                        patch_count += 1
            except Exception as e:
                 # Log the error and continue with the next tile
                 self.logger.error(f"Error processing tile {tile_path}: {e}", exc_info=True) # Add exc_info for traceback
        return all_patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        features = torch.from_numpy(patch['features']).float()
        label_np = patch['label'].astype(np.int64)
        label = torch.from_numpy(label_np).long()
        weight_value = patch.get('weight', 1.0)
        weight = torch.tensor(weight_value, dtype=torch.float32)
        
        if self.transform:
            # Apply spatial transforms to features and labels
            # Assume transform does NOT handle scalar weight
            features, label = self.transform(features, label, weight=None)
        
        return features, label, weight # Return scalar weight


# --- Loss Function (Common) ---
def scaled_masked_loss(outputs, targets, weight=None, sample_weights=None, weight_scale=1.0):
    """
    Custom loss: masked cross-entropy with optional class weights and sample weights.
    Assumes targets are 0=NoData, 1=Deciduous, 2=Evergreen.
    Assumes class weights are for [Deciduous, Evergreen].
    Assumes sample_weights are per-sample (shape [B]).
    """
    valid_mask = (targets > 0) # Mask for Deciduous (1) and Evergreen (2)
    if not torch.any(valid_mask):
        loss = torch.tensor(0.0, device=outputs.device, dtype=torch.float32)
        return loss.requires_grad_() if outputs.requires_grad else loss

    outputs_masked = outputs.permute(0, 2, 3, 1)[valid_mask] # (N_valid, N_CLASSES=3)
    targets_masked = targets[valid_mask] # (N_valid,), values are 1 or 2
    targets_masked_shifted = targets_masked - 1 # Shift to 0 (Deciduous), 1 (Evergreen)
    outputs_masked_valid_classes = outputs_masked[:, 1:] # Shape: (N_valid, 2) - Logits for D, E

    pixel_losses = F.cross_entropy(outputs_masked_valid_classes, targets_masked_shifted, weight=weight, reduction='none')

    if sample_weights is not None and weight_scale > 0.0:
        sample_weights_expanded = sample_weights[:, None, None].expand_as(targets)
        pixel_sample_weights = sample_weights_expanded[valid_mask]
        scaled_pixel_sample_weights = 1.0 + (pixel_sample_weights - 1.0) * weight_scale
        scaled_pixel_sample_weights = scaled_pixel_sample_weights.clamp(min=0)
        weighted_loss = (pixel_losses * scaled_pixel_sample_weights).sum() / scaled_pixel_sample_weights.sum().clamp(min=1e-8)
    else:
        weighted_loss = pixel_losses.mean()
    
    return weighted_loss

# --- Evaluation Function (Common) ---
def evaluate_unet_model(model, dataloader, device, criterion=None, weight_scale=1.0, class_weights=None, logger=None):
    """Evaluates U-Net model, returns metrics dict for classes 1 & 2."""

    logger = logger or logging.getLogger(__name__)

    model.eval()
    running_loss = 0.0
    all_true_labels, all_pred_labels = [], []
    
    if len(dataloader.dataset) == 0:
        logger.warning("Evaluation dataloader is empty.")
        return {'loss': 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'confusion_matrix': np.zeros((2, 2), dtype=int)}
    
    with torch.no_grad():
        for inputs, labels, sample_weights in dataloader:
            inputs, labels, sample_weights = inputs.to(device), labels.to(device), sample_weights.to(device)
            outputs = model(inputs)
            if criterion: loss = criterion(outputs, labels, sample_weights); running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            valid_mask = (labels > 0)
            all_true_labels.extend(labels[valid_mask].cpu().numpy())
            all_pred_labels.extend(preds[valid_mask].cpu().numpy())

    y_true, y_pred = np.array(all_true_labels), np.array(all_pred_labels)
    if len(y_true) == 0:
        logger.warning("No valid labels (1 or 2) found in evaluation data.")
        return {'loss': running_loss / len(dataloader) if dataloader else 0.0, 'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0, 'confusion_matrix': np.zeros((2, 2), dtype=int)}

    labels_for_metrics = [1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels_for_metrics)
    f1 = f1_score(y_true, y_pred, labels=labels_for_metrics, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, labels=labels_for_metrics, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels_for_metrics, average='weighted', zero_division=0)
    accuracy = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0
    
    return {'loss': running_loss / len(dataloader) if dataloader else 0.0,
            'accuracy': accuracy, 'f1_score': f1, 'precision': precision, 'recall': recall,
            'confusion_matrix': cm}

# --- Visualization Functions (Common) ---
def visualize_unet_predictions(model, dataloader, device, output_feature_names, output_dir, fold_tag, num_samples=5, logger=None):
    """Visualizes model predictions for a few validation samples."""

    logger = logger or logging.getLogger(__name__)

    model.eval()
    samples = []
    if len(dataloader.dataset) == 0: return
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            if len(samples) >= num_samples: break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(min(inputs.size(0), num_samples - len(samples))):
                 # Use index 0 for first feature visualization, regardless of name
                 samples.append({'input': inputs[i, 0].cpu().numpy(), 'true': labels[i].cpu().numpy(), 'pred': preds[i].cpu().numpy()})
            if len(samples) >= num_samples: break
    
    if not samples: return
    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)), squeeze=False)
    first_feature_name = output_feature_names[0] if output_feature_names else "Feature 1"
    
    for i, sample in enumerate(samples):
        ax=axes[i,0]; im=ax.imshow(sample['input'], cmap='viridis'); ax.set_title(f'Input ({first_feature_name})'); fig.colorbar(im, ax=ax)
        ax=axes[i,1]; im=ax.imshow(sample['true'], cmap='tab10', vmin=0, vmax=2); ax.set_title('True Labels (0:ND, 1:Dec, 2:Evg)'); fig.colorbar(im, ax=ax, ticks=[0,1,2])
        ax=axes[i,2]; im=ax.imshow(sample['pred'], cmap='tab10', vmin=0, vmax=2); ax.set_title('Predicted Labels'); fig.colorbar(im, ax=ax, ticks=[0,1,2])
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f"phenology_unet_{fold_tag}_predictions.png")
    try: plt.savefig(vis_path); logger.info(f"Saved prediction visualization to {vis_path}")
    except Exception as e: logger.error(f"Failed to save prediction plot: {e}")
    plt.close(fig)

def prepare_unet_images_for_tensorboard(batch_inputs, batch_labels, batch_preds, output_feature_names, tag_prefix):
    """Prepares images for TensorBoard logging."""
    inputs_np, labels_np, preds_np = batch_inputs.cpu().numpy(), batch_labels.cpu().numpy(), batch_preds.cpu().numpy()
    batch_size = inputs_np.shape[0]
    cmap = np.array([[0,0,0],[0,255,0],[0,0,255]], dtype=np.uint8)
    vis_images = {}
    num_samples_to_log = min(batch_size, 4)
    first_feature_name = output_feature_names[0] if output_feature_names else "Feature_1"

    for i in range(num_samples_to_log):
        input_ch1 = inputs_np[i, 0]
        input_ch1_norm = (input_ch1 - np.nanmin(input_ch1)) / (np.nanmax(input_ch1) - np.nanmin(input_ch1) + 1e-8)
        input_rgb = (plt.cm.viridis(input_ch1_norm)[:, :, :3] * 255).astype(np.uint8)
        vis_images[f'{tag_prefix}/Input_{first_feature_name}/{i}'] = torch.from_numpy(input_rgb).permute(2, 0, 1)

        label_rgb = cmap[labels_np[i]]; vis_images[f'{tag_prefix}/Ground_Truth/{i}'] = torch.from_numpy(label_rgb).permute(2, 0, 1)
        pred_rgb = cmap[preds_np[i]]; vis_images[f'{tag_prefix}/Prediction/{i}'] = torch.from_numpy(pred_rgb).permute(2, 0, 1)
    return vis_images

def count_rf_parameters(rf_model):
    """
    Count the number of parameters in a RandomForest model.
    (Copied from train_rf_selected_features.py)
    """
    try:
        # Handle cases where the model might not be fitted or has no estimators
        if not hasattr(rf_model, 'estimators_') or not rf_model.estimators_:
            return {
                'n_estimators': 0,
                'n_nodes': 0,
                'params_per_node': 0,
                'total_parameters': 0
            }
            
        n_estimators = len(rf_model.estimators_)
        total_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
        n_classes = rf_model.n_classes_
        
        # Simplified parameter count: nodes store split criteria (feature, threshold) and value per class at leaves.
        # A rough estimate might consider 2 values per internal node + n_classes per leaf node.
        # A simpler approach used in the original script:
        params_per_node = 2 + n_classes # Assuming 2 params for split + n_classes for output values
        total_parameters = total_nodes * params_per_node
        
        return {
            'n_estimators': n_estimators,
            'n_nodes': total_nodes,
            'params_per_node': params_per_node,
            'total_parameters': total_parameters
        }
    except AttributeError as e:
        # Log error if attributes like n_classes_ or estimators_ are missing
        logger = logging.getLogger(__name__) # Get logger instance
        logger.warning(f"Could not count RF parameters, model might not be fitted correctly: {e}")
        return {
            'n_estimators': 'N/A',
            'n_nodes': 'N/A',
            'params_per_node': 'N/A',
            'total_parameters': 'N/A'
        }
