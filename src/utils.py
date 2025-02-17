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

from src.constants import PHASE_RANGE, AMP_RANGE, OFFSET_RANGE

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

def scale_amplitude(amp: np.ndarray) -> np.ndarray:
    """Scale amplitude array to [0..65535] based on AMP_RANGE = (0,2)."""
    return scale_array_to_uint16(amp, AMP_RANGE[0], AMP_RANGE[1])

def scale_offset(offset: np.ndarray) -> np.ndarray:
    """Scale offset array to [0..65535]."""
    return scale_array_to_uint16(offset, OFFSET_RANGE[0], OFFSET_RANGE[1])

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

    # NDVI
    ndvi = (b8 - b4) / np.maximum(b8 + b4, eps)

    # EVI (some standard constants)
    #   EVI = 2.5 * (b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1)
    evi_denom = (b8 + 6.0*b4 - 7.5*b2 + 1.0)
    evi = 2.5 * (b8 - b4) / np.maximum(evi_denom, eps)

    # NBR
    nbr = (b8 - b12) / np.maximum(b8 + b12, eps)

    # CRSWIR as example
    #   crswir = b11 / (b12 + eps)
    crswir = b11 / np.maximum(b12, eps)

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

