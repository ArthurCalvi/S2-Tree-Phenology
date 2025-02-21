import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

@dataclass
class BandData:
    """
    Data class to hold Sentinel-2 time-series band data + optional DEM + raw msk_cldprb.
    
    Shapes:
       b2, b4, b8, b11, b12: (T, H, W)
       msk_cldprb: (T, H, W)
       dem: (H, W) or None
       dates: list of length T
    """
    b2: np.ndarray
    b4: np.ndarray
    b8: np.ndarray
    b11: np.ndarray
    b12: np.ndarray
    msk_cldprb: np.ndarray  # raw cloud-prob band, range ~0..100 after scaling
    dates: List[datetime]

    dem: Optional[np.ndarray] = None  # shape (H,W) if provided

    def __post_init__(self):
        # All 5 S2 bands must share the same shape
        shapes = {
            'b2': self.b2.shape,
            'b4': self.b4.shape,
            'b8': self.b8.shape,
            'b11': self.b11.shape,
            'b12': self.b12.shape,
            'msk_cldprb': self.msk_cldprb.shape
        }
        shape_set = set(shapes.values())
        if len(shape_set) != 1:
            msg = f"All S2 bands + msk_cldprb must share shape (T,H,W). Got: {shapes}"
            raise ValueError(msg)

        # Check time dimension
        (T, H, W) = self.b2.shape
        if T != len(self.dates):
            raise ValueError(
                f"Time dimension in band arrays = {T}, but got {len(self.dates)} dates."
            )

        # If DEM is provided, must match (H,W)
        if self.dem is not None:
            if self.dem.shape != (H, W):
                raise ValueError(f"DEM shape {self.dem.shape} must match (H,W)={(H,W)}")


# Define valid ranges for amplitude, offset, and phase
PHASE_RANGE = (-np.pi, np.pi)          # We'll shift/scale to [0..65535]

TARGET_AMP_RANGE = {
    'ndvi': (0.0, 1.0),
    'evi': (0.0, 1.0),
    'nbr': (0.0, 1.0),
    'crswir': (0.0, 2.5)
}
TARGET_OFFSET_RANGE = {
    'ndvi': (0.0, 1.0),
    'evi': (0.0, 1.0),
    'nbr': (0.0, 1.0),
    'crswir': (0.0, 2.5)
}

# Number of harmonics: typically set from command-line
# but you can define a default here if you want
DEFAULT_NUM_HARMONICS = 2
DEFAULT_MAX_ITER = 5

# Default band names in the input mosaic
BAND_NAMES = ["B2", "B4", "B8", "B11", "B12", "MSK_CLDPRB"]

# For convenience, define which index computations we want
AVAILABLE_INDICES = ["ndvi", "evi", "nbr", "crswir"]


SCALE_FACTOR_BANDS = 65535 / 3000  # approximate

band_preprocess = {
    "B2":         lambda arr_uint16: (arr_uint16.astype(np.float32) / SCALE_FACTOR_BANDS),
    "B4":         lambda arr_uint16: (arr_uint16.astype(np.float32) / SCALE_FACTOR_BANDS),
    "B8":         lambda arr_uint16: (arr_uint16.astype(np.float32) / SCALE_FACTOR_BANDS),
    "B11":        lambda arr_uint16: (arr_uint16.astype(np.float32) / SCALE_FACTOR_BANDS),
    "B12":        lambda arr_uint16: (arr_uint16.astype(np.float32) / SCALE_FACTOR_BANDS),
    "MSK_CLDPRB": lambda arr_uint16: (arr_uint16.astype(np.float32) / SCALE_FACTOR_BANDS),
}
