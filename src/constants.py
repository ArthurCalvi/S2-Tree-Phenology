"""
constants.py
-------------
Holds constants for scaling, band names, etc.
"""

import numpy as np

# Define valid ranges for amplitude, offset, and phase
PHASE_RANGE = (-np.pi, np.pi)          # We'll shift/scale to [0..65535]
AMP_RANGE   = (0, 2.0)                 # Example guess: clamp amplitude to [0..2]
OFFSET_RANGE= (0, 2.0)                 # Example guess: clamp offset to [0..2]

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
