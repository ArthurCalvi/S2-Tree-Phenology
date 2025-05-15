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

# Mapping from detailed eco-region names to general categories
MAPPING_SER_ECO_REGIONS = {
    'Côtes_et_plateaux_de_la_Manche': 'Centre Nord semi-océanique',
    'Ardenne_primaire': 'Grand Est semi-continental',
    'Préalpes_du_Nord': 'Alpes',
    'Garrigues': 'Méditerranée',
    'Massif_vosgien_central': 'Vosges',
    'Premier_plateau_du_Jura': 'Jura',
    'Piémont_pyrénéen': 'Pyrénées',
    'Terres_rouges': 'Sud-Ouest océanique',
    'Corse_occidentale': 'Corse',
    "Châtaigneraie_du_Centre_et_de_l'Ouest": 'Massif central',
    'Ouest-Bretagne_et_Nord-Cotentin': 'Grand Ouest cristallin et océanique',
    'Total': 'Total'
}

MAPPING_ECO_REGIONS_FR_EN = {
    "Grand Ouest cristallin et océanique": "Greater Crystalline and Oceanic West",
    "Centre Nord semi-océanique": "Semi-Oceanic North Center",
    "Grand Est semi-continental": "Greater Semi-Continental East",
    "Vosges": "Vosges",
    "Jura": "Jura",
    "Sud-Ouest océanique": "Oceanic Southwest",
    "Massif central": "Central Massif",
    "Alpes": "Alps",
    "Pyrénées": "Pyrenees",
    "Méditerranée": "Mediterranean",
    "Corse": "Corsica"
}

# Forest cover ratio by eco-region (percentage of land covered by forest)
# Based on data from GRECO-Tauxdeboisementmoyen-Dpartementsreprsentatifs-Caractristiquescologiquesdominantes.csv
FOREST_COVER_RATIO_BY_REGION = {
    "Greater Crystalline and Oceanic West": 0.20,  # Plaines Atlantiques: 15-25%
    "Semi-Oceanic North Center": 0.30,  # Bassin Parisien: 25-35%
    "Greater Semi-Continental East": 0.75,  # Ardenne Primaire: 70-80%
    "Vosges": 0.75,  # Massif Vosgien Central: 70-80%
    "Jura": 0.60,  # Jura: 55-65%
    "Oceanic Southwest": 0.75,  # Landes de Gascogne: 70-80%
    "Central Massif": 0.49,  # Massif Central: 49%
    "Alps": 0.45,  # Alpes Internes: 40-50%
    "Pyrenees": 0.65,  # Pyrénées: 60-70%
    "Mediterranean": 0.75,  # Cévennes/Alpes Externes du Sud: 70-80%
    "Corsica": 0.75  # Assuming similar to Mediterranean regions
}

# Effective forest area by eco-region in square kilometers
# Calculated by multiplying the total area of each eco-region by its forest cover ratio
EFFECTIVE_FOREST_AREA_BY_REGION = {
    "Alps": 13007.42,  # Total area: 28905.38 km² * forest ratio: 0.45
    "Central Massif": 38632.55,  # Total area: 78841.95 km² * forest ratio: 0.49
    "Corsica": 6569.81,  # Total area: 8759.75 km² * forest ratio: 0.75
    "Greater Crystalline and Oceanic West": 11947.86,  # Total area: 59739.31 km² * forest ratio: 0.20
    "Greater Semi-Continental East": 52825.64,  # Total area: 70434.18 km² * forest ratio: 0.75
    "Jura": 5788.60,  # Total area: 9647.66 km² * forest ratio: 0.60
    "Mediterranean": 26838.60,  # Total area: 35784.80 km² * forest ratio: 0.75
    "Oceanic Southwest": 61409.33,  # Total area: 81879.11 km² * forest ratio: 0.75
    "Pyrenees": 9995.91,  # Total area: 15378.33 km² * forest ratio: 0.65
    "Semi-Oceanic North Center": 44940.11,  # Total area: 149800.37 km² * forest ratio: 0.30
    "Vosges": 7002.68,  # Total area: 9336.91 km² * forest ratio: 0.75
}

# List of all eco-regions (English names, consistent with dataset)
ALL_ECO_REGIONS = list(EFFECTIVE_FOREST_AREA_BY_REGION.keys())

# Genus mapping based on provided categorical_mappings.json
GENUS_MAPPING = {
    1: "alder",
    2: "ash",
    3: "beech",
    4: "birch",
    5: "chestnut",
    6: "douglas",
    7: "fir",
    8: "hornbeam",
    9: "larch",
    10: "locusts",
    11: "oak",
    12: "pine",
    13: "poplar",
    14: "spruce"
}

# === U-Net Specific Constants ===

# Define the phenology mapping (label -> name)
PHENOLOGY_MAPPING = {0: 'No Data', 1: 'Deciduous', 2: 'Evergreen'}

# Define mapping from feature names to 1-based band indices in the TIFF file
# This assumes the band order described previously
ALL_FEATURE_BAND_INDICES = {
    'ndvi_amplitude_h1': 1, 'ndvi_amplitude_h2': 2, 'ndvi_phase_h1': 3, 'ndvi_phase_h2': 4, 'ndvi_offset': 5, 'ndvi_var_residual': 6,
    'evi_amplitude_h1': 7, 'evi_amplitude_h2': 8, 'evi_phase_h1': 9, 'evi_phase_h2': 10, 'evi_offset': 11, 'evi_var_residual': 12,
    'nbr_amplitude_h1': 13, 'nbr_amplitude_h2': 14, 'nbr_phase_h1': 15, 'nbr_phase_h2': 16, 'nbr_offset': 17, 'nbr_var_residual': 18,
    'crswir_amplitude_h1': 19, 'crswir_amplitude_h2': 20, 'crswir_phase_h1': 21, 'crswir_phase_h2': 22, 'crswir_offset': 23, 'crswir_var_residual': 24,
}
PHENOLOGY_BAND = 25 # Phenology label is the 25th band

# Map feature suffixes to unscaling types
FEATURE_SUFFIX_TO_TYPE = {
    '_amplitude_h1': 'amplitude',
    '_amplitude_h2': 'amplitude',
    '_phase_h1': 'phase',
    '_phase_h2': 'phase',
    '_offset': 'offset',
    '_var_residual': 'variance'
}

# Base feature names for each index (before cos/sin transform)
BASE_FEATURE_NAMES_PER_INDEX = {
    'ndvi': ['ndvi_amplitude_h1', 'ndvi_amplitude_h2', 'ndvi_phase_h1', 'ndvi_phase_h2', 'ndvi_offset', 'ndvi_var_residual'],
    'evi': ['evi_amplitude_h1', 'evi_amplitude_h2', 'evi_phase_h1', 'evi_phase_h2', 'evi_offset', 'evi_var_residual'],
    'nbr': ['nbr_amplitude_h1', 'nbr_amplitude_h2', 'nbr_phase_h1', 'nbr_phase_h2', 'nbr_offset', 'nbr_var_residual'],
    'crswir': ['crswir_amplitude_h1', 'crswir_amplitude_h2', 'crswir_phase_h1', 'crswir_phase_h2', 'crswir_offset', 'crswir_var_residual']
}

# Output feature names (after cos/sin transform)
OUTPUT_FEATURE_NAMES_PER_INDEX = {
    'ndvi': ['ndvi_amplitude_h1', 'ndvi_amplitude_h2', 'ndvi_phase_h1_cos', 'ndvi_phase_h1_sin', 'ndvi_phase_h2_cos', 'ndvi_phase_h2_sin', 'ndvi_offset', 'ndvi_var_residual'],
    'evi': ['evi_amplitude_h1', 'evi_amplitude_h2', 'evi_phase_h1_cos', 'evi_phase_h1_sin', 'evi_phase_h2_cos', 'evi_phase_h2_sin', 'evi_offset', 'evi_var_residual'],
    'nbr': ['nbr_amplitude_h1', 'nbr_amplitude_h2', 'nbr_phase_h1_cos', 'nbr_phase_h1_sin', 'nbr_phase_h2_cos', 'nbr_phase_h2_sin', 'nbr_offset', 'nbr_var_residual'],
    'crswir': ['crswir_amplitude_h1', 'crswir_amplitude_h2', 'crswir_phase_h1_cos', 'crswir_phase_h1_sin', 'crswir_phase_h2_cos', 'crswir_phase_h2_sin', 'crswir_offset', 'crswir_var_residual']
}

# Identify phase features for transformation
PHASE_FEATURE_SUFFIXES = ['_phase_h1', '_phase_h2']
PHASE_TRANSFORM_SUFFIXES = ['_cos', '_sin']