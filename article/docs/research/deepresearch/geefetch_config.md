# GeeFetch Configuration for S2-Tree-Phenology

## Configuration Structure

Based on `config_corsica_20231215.yaml`, the GeeFetch config follows this structure:

```yaml
data_dir: <output_directory>
s2:
  cloud_prb_threshold: <percentage>  # Pixel-level cloud probability threshold
  cloudless_portion: <percentage>     # Tile-level cloudless requirement
  selected_bands: [list of bands]     # e.g., B2, B4, B8, B11, B12, MSK_CLDPRB
satellite_default:
  aoi:
    spatial: {left, right, top, bottom, epsg}
    temporal: {start_date, end_date}
  composite_method: <METHOD>          # e.g., MEDIAN
  dtype: <data_type>                  # e.g., UInt16
  gee:
    ee_project_id: <project_id>
    max_tile_size: <size>
  resolution: <meters>                # e.g., 10
  tile_size: <pixels>                 # e.g., 10240
```

## Cloud Masking Parameters

**Method**: s2cloudless cloud probability (COPERNICUS/S2_CLOUD_PROBABILITY) + QA60 bitmask

**Thresholds (from actual config)**:
- **cloud_prb_threshold**: 75 (pixel-level - masks pixels with cloud probability > 75%)
- **cloudless_portion**: 5 (tile-level - drops scenes with CLOUDY_PIXEL_PERCENTAGE > 95% or HIGH_PROBA_CLOUDS_PERCENTAGE > 97.5%)

**Image-level prefilter formula**:
- Drop scenes where: CLOUDY_PIXEL_PERCENTAGE > (100 − cloudless_portion)
- AND: HIGH_PROBA_CLOUDS_PERCENTAGE > (50 − cloudless_portion/2)

**Additional QA**: QA60 bitmask removes opaque (bit 10) and cirrus (bit 11) clouds

## Compositing Method

**Default**: MEDIAN (specified in satellite_default.composite_method)

## Temporal Window

User-defined via `satellite_default.aoi.temporal`:
- `start_date`: e.g., '2023-12-01'
- `end_date`: e.g., '2023-12-31'

Can create monthly composites or switch to time-series mode (CompositeMethod.TIMESERIES)

## Spatial AOI

Defined in `satellite_default.aoi.spatial`:
- Bounding box: left, right, top, bottom coordinates
- Projection: EPSG code (e.g., 2154 for Lambert-93 France)

## Processing Parameters

- **resolution**: 10 meters (Sentinel-2 native resolution for visible/NIR bands)
- **tile_size**: 10240 pixels (102.4 km × 102.4 km at 10m resolution)
- **dtype**: UInt16 (16-bit unsigned integer for reflectance values)
- **max_tile_size**: 8 (GEE parameter for chunking)

## Sources

- **GeeFetch Sentinel-2 docs**: https://geefetch.readthedocs.io/en/latest/satellites/sentinel2/
- **GeeFetch API (S2)**: https://geefetch.readthedocs.io/en/latest/api/satellites/
- **GeeFetch CLI configuration**: https://geefetch.readthedocs.io/en/latest/api/cli/configuration/
- **Earth Engine S2 dataset**: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED
- **Project config example**: `ops/geefetch/configs/config_corsica_20231215.yaml`
