#!/usr/bin/env python3
"""Download ancillary DEM / climate / soil features aggregated per training tile using Google Earth Engine."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

import ee
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv
from download_embeddings import init_ee_from_env
from shapely.geometry import mapping


def load_tiles(parquet_path: Path) -> tuple[ee.FeatureCollection, pd.DataFrame]:
    gdf = gpd.read_parquet(parquet_path)
    if gdf.crs is None:
        raise ValueError("Tiles parquet must have a CRS")
    gdf = gdf.to_crs(4326)
    rows = []
    features = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        tile_id = row.tile_id if 'tile_id' in gdf.columns else idx
        geojson = mapping(geom)
        feature = ee.Feature(ee.Geometry(geojson), {'tile_id': int(tile_id)})
        features.append(feature)
        rows.append({'tile_id': int(tile_id)})
    fc = ee.FeatureCollection(features)
    return fc, pd.DataFrame(rows)


def fc_to_dataframe(fc: ee.FeatureCollection, select_props: List[str] | None = None) -> pd.DataFrame:
    records = fc.getInfo()['features']
    data = []
    for feat in records:
        props = feat['properties']
        if select_props is not None:
            props = {k: props.get(k) for k in select_props}
        data.append(props)
    df = pd.DataFrame(data)
    drop_cols = [c for c in df.columns if c.startswith('system:')]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def dem_features(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    dem = ee.Image('CGIAR/SRTM90_V4').rename('elevation')
    terrain = ee.Terrain.products(dem)
    slope = terrain.select('slope').rename('slope')
    aspect = terrain.select('aspect').rename('aspect')
    image = dem.addBands(slope).addBands(aspect)
    reducer = (ee.Reducer.mean()
               .combine(ee.Reducer.minMax(), sharedInputs=True)
               .combine(ee.Reducer.stdDev(), sharedInputs=True)
               .combine(ee.Reducer.percentile([90]), sharedInputs=True))
    return image.reduceRegions(collection=fc, reducer=reducer, scale=100)


def soil_texture_features(fc: ee.FeatureCollection) -> ee.FeatureCollection:
    soil = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02').select(0).rename('soil_texture_class')
    reducer = ee.Reducer.mode()
    return soil.reduceRegions(collection=fc, reducer=reducer, scale=250)


def soilgrids_features(fc: ee.FeatureCollection, depth: str = '0-5cm') -> ee.FeatureCollection:
    def sg_band(asset: str, prefix: str, rename: str) -> ee.Image:
        image = ee.Image(f'projects/soilgrids-isric/{asset}')
        band = f"{prefix}_{depth}_mean"
        return image.select(band).rename(rename)

    clay = sg_band('clay_mean', 'clay', 'soil_clay_mean_gkg')
    sand = sg_band('sand_mean', 'sand', 'soil_sand_mean_gkg')
    silt = sg_band('silt_mean', 'silt', 'soil_silt_mean_gkg')
    bulk_density = sg_band('bdod_mean', 'bdod', 'soil_bulk_density_mean_kgm3')
    organic_carbon = sg_band('soc_mean', 'soc', 'soil_organic_carbon_mean_gkg')

    image = clay.addBands(sand).addBands(silt).addBands(bulk_density).addBands(organic_carbon)
    reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
    return image.reduceRegions(collection=fc, reducer=reducer, scale=250)


def era5_features(fc: ee.FeatureCollection, year: int) -> ee.FeatureCollection:
    era = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR').filter(ee.Filter.calendarRange(year, year, 'year'))

    temp_mean = era.select('temperature_2m').mean().subtract(273.15).rename('era5_temp_mean_degC')
    temp_min = era.select('temperature_2m_min').min().subtract(273.15).rename('era5_temp_min_degC')
    temp_max = era.select('temperature_2m_max').max().subtract(273.15).rename('era5_temp_max_degC')

    dew_mean = era.select('dewpoint_temperature_2m').mean().subtract(273.15).rename('era5_dewpoint_mean_degC')
    dew_min = era.select('dewpoint_temperature_2m_min').min().subtract(273.15).rename('era5_dewpoint_min_degC')
    dew_max = era.select('dewpoint_temperature_2m_max').max().subtract(273.15).rename('era5_dewpoint_max_degC')

    precip_sum = era.select('total_precipitation_sum').sum().multiply(1000).rename('era5_precip_total_mm')
    precip_min = era.select('total_precipitation_min').min().multiply(1000).rename('era5_precip_min_mm')
    precip_max = era.select('total_precipitation_max').max().multiply(1000).rename('era5_precip_max_mm')

    surface_mean = era.select('surface_pressure').mean().divide(100).rename('era5_surface_pressure_mean_hPa')
    surface_min = era.select('surface_pressure_min').min().divide(100).rename('era5_surface_pressure_min_hPa')
    surface_max = era.select('surface_pressure_max').max().divide(100).rename('era5_surface_pressure_max_hPa')

    soil_moist = era.select('volumetric_soil_water_layer_1').mean().rename('era5_soil_moisture_mean')

    image = (temp_mean.addBands(temp_min).addBands(temp_max)
             .addBands(dew_mean).addBands(dew_min).addBands(dew_max)
             .addBands(precip_sum).addBands(precip_min).addBands(precip_max)
             .addBands(surface_mean).addBands(surface_min).addBands(surface_max)
             .addBands(soil_moist))

    reducer = ee.Reducer.mean()
    return image.reduceRegions(collection=fc, reducer=reducer, scale=1000)


def rename_dem_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'elevation_mean' in df.columns:
        return df
    mapping = {
        'mean': 'elevation_mean',
        'mean_1': 'slope_mean',
        'mean_2': 'aspect_mean',
        'min': 'elevation_min',
        'min_1': 'slope_min',
        'min_2': 'aspect_min',
        'max': 'elevation_max',
        'max_1': 'slope_max',
        'max_2': 'aspect_max',
        'stdDev': 'elevation_std',
        'stdDev_1': 'slope_std',
        'stdDev_2': 'aspect_std',
        'p90': 'elevation_p90',
        'p90_1': 'slope_p90',
        'p90_2': 'aspect_p90'
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def rename_era_columns(df: pd.DataFrame) -> pd.DataFrame:
    if 'era5_temp_mean_degC' in df.columns:
        return df
    mapping = {
        'era5_temp_mean_degC_mean': 'era5_temp_mean_degC',
        'era5_temp_min_degC_mean': 'era5_temp_min_degC',
        'era5_temp_max_degC_mean': 'era5_temp_max_degC',
        'era5_dewpoint_mean_degC_mean': 'era5_dewpoint_mean_degC',
        'era5_dewpoint_min_degC_mean': 'era5_dewpoint_min_degC',
        'era5_dewpoint_max_degC_mean': 'era5_dewpoint_max_degC',
        'era5_precip_total_mm_mean': 'era5_precip_total_mm',
        'era5_precip_min_mm_mean': 'era5_precip_min_mm',
        'era5_precip_max_mm_mean': 'era5_precip_max_mm',
        'era5_surface_pressure_mean_hPa_mean': 'era5_surface_pressure_mean_hPa',
        'era5_surface_pressure_min_hPa_mean': 'era5_surface_pressure_min_hPa',
        'era5_surface_pressure_max_hPa_mean': 'era5_surface_pressure_max_hPa',
        'era5_soil_moisture_mean_mean': 'era5_soil_moisture_mean'
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def rename_soilgrids_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'soil_clay_mean_gkg_mean': 'soil_clay_mean_gkg',
        'soil_clay_mean_gkg_stdDev': 'soil_clay_std_gkg',
        'soil_sand_mean_gkg_mean': 'soil_sand_mean_gkg',
        'soil_sand_mean_gkg_stdDev': 'soil_sand_std_gkg',
        'soil_silt_mean_gkg_mean': 'soil_silt_mean_gkg',
        'soil_silt_mean_gkg_stdDev': 'soil_silt_std_gkg',
        'soil_bulk_density_mean_kgm3_mean': 'soil_bulk_density_mean_kgm3',
        'soil_bulk_density_mean_kgm3_stdDev': 'soil_bulk_density_std_kgm3',
        'soil_organic_carbon_mean_gkg_mean': 'soil_organic_carbon_mean_gkg',
        'soil_organic_carbon_mean_gkg_stdDev': 'soil_organic_carbon_std_gkg'
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Download tile-level ancillary features from Earth Engine")
    parser.add_argument('--tiles-parquet', default='results/datasets/tiles_2_5_km_final.parquet')
    parser.add_argument('--years', default='2023', help='Comma-separated list of ERA5 years to aggregate (e.g., 2018,2020,2022,2023)')
    parser.add_argument('--output', default='results/analysis_context/tile_context_features.parquet')
    parser.add_argument('--project', default=os.getenv('EE_PROJECT'), help='Earth Engine project ID (defaults to $EE_PROJECT)')
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    tiles_parquet = Path(args.tiles_parquet)
    output_path = Path(args.output)

    init_ee_from_env(project=args.project)
    fc, tile_df = load_tiles(tiles_parquet)

    dem_fc = dem_features(fc)
    soil_fc = soil_texture_features(fc)
    soilgrids_fc = soilgrids_features(fc)
    years = sorted({int(y) for y in str(args.years).split(',')})

    dem_df = rename_dem_columns(fc_to_dataframe(dem_fc))
    soil_df = fc_to_dataframe(soil_fc)
    if 'mode' in soil_df.columns and 'soil_texture_class_mode' not in soil_df.columns:
        soil_df = soil_df.rename(columns={'mode': 'soil_texture_class_mode'})
    if 'tile_id' not in soil_df.columns and 'tileid' in soil_df.columns:
        soil_df = soil_df.rename(columns={'tileid': 'tile_id'})
    soilgrids_df = rename_soilgrids_columns(fc_to_dataframe(soilgrids_fc))
    if 'tile_id' not in soilgrids_df.columns and 'tileid' in soilgrids_df.columns:
        soilgrids_df = soilgrids_df.rename(columns={'tileid': 'tile_id'})

    df = tile_df.merge(dem_df, on='tile_id', how='left')

    for year in years:
        era_fc = era5_features(fc, year)
        era_df = rename_era_columns(fc_to_dataframe(era_fc))
        suffix = f"_{year}"
        cols = [c for c in era_df.columns if c != 'tile_id']
        era_df = era_df[['tile_id'] + cols]
        era_df = era_df.rename(columns={c: f"{c}{suffix}" for c in cols})
        df = df.merge(era_df, on='tile_id', how='left')

    if 'soil_texture_class_mode' in soil_df.columns:
        df = df.merge(soil_df[['tile_id', 'soil_texture_class_mode']], on='tile_id', how='left')
    if not soilgrids_df.empty:
        df = df.merge(soilgrids_df, on='tile_id', how='left')

    numeric_cols = [c for c in df.columns if c != 'tile_id']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved ancillary tile features to {args.output}")


if __name__ == '__main__':
    main()
