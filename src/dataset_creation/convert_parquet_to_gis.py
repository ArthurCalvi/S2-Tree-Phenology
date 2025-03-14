import pandas as pd
import geopandas as gpd
import os

def convert_parquet_to_gis_formats(parquet_file, output_dir=None):
    """
    Convert a Parquet file to GeoJSON and Shapefile formats for QGIS.
    
    Parameters:
    -----------
    parquet_file : str
        Path to the input Parquet file
    output_dir : str, optional
        Directory to save output files. If None, uses the directory of the input file.
    """
    print(f"Reading Parquet file: {parquet_file}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(parquet_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(parquet_file))[0]
    
    # Read Parquet file
    try:
        # Try to read as GeoDataFrame first (if it already has geometry)
        gdf = gpd.read_parquet(parquet_file)
        has_geometry = True
    except:
        # If it fails, read as regular DataFrame
        df = pd.read_parquet(parquet_file)
        
        # Check if it has latitude and longitude columns
        if 'latitude' in df.columns and 'longitude' in df.columns:
            print("Converting latitude/longitude to geometry...")
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"  # WGS84
            )
            has_geometry = True
        elif 'lat' in df.columns and 'lon' in df.columns:
            print("Converting lat/lon to geometry...")
            gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(df.lon, df.lat),
                crs="EPSG:4326"  # WGS84
            )
            has_geometry = True
        else:
            print("No recognized coordinate columns found. Saving as regular CSV.")
            output_csv = os.path.join(output_dir, f"{base_name}.csv")
            df.to_csv(output_csv, index=False)
            print(f"Saved as CSV: {output_csv}")
            has_geometry = False
            gdf = df  # Just for code flow, will not use geometry methods
    
    # Export to GIS formats if we have geometry
    if has_geometry:
        # Save as GeoJSON
        geojson_output = os.path.join(output_dir, f"{base_name}.geojson")
        print(f"Saving as GeoJSON: {geojson_output}")
        gdf.to_file(geojson_output, driver="GeoJSON")
        
        # Save as Shapefile
        shp_output = os.path.join(output_dir, f"{base_name}.shp")
        print(f"Saving as Shapefile: {shp_output}")
        gdf.to_file(shp_output)
        
        # Save as GeoPackage (another common QGIS format)
        gpkg_output = os.path.join(output_dir, f"{base_name}.gpkg")
        print(f"Saving as GeoPackage: {gpkg_output}")
        gdf.to_file(gpkg_output, driver="GPKG")
    
    print("Conversion completed!")

if __name__ == "__main__":
    parquet_file = "results/datasets/tiles_2_5_km_final.parquet"
    convert_parquet_to_gis_formats(parquet_file) 