import geopandas as gpd
import pandas as pd
from src.constants import MAPPING_SER_ECO_REGIONS, MAPPING_ECO_REGIONS_FR_EN

# Load eco-regions
eco_gdf = gpd.read_file("./data/species/ser_l93_new")

# Process eco-regions (same as in tiles_selection.py)
if "codeser" in eco_gdf.columns:
    eco_gdf["greco"] = eco_gdf["codeser"].apply(
        lambda x: x[0] if isinstance(x, str) and len(x) > 0 else x
    )
    eco_gdf = eco_gdf.dissolve(by="greco", aggfunc="first").reset_index().iloc[1:]
    eco_gdf["NomSER"] = eco_gdf["NomSER"].apply(
        lambda x: MAPPING_SER_ECO_REGIONS.get(x.replace(" ", "_"), x.replace(" ", "_")) if isinstance(x, str) else x
    )
    eco_gdf["NomSER"] = eco_gdf["NomSER"].apply(
        lambda x: MAPPING_ECO_REGIONS_FR_EN.get(x, x) if isinstance(x, str) else x
    )

# Calculate areas
eco_gdf["area_km2"] = eco_gdf.geometry.area / 1_000_000  # Convert to km²
eco_areas = eco_gdf.set_index("NomSER")["area_km2"].to_dict()

# Print areas by eco-region
print("\nAreas by eco-region:")
for region, area in sorted(eco_areas.items(), key=lambda x: 0 if x[1] is None else x[1], reverse=True):
    if region is not None:
        print(f"{region}: {area:.2f} km²")
        
# Print dictionary format for EFFECTIVE_FOREST_AREA_BY_REGION
print("\nEFFECTIVE_FOREST_AREA_BY_REGION = {")
for region, area in sorted(eco_areas.items(), key=lambda x: str(x[0])):
    if region is not None:
        from src.constants import FOREST_COVER_RATIO_BY_REGION
        forest_ratio = FOREST_COVER_RATIO_BY_REGION.get(region, 0.5)
        effective_area = area * forest_ratio
        print(f'    "{region}": {effective_area:.2f},  # Total area: {area:.2f} km² * forest ratio: {forest_ratio:.2f}')
print("}") 