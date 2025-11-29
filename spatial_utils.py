from __future__ import annotations
import geopandas as gpd

def load_counties(path: str, fips_col: str = "GEOID"):
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf = gdf.rename(columns={fips_col: "county_fips"})
    gdf["county_fips"] = gdf["county_fips"].astype(str).str.zfill(5)
    return gdf[["county_fips","geometry"]]
