from pandas import DataFrame as df
import geopandas as gpd
from shapely.geometry import LineString

def create(xcol:str, ycol:str, line_no:list, index:int, data:df, crs_epsg:int = 32612, path:str = "./"):
    """
    - Create a GeoJSON file.
    - Read UTM coordinates from the DataFrame, convert to WGS84(lag,lon), and write to GeoJSON.
    args:
        - xcol: column name for x coordinates (e.g., "UTM_X")
        - ycol: column name for y coordinates (e.g., "UTM_Y")
        - line_no: list of line numbers (e.g., [1, 2, 3])
        - index: index to select a line from line_no (e.g., 0)
        - data: DataFrame containing the coordinates and optional line_id column
        - crs_epsg: EPSG code for the input CRS (default is 32612 for UTM zone 12N)
        - path: path to the input CSV file (default is "./")
    """
    # --- INPUTS ---
    out_shp  = f"{path}{line_no[index]}_lines.geojson"

    # Optional columns:
    # - line_id: to create multiple line features
    line_id_col = "Line" if "Line" in data.columns else None

    # --- BUILD LINES ---
    lines = []
    attrs = []

    if line_id_col:
        grouped = data.groupby(line_id_col)
        for lid, g in grouped:
            coords = list(zip(g[xcol].to_numpy(), g[ycol].to_numpy()))
            if len(coords) < 2:
                continue
            lines.append(LineString(coords))
            attrs.append({line_id_col: lid})

    # --- WRITE SHAPEFILE ---
    wgs84_crs_id:int = 4326
    gdf = gpd.GeoDataFrame(attrs, geometry=lines, crs=crs_epsg)
    gdf_4326 = gdf.to_crs(wgs84_crs_id)
    gdf_4326.to_file(out_shp)  # writes .shp/.shx/.dbf/.prj
    print(f"Wrote {len(gdf)} line(s) to {out_shp}")

def main(line_no, data):
    create(
        xcol="x_wgs84",
        ycol="y_wgs84",
        line_no=line_no,  # Example line numbers
        index=0,  # Select the first line number for processing
        data=data,  # Placeholder for the actual DataFrame
        path="../data/"  # Path to the input CSV file
    )

if __name__ == "__main__":
    import pandas as pd
    from rich import print
    
    path:str = "../data/11-024_Alberta_NE.csv"
    data = pd.read_csv(path)[["Line","x_wgs84", "y_wgs84"]]
    line_no:list = data["Line"].unique().tolist()
    print(data.isna().sum())  # Check for missing values
    data.fillna(1e-20, inplace=True)  # Replace NaN with a small value to avoid issues in LineString creation
    main(line_no, data)