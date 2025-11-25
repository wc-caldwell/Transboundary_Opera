import os
import geopandas as gpd
import asf_search as asf
import re


def get_opera_frame_ids(shapefile_path):
    """
    Identifies OPERA DISP-S1 Frame IDs intersecting a shapefile.
    """
    # 1. Read the shapefile and convert to WKT (Well-Known Text)
    # asf_search requires WKT for spatial queries
    gdf = gpd.read_file(shapefile_path)
    
    # Ensure we are in WGS84 (Lat/Lon)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
        
    # Combine all features into a single geometry (convex hull) for the search
    aoi_wkt = gdf.unary_union.convex_hull.wkt

    print(f"Searching ASF for OPERA-DISP products intersecting your AOI...")

    # 2. Search ASF for OPERA DISP-S1 products
    # We only need metadata, so we limit results to avoid long wait times.
    # We filter by the OPERA-S1 dataset and DISP product type.
    results = asf.geo_search(
        intersectsWith=aoi_wkt,
        dataset=asf.DATASET.OPERA_S1,
        processingLevel=asf.PRODUCT_TYPE.DISP_S1,
        maxResults=50  # We only need a few results to identify the Frame IDs
    )

    if not results:
        print("No intersecting OPERA DISP products found.")
        return []

    # 3. Extract Frame IDs from Filenames
    # Filename format: OPERA_L3_DISP-S1_IW_Fxxxxx_VV_...
    # We use regex to find the pattern _F followed by 5 digits
    frame_ids = set()
    pattern = re.compile(r'_F(\d{5})_')

    for product in results:
        filename = product.properties['fileName']
        match = pattern.search(filename)
        if match:
            # We strip the underscore and 'F' to get the raw ID number if needed, 
            # but usually, you want the integer or the string. 
            # The regex group matches the digits inside (e.g., '12345').
            frame_id_str = match.group(1) 
            frame_ids.add(int(frame_id_str))

    return sorted(list(frame_ids))