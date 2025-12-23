import os
import geopandas as gpd
import asf_search as asf
import re
import numpy as np
import h5py
import rasterio as rio


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

def create_geom_h5_with_ref(east_tif, north_tif, ref_h5):
    """
    Creates geom.h5 by calculating angles from TIFFs and 
    copying all attributes from a reference timeseries.h5.
    """
    # 1. Read the Data from TIFFs
    with rio.open(east_tif) as src_e, rio.open(north_tif) as src_n:
        east = src_e.read(1)
        north = src_n.read(1)

    # 2. Calculate Geometry
    up = np.sqrt(np.clip(1 - east**2 - north**2, 0, 1))
    inc_angle = np.degrees(np.arccos(up))
    az_angle = np.degrees(np.arctan2(-east, north))

    # 3. Create Geometry H5 and Copy Attributes
    with h5py.File(ref_h5, 'r') as f_ref:
        out_h5 = str(ref_h5).replace('timeseries', 'geom')
        with h5py.File(out_h5, 'w') as f_out:
            # Create datasets
            f_out.create_dataset('incidenceAngle', data=inc_angle.astype(np.float32))
            f_out.create_dataset('azimuthAngle', data=az_angle.astype(np.float32))
            
            # Copy all attributes from the reference file
            for key, val in f_ref.attrs.items():
                f_out.attrs[key] = val
                
            # Double check: ensure dataset names are correct for MintPy
            # and that 'FILE_TYPE' reflects geometry
            f_out.attrs['FILE_TYPE'] = 'geometry'

    print(f"Created geom file synced with {ref_h5}")