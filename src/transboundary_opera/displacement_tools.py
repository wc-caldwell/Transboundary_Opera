import os
import geopandas as gpd
import asf_search as asf
import re
import numpy as np
import h5py
import rasterio as rio


def get_unique_frame_ids(gdf):
    """
    Extract unique OPERA frame IDs from a GeoDataFrame by searching ASF.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame with geometry column
    search_start : datetime
        Start date for ASF search
    search_end : datetime
        End date for ASF search
    
    Returns:
    --------
    list : Sorted list of unique frame IDs
    """
    all_frame_ids = []
    pattern = re.compile(r'_F(\d{5})_')

    for entry, row in gdf.iterrows():
        results = asf.geo_search(
            intersectsWith=row.geometry.convex_hull.wkt,
            dataset=asf.DATASET.OPERA_S1,
            processingLevel=asf.PRODUCT_TYPE.DISP_S1,
            maxResults=50
        )
        
        current_frame_ids = set()
        for product in results:
            filename = product.properties['fileName']
            match = pattern.search(filename)
            if match:
                frame_id_str = match.group(1) 
                current_frame_ids.add(int(frame_id_str))
        
        all_frame_ids.append(sorted(list(current_frame_ids)))

    gdf['frame_ids'] = all_frame_ids
    unique_frame_ids = sorted({fid for ids in gdf['frame_ids'] if isinstance(ids, (list, tuple, set)) for fid in ids})
    
    return unique_frame_ids

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
            f_out.create_dataset('losEast', data=east.astype(np.float32))
            f_out.create_dataset('losNorth', data=north.astype(np.float32))
            f_out.create_dataset('incidenceAngle', data=inc_angle.astype(np.float32))
            f_out.create_dataset('azimuthAngle', data=az_angle.astype(np.float32))
            
            # Copy all attributes from the reference file
            for key, val in f_ref.attrs.items():
                f_out.attrs[key] = val
                
            # Double check: ensure dataset names are correct for MintPy
            # and that 'FILE_TYPE' reflects geometry
            f_out.attrs['FILE_TYPE'] = 'geometry'

    print(f"Created geom file synced with {ref_h5}")