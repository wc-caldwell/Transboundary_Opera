import os
import geopandas as gpd
import numpy as np
import h5py
import rasterio as rio
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import asf_search as asf

# Compile once at module level
FRAME_PATTERN = re.compile(r'_F(\d{5})_')


def extract_frame_ids(results):
    """Extract frame IDs from ASF search results."""
    return {
        int(match.group(1))
        for product in results
        if (match := FRAME_PATTERN.search(product.properties['fileName']))
    }


def search_single_geometry(geometry):
    """Search ASF for a single geometry."""
    results = asf.geo_search(
        intersectsWith=geometry.convex_hull.wkt,
        dataset=asf.DATASET.OPERA_S1,
        processingLevel=asf.PRODUCT_TYPE.DISP_S1,
        maxResults=50
    )
    return extract_frame_ids(results)


def get_unique_frame_ids(gdf, track_per_row=True, max_workers=8):
    """
    Extract unique OPERA frame IDs from a GeoDataFrame by searching ASF.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with geometry column
    track_per_row : bool
        If True, stores frame_ids per row (slower but preserves mapping).
        If False, uses unified geometry search (faster).
    max_workers : int
        Maximum parallel requests for per-row searching
    
    Returns
    -------
    list : Sorted list of unique frame IDs
    """
    if not track_per_row:
        # FAST PATH: Combine all geometries into one search
        unified_geom = gdf.geometry.union_all()  # or .unary_union for older geopandas
        results = asf.geo_search(
            intersectsWith=unified_geom.convex_hull.wkt,
            dataset=asf.DATASET.OPERA_S1,
            processingLevel=asf.PRODUCT_TYPE.DISP_S1,
            maxResults=437  # Increase limit for unified search
        )

        return sorted(extract_frame_ids(results))
    
    # PER-ROW PATH: Parallel requests with per-geometry tracking
    frame_ids_by_idx = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(search_single_geometry, row.geometry): idx
            for idx, row in gdf.iterrows()
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                frame_ids_by_idx[idx] = sorted(future.result())
            except Exception as e:
                print(f"Search failed for index {idx}: {e}")
                frame_ids_by_idx[idx] = []
    
    # Assign results back in original order
    gdf['frame_ids'] = [frame_ids_by_idx.get(idx, []) for idx in gdf.index]
    
    # Flatten and deduplicate
    unique_frame_ids = sorted({
        fid 
        for ids in gdf['frame_ids'] 
        for fid in ids
    })
    
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