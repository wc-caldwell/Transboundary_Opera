import numpy as np
import matplotlib.pyplot as plt
import h5py
from mintpy.asc_desc2horz_vert import get_overlap_lalo
from mintpy.utils import readfile

def cache_valid_extent(ts_file, atr):
    """Cache valid data mask, geographic transform, and dates for a file."""
    # Read coherence (2D) instead of timeseries (3D)
    coh_file = ts_file.parent / 'avgSpatialCoh.h5'
    data, _ = readfile.read(coh_file)
    valid_mask = ~np.isnan(data) & (data > 0)
    
    # Read dates from timeseries file
    with h5py.File(ts_file, 'r') as f:
        dates = [d.decode() for d in f['date'][:]]
    
    # Store geographic transform info and dates
    return {
        'mask': valid_mask,
        'N': float(atr['Y_FIRST']),
        'W': float(atr['X_FIRST']),
        'lat_step': float(atr['Y_STEP']),
        'lon_step': float(atr['X_STEP']),
        'dates': dates,
    }


def count_overlap_from_cache(cache1, cache2, bbox):
    """Count overlapping valid pixels using cached masks."""
    S, N, W, E = bbox
    
    masks_in_bbox = []
    for cache in [cache1, cache2]:
        # Convert bbox to pixel indices for this cache
        y0 = int(round((N - cache['N']) / cache['lat_step']))
        x0 = int(round((W - cache['W']) / cache['lon_step']))
        y1 = int(round((S - cache['N']) / cache['lat_step']))
        x1 = int(round((E - cache['W']) / cache['lon_step']))
        
        # Bounds check
        mask = cache['mask']
        if y0 < 0 or x0 < 0 or y1 > mask.shape[0] or x1 > mask.shape[1]:
            return 0
        
        masks_in_bbox.append(mask[y0:y1, x0:x1])
    
    # Ensure same shape (might differ by 1 pixel due to rounding)
    min_shape = (
        min(masks_in_bbox[0].shape[0], masks_in_bbox[1].shape[0]),
        min(masks_in_bbox[0].shape[1], masks_in_bbox[1].shape[1])
    )
    
    m1 = masks_in_bbox[0][:min_shape[0], :min_shape[1]]
    m2 = masks_in_bbox[1][:min_shape[0], :min_shape[1]]
    
    return int(np.sum(m1 & m2))

def get_asc_desc_pairs(data_dir, min_overlap_pixels=100, min_common_dates=2):
    """Find ascending/descending pairs with valid data and temporal overlap.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing track subdirectories
    min_overlap_pixels : int
        Minimum number of overlapping valid pixels required (default: 100)
    min_common_dates : int
        Minimum number of common dates required (default: 2)
    
    Returns
    -------
    overlapping_pairs : list of dict
        List of valid overlapping pairs with bbox, metadata, and common dates
    """
    asc_list = []
    desc_list = []
    
    for ts_file in list(data_dir.glob('*/*/timeseries.h5')):
        orb_dir = dict(h5py.File(ts_file, 'r').attrs)['ORBIT_DIRECTION']
        if orb_dir == 'Ascending':
            asc_list.append(ts_file)
        else:
            desc_list.append(ts_file)

    # Get attributes for all files
    asc_attrs = [readfile.read_attribute(str(f)) for f in asc_list]
    desc_attrs = [readfile.read_attribute(str(f)) for f in desc_list]
    
    # Pre-cache valid data masks, extents, and dates for all files (read each file ONCE)
    print("Caching valid data masks and dates...")
    asc_cache = [cache_valid_extent(f, a) for f, a in zip(asc_list, asc_attrs)]
    desc_cache = [cache_valid_extent(f, a) for f, a in zip(desc_list, desc_attrs)]

    overlapping_pairs = []

    for i, asc_file in enumerate(asc_list):
        for j, desc_file in enumerate(desc_list):
            try:
                bbox = get_overlap_lalo([asc_attrs[i], desc_attrs[j]])
                if bbox is None:
                    continue
                
                S, N, W, E = bbox
                if S >= N or W >= E:
                    continue
                
                # Check temporal overlap
                common_dates = sorted(set(asc_cache[i]['dates']) & set(desc_cache[j]['dates']))
                if len(common_dates) < min_common_dates:
                    continue
                
                # Fast spatial overlap check using cached masks
                n_overlap = count_overlap_from_cache(
                    asc_cache[i], desc_cache[j], bbox
                )
                
                if n_overlap < min_overlap_pixels:
                    continue
                
                overlapping_pairs.append({
                    'asc_file': asc_file,
                    'desc_file': desc_file,
                    'asc_frame': asc_file.parent.parent.name,
                    'desc_frame': desc_file.parent.parent.name,
                    'bbox': bbox,
                    'n_overlap_pixels': n_overlap,
                    'common_dates': common_dates,
                    'n_common_dates': len(common_dates),
                })
                
            except Exception:
                continue
            
    return overlapping_pairs

def plot_displacements(horz_file, vert_file, time_idx=None, date=None, vlim=None, cmap='RdBu', gdf=None, boundary_color='black', boundary_linewidth=1.5):
    """
    Plot horizontal and vertical displacement maps.
    
    Parameters
    ----------
    horz_file : str or Path
        Path to horizontal displacement HDF5 file
    vert_file : str or Path
        Path to vertical displacement HDF5 file
    time_idx : int, optional
        Time index to plot. If None and date is None, plots all time steps.
    date : str, optional
        Date string (YYYYMMDD) to plot. Takes precedence over time_idx.
    vlim : tuple, optional
        Color limits as (vmin, vmax). If None, uses symmetric limits.
    cmap : str
        Colormap name (default: 'RdBu')
    gdf : GeoDataFrame, optional
        GeoDataFrame with polygon geometries to overlay as boundaries (assumed EPSG:4326)
    boundary_color : str
        Color for polygon boundaries (default: 'black')
    boundary_linewidth : float
        Line width for polygon boundaries (default: 1.5)
    
    Returns
    -------
    fig : matplotlib Figure
    """
    # Read data directly from HDF5 to avoid MintPy parsing issues
    with h5py.File(horz_file, 'r') as f:
        horz = f['timeseries'][:]
        if 'date' in f:
            dates = [d.decode() if isinstance(d, bytes) else d for d in f['date'][:]]
        else:
            dates = [f't{i}' for i in range(horz.shape[0])]
        
        attrs = dict(f.attrs)
        ref_y = int(attrs.get('REF_Y', -1))
        ref_x = int(attrs.get('REF_X', -1))
        
        # Get georeferencing info
        epsg = attrs.get('EPSG', None)
        if isinstance(epsg, bytes):
            epsg = epsg.decode()
        x_first = float(attrs.get('X_FIRST', 0))
        y_first = float(attrs.get('Y_FIRST', 0))
        x_step = float(attrs.get('X_STEP', 1))
        y_step = float(attrs.get('Y_STEP', -1))
    
    with h5py.File(vert_file, 'r') as f:
        vert = f['timeseries'][:]
    
    if date is not None:
        if date in dates:
            time_idx = dates.index(date)
        else:
            raise ValueError(f"Date {date} not found. Available dates: {dates}")
    
    if vlim is None:
        max_abs = max(np.nanmax(np.abs(horz)), np.nanmax(np.abs(vert)))
        vlim = (-max_abs, max_abs)
    
    # Calculate image extent for geographic coordinates
    n_rows = horz.shape[-2]
    n_cols = horz.shape[-1]
    extent = [
        x_first,                        # left
        x_first + x_step * n_cols,      # right
        y_first + y_step * n_rows,      # bottom
        y_first                         # top
    ]
    
    # Reproject GeoDataFrame and filter to raster extent
    gdf_plot = None
    if gdf is not None and epsg is not None:
        from shapely.geometry import box
        gdf_reproj = gdf.to_crs(f'EPSG:{epsg}')
        raster_bounds = box(extent[0], extent[2], extent[1], extent[3])
        gdf_plot = gdf_reproj[gdf_reproj.intersects(raster_bounds)]
    
    def overlay_gdf(ax):
        """Helper to overlay GeoDataFrame boundaries on an axis."""
        if gdf_plot is not None:
            gdf_plot.boundary.plot(ax=ax, color=boundary_color, linewidth=boundary_linewidth)
    
    if horz.ndim == 3:
        n_times = horz.shape[0]
        
        if time_idx is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            date_str = dates[time_idx] if time_idx < len(dates) else f't={time_idx}'
            
            im0 = axes[0].imshow(horz[time_idx], cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=extent)
            axes[0].set_title(f'Horizontal ({date_str})')
            plt.colorbar(im0, ax=axes[0], label='m')
            overlay_gdf(axes[0])
            
            im1 = axes[1].imshow(vert[time_idx], cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=extent)
            axes[1].set_title(f'Vertical ({date_str})')
            plt.colorbar(im1, ax=axes[1], label='m')
            overlay_gdf(axes[1])
            
            if ref_y >= 0 and ref_x >= 0:
                ref_x_geo = x_first + ref_x * x_step
                ref_y_geo = y_first + ref_y * y_step
                for ax in axes:
                    ax.plot(ref_x_geo, ref_y_geo, 'k*', markersize=10)
        else:
            max_cols = 8
            if n_times > max_cols:
                print(f"  Showing first {max_cols} of {n_times} time steps. Use time_idx or date to select specific times.")
                n_plot = max_cols
            else:
                n_plot = n_times
            
            fig, axes = plt.subplots(2, n_plot, figsize=(2.5 * n_plot, 5))
            
            if n_plot == 1:
                axes = axes.reshape(2, 1)
            
            for t in range(n_plot):
                date_str = dates[t] if t < len(dates) else f't={t}'
                
                im0 = axes[0, t].imshow(horz[t], cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=extent)
                axes[0, t].set_title(f'{date_str}', fontsize=8)
                axes[0, t].axis('off')
                overlay_gdf(axes[0, t])
                
                im1 = axes[1, t].imshow(vert[t], cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=extent)
                axes[1, t].axis('off')
                overlay_gdf(axes[1, t])
            
            axes[0, 0].set_ylabel('Horizontal', fontsize=10)
            axes[1, 0].set_ylabel('Vertical', fontsize=10)
            
            fig.colorbar(im0, ax=axes[0, :], label='m', shrink=0.8)
            fig.colorbar(im1, ax=axes[1, :], label='m', shrink=0.8)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im0 = axes[0].imshow(horz, cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=extent)
        axes[0].set_title('Horizontal Displacement')
        plt.colorbar(im0, ax=axes[0], label='m')
        overlay_gdf(axes[0])
        
        im1 = axes[1].imshow(vert, cmap=cmap, vmin=vlim[0], vmax=vlim[1], extent=extent)
        axes[1].set_title('Vertical Displacement')
        plt.colorbar(im1, ax=axes[1], label='m')
        overlay_gdf(axes[1])
        
        if ref_y >= 0 and ref_x >= 0:
            ref_x_geo = x_first + ref_x * x_step
            ref_y_geo = y_first + ref_y * y_step
            for ax in axes:
                ax.plot(ref_x_geo, ref_y_geo, 'k*', markersize=10)
    
    plt.tight_layout()
    plt.show() 
    
    return fig

def list_dates(horz_file):
    """List available dates in a timeseries file.
    
    Parameters
    ----------
    horz_file : str or Path
        Path to displacement HDF5 file
        
    Returns
    -------
    dates : list
        List of date strings
    """
    with h5py.File(horz_file, 'r') as f:
        if 'date' in f:
            dates = [d.decode() if isinstance(d, bytes) else d for d in f['date'][:]]
        else:
            dates = []
    
    print(f"Available dates ({len(dates)}):")
    for i, d in enumerate(dates):
        print(f"  [{i}] {d}")
    
    return dates