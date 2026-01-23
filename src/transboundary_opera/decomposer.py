from pathlib import Path
import numpy as np
from pyproj import Transformer
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert
from mintpy.utils import ptime, readfile, utils as ut, writefile


class InSARDecomposer:
    """Process overlapping InSAR pairs and decompose LOS to horizontal/vertical components."""
    
    def __init__(self, overlapping_pairs, ds_name='timeseries', azimuth=-90):
        self.overlapping_pairs = overlapping_pairs
        self.ds_name = ds_name
        self.azimuth = azimuth
        self.failed_pairs = []
        self.successful_pairs = []
    
    def _get_file_paths(self, timeseries_files):
        """Generate all related file paths from timeseries files."""
        return {
            'timeseries': timeseries_files,
            'velocity': [Path(str(f).replace('timeseries', 'velocity')) for f in timeseries_files],
            'geometry': [Path(str(f).replace('timeseries', 'geometryGeo')) for f in timeseries_files],
            'coherence': [f.parent / 'avgSpatialCoh.h5' for f in timeseries_files],
        }
    
    def _compute_grid_params(self, atr, bbox):
        """Compute grid dimensions and steps from attributes and bounding box."""
        S, N, W, E = bbox
        lat_step = float(atr['Y_STEP'])
        lon_step = float(atr['X_STEP'])
        length = int(round((S - N) / lat_step))
        width = int(round((E - W) / lon_step))
        return {'length': length, 'width': width, 'lat_step': lat_step, 'lon_step': lon_step}
    
    def _find_max_coherence_location(self, coh_data, bbox, grid):
        """Find the location of maximum coherence within the processed region.
        
        Args:
            coh_data: 2D coherence array (already subsetted to bbox)
            bbox: Bounding box (S, N, W, E)
            grid: Grid parameters dict with lat_step, lon_step
            
        Returns:
            (lat, lon) of max coherence location in geographic coordinates
        """
        max_idx = np.unravel_index(np.nanargmax(coh_data), coh_data.shape)
        row, col = max_idx
        
        S, N, W, E = bbox
        
        # Convert pixel indices to geographic coordinates
        # Row 0 corresponds to N (northern edge), increasing row goes south
        # Col 0 corresponds to W (western edge), increasing col goes east
        ref_lat = N + row * grid['lat_step']  # lat_step is negative
        ref_lon = W + col * grid['lon_step']  # lon_step is positive
        
        return ref_lat, ref_lon
    
    def _build_output_metadata(self, atr, grid, bbox, ref_coords):
        """Build metadata dictionary for output files."""
        _, N, W, _ = bbox
        ref_lat, ref_lon = ref_coords
        
        out_atr = atr.copy()
        out_atr.update({
            'FILE_TYPE': 'displacement',
            'WIDTH': str(grid['width']),
            'LENGTH': str(grid['length']),
            'X_STEP': str(grid['lon_step']),
            'Y_STEP': str(grid['lat_step']),
            'X_FIRST': str(W),
            'Y_FIRST': str(N),
        })
        
        ref_y, ref_x = ut.coordinate(out_atr).geo2radar(ref_lat, ref_lon)[0:2]
        out_atr['REF_Y'] = int(ref_y)
        out_atr['REF_X'] = int(ref_x)
        
        return out_atr
    
    def _read_pair_data(self, paths, atr_list, grid, bbox):
        """Read all data arrays for a pair of files."""
        N, W = bbox[1], bbox[2]
        shape = (len(paths['timeseries']), grid['length'], grid['width'])
        
        dlos = np.zeros(shape, dtype=np.float32)
        los_inc_angle = np.zeros(shape, dtype=np.float32)
        los_az_angle = np.zeros(shape, dtype=np.float32)
        coh = np.zeros(shape, dtype=np.float32)
        
        for i, (atr, ts_file) in enumerate(zip(atr_list, paths['timeseries'])):
            coord = ut.coordinate(atr)
            y0, x0 = coord.lalo2yx(N, W)
            box = (x0, y0, x0 + grid['width'], y0 + grid['length'])
            
            coh[i] = readfile.read(paths['coherence'][i], box=box, datasetName='avgSpatialCoh')[0]
            data, _ = readfile.read(ts_file, box=box, datasetName=self.ds_name)
            dlos[i] = data[0]
            los_inc_angle[i] = readfile.read(paths['geometry'][i], box=box, datasetName='incidenceAngle')[0]
            los_az_angle[i] = readfile.read(paths['geometry'][i], box=box, datasetName='azimuthAngle')[0]
        
        return {'dlos': dlos, 'inc': los_inc_angle, 'az': los_az_angle, 'coh': coh}
    
    def _write_outputs(self, dhorz, dvert, out_atr, ref_file, tracks):
        """Write horizontal and vertical components to HDF5 files."""
        out_dir = ref_file.parent.parent.parent
        outfiles = {
            'horz': out_dir / f'{tracks[0]}_{tracks[1]}_dhorz.h5',
            'vert': out_dir / f'{tracks[0]}_{tracks[1]}_dvert.h5',
        }
        
        print(f'  Writing horizontal: {outfiles["horz"]}')
        writefile.write(dhorz, out_file=str(outfiles['horz']), metadata=out_atr, ref_file=ref_file)
        
        print(f'  Writing vertical:   {outfiles["vert"]}')
        writefile.write(dvert, out_file=str(outfiles['vert']), metadata=out_atr, ref_file=ref_file)
        
        return outfiles
    
    def process_pair(self, pair):
        """Process a single overlapping pair and return output paths."""
        timeseries_files = list(pair.values())[:2]
        paths = self._get_file_paths(timeseries_files)
        tracks = [timeseries_files[0].parts[-3], timeseries_files[1].parts[-3]]
        ref_file = timeseries_files[0]
        
        atr_list = [readfile.read_attribute(f) for f in timeseries_files]
        grid = self._compute_grid_params(atr_list[0], pair['bbox'])
        
        # Read all data (subsetted to bbox)
        data = self._read_pair_data(paths, atr_list, grid, pair['bbox'])
        
        # Decompose LOS to horizontal/vertical
        dlos_clean = np.nan_to_num(data['dlos'], nan=0.0)
        dhorz, dvert = asc_desc2horz_vert(dlos_clean, data['inc'], data['az'], self.azimuth)
        
        # Find reference point from max coherence WITHIN the processed region
        # Use mean coherence across both tracks for more robust reference selection
        mean_coh = np.nanmean(data['coh'], axis=0)
        ref_coords = self._find_max_coherence_location(mean_coh, pair['bbox'], grid)
        
        # Build metadata and write
        out_atr = self._build_output_metadata(atr_list[0], grid, pair['bbox'], ref_coords)
        return self._write_outputs(dhorz, dvert, out_atr, ref_file, tracks)
    
    def run(self, verbose=True):
        """Process all overlapping pairs."""
        self.failed_pairs = []
        self.successful_pairs = []
        
        for idx, pair in enumerate(self.overlapping_pairs):
            if verbose:
                print(f'Processing pair {idx + 1}/{len(self.overlapping_pairs)}')
            try:
                outfiles = self.process_pair(pair)
                self.successful_pairs.append(outfiles)
            except Exception as e:
                if verbose:
                    print(f'  Failed: {e}')
                self.failed_pairs.append({'index': idx, 'pair': pair, 'error': str(e)})
        
        if verbose:
            self._print_summary()
        
        return self
    
    def _print_summary(self):
        """Print processing summary."""
        print(f'\n{"="*50}')
        print(f'Complete: {len(self.successful_pairs)} succeeded, {len(self.failed_pairs)} failed')
        if self.failed_pairs:
            print('Failed pairs:')
            for item in self.failed_pairs:
                print(f'  Pair {item["index"]}: {item["error"]}')