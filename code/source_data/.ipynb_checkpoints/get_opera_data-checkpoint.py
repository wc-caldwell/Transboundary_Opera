#!/usr/bin/env python3
import os
import time
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from transboundary_opera import displacement_tools as dt
from opera_utils.disp import _download

MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds, will increase exponentially
SEARCH_START = datetime(2016, 1, 1)
SEARCH_END = datetime(2026, 1, 1)


def main():
    parser = argparse.ArgumentParser(description="Download OPERA displacement data")
    parser.add_argument(
        "--data-storage",
        type=Path,
        default= Path(os.getcwd()).parent.parent / 'analysis_data' / 'OPERA_data',
        help="Directory to store downloaded data",
    )
    args = parser.parse_args()
    data_storage = args.data_storage
    os.makedirs(data_storage, exist_ok=True)

    gdf = gpd.read_file(Path(os.getcwd()).parent.parent / "raw_data/TBA_full.shp")
    frame_ids = dt.get_unique_frame_ids(gdf, track_per_row=True, max_workers=8)

    for idx, data in gdf.iterrows():
        print(f"Processing aquifer {data.CODE_2021}...", flush=True)
        bbox = data.geometry.bounds
        aquifer_frame_ids = data.frame_ids
        print(f"  Fetching frame geometries for {len(aquifer_frame_ids)} frames...", flush=True)
        geom_frames = dt.get_frame_geometries(
            aquifer_frame_ids,
            gdf_bounds=data.geometry.bounds
        )
        print(f"  Got geometries, starting downloads...", flush=True)

        for frame in aquifer_frame_ids:
            out_path = data_storage / data.CODE_2021 / str(frame)
            subset_dir = out_path / 'subset-ncs'
            mintpy_dir = out_path / 'mintpy'
            done_marker = out_path / '.download_complete'

            # Skip if already processed by mintpy
            if mintpy_dir.exists():
                print(f'  Frame {frame} for aquifer {data.CODE_2021} already processed (mintpy exists).', flush=True)
                continue

            # Skip if download already completed
            if done_marker.exists():
                print(f'  Data from OPERA frame {frame} for aquifer {data.CODE_2021} already downloaded locally.', flush=True)
                continue

            # Clean up any partial downloads from previous interrupted runs
            if subset_dir.exists():
                print(f"  Cleaning up partial download for frame {frame}...", flush=True)
                shutil.rmtree(subset_dir)

            os.makedirs(out_path, exist_ok=True)
            # clip the target aquifer by the current opera frame
            clipped = gpd.clip(
                gpd.GeoSeries([data.geometry], crs=gdf.crs),
                geom_frames[geom_frames['frame_id'] == frame]
            )
            # if the clipped aquifer is empty, then the frame does not actually overlap and can be removed and skipped
            if clipped.empty:
                print(f"  No overlap for frame {frame}, skipping...", flush=True)
                shutil.rmtree(out_path)
                continue

            clipped_bbox = clipped.geometry.iloc[0].bounds
            print(f"  Downloading frame {frame}...", flush=True)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    _download.run_download(
                        frame,
                        start_datetime=SEARCH_START,
                        end_datetime=SEARCH_END,
                        num_workers=8,
                        output_dir=out_path / 'subset-ncs',
                        bbox=clipped_bbox
                    )
                    # Mark as complete only after successful download
                    done_marker.touch()
                    print(f"  Frame {frame} done.", flush=True)
                    break  # success
                except (ConnectionError, OSError) as e:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_DELAY * (2 ** (attempt - 1))
                        print(f"  Attempt {attempt} failed for frame {frame}: {e}", flush=True)
                        print(f"  Retrying in {wait}s...", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"  All {MAX_RETRIES} attempts failed for frame {frame}. Skipping.", flush=True)


if __name__ == "__main__":
    main()