# Author: Laura Paccini (laura.paccini@pnnl.gov)
# Date: May 27, 2025
# Description: Script to extract environmental variables around MCS locations
import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import healpy as hp
from easygems import healpix as egh
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def convert_time(time_array):
    """Convert cftime to standard datetime64"""
    if hasattr(time_array[0], 'year'):  # It's a cftime object
        return np.array([np.datetime64(datetime(t.year, t.month, t.day, t.hour)) 
                         for t in time_array])
    return time_array

def add_circular_trigger_areas_df(filtered_df, RADII, hp_grid, ocean_mask=None, remove_land=False):
    """DataFrame-optimized version of circular area calculation with optional land filtering
    
    Parameters:
    -----------
    filtered_df : pandas.DataFrame
        DataFrame with MCS track information
    RADII : np.ndarray
        Array of radii in degrees
    hp_grid : xarray.Dataset
        HEALPix grid with lat/lon information
    ocean_mask : xarray.DataArray, optional
        Ocean mask (0=land, non-NaN=ocean)
    remove_land : bool, default=False
        If True, filter out areas that contain any land
    
    Returns:
    --------
    tuple
        (trigger_areas, filtered_df)
        - trigger_areas: Dictionary mapping (track_idx, radius) to pixel arrays
        - filtered_df: DataFrame filtered to ocean-only areas if remove_land=True
    """
    # Get HEALPix grid parameters
    nside = egh.get_nside(hp_grid)
    nest = True if egh.get_nest(hp_grid) else False
    
    # Create dictionary to store trigger areas
    trigger_areas = {}
    
    # For ocean filtering
    ocean_tracks = set()
    
    # Process tracks in batches for better memory management
    batch_size = 100
    total_tracks = len(filtered_df)
    
    for batch_start in range(0, total_tracks, batch_size):
        batch_end = min(batch_start + batch_size, total_tracks)
        
        # Get batch of trigger indices
        batch_df = filtered_df.iloc[batch_start:batch_end]
        
        # Process all radii for each track in the batch
        def process_track(row):
            idx = row.Index
            cell_idx = int(row.trigger_idx)
            
            # Store area indices per radius
            track_areas = {}
            is_ocean_track = True  # Assume ocean track until we find land
            
            # Process each radius (largest first for efficiency when filtering)
            for radius in sorted(RADII, reverse=True):
                # Get pixels within radius
                area_idxs = hp.query_disc(
                    nside, 
                    hp.pix2vec(nside, cell_idx, nest=nest), 
                    np.radians(radius),
                    inclusive=False, 
                    nest=nest
                )
                
                # Store result - use track ID without tuple if it's a tuple
                if isinstance(idx, tuple):
                    track_id = idx[0]  # Extract just the track ID part
                    track_areas[(track_id, radius)] = area_idxs
                else:
                    track_areas[(idx, radius)] = area_idxs
                
                # Check if this is a largest radius and we need to filter for ocean
                if remove_land and radius == max(RADII):
                    # Check if all areas are ocean
                    if ocean_mask is not None:
                        try:
                            # Get ocean mask values for these pixels
                            mask_values = ocean_mask.sel(cell=area_idxs).values
                            # If any are NaN, there's land in this area
                            if np.any(np.isnan(mask_values)):
                                is_ocean_track = False
                        except Exception as e:
                            print(f"Error checking ocean mask for track {idx}: {e}")
                            # If selection fails, be conservative and exclude
                            is_ocean_track = False
            
            return idx, track_areas, is_ocean_track
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_track, batch_df.itertuples()))
        
        # Store results
        for idx, areas, is_ocean in results:
            # Store areas in the dictionary
            trigger_areas.update(areas)
            
            # Mark as ocean track if needed
            if is_ocean:
                ocean_tracks.add(idx)
    
    # Print out a few example keys to check format
    sample_keys = list(trigger_areas.keys())[:5] if trigger_areas else []
    print(f"Sample area keys: {sample_keys}")
    
    # Filter DataFrame if requested
    if remove_land:
        filtered_ocean_df = filtered_df.loc[list(ocean_tracks)].copy()
        print(f"Filtered from {len(filtered_df)} to {len(filtered_ocean_df)} ocean-only tracks")
        return trigger_areas, filtered_ocean_df
    
    return trigger_areas, filtered_df

def extract_spatial_data(trigger_areas, filtered_df, ds_variable, RADII, hp_grid, time_tolerance=pd.Timedelta('1H'), 
                         times_before_init=None, include_full_evolution=False, latvar='meanlat', lonvar='meanlon'):
    """Extract full spatial data for each track and time
    
    Parameters:
    -----------
    trigger_areas : dict
        Dictionary mapping (track_idx, radius) to pixel arrays
    filtered_df : pandas.DataFrame
        DataFrame with MCS track information
    ds_variable : xarray.DataArray
        Data variable to extract values from
    RADII : np.ndarray
        Array of radii in degrees
    hp_grid : xarray.Dataset
        HEALPix grid with lat/lon information (needed for cell coordinates)
    time_tolerance : pd.Timedelta, optional
        Tolerance for matching times (default: 1 hour)
    times_before_init : pd.Timedelta, optional
        If provided, extract data for this time period before each track's start time (initiation)
    include_full_evolution : bool, optional
        If True, include all times in the track's lifecycle after initiation (default: False)
    latvar : str, optional
        Name of latitude variable (default: 'meanlat')
    lonvar : str, optional
        Name of longitude variable (default: 'meanlon')
    
    Returns:
    --------
    xarray.Dataset
        Dataset containing full spatial data for each track, radius, and time point
    """
    # Pre-calculate available times once (huge speedup)
    available_times = pd.DatetimeIndex(ds_variable.time.values)
    
    # Extract the unique track IDs (ignoring time indices in case tracks are tuples)
    if isinstance(filtered_df.index[0], tuple):
        # If index is a tuple (track_id, time_idx), extract just the track IDs
        unique_track_ids = sorted(set([idx[0] for idx in filtered_df.index]))
    else:
        # Otherwise use the indices directly
        unique_track_ids = filtered_df.index.unique()
    
    # Create dictionary to map track ID to a sequential integer (needed for xarray)
    track_id_to_idx = {track_id: i for i, track_id in enumerate(unique_track_ids)}
    
    # First pass: identify all times needed for each track
    track_time_mapping = {}
    for track_id in unique_track_ids:
        # Get all rows for this track ID
        if isinstance(filtered_df.index[0], tuple):
            track_indices = [idx for idx in filtered_df.index if idx[0] == track_id]
            if not track_indices:
                continue
                
            track_rows = filtered_df.loc[track_indices]
            if isinstance(track_rows, pd.Series):
                track_data = track_rows
                track_rows = pd.DataFrame([track_rows])
            else:
                track_data = track_rows.iloc[0]  # Use first row for reference info
            
            # If we need full evolution, gather all base_times for this track
            if include_full_evolution:
                track_base_times = pd.DatetimeIndex([pd.Timestamp(row['base_time']) for _, row in track_rows.iterrows()])
        else:
            if track_id not in filtered_df.index:
                continue
                
            track_data = filtered_df.loc[track_id]
            
            # For non-tuple indices, we don't have multiple times per track
            if include_full_evolution:
                if isinstance(track_data, pd.DataFrame):
                    track_base_times = pd.DatetimeIndex([pd.Timestamp(row['base_time']) for _, row in track_data.iterrows()])
                else:
                    track_base_times = pd.DatetimeIndex([pd.Timestamp(track_data['base_time'])])
        
        # Get times needed for analysis
        analysis_times = []
        
        # 1. Add pre-convective times if requested
        if times_before_init is not None:
            # Get the initiation time of the MCS track
            init_time = pd.Timestamp(track_data['start_basetime'])
            start_time = init_time - times_before_init
            
            # Find all available times within this range
            preconv_times = available_times[(available_times >= start_time) & 
                                         (available_times <= init_time)]
            analysis_times.extend(preconv_times)
        
        # 2. Add full evolution times if requested
        if include_full_evolution:
            # Get the initiation time and find the last time in the track
            init_time = pd.Timestamp(track_data['start_basetime'])
            
            # Determine the end time from track data
            if isinstance(filtered_df.index[0], tuple):
                # Get all track times and find latest
                track_times = [pd.Timestamp(row['base_time']) for _, row in track_rows.iterrows()]
                if track_times:
                    end_time = max(track_times)
                    
                    # Find ALL available times between init_time and end_time
                    post_init_times = available_times[(available_times >= init_time) & 
                                                    (available_times <= end_time)]
                    analysis_times.extend(post_init_times)
            else:
                # Single time track handling
                base_time = pd.Timestamp(track_data['base_time'])
                if base_time > init_time:  # Only if base_time is after initiation
                    # Use a wider window to ensure we get data
                    post_init_times = available_times[(available_times >= init_time) & 
                                                    (available_times <= base_time)]
                    analysis_times.extend(post_init_times)
        
        # 3. If neither pre-convective nor full evolution requested, just use base_time
        if not times_before_init and not include_full_evolution:
            request_time = pd.Timestamp(track_data['base_time'])
            
            # Find closest available time within tolerance
            if time_tolerance is not None:
                closest_idx = np.abs(available_times - request_time).argmin()
                time_diff = abs(available_times[closest_idx] - request_time)
                if time_diff <= time_tolerance:
                    analysis_times.append(available_times[closest_idx])
            elif request_time in available_times:
                analysis_times.append(request_time)
        
        # Skip if no valid times found
        if not analysis_times:
            continue
        
        # Store unique times for this track
        track_time_mapping[track_id] = sorted(set(analysis_times))
    
    # Get maximum number of time points across all tracks
    max_time_points = max([len(times) for times in track_time_mapping.values()]) if track_time_mapping else 0
    
    print(f"Found {len(track_time_mapping)} tracks with valid times, max time points: {max_time_points}")
    
    # Second pass: collect all unique cells across all track areas for global indexing
    print("Creating global cell mapping...")
    all_cells = set()
    
    # Print a few keys from trigger_areas to debug
    print(f"Sample trigger area keys: {list(trigger_areas.keys())[:5]}")
    
    # Count successful tracks
    success_count = 0
    for track_id in unique_track_ids:
        # Check if track exists in trigger_areas
        found = False
        for radius in RADII:
            # The key format should be (track_id, radius)
            key = (track_id, radius)
            if key in trigger_areas:
                all_cells.update(trigger_areas[key])
                found = True
                
        if found:
            success_count += 1
    
    print(f"Found areas for {success_count} out of {len(unique_track_ids)} tracks")
    
    # Create array of unique cell IDs (sorted for consistency)
    unique_cell_ids = np.array(sorted(all_cells))
    
    # Map global cell IDs to sequential indices (0, 1, 2...)
    cell_id_to_index = {int(cell_id): idx for idx, cell_id in enumerate(unique_cell_ids)}
    print(f"Created mapping for {len(cell_id_to_index)} unique cells")
    
    # For each radius, get the maximum number of cells
    max_cells_per_radius = {}
    for radius in RADII:
        max_cells = 0
        for track_id in unique_track_ids:
            key = (track_id, radius)
            if key in trigger_areas:
                max_cells = max(max_cells, len(trigger_areas[key]))
        
        # Store maximum for this radius (ensure at least 100 cells)
        max_cells_per_radius[radius] = max(max_cells, 100)
        print(f"Radius {radius}: maximum {max_cells} cells")
    
    # Collect all unique times needed across all tracks
    all_times_needed = []
    for times in track_time_mapping.values():
        all_times_needed.extend(times)
    unique_times_needed = sorted(set(all_times_needed))
    print(f"Need to fetch data for {len(unique_times_needed)} unique time points")
    
    # Create output dataset structure using global cell mapping
    ds_output = xr.Dataset(
        coords={
            'track': range(len(track_id_to_idx)),
            'time': range(max_time_points),
            'cell_id': unique_cell_ids,  # Store actual cell IDs as a coordinate
        },
        attrs={
            'description': f'Spatial data around MCS tracks for variable: {ds_variable.name}',
            'created': datetime.now().isoformat(),
            'variable_units': getattr(ds_variable, 'units', 'unknown'),
        }
    )
    
    # Add track_id as a coordinate variable (maps index to actual track ID)
    ds_output = ds_output.assign_coords(track_id=('track', [tid for tid in track_id_to_idx.keys()]))
    
    # Create maps to store actual times for each track/time index
    time_values = np.full((len(track_id_to_idx), max_time_points), 
                          np.datetime64('NaT'), dtype='datetime64[ns]')
    time_offsets = np.full((len(track_id_to_idx), max_time_points), np.nan, dtype=np.float32)    

    # Add data variables - one for each radius
    for radius in RADII:
        ds_output[f'data_radius_{radius:.1f}'] = xr.DataArray(
            np.full((len(track_id_to_idx), max_time_points, len(unique_cell_ids)),
                   np.nan, dtype=np.float32),
            dims=['track', 'time', 'cell_id'],
            coords={
                'track': ds_output['track'],
                'time': ds_output['time'],
                'cell_id': ds_output['cell_id']
            },
            attrs={
                'radius': radius,
                'radius_units': 'degrees',
                'description': f'Data values for radius {radius} degrees'
            }
        )
        
        # Create mask to indicate which cells are within this radius for each track
        ds_output[f'mask_radius_{radius:.1f}'] = xr.DataArray(
            np.zeros((len(track_id_to_idx), len(unique_cell_ids)),
                    dtype=np.bool_),
            dims=['track', 'cell_id'],
            coords={
                'track': ds_output['track'],
                'cell_id': ds_output['cell_id']
            },
            attrs={
                'description': f'Mask indicating cells within radius {radius} degrees'
            }
        )
    
    # Create arrays for track metadata
    track_lats = np.full((len(track_id_to_idx), max_time_points), np.nan, dtype=np.float32)
    track_lons = np.full((len(track_id_to_idx), max_time_points), np.nan, dtype=np.float32)
    
    # Add to dataset
    ds_output['track_lat'] = xr.DataArray(
        track_lats,
        dims=['track', 'time'],
        coords={
            'track': ds_output['track'],
            'time': ds_output['time']
        },
        attrs={'units': 'degrees_north', 'description': 'Latitude of MCS track center'}
    )
    
    ds_output['track_lon'] = xr.DataArray(
        track_lons,
        dims=['track', 'time'],
        coords={
            'track': ds_output['track'],
            'time': ds_output['time']
        },
        attrs={'units': 'degrees_east', 'description': 'Longitude of MCS track center'}
    )
    
    # Fetch ALL data at once in a single operation
    print(f"Fetching data for {len(unique_times_needed)} times × {len(unique_cell_ids)} cells...")
    try:
        # Get all data at once
        full_data_chunk = ds_variable.sel(
            time=unique_times_needed,
            cell=unique_cell_ids
        ).compute()
        
        print(f"Successfully fetched data array with shape {full_data_chunk.shape}")
        
        # Check if we got any valid data
        has_data = not np.all(np.isnan(full_data_chunk.values))
        if not has_data:
            print("WARNING: No valid data found in the full data extraction!")
            
    except Exception as e:
        print(f"ERROR: Failed to extract full data chunk: {e}")
        traceback.print_exc()
        # Create an empty chunk
        full_data_chunk = xr.DataArray(
            np.full((len(unique_times_needed), len(unique_cell_ids)), np.nan),
            dims=['time', 'cell'],
            coords={'time': unique_times_needed, 'cell': unique_cell_ids}
        )

    # Create a lookup dictionary from time to index in full_data_chunk
    time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(full_data_chunk.time.values)}
    
    # Now populate the output dataset with data
    for track_id, track_times in track_time_mapping.items():
        # Skip if track is not in trigger_areas anymore
        if (track_id, RADII[0]) not in trigger_areas:
            print(f"  Skipping track {track_id} - no area found for radius {RADII[0]}")
            continue
            
        # Get index for this track
        track_idx = track_id_to_idx[track_id]
        
        # Get track data
        if isinstance(filtered_df.index[0], tuple):
            track_indices = [idx for idx in filtered_df.index if idx[0] == track_id]
            track_data = filtered_df.loc[track_indices[0]]
            track_rows = filtered_df.loc[[idx for idx in filtered_df.index if idx[0] == track_id]]
            if isinstance(track_rows, pd.Series):
                track_rows = pd.DataFrame([track_rows])
        else:
            track_data = filtered_df.loc[track_id]
            if isinstance(track_data, pd.DataFrame):
                track_rows = track_data
            else:
                track_rows = pd.DataFrame([track_data])
        
        # Get initiation time
        init_time = pd.Timestamp(track_data['start_basetime'])
        
        # Process each time for this track
        print(f"Processing track {track_id} with {len(track_times)} times")
        for t_idx, current_time in enumerate(track_times):
            if t_idx >= max_time_points:
                break
                
            # Store time info
            time_values[track_idx, t_idx] = current_time
            time_offsets[track_idx, t_idx] = float((current_time - init_time).total_seconds() / 3600)
            
            # Find track position at this time
            if time_offsets[track_idx, t_idx] < 0:  # Pre-convective time
                lat = float(track_data[latvar]) if latvar in track_data else np.nan
                lon = float(track_data[lonvar]) if lonvar in track_data else np.nan
            else:
                # For post-initiation times, find the closest matching time in track_rows
                time_diffs = []
                for _, row in track_rows.iterrows():
                    if 'base_time' in row:
                        row_time = pd.Timestamp(row['base_time'])
                        time_diffs.append((abs(row_time - current_time), row))
                
                if time_diffs:
                    time_diffs.sort(key=lambda x: x[0])
                    closest_row = time_diffs[0][1]
                    lat = float(closest_row[latvar]) if latvar in closest_row else np.nan
                    lon = float(closest_row[lonvar]) if lonvar in closest_row else np.nan
                else:
                    lat = float(track_data[latvar]) if latvar in track_data else np.nan
                    lon = float(track_data[lonvar]) if lonvar in track_data else np.nan
            
            # Store position
            track_lats[track_idx, t_idx] = lat
            track_lons[track_idx, t_idx] = lon
            
            # Find this time in the full_data_chunk
            if pd.Timestamp(current_time) in time_to_idx:
                time_idx_in_chunk = time_to_idx[pd.Timestamp(current_time)]
                
                # Process each radius
                for radius in RADII:
                    # Get the area key (should be (track_id, radius))
                    area_key = (track_id, radius)
                    
                    if area_key not in trigger_areas:
                        print(f"  No area key found for track {track_id}, radius {radius}")
                        continue
                        
                    # Get pixels for this area
                    area_pixels = trigger_areas[area_key]
                    
                    # Update mask for this track and radius
                    mask_array = ds_output[f'mask_radius_{radius:.1f}'].values
                    for pixel in area_pixels:
                        if int(pixel) in cell_id_to_index:
                            pixel_idx = cell_id_to_index[int(pixel)]
                            mask_array[track_idx, pixel_idx] = True
                    ds_output[f'mask_radius_{radius:.1f}'].values = mask_array
                    
                    # Extract data for this area at this time directly
                    try:
                        # Get subset of data for this time and these cells
                        area_data = full_data_chunk.sel(time=current_time, cell=area_pixels)
                        
                        # Update the data array for this track, time, and cells
                        for cell_idx, cell_id in enumerate(area_data.cell.values):
                            mapped_idx = cell_id_to_index[int(cell_id)]
                            value = area_data.values[cell_idx]
                            ds_output[f'data_radius_{radius:.1f}'].values[track_idx, t_idx, mapped_idx] = value
                            
                    except Exception as e:
                        print(f"  Error extracting data for track {track_id} at time {current_time}: {e}")
                        continue
            else:
                print(f"  Time {current_time} not found in full_data_chunk")
    
    # Add the time values and offsets as coordinates
    ds_output = ds_output.assign_coords(
        time_values=xr.DataArray(time_values, dims=['track', 'time'],
                                attrs={'description': 'Actual time values'}),
        time_offset_hours=xr.DataArray(time_offsets, dims=['track', 'time'],
                                     attrs={'units': 'hours', 
                                           'description': 'Time offset from initiation (negative=before, positive=after)'})
    )
    
    # Add cell coordinate information - get lat/lon for all cells
    if len(unique_cell_ids) > 0:
        try:
            cell_lats = hp_grid.lat.sel(cell=unique_cell_ids).values
            cell_lons = hp_grid.lon.sel(cell=unique_cell_ids).values 
            
            # Add lat/lon as data variables
            ds_output['cell_lat'] = xr.DataArray(
                cell_lats,
                dims=['cell_id'],
                coords={'cell_id': ds_output['cell_id']},
                attrs={'units': 'degrees_north', 'description': 'Latitude of cell center'}
            )
            
            ds_output['cell_lon'] = xr.DataArray(
                cell_lons,
                dims=['cell_id'],
                coords={'cell_id': ds_output['cell_id']},
                attrs={'units': 'degrees_east', 'description': 'Longitude of cell center'}
            )
        except Exception as e:
            print(f"Error adding cell coordinates: {e}")
            traceback.print_exc()
    
    print(f"Created dataset with {len(track_id_to_idx)} tracks, {max_time_points} max time points")
    
    # Verify data was written correctly
    for var in ds_output.data_vars:
        if 'data_radius' in var:
            data = ds_output[var].values
            non_nan_count = np.sum(~np.isnan(data))
            print(f"Variable {var} has {non_nan_count} non-NaN values out of {data.size}")
    
    return ds_output

def filter_mcs_tracks(subset_mcs_stats, mcs_status_filter="base"):
    """Filter MCS tracks based on different criteria
    
    Parameters:
    -----------
    subset_mcs_stats : xarray.Dataset
        Dataset with MCS track information
    mcs_status_filter : str, default="base"
        Type of filtering to apply:
        - "base": Only filter out tracks that start as a splitter
        - "0": Require times=0 to have mcs_status=0
        - "01": Require times=0,1 to have mcs_status=0
        - "012": Require times=0,1,2 to have mcs_status=0
    
    Returns:
    --------
    xarray.Dataset
        Filtered dataset
    """
    # First, get base tracks (no splitting)
    mcs_tracks_base = subset_mcs_stats.where(
        np.isnan(subset_mcs_stats["start_split_cloudnumber"]), 
        drop=True
    )
    
    if mcs_status_filter == "base":
        print(f"Using base filter (no splitting): {len(mcs_tracks_base.tracks)} tracks")
        return mcs_tracks_base
    
    # Filter based on initial mcs_status=0
    if mcs_status_filter == "0":
        # Select tracks where at least times=0 has mcs_status = 0
        condition = (mcs_tracks_base.mcs_status.isel(times=0) == 0)
        tracks = mcs_tracks_base.tracks.where(condition, drop=True)
        filtered_tracks = mcs_tracks_base.sel(tracks=tracks.values)
        print(f"Filtered to tracks with times=0 having status=0: {len(filtered_tracks.tracks)} tracks")
        return filtered_tracks
    
    elif mcs_status_filter == "01":
        # Select tracks where times=0 AND times=1 have mcs_status = 0
        condition = (
            (mcs_tracks_base.mcs_status.isel(times=0) == 0) & 
            (mcs_tracks_base.mcs_status.isel(times=1) == 0)
        )
        tracks = mcs_tracks_base.tracks.where(condition, drop=True)
        filtered_tracks = mcs_tracks_base.sel(tracks=tracks.values)
        print(f"Filtered to tracks with times=0,1 having status=0: {len(filtered_tracks.tracks)} tracks")
        return filtered_tracks
    
    elif mcs_status_filter == "012":
        # Select tracks where times=0, times=1, AND times=2 have mcs_status = 0
        condition = (
            (mcs_tracks_base.mcs_status.isel(times=0) == 0) &
            (mcs_tracks_base.mcs_status.isel(times=1) == 0) &
            (mcs_tracks_base.mcs_status.isel(times=2) == 0)
        )
        tracks = mcs_tracks_base.tracks.where(condition, drop=True)
        filtered_tracks = mcs_tracks_base.sel(tracks=tracks.values)
        print(f"Filtered to tracks with times=0,1,2 having status=0: {len(filtered_tracks.tracks)} tracks")
        return filtered_tracks
    
    # Default to base filter
    print(f"Unknown filter '{mcs_status_filter}', using base filter: {len(mcs_tracks_base.tracks)} tracks")
    return mcs_tracks_base

def parse_radii(radii_str):
    """Parse radii string from bash to numpy array"""
    try:
        # Get first value only
        radius = float(radii_str.split(',')[0])
        return np.array([radius])
    except:
        # Default if parsing fails
        return np.array([5.0])

def prepare_data(filtered_df, lat_bounds, lonvar, latvar, hp_grid):
    """Prepare the data for processing"""
    print("Preparing data...")
    
    # Calculate the HEALPix indices
    nside = egh.get_nside(hp_grid)
    pixel_indices = hp.ang2pix(
        nside,
        filtered_df[lonvar].values,
        filtered_df[latvar].values,
        nest=True, 
        lonlat=True
    )
    
    filtered_df['trigger_idx'] = pixel_indices
    return filtered_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract full spatial data around MCS locations.')
    
    # Input/output options
    parser.add_argument('--catalog_url', default="https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml",
                        help='URL of the intake catalog')
    parser.add_argument('--current_location', default="NERSC", help='Current location in catalog')
    parser.add_argument('--catalog_model', default="scream_ne120", 
                        help='Model name in the catalog (e.g., scream_ne120)')
    parser.add_argument('--trackfile', required=True, help='Path to MCS track file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--variable', required=True, help='Variable to extract from dataset')
    
    # Date filtering options
    parser.add_argument('--start_date', help='Start date for filtering (YYYY-MM-DD)')
    parser.add_argument('--end_date', help='End date for filtering (YYYY-MM-DD)')
    
    # Spatial filtering options
    parser.add_argument('--min_lon', type=float, default=None, help='Minimum longitude')
    parser.add_argument('--max_lon', type=float, default=None, help='Maximum longitude')
    parser.add_argument('--min_lat', type=float, default=None, help='Minimum latitude')
    parser.add_argument('--max_lat', type=float, default=None, help='Maximum latitude')
    
    # Processing options
    parser.add_argument('--radii', default="5,2", help='Comma-separated list of radii in degrees')
    parser.add_argument('--lat_var', default='meanlat', help='Latitude variable name in tracks')
    parser.add_argument('--lon_var', default='meanlon', help='Longitude variable name in tracks')
    parser.add_argument('--remove_land', action='store_true', help='Filter out track areas that contain land')
    parser.add_argument('--hours_before_init', type=int, default=12, 
                        help='Hours before initiation to extract data')
    parser.add_argument('--include_evolution', action='store_true',
                        help='Include full temporal evolution of MCS')
    parser.add_argument('--mcs_status_filter', default="base", 
                      choices=["base", "0", "01", "012"],
                      help='Filter tracks based on initial mcs_status=0')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for processing')
    parser.add_argument('--max_tracks', type=int, default=None,
                        help='Maximum number of tracks to process (for testing)')
    parser.add_argument('--catalog_params', default='{"zoom": 8}', 
                        help='JSON string of catalog parameters')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    print(f"Starting spatial data extraction for variable: {args.variable}")
    
    # Import modules here to avoid importing unnecessary modules
    import intake
    import json
    
    # Parse RADII from command line
    RADII = parse_radii(args.radii)
    if len(RADII) > 1:
        print("Warning: Only the first radius value will be used")
        RADII = np.array([RADII[0]])
    print(f"Using radii: {RADII}")
    
    # Parse catalog parameters
    try:
        catalog_params = json.loads(args.catalog_params)
    except:
        print(f"Warning: Could not parse catalog_params '{args.catalog_params}'. Using default zoom=8.")
        catalog_params = {'zoom': 8}
    
    # Open catalog and get dataset
    print(f"Opening catalog from {args.catalog_url}")
    cat = intake.open_catalog(args.catalog_url)[args.current_location]
    
    # Set latitude bounds considering RADII.max() if not specified
    if args.min_lat is None:
        min_lat = -90 + RADII.max()
        print(f"No minimum latitude specified. Using {min_lat}")
    else:
        min_lat = args.min_lat
    
    if args.max_lat is None:
        max_lat = 90 - RADII.max()
        print(f"No maximum latitude specified. Using {max_lat}")
    else:
        max_lat = args.max_lat
    
    lat_bounds = (min_lat, max_lat)
    
    # Load dataset
    print(f"Loading dataset {args.catalog_model} from catalog...")
    ds = cat[args.catalog_model](**catalog_params).to_dask().pipe(egh.attach_coords, signed_lon=True)
    ds = ds.assign_coords(time=convert_time(ds.time.values))
    
    # Create spatial filter conditions
    spatial_filters = []
    
    # Add latitude bounds considering RADII.max() buffer
    spatial_filters.append(ds['lat'] > lat_bounds[0] - RADII.max())
    spatial_filters.append(ds['lat'] < lat_bounds[1] + RADII.max())
    
    # Add longitude bounds if specified
    if args.min_lon is not None and args.max_lon is not None:
        print(f"Using longitude bounds: {args.min_lon} to {args.max_lon}")
        # Handle special case crossing the -180/180 boundary
        if args.min_lon > args.max_lon:
            # e.g., min=170, max=-170
            spatial_filters.append((ds['lon'] > args.min_lon - RADII.max()) | 
                                  (ds['lon'] < args.max_lon + RADII.max()))
        else:
            # Normal case
            spatial_filters.append(ds['lon'] > args.min_lon - RADII.max())
            spatial_filters.append(ds['lon'] < args.max_lon + RADII.max())
    
    # Filter dataset
    print("Filtering dataset...")
    if spatial_filters:
        # Start with first filter
        combined_filter = spatial_filters[0]
        # Combine additional filters
        for filter_condition in spatial_filters[1:]:
            combined_filter = combined_filter & filter_condition
        ds_var = ds.where(combined_filter, drop=True)
    else:
        ds_var = ds  # No filters applied

    # Diagnostic check to see what's available in the dataset
    print(f"\nDIAGNOSTIC - Dataset Info for {args.variable}")
    if args.variable in ds:
        var = ds[args.variable]
        sample_data = var.isel(time=0).head()
        print(f"Variable shape: {var.shape}")
        print(f"Variable dimensions: {var.dims}")
        print(f"Time values range: {var.time.values[0]} to {var.time.values[-1]}")
        print(f"Sample data available: {not np.all(np.isnan(sample_data.values))}")
        # Try to compute a sample
        try:
            test_value = var.isel(time=0, cell=slice(0, 10)).compute()
            print(f"Sample computed values (first 3): {test_value.values.flatten()[:3]}")
        except Exception as e:
            print(f"Error computing sample data: {e}")
    else:
        print(f"WARNING: Variable '{args.variable}' not found in dataset!")
        print(f"Available variables: {list(ds.data_vars)}")

    # Get the lat/lon coordinates of the healpix grid
    print("Computing HEALPix grid...")
    hp_grid = ds[['lat', 'lon']].compute()
    
    # Get land-sea-mask if needed
    ocean_mask = None
    if args.remove_land:
        print("Computing ocean mask...")
        lf = ds['LANDFRAC']
        ocean_mask = lf.where(lf==0).compute()
    
    # Load MCS track data
    print(f"Loading MCS track data from {args.trackfile}")
    mcs_trackstats = xr.open_dataset(args.trackfile)
    
    # Subsample relevant information
    subset_mcs_stats = mcs_trackstats[
        ['start_split_cloudnumber', 'start_basetime', 'base_time', 'meanlon', 'meanlat','mcs_status',
         'mcs_duration', 'meanlon_smooth', 'meanlat_smooth']
    ].compute()
    
    # Select tracks based on conditions (those that don't start as a splitter and have at least one initial mcs_status=0)
    mcs_tracks_triggered = filter_mcs_tracks(subset_mcs_stats, args.mcs_status_filter)
    
    # Use lat/lon variable names from command line
    latvar = args.lat_var
    lonvar = args.lon_var
    
    # Save start location of tracks
    mcs_tracks_triggered['start_lat'] = mcs_tracks_triggered[latvar].isel(times=0)
    mcs_tracks_triggered['start_lon'] = mcs_tracks_triggered[lonvar].isel(times=0)
    
    # Convert to DataFrame and filter
    df = mcs_tracks_triggered.to_dataframe()
    
    # Apply filters with pandas
    filter_conditions = [
        (df['start_lat'] > lat_bounds[0]),
        (df['start_lat'] < lat_bounds[1]),
        (df[latvar] > lat_bounds[0]),
        (df[latvar] < lat_bounds[1]),
        df[latvar].notna(),
        df[lonvar].notna()
    ]
    
    # Add date filters if provided
    if args.start_date:
        start_date = pd.Timestamp(args.start_date)
        filter_conditions.append(pd.to_datetime(df['base_time']) >= start_date)
    
    if args.end_date:
        end_date = pd.Timestamp(args.end_date)
        filter_conditions.append(pd.to_datetime(df['base_time']) <= end_date)
    
    # Add longitude filters if provided
    if args.min_lon is not None and args.max_lon is not None:
        # Handle special case crossing the -180/180 boundary
        if args.min_lon > args.max_lon:
            filter_conditions.append((df[lonvar] >= args.min_lon) | (df[lonvar] <= args.max_lon))
        else:
            filter_conditions.append(df[lonvar] >= args.min_lon)
            filter_conditions.append(df[lonvar] <= args.max_lon)
    
    # Apply all filters
    filtered_df = df[np.logical_and.reduce(filter_conditions)].copy()
    
    print(f"Filtered to {len(filtered_df)} tracks")

    # Limit tracks for testing if specified
    if args.max_tracks is not None and args.max_tracks > 0:
        # Get unique tracks first, then limit
        unique_track_ids = filtered_df.index.get_level_values('tracks').unique()
        if len(unique_track_ids) > args.max_tracks:
            unique_track_ids = unique_track_ids[:args.max_tracks]
        
        # Keep only rows for these tracks
        filtered_df = filtered_df.loc[filtered_df.index.get_level_values('tracks').isin(unique_track_ids)]
        print(f"Limited to {len(unique_track_ids)} unique tracks ({len(filtered_df)} rows)")

    # Add HEALPix indices to filtered_df
    filtered_df = prepare_data(filtered_df, lat_bounds, lonvar, latvar, hp_grid)
    
    # Calculate circular areas around triggers
    print("Calculating circular areas...")
    trigger_areas, filtered_df = add_circular_trigger_areas_df(
        filtered_df,
        RADII,
        hp_grid,
        ocean_mask=ocean_mask,
        remove_land=args.remove_land
    )
    
    # Extract spatial data
    print(f"Extracting spatial data for variable {args.variable}...")
    times_before_init = pd.Timedelta(f'{args.hours_before_init}H') if args.hours_before_init > 0 else None
    
    spatial_data = extract_spatial_data(
        trigger_areas,
        filtered_df,
        ds_var[args.variable],
        RADII,
        hp_grid,  # Pass hp_grid here - this was missing
        time_tolerance=pd.Timedelta('1H'),
        times_before_init=times_before_init,
        include_full_evolution=args.include_evolution
    )
    
    # Create output filename
    start_date_str = args.start_date.replace('-', '') if args.start_date else "unknown"
    end_date_str = args.end_date.replace('-', '') if args.end_date else "unknown"
    output_file = os.path.join(
        args.output_dir,
        f"{args.variable}_spatial_{start_date_str}_{end_date_str}_mcs{args.mcs_status_filter}.nc"
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save results
    print(f"Saving spatial data to {output_file}...")
    # Ensure dataset is computed before saving
    print("Computing final dataset...")
    spatial_data = spatial_data.compute()

    # Final verification of data before saving
    for var in spatial_data.data_vars:
        if 'data_radius' in var:
            data = spatial_data[var].values
            non_nan_count = np.sum(~np.isnan(data))
            print(f"Variable {var} has {non_nan_count} non-NaN values out of {data.size}")

    # Save the dataset
    spatial_data.to_netcdf(output_file)

    # Verify data was written correctly
    print("Verifying saved data...")
    try:
        test_ds = xr.open_dataset(output_file)
        for var in test_ds.data_vars:
            if 'data_radius' in var:
                has_data = not np.all(np.isnan(test_ds[var].values))
                print(f"Variable {var} has data: {has_data}")
                if has_data:
                    print(f"First valid value: {np.nanmean(test_ds[var].values)}")
    except Exception as e:
        print(f"Error verifying data: {e}")
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"Extraction completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
if __name__ == "__main__":
    main()