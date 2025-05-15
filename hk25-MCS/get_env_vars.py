# Author: Laura Paccini (laura.paccini@pnnl.gov)
# Date: May 15, 2025
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
        #print(f"Processing batch {batch_start//batch_size + 1}/{(total_tracks//batch_size) + 1}")
        
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
                
                # Store result
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
                        except:
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
    
    # Filter DataFrame if requested
    if remove_land:
        filtered_ocean_df = filtered_df.loc[list(ocean_tracks)].copy()
        print(f"Filtered from {len(filtered_df)} to {len(filtered_ocean_df)} ocean-only tracks")
        return trigger_areas, filtered_ocean_df
    
    return trigger_areas, filtered_df

def extract_var_statistics_fast(trigger_areas, filtered_df, ds_variable, RADII, time_tolerance=pd.Timedelta('1H'), 
                              times_before_init=None, include_full_evolution=False):
    """Ultra-optimized function that extracts only summary statistics for each area
    
    Parameters:
    -----------
    trigger_areas : dict
        Dictionary mapping (track_idx, radius) to pixel arrays
    filtered_df : pandas.DataFrame
        DataFrame with MCS track information
    ds_variable : xarray.DataArray
        Data variable to extract statistics from
    RADII : np.ndarray
        Array of radii in degrees
    time_tolerance : pd.Timedelta, optional
        Tolerance for matching times (default: 1 hour)
    times_before_init : pd.Timedelta, optional
        If provided, extract data for this time period before each track's start time (initiation)
    include_full_evolution : bool, optional
        If True, include all times in the track's lifecycle after initiation (default: False)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing statistics for each track, radius, and time offset
    """
    results = []
    
    # Pre-calculate available times once (huge speedup)
    available_times = pd.DatetimeIndex(ds_variable.time.values)
    
    # Process in larger batches
    batch_size = 250
    
    # Extract the unique track IDs (ignoring time indices in case tracks are tuples)
    if isinstance(filtered_df.index[0], tuple):
        # If index is a tuple (track_id, time_idx), extract just the track IDs
        unique_track_ids = sorted(set([idx[0] for idx in filtered_df.index]))
    else:
        # Otherwise use the indices directly
        unique_track_ids = filtered_df.index.unique()
        
    total_tracks = len(unique_track_ids)
    
    for batch_start in range(0, total_tracks, batch_size):
        batch_end = min(batch_start + batch_size, total_tracks)
        batch_track_ids = unique_track_ids[batch_start:batch_end]
        
        #print(f"Processing batch {batch_start//batch_size + 1}/{(total_tracks//batch_size) + 1}")
        
        # Create a dictionary to store all pixel indices for this batch
        all_pixels = {}
        track_times_map = {}
        
        # First gather all pixels and times needed for this batch
        for track_id in batch_track_ids:
            # Get all rows for this track ID (regardless of time index)
            if isinstance(filtered_df.index[0], tuple):
                # Find all rows where the first element of the index tuple matches track_id
                track_rows = filtered_df.loc[[idx for idx in filtered_df.index if idx[0] == track_id]]
                track_data = track_rows.iloc[0]  # Use first row for reference info
                
                # If we need full evolution, gather all base_times for this track
                if include_full_evolution:
                    track_base_times = pd.DatetimeIndex([pd.Timestamp(row['base_time']) for _, row in track_rows.iterrows()])
            else:
                track_data = filtered_df.loc[track_id]
                
                # For non-tuple indices, we don't have multiple times per track
                if include_full_evolution:
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
                # Get all available times from the track's lifecycle
                if isinstance(filtered_df.index[0], tuple):
                    # For tuple indices, we need to find all timestamps in the dataset
                    # that match any of this track's base_times (within tolerance)
                    for base_time in track_base_times:
                        if time_tolerance is not None:
                            # Find closest time within tolerance
                            closest_idx = np.abs(available_times - base_time).argmin()
                            if abs(available_times[closest_idx] - base_time) <= time_tolerance:
                                analysis_times.append(available_times[closest_idx])
                        elif base_time in available_times:
                            analysis_times.append(base_time)
                else:
                    # For single index, just add the base_time
                    base_time = pd.Timestamp(track_data['base_time'])
                    if time_tolerance is not None:
                        closest_idx = np.abs(available_times - base_time).argmin()
                        if abs(available_times[closest_idx] - base_time) <= time_tolerance:
                            analysis_times.append(available_times[closest_idx])
                    elif base_time in available_times:
                        analysis_times.append(base_time)
            
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
            track_times_map[track_id] = pd.DatetimeIndex(sorted(set(analysis_times)))
            
            # Collect all pixels for each radius
            for radius in RADII:
                # Try different key formats depending on how trigger_areas is structured
                area_keys_to_try = [
                    (track_id, radius),  # Simple (track_id, radius) key
                    ((track_id, 0), radius)  # (track_id, time_idx=0, radius) key
                ]
                
                # Try each possible key format
                for area_key in area_keys_to_try:
                    if area_key in trigger_areas:
                        pixels = trigger_areas[area_key]
                        all_pixels[area_key] = pixels
                        break
        
        if not all_pixels:
            print("  No valid areas found in this batch")
            continue
            
        # Now fetch all data at once for this batch
        all_track_pixels = np.concatenate(list(all_pixels.values()))
        all_track_times = []
        for times in track_times_map.values():
            all_track_times.extend(times)
        all_track_times = list(set(all_track_times))
        
        # Get all data in one big selection
        try:
            data_chunk = ds_variable.sel(
                time=all_track_times,
                cell=np.unique(all_track_pixels)
            ).compute()
        except Exception as e:
            print(f"  Error fetching data: {e}")
            continue
        
        # Process statistics for each track, radius, and time
        for track_id in batch_track_ids:
            if track_id not in track_times_map:
                continue
                
            # Get all times for this track
            track_times = track_times_map[track_id]
            
            # Get track data (use first row if multiple exist)
            if isinstance(filtered_df.index[0], tuple):
                track_indices = [idx for idx in filtered_df.index if idx[0] == track_id]
                track_data = filtered_df.loc[track_indices[0]]
            else:
                track_data = filtered_df.loc[track_id]
            
            # Use start_basetime as the reference time for all time offsets
            reference_time = pd.Timestamp(track_data['start_basetime'])
            
            for radius in RADII:
                # Try different key formats
                area_key = None
                for key_to_try in [(track_id, radius), ((track_id, 0), radius)]:
                    if key_to_try in all_pixels:
                        area_key = key_to_try
                        break
                        
                if area_key is None:
                    continue
                    
                pixels = all_pixels[area_key]
                
                # Process each time point in the range
                for current_time in track_times:
                    # Calculate time offset in hours (negative = before init, positive = after init)
                    time_offset = (current_time - reference_time).total_seconds() / 3600
                    
                    # Extract just this area's data for this time
                    try:
                        area_data = data_chunk.sel(time=current_time, cell=pixels)
                        
                        # Calculate statistics
                        stats = {
                            'track': track_id,  # Store just the track ID, not the tuple
                            'radius': radius,
                            'matched_time': current_time,
                            'reference_time': reference_time,
                            'time_offset_hours': time_offset,
                            'mean': float(area_data.mean().values),
                            'median': float(np.nanmedian(area_data.values)),
                            'min': float(area_data.min().values),
                            'max': float(area_data.max().values),
                            'std': float(area_data.std().values),
                            'count': len(pixels),
                            'valid_count': int(np.sum(~np.isnan(area_data.values)))
                        }
                        results.append(stats)
                    except Exception as e:
                        print(f"  Error processing area {area_key} at time {current_time}: {e}")
                        continue
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    if len(result_df) > 0:
        # Add a column for relative time index (negative = before init, 0 = init, positive = after init)
        result_df['time_index'] = result_df['time_offset_hours'].round().astype(int)
    
    print(f"Processed {len(result_df)} track-radius-time combinations")
    return result_df

def save_results(result_df, output_path, format='netcdf'):
    """Save the results to a file in the specified format
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with the results
    output_path : str
        Path where to save the results
    format : str
        Format to use for saving the results ('parquet', 'csv', or 'netcdf')
    
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format.lower() == 'parquet':
        if not output_path.endswith('.parquet'):
            output_path += '.parquet'
        result_df.to_parquet(output_path, index=False)
    elif format.lower() == 'csv':
        if not output_path.endswith('.csv'):
            output_path += '.csv'
        result_df.to_csv(output_path, index=False)
    elif format.lower() == 'netcdf':
        if not output_path.endswith('.nc'):
            output_path += '.nc'
        # Convert DataFrame to xarray Dataset
        result_ds = result_df.to_xarray()
        result_ds.to_netcdf(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet', 'csv', or 'netcdf'")
    
    print(f"Results saved to {output_path}")
    return output_path

def prepare_data(filtered_df, lat_bounds, lonvar, latvar, hp_grid):
    """Prepare the data for processing"""
    print("Preparing data...")
    
    # Calculate the HEALPix indices
    nside = egh.get_nside(hp_grid)
    pixel_indices_smooth = hp.ang2pix(
        nside,
        filtered_df[lonvar].values,
        filtered_df[latvar].values,
        nest=True, 
        lonlat=True
    )
    
    filtered_df['trigger_idx'] = pixel_indices_smooth
    return filtered_df

def parse_radii(radii_str):
    """Parse radii string from bash to numpy array"""
    try:
        # Format example: "5,3.5,2"
        return np.array([float(r) for r in radii_str.split(',')])
    except:
        # Default if parsing fails
        return np.arange(5, 0, -1.5)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract environmental variables around MCS locations.')
    
    # Input/output options
    parser.add_argument('--catalog_url', default="https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml",
                        help='URL of the intake catalog')
    parser.add_argument('--current_location', default="NERSC", help='Current location in catalog')
    parser.add_argument('--catalog_model', default="scream_ne120", 
                        help='Model name in the catalog (e.g., scream_ne120)')
    parser.add_argument('--trackfile', required=True, help='Path to MCS track file')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--output_format', default='netcdf', choices=['parquet', 'csv', 'netcdf'],
                        help='Format to save results (parquet, csv, or netcdf)')
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
    parser.add_argument('--radii', default="5,3.5,2", help='Comma-separated list of radii in degrees')
    parser.add_argument('--lat_var', default='meanlat', help='Latitude variable name in tracks')
    parser.add_argument('--lon_var', default='meanlon', help='Longitude variable name in tracks')
    parser.add_argument('--remove_land', action='store_true', help='Filter out track areas that contain land')
    parser.add_argument('--hours_before_init', type=int, default=24, 
                        help='Hours before initiation to extract data')
    parser.add_argument('--include_evolution', action='store_true',
                        help='Include full temporal evolution of MCS')
    parser.add_argument('--batch_size', type=int, default=250,
                        help='Batch size for processing')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--catalog_params', default='{"zoom": 8}', 
                        help='JSON string of catalog parameters')
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    print(f"Starting environmental variables extraction for variable: {args.variable}")
    
    # Import modules here to avoid importing unnecessary modules
    import intake
    import json
    
    # Parse RADII from command line
    RADII = parse_radii(args.radii)
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
        ['start_split_cloudnumber', 'start_basetime', 'base_time', 'meanlon', 'meanlat',
         'mcs_duration', 'meanlon_smooth', 'meanlat_smooth']
    ].compute()
    
    # Select tracks that don't start as a splitter
    mcs_tracks_triggered = subset_mcs_stats.where(
        np.isnan(subset_mcs_stats["start_split_cloudnumber"]), drop=True
    )
    
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
    
    # Extract variable statistics
    print(f"Extracting statistics for variable {args.variable}...")
    times_before_init = pd.Timedelta(f'{args.hours_before_init}H') if args.hours_before_init > 0 else None
    
    var_stats = extract_var_statistics_fast(
        trigger_areas,
        filtered_df,
        ds_var[args.variable],
        RADII,
        time_tolerance=pd.Timedelta('1H'),
        times_before_init=times_before_init,
        include_full_evolution=args.include_evolution
    )
    
    # Create output filename
    start_date_str = args.start_date.replace('-', '') if args.start_date else "unknown"
    end_date_str = args.end_date.replace('-', '') if args.end_date else "unknown"
    output_file = os.path.join(
    args.output_dir,
    f"{args.variable}_stats_{start_date_str}_{end_date_str}"
)
    
    # Save results
    save_results(var_stats, output_file, format=args.output_format)
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"Extraction completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
if __name__ == "__main__":
    main()