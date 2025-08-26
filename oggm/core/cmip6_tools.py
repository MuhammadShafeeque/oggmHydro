"""Functions for climate data processing and bias correction.

This module provides utilities for processing climate data for OGGM, 
including bias correction of GCM data using the delta method.
"""

# Built-ins
import logging
import os
import warnings

# External libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def apply_delta_method(gdir, gcm_filename, reference_filename=None, 
                      reference_period=(1981, 2020), 
                      variables=None):
    """Apply delta method for bias correction of GCM data.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    gcm_filename : str
        path to the GCM file
    reference_filename : str, optional
        path to the reference file (e.g., ERA5)
        if None, uses the default climate file in the GDir
    reference_period : tuple
        reference period for bias correction (default: 1981-2020)
    variables : list
        variables to bias correct (default: ['temp', 'prcp'])
        
    Returns
    -------
    xr.Dataset
        bias-corrected GCM data
    """
    # Default variables
    if variables is None:
        variables = ['temp', 'prcp']
    
    # Load GCM data
    with xr.open_dataset(gcm_filename) as ds_gcm:
        # Get reference data
        if reference_filename is None:
            reference_filename = gdir.get_filepath('climate_historical')
        
        with xr.open_dataset(reference_filename) as ds_ref:
            # Ensure variables exist
            for var in variables:
                if var not in ds_gcm or var not in ds_ref:
                    raise InvalidParamsError(f"Variable {var} not found in datasets")
            
            # Extract reference period data
            ref_start, ref_end = reference_period
            ds_ref_period = ds_ref.sel(time=slice(f'{ref_start}-01-01', f'{ref_end}-12-31'))
            
            # Calculate monthly means for the reference period
            ref_monthly = ds_ref_period.groupby('time.month').mean()
            
            # Calculate reference climatology for the GCM
            # Try to get the same reference period from GCM
            try:
                ds_gcm_period = ds_gcm.sel(time=slice(f'{ref_start}-01-01', f'{ref_end}-12-31'))
                gcm_monthly = ds_gcm_period.groupby('time.month').mean()
            except (KeyError, ValueError):
                # If the reference period doesn't exist in GCM, use a fallback period
                log.warning(f"Reference period {ref_start}-{ref_end} not found in GCM data")
                log.warning("Using available GCM data for calculating reference climatology")
                # Use first 30 years of GCM data as reference
                first_year = pd.to_datetime(ds_gcm.time.values[0]).year
                gcm_monthly = ds_gcm.sel(
                    time=slice(f'{first_year}-01-01', f'{first_year+29}-12-31')
                ).groupby('time.month').mean()
            
            # Create output dataset
            ds_corrected = ds_gcm.copy()
            
            # Apply delta method for each variable
            for var in variables:
                if var == 'temp':
                    # For temperature: additive correction
                    # Calculate monthly temperature bias
                    temp_bias = ref_monthly[var] - gcm_monthly[var]
                    
                    # Apply correction to each month
                    for month in range(1, 13):
                        month_idx = ds_gcm['time.month'] == month
                        ds_corrected[var].loc[{'time': month_idx}] += float(temp_bias.sel(month=month))
                
                elif var == 'prcp':
                    # For precipitation: multiplicative correction
                    # Calculate monthly precipitation ratio
                    # Add small epsilon to avoid division by zero
                    epsilon = 1e-6
                    prcp_ratio = (ref_monthly[var] + epsilon) / (gcm_monthly[var] + epsilon)
                    
                    # Apply correction to each month
                    for month in range(1, 13):
                        month_idx = ds_gcm['time.month'] == month
                        ds_corrected[var].loc[{'time': month_idx}] *= float(prcp_ratio.sel(month=month))
                        
                        # Ensure no negative precipitation
                        ds_corrected[var] = ds_corrected[var].clip(min=0)
                
                else:
                    # For other variables: use additive correction as default
                    var_bias = ref_monthly[var] - gcm_monthly[var]
                    
                    for month in range(1, 13):
                        month_idx = ds_gcm['time.month'] == month
                        ds_corrected[var].loc[{'time': month_idx}] += float(var_bias.sel(month=month))
            
            # Add metadata about bias correction
            ds_corrected.attrs['bias_correction'] = 'delta_method'
            ds_corrected.attrs['reference_period'] = f'{ref_start}-{ref_end}'
            ds_corrected.attrs['reference_dataset'] = os.path.basename(reference_filename)
            
            return ds_corrected


@entity_task(log)
def process_cmip6_data(gdir, gcm_filenames, scenario, 
                      reference_filename=None,
                      reference_period=(1981, 2020),
                      output_dir=None):
    """Process CMIP6 data for OGGM.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    gcm_filenames : dict
        dictionary of GCM filenames {gcm_name: filename}
    scenario : str
        SSP scenario (e.g., 'ssp126', 'ssp245', 'ssp370', 'ssp585')
    reference_filename : str, optional
        path to the reference file (e.g., ERA5)
        if None, uses the default climate file in the GDir
    reference_period : tuple
        reference period for bias correction (default: 1981-2020)
    output_dir : str, optional
        directory to save processed files
        if None, uses the GDir's working directory
        
    Returns
    -------
    dict
        dictionary of processed filenames {gcm_name: filename}
    """
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(gdir.dir, 'climate')
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each GCM
    processed_files = {}
    
    for gcm_name, filename in gcm_filenames.items():
        # Apply bias correction
        ds_corrected = apply_delta_method(
            gdir, filename, reference_filename, reference_period
        )
        
        # Save processed file
        output_filename = os.path.join(
            output_dir, 
            f'{gdir.rgi_id}_{gcm_name}_{scenario}_bias_corrected.nc'
        )
        
        ds_corrected.to_netcdf(output_filename)
        
        processed_files[gcm_name] = output_filename
        
    return processed_files


def extract_cmip6_grid_point(ds, lon, lat, method='nearest'):
    """Extract CMIP6 data for a specific grid point.
    
    Parameters
    ----------
    ds : xr.Dataset
        CMIP6 dataset
    lon : float
        longitude of the grid point
    lat : float
        latitude of the grid point
    method : str
        method for selecting grid point ('nearest' or 'linear')
        
    Returns
    -------
    xr.Dataset
        CMIP6 data for the grid point
    """
    # Check which coordinate system is used
    if 'lon' in ds.dims and 'lat' in ds.dims:
        # Regular grid
        return ds.sel(lon=lon, lat=lat, method=method)
    
    elif 'longitude' in ds.dims and 'latitude' in ds.dims:
        # Some CMIP6 models use these names
        return ds.sel(longitude=lon, latitude=lat, method=method)
    
    elif 'x' in ds.dims and 'y' in ds.dims:
        # Some models might use x, y coordinates
        # Would need to project to lat/lon first
        raise NotImplementedError("Extraction from x,y coordinates not implemented")
    
    else:
        # Try to find suitable coordinates
        lon_vars = [v for v in ds.variables if 'lon' in v.lower()]
        lat_vars = [v for v in ds.variables if 'lat' in v.lower()]
        
        if lon_vars and lat_vars:
            lon_var = lon_vars[0]
            lat_var = lat_vars[0]
            return ds.sel({lon_var: lon, lat_var: lat}, method=method)
        
        else:
            raise ValueError("Could not find longitude/latitude coordinates in dataset")
