"""Functions for computing freshwater runoff from glaciers.

This module contains functions to calculate and analyze freshwater
runoff from glaciers, with a specific focus on Greenland's peripheral
glaciers.
"""

# Built-ins
import logging
import os
import copy
from functools import partial

# External libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage import gaussian_filter1d

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)


def _get_peak_water(runoff_series, window=11):
    """Identify peak water in a runoff time series.
    
    Parameters
    ----------
    runoff_series : pd.Series or np.ndarray
        The annual runoff time series
    window : int
        Window size for rolling mean (default: 11 years)
        
    Returns
    -------
    peak_year : int or None
        The year of peak water, or None if no peak is found
    smoothed_series : pd.Series
        The smoothed runoff series
    """
    if isinstance(runoff_series, np.ndarray):
        runoff_series = pd.Series(runoff_series)
    
    # Apply rolling mean to reduce interannual variability
    smoothed_series = runoff_series.rolling(window=window, center=True).mean()
    
    # Find the year of maximum runoff after smoothing
    if np.isnan(smoothed_series).all():
        return None, smoothed_series
    
    # Get peak year from the smoothed series
    peak_idx = smoothed_series.argmax()
    peak_year = runoff_series.index[peak_idx]
    
    # Only return a peak if it's not at the edges of the time series
    if peak_idx == 0 or peak_idx == len(smoothed_series) - 1:
        return None, smoothed_series
    
    return peak_year, smoothed_series


@entity_task(log)
def compute_glacier_runoff(gdir, year=None, fl_id=0):
    """Compute the total freshwater runoff from a glacier.
    
    This function calculates the total freshwater runoff including:
    - On-glacier ice melt
    - On-glacier snow melt
    - On-glacier liquid precipitation
    - Off-glacier snow melt
    - Off-glacier liquid precipitation
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    year : int, optional
        the year to compute the runoff for. If None, compute for all years
    fl_id : int
        the flowline ID to compute the runoff for
        
    Returns
    -------
    runoff : float or pd.Series
        The total annual runoff in m³ water equivalent
    """
    # Check if we have the necessary files
    if not os.path.exists(gdir.get_filepath('model_diagnostics')):
        raise InvalidWorkflowError('Run a model simulation first!')
    
    with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
        # Get necessary variables
        time = ds.time.values
        years = pd.DatetimeIndex(time).year
        
        # Check if we have the necessary variables
        if 'runoff' not in ds:
            # Need to compute runoff from components
            
            # 1. On-glacier components
            glacier_ice_melt = ds.get('ice_flux', 0)  # Ice melt
            glacier_snow_melt = ds.get('snow_melt', 0)  # Snow melt
            glacier_rain = ds.get('liq_prcp_on_glacier', 0)  # Liquid precipitation on glacier
            
            # 2. Off-glacier components (in deglaciated areas)
            # Areas where the glacier has retreated or disappeared
            offglacier_snow_melt = ds.get('offglacier_snow_melt', 0)
            offglacier_rain = ds.get('offglacier_liq_prcp', 0)
            
            # Sum all components
            total_runoff = (glacier_ice_melt + glacier_snow_melt + glacier_rain + 
                           offglacier_snow_melt + offglacier_rain)
        else:
            # Runoff already computed
            total_runoff = ds.runoff
        
        # Convert to m³ water equivalent (if needed)
        if total_runoff.attrs.get('units', '') == 'kg':
            # Convert kg to m³ w.e.
            total_runoff = total_runoff / 1000
        
        # Return value for specific year or entire series
        if year is not None:
            if year not in years:
                raise ValueError(f'Year {year} not in model simulation')
            idx = np.where(years == year)[0][0]
            return float(total_runoff.values[idx])
        else:
            # Return the entire series
            return pd.Series(total_runoff.values, index=years)


@entity_task(log)
def analyze_peak_water(gdir, window=11, return_series=False):
    """Analyze the peak water for a glacier.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    window : int
        Window size for rolling mean (default: 11 years)
    return_series : bool
        Whether to return the full runoff series
        
    Returns
    -------
    dict
        A dictionary containing peak water information:
        - peak_year: year of peak water
        - peak_runoff: runoff at peak water (m³ w.e.)
        - runoff_series: full runoff series (if return_series=True)
    """
    # Get runoff series
    runoff_series = compute_glacier_runoff(gdir)
    
    # Find peak water
    peak_year, smoothed_series = _get_peak_water(runoff_series, window=window)
    
    result = {
        'peak_year': peak_year,
        'has_peaked': peak_year is not None
    }
    
    if peak_year is not None:
        result['peak_runoff'] = float(smoothed_series.loc[peak_year])
    
    if return_series:
        result['runoff_series'] = runoff_series
        result['smoothed_series'] = smoothed_series
    
    return result


@entity_task(log)
def compute_runoff_components(gdir, year=None):
    """Compute detailed runoff components for a glacier.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    year : int, optional
        the year to compute the runoff for. If None, compute for all years
        
    Returns
    -------
    dict or pd.DataFrame
        A dictionary or DataFrame with runoff components
    """
    # Check if we have the necessary files
    if not os.path.exists(gdir.get_filepath('model_diagnostics')):
        raise InvalidWorkflowError('Run a model simulation first!')
    
    with xr.open_dataset(gdir.get_filepath('model_diagnostics')) as ds:
        # Get time information
        time = ds.time.values
        years = pd.DatetimeIndex(time).year
        
        # Extract runoff components
        components = {
            'glacier_ice_melt': ds.get('ice_flux', np.zeros_like(time)).values,
            'glacier_snow_melt': ds.get('snow_melt', np.zeros_like(time)).values,
            'glacier_rain': ds.get('liq_prcp_on_glacier', np.zeros_like(time)).values,
            'offglacier_snow_melt': ds.get('offglacier_snow_melt', np.zeros_like(time)).values,
            'offglacier_rain': ds.get('offglacier_liq_prcp', np.zeros_like(time)).values
        }
        
        # Calculate total runoff
        components['total_runoff'] = (
            components['glacier_ice_melt'] + 
            components['glacier_snow_melt'] + 
            components['glacier_rain'] + 
            components['offglacier_snow_melt'] + 
            components['offglacier_rain']
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(components, index=years)
        
        # Return value for specific year or entire series
        if year is not None:
            if year not in years:
                raise ValueError(f'Year {year} not in model simulation')
            return df.loc[year].to_dict()
        else:
            return df
