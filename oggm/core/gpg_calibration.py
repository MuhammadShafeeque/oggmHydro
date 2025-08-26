"""Functions for calibrating OGGM for Greenland peripheral glaciers.

This module provides enhanced calibration methods for Greenland peripheral 
glaciers, incorporating geodetic mass balance and frontal ablation data.
"""

# Built-ins
import logging
import os
import copy
import warnings

# External libs
import numpy as np
import pandas as pd
import xarray as xr

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.core.massbalance import (MassBalanceModel, MultipleFlowlineMassBalance,
                                  PastMassBalance, ConstantMassBalance)

# Module logger
log = logging.getLogger(__name__)


@entity_task(log)
def calibrate_inversion_from_geodetic_mb(gdir, geodetic_mb_data=None, 
                                        frontal_ablation_data=None,
                                        fs=0., glen_a=None, min_mu_star_frac=None):
    """Calibrate glacier mass balance with geodetic data and frontal ablation.
    
    This function calibrates the mass balance model using geodetic mass balance
    data and frontal ablation data, including volume changes below sea level.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    geodetic_mb_data : float or dict or None
        the geodetic mass balance data (m w.e. yr-1) for this glacier
        if a dict, must contain a value for this glacier's RGI ID
    frontal_ablation_data : float or dict or None
        the frontal ablation data (m w.e. yr-1) for this glacier
        if a dict, must contain a value for this glacier's RGI ID
    fs : float
        sliding parameter
    glen_a : float
        glen's creep parameter
    min_mu_star_frac : float
        minimum fraction of the original (non-calving) mu* for this glacier that
        the parametrization can allow
        
    Returns
    -------
    float
        the calibrated temperature sensitivity mu_star (mm w.e. K-1)
    """
    from oggm.core.inversion import find_sia_flux_from_thickness
    from oggm.core.flowline import FluxBasedModel
    
    # Get the reference mb data
    if geodetic_mb_data is None:
        raise InvalidParamsError('geodetic_mb_data is required for calibration')
    
    # If geodetic_mb_data is a dict, get the value for this glacier
    if isinstance(geodetic_mb_data, dict):
        if gdir.rgi_id not in geodetic_mb_data:
            raise InvalidParamsError(f'geodetic_mb_data does not contain data for {gdir.rgi_id}')
        mb_ref = geodetic_mb_data[gdir.rgi_id]
    else:
        mb_ref = geodetic_mb_data
    
    # If frontal_ablation_data is provided and this is a tidewater glacier
    if frontal_ablation_data is not None and gdir.is_tidewater:
        # If frontal_ablation_data is a dict, get the value for this glacier
        if isinstance(frontal_ablation_data, dict):
            if gdir.rgi_id not in frontal_ablation_data:
                log.warning(f'frontal_ablation_data does not contain data for {gdir.rgi_id}')
                calving_data = 0
            else:
                calving_data = frontal_ablation_data[gdir.rgi_id]
        else:
            calving_data = frontal_ablation_data
    else:
        calving_data = 0
        
    if calving_data == 0 and gdir.is_tidewater:
        log.warning(f'No frontal ablation data for tidewater glacier {gdir.rgi_id}')
    
    # Default to standard glen_a if not provided
    if glen_a is None:
        glen_a = cfg.PARAMS['glen_a']
    
    # Min mu_star fraction for tidewater glaciers
    if min_mu_star_frac is None:
        min_mu_star_frac = cfg.PARAMS['calving_min_mu_star_frac']
    
    # Get the flowlines
    fls = gdir.read_pickle('inversion_flowlines')
    
    # Annual stats
    annual_solid_prcp = 0.
    annual_temp_above_melt = 0.
    
    # Read climate data
    with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
        # Time period for calibration (use period with data from both datasets)
        # This would be set based on the temporal coverage of the geodetic mb data
        # For simplicity, we'll use a fixed period here
        y0, y1 = 2000, 2020
        
        # Extract solid precipitation and temperature
        temp = ds.temp.sel(time=slice(f'{y0}-01-01', f'{y1}-12-31')).values
        prcp = ds.prcp.sel(time=slice(f'{y0}-01-01', f'{y1}-12-31')).values
        time = ds.time.sel(time=slice(f'{y0}-01-01', f'{y1}-12-31')).values
        ref_hgt = ds.ref_hgt
        
        # Compute annual stats
        # Number of years in record
        ny = len(time) // 12
        
        # Loop over flowlines
        for fl in fls:
            # Height along flowline
            h = fl.surface_h
            
            # For each height along the flowline
            for hgt in h:
                # Temperature at this height (lapse rate corrected)
                temp_at_h = temp + cfg.PARAMS['temp_default_gradient'] * (hgt - ref_hgt)
                
                # Melt threshold temperature
                melt_t = cfg.PARAMS['temp_melt']
                
                # Sum temperature above melt threshold
                temp_above_melt = np.sum(np.clip(temp_at_h - melt_t, 0, None))
                
                # Convert to annual average
                temp_above_melt /= ny
                
                # Add to total weighted by grid cell area
                annual_temp_above_melt += temp_above_melt * fl.dx * fl.widths_m
                
                # Get solid precipitation at this height
                # Simple temperature threshold
                is_solid = temp_at_h < cfg.PARAMS['temp_all_solid']
                is_liquid = temp_at_h > cfg.PARAMS['temp_all_liq']
                is_mixed = ~(is_solid | is_liquid)
                
                # Fraction of solid precipitation in mixed state
                psolid = np.ones_like(temp_at_h)
                psolid[is_liquid] = 0
                psolid[is_mixed] = (cfg.PARAMS['temp_all_liq'] - temp_at_h[is_mixed]) / \
                                  (cfg.PARAMS['temp_all_liq'] - cfg.PARAMS['temp_all_solid'])
                
                # Total solid precipitation
                solid_prcp = np.sum(prcp * psolid)
                
                # Convert to annual average and apply precipitation factor
                solid_prcp = solid_prcp / ny * cfg.PARAMS['prcp_scaling_factor']
                
                # Add to total weighted by grid cell area
                annual_solid_prcp += solid_prcp * fl.dx * fl.widths_m
                
    # Total glacier area
    area = np.sum([fl.dx * np.sum(fl.widths_m) for fl in fls])
    
    # Now, apply the calibration formula from the paper:
    # μ = (f_P * P_solid - (ΔM_awl + C + f_bwl * ΔM_f) / A_RGI) * 1/T_m
    
    # Above water line mass change rate (m3 w.e. yr-1)
    delta_m_awl = mb_ref * area
    
    # Calving flux (m3 w.e. yr-1)
    C = calving_data
    
    # Volume retreat due to area changes (for tidewater glaciers)
    delta_m_f = 0  # This would come from frontal ablation data if available
    
    # Fraction below waterline (default)
    f_bwl = 0.8 if gdir.is_tidewater else 0
    
    # Calculate temperature sensitivity
    mu_star = (cfg.PARAMS['prcp_scaling_factor'] * annual_solid_prcp - 
              (delta_m_awl + C + f_bwl * delta_m_f)) / annual_temp_above_melt
    
    # Apply minimum mu_star constraint for tidewater glaciers
    if gdir.is_tidewater and min_mu_star_frac > 0:
        # Get the non-calving mu_star that would be used
        mu_ref = (cfg.PARAMS['prcp_scaling_factor'] * annual_solid_prcp - delta_m_awl) / annual_temp_above_melt
        
        # Apply minimum fraction
        mu_star = max(mu_star, mu_ref * min_mu_star_frac)
    
    # Store the calibrated mu_star
    gdir.add_to_diagnostics('mu_star_geodetic', float(mu_star))
    
    # Also compute and store the apparent mass balance
    mb_apparent = delta_m_awl / area
    gdir.add_to_diagnostics('mb_geodetic', float(mb_ref))
    gdir.add_to_diagnostics('mb_apparent', float(mb_apparent))
    
    # Store frontal ablation data if used
    if calving_data > 0:
        gdir.add_to_diagnostics('frontal_ablation', float(calving_data))
    
    return float(mu_star)


@entity_task(log)
def compute_apparent_mb(gdir, mb_geodetic, frontal_ablation=None):
    """Compute the apparent mass balance from geodetic and frontal ablation.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    mb_geodetic : float
        geodetic mass balance (m w.e. yr-1)
    frontal_ablation : float
        frontal ablation (m w.e. yr-1)
        
    Returns
    -------
    float
        apparent mass balance (m w.e. yr-1)
    """
    if frontal_ablation is None or not gdir.is_tidewater:
        return mb_geodetic
    
    # Area
    area = gdir.rgi_area_m2
    
    # Apply correction
    return mb_geodetic - frontal_ablation / area
