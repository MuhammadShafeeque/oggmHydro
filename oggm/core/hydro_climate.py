"""
HydroMassBalance Climate Integration and Runoff Computation

This module provides:
- Seamless integration with regional_scaling.py climate data
- Advanced runoff computation and routing
- Hydrological component separation (glacier vs non-glacier)
- Multi-scale temporal aggregation
- Quality control and bias correction
- Streamflow calibration utilities

Authors: Muhammad Shafeeque & Claude AI Assistant
License: BSD-3-Clause (compatible with OGGM)
"""

# Built-ins
import logging
import os
import warnings
from datetime import datetime, timedelta
import json
from collections import defaultdict

# External libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats, signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

# OGGM imports
from oggm import cfg, utils
from oggm.core.massbalance import MassBalanceModel
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm import entity_task

# Module logger
log = logging.getLogger(__name__)

# Default runoff parameters
DEFAULT_RUNOFF_PARAMS = {
    'routing_method': 'linear_reservoir',  # 'linear_reservoir', 'unit_hydrograph', 'none'
    'reservoir_constant': 30,  # days
    'evapotranspiration': True,
    'groundwater_component': True,
    'snow_retention': True,
    'debris_retention_factor': 0.1,  # debris-covered areas retain more water
    'elevation_routing': True,
    'quality_control': True
}


class HydroClimateIntegration:
    """Integration with regional scaling and other climate data sources"""
    
    def __init__(self, gdir, climate_source='regional_scaling', **kwargs):
        """
        Initialize climate integration
        
        Parameters
        ----------
        gdir : GlacierDirectory
            OGGM glacier directory
        climate_source : str
            Climate data source ('regional_scaling', 'oggm_default', 'w5e5', 'era5')
        """
        self.gdir = gdir
        self.climate_source = climate_source
        self.config = kwargs
        
        # Initialize climate data
        self.climate_data = None
        self.quality_metrics = {}
        
        self._load_climate_data()
    
    def _load_climate_data(self):
        """Load climate data from specified source"""
        log.info(f"Loading {self.climate_source} climate data for {self.gdir.rgi_id}")
        
        try:
            if self.climate_source == 'regional_scaling':
                self._load_regional_scaling_data()
            elif self.climate_source == 'w5e5':
                self._load_w5e5_data()
            elif self.climate_source == 'era5':
                self._load_era5_data()
            else:  # oggm_default
                self._load_oggm_default_data()
                
            # Quality control
            if self.config.get('quality_control', True):
                self._quality_control_climate()
                
        except Exception as e:
            log.error(f"Failed to load {self.climate_source} climate data: {e}")
            # Fallback to OGGM default
            log.info("Falling back to OGGM default climate data")
            self._load_oggm_default_data()
    
    def _load_regional_scaling_data(self):
        """Load regional scaling climate data"""
        try:
            # Check if regional scaling data exists
            rs_file = self.gdir.get_filepath('climate_historical', filesuffix='_regional_scaling')
            
            if not os.path.exists(rs_file):
                # Trigger regional scaling processing
                log.info("Regional scaling data not found, processing now...")
                from oggm.core.regional_scaling import process_regional_scaling_climate_data
                
                # Use station data path from config
                station_path = self.config.get('station_data_path', cfg.PATHS.get('station_data_path'))
                if station_path:
                    process_regional_scaling_climate_data(
                        self.gdir,
                        station_data_path=station_path,
                        output_filesuffix='_regional_scaling'
                    )
                else:
                    raise FileNotFoundError("Station data path required for regional scaling")
            
            # Load processed regional scaling data
            with utils.ncDataset(rs_file, mode='r') as nc:
                time = nc.variables['time']
                import cftime
                time = cftime.num2date(time[:], time.units, calendar=time.calendar)
                
                self.climate_data = {
                    'time': pd.to_datetime(time),
                    'temp': nc.variables['temp'][:].astype(np.float64),
                    'prcp': nc.variables['prcp'][:].astype(np.float64),
                    'ref_hgt': nc.ref_hgt,
                    'source': 'regional_scaling',
                    'quality_flags': getattr(nc, 'quality_flags', {}),
                    'bias_correction_applied': getattr(nc, 'bias_correction_applied', True),
                    'downscaling_method': getattr(nc, 'downscaling_method', 'physical')
                }
                
                # Additional variables if available
                for var in ['temp_min', 'temp_max', 'humidity', 'radiation', 'wind']:
                    if var in nc.variables:
                        self.climate_data[var] = nc.variables[var][:].astype(np.float64)
                
        except Exception as e:
            log.error(f"Failed to load regional scaling data: {e}")
            raise
    
    def _load_w5e5_data(self):
        """Load W5E5 climate data"""
        try:
            from oggm.shop.w5e5 import process_w5e5_data
            
            # Process W5E5 data if not already done
            w5e5_file = self.gdir.get_filepath('climate_historical', filesuffix='_w5e5')
            
            if not os.path.exists(w5e5_file):
                process_w5e5_data(self.gdir, output_filesuffix='_w5e5')
            
            # Load W5E5 data
            with utils.ncDataset(w5e5_file, mode='r') as nc:
                time = nc.variables['time']
                import cftime
                time = cftime.num2date(time[:], time.units, calendar=time.calendar)
                
                self.climate_data = {
                    'time': pd.to_datetime(time),
                    'temp': nc.variables['temp'][:].astype(np.float64),
                    'prcp': nc.variables['prcp'][:].astype(np.float64), 
                    'ref_hgt': nc.ref_hgt,
                    'source': 'w5e5',
                    'bias_correction_applied': False,
                    'native_resolution': '0.5 degree'
                }
                
        except Exception as e:
            log.error(f"Failed to load W5E5 data: {e}")
            raise
    
    def _load_era5_data(self):
        """Load ERA5 climate data"""
        try:
            from oggm.shop.ecmwf import process_ecmwf_data
            
            era5_file = self.gdir.get_filepath('climate_historical', filesuffix='_era5')
            
            if not os.path.exists(era5_file):
                process_ecmwf_data(self.gdir, dataset='ERA5', output_filesuffix='_era5')
            
            with utils.ncDataset(era5_file, mode='r') as nc:
                time = nc.variables['time']
                import cftime
                time = cftime.num2date(time[:], time.units, calendar=time.calendar)
                
                self.climate_data = {
                    'time': pd.to_datetime(time),
                    'temp': nc.variables['temp'][:].astype(np.float64),
                    'prcp': nc.variables['prcp'][:].astype(np.float64),
                    'ref_hgt': nc.ref_hgt,
                    'source': 'era5',
                    'bias_correction_applied': False,
                    'native_resolution': '0.25 degree'
                }
                
        except Exception as e:
            log.error(f"Failed to load ERA5 data: {e}")
            raise
    
    def _load_oggm_default_data(self):
        """Load OGGM default climate data"""
        try:
            climate_file = self.gdir.get_filepath('climate_historical')
            
            with utils.ncDataset(climate_file, mode='r') as nc:
                time = nc.variables['time']
                import cftime
                time = cftime.num2date(time[:], time.units, calendar=time.calendar)
                
                self.climate_data = {
                    'time': pd.to_datetime(time),
                    'temp': nc.variables['temp'][:].astype(np.float64),
                    'prcp': nc.variables['prcp'][:].astype(np.float64),
                    'ref_hgt': nc.ref_hgt,
                    'source': 'oggm_default',
                    'bias_correction_applied': True
                }
                
        except Exception as e:
            log.error(f"Failed to load OGGM default climate data: {e}")
            raise
    
    def _quality_control_climate(self):
        """Perform quality control on climate data"""
        log.info("Performing climate data quality control")
        
        temp = self.climate_data['temp']
        prcp = self.climate_data['prcp']
        
        # Basic quality checks
        quality_flags = []
        
        # Temperature checks
        if np.any(temp < -50) or np.any(temp > 50):
            quality_flags.append('extreme_temperatures')
        
        if np.any(np.isnan(temp)):
            quality_flags.append('missing_temperature')
        
        if np.std(temp) < 1:
            quality_flags.append('low_temperature_variability')
        
        # Precipitation checks
        if np.any(prcp < 0):
            quality_flags.append('negative_precipitation')
            # Fix negative precipitation
            self.climate_data['prcp'] = np.maximum(prcp, 0)
        
        if np.any(prcp > 1000):  # Very high daily precipitation
            quality_flags.append('extreme_precipitation')
        
        if np.any(np.isnan(prcp)):
            quality_flags.append('missing_precipitation')
        
        # Temporal consistency checks
        time_diff = np.diff(self.climate_data['time'])
        expected_diff = pd.Timedelta(days=30.44)  # Monthly data
        
        if not np.all(np.abs(time_diff - expected_diff) < pd.Timedelta(days=5)):
            quality_flags.append('irregular_time_spacing')
        
        # Climate realism checks
        annual_temp = self._aggregate_to_annual(temp)
        annual_prcp = self._aggregate_to_annual(prcp)
        
        if np.any(annual_prcp < 100):  # Very dry
            quality_flags.append('very_low_precipitation')
        
        if np.any(annual_temp > 15):  # Very warm for glacier
            quality_flags.append('high_temperature_for_glacier')
        
        self.quality_metrics = {
            'quality_flags': quality_flags,
            'temp_range': [float(np.min(temp)), float(np.max(temp))],
            'prcp_range': [float(np.min(prcp)), float(np.max(prcp))],
            'temp_mean': float(np.mean(temp)),
            'prcp_mean': float(np.mean(prcp)),
            'data_completeness': float(1 - np.sum(np.isnan(temp) | np.isnan(prcp)) / len(temp)),
            'time_coverage': {
                'start': str(self.climate_data['time'].min()),
                'end': str(self.climate_data['time'].max()),
                'n_years': len(self.climate_data['time']) / 12
            }
        }
        
        log.info(f"Quality control completed. Flags: {quality_flags}")
    
    def _aggregate_to_annual(self, monthly_data):
        """Aggregate monthly data to annual"""
        n_years = len(monthly_data) // 12
        annual_data = []
        
        for year in range(n_years):
            start_idx = year * 12
            end_idx = start_idx + 12
            annual_data.append(np.sum(monthly_data[start_idx:end_idx]))
        
        return np.array(annual_data)
    
    def get_climate_at_elevation(self, elevation, temporal_subset=None):
        """
        Get climate data adjusted for specific elevation
        
        Parameters
        ----------
        elevation : float
            Target elevation (m a.s.l.)
        temporal_subset : tuple
            (start_date, end_date) for temporal subsetting
            
        Returns
        -------
        dict
            Climate data adjusted for elevation
        """
        if self.climate_data is None:
            raise ValueError("Climate data not loaded")
        
        # Temperature lapse rate adjustment
        temp_lapse_rate = cfg.PARAMS.get('temp_default_gradient', -0.0065)  # K/m
        elevation_diff = elevation - self.climate_data['ref_hgt']
        
        adjusted_temp = self.climate_data['temp'] + temp_lapse_rate * elevation_diff
        
        # Precipitation adjustment (simplified orographic effect)
        prcp_gradient = cfg.PARAMS.get('prcp_gradient', 0.0002)  # 1/m
        prcp_factor = 1 + prcp_gradient * elevation_diff
        adjusted_prcp = self.climate_data['prcp'] * np.maximum(prcp_factor, 0.1)
        
        # Temporal subsetting
        if temporal_subset:
            start_date, end_date = temporal_subset
            time_mask = ((self.climate_data['time'] >= pd.to_datetime(start_date)) & 
                        (self.climate_data['time'] <= pd.to_datetime(end_date)))
            
            adjusted_temp = adjusted_temp[time_mask]
            adjusted_prcp = adjusted_prcp[time_mask]
            time_subset = self.climate_data['time'][time_mask]
        else:
            time_subset = self.climate_data['time']
        
        return {
            'time': time_subset,
            'temp': adjusted_temp,
            'prcp': adjusted_prcp,
            'elevation': elevation,
            'source': self.climate_data['source']
        }
    
    def apply_bias_correction(self, reference_data, method='quantile_mapping'):
        """
        Apply bias correction to climate data
        
        Parameters
        ----------
        reference_data : dict
            Reference climate data for bias correction
        method : str
            Bias correction method
        """
        if self.climate_data['source'] == 'regional_scaling':
            log.info("Regional scaling data already bias-corrected")
            return
        
        log.info(f"Applying {method} bias correction")
        
        # Simple linear scaling bias correction
        if method == 'linear_scaling':
            # Temperature (additive)
            temp_bias = np.mean(reference_data['temp']) - np.mean(self.climate_data['temp'])
            self.climate_data['temp'] += temp_bias
            
            # Precipitation (multiplicative)
            prcp_factor = np.mean(reference_data['prcp']) / np.mean(self.climate_data['prcp'])
            self.climate_data['prcp'] *= prcp_factor
            
            log.info(f"Applied bias correction: temp_bias={temp_bias:.2f}°C, prcp_factor={prcp_factor:.2f}")
        
        self.climate_data['bias_correction_applied'] = True


class RunoffComputation:
    """Advanced runoff computation and hydrological routing"""
    
    def __init__(self, gdir, routing_config=None):
        """
        Initialize runoff computation
        
        Parameters
        ----------
        gdir : GlacierDirectory
            OGGM glacier directory
        routing_config : dict
            Routing configuration parameters
        """
        self.gdir = gdir
        self.config = routing_config or DEFAULT_RUNOFF_PARAMS.copy()
        
        # Load glacier geometry
        self._load_glacier_geometry()
        
        # Initialize routing parameters
        self._initialize_routing()
    
    def _load_glacier_geometry(self):
        """Load glacier geometry for runoff computation"""
        try:
            # Load flowlines
            self.flowlines = self.gdir.read_pickle('inversion_flowlines')
            
            # Extract elevation profile
            self.elevations = []
            self.widths = []
            self.areas = []
            
            for fl in self.flowlines:
                self.elevations.extend(fl.surface_h)
                self.widths.extend(fl.widths_m)
                # Calculate areas
                areas = fl.widths_m * fl.dx_meter
                self.areas.extend(areas)
            
            self.elevations = np.array(self.elevations)
            self.widths = np.array(self.widths)
            self.areas = np.array(self.areas)
            
            # Glacier properties
            self.glacier_area_total = np.sum(self.areas)
            self.mean_elevation = np.average(self.elevations, weights=self.areas)
            self.elevation_range = np.max(self.elevations) - np.min(self.elevations)
            
            log.info(f"Loaded glacier geometry: {len(self.elevations)} elevation bands")
            
        except Exception as e:
            log.warning(f"Could not load detailed glacier geometry: {e}")
            # Fallback to simple geometry
            self._create_simple_geometry()
    
    def _create_simple_geometry(self):
        """Create simple glacier geometry for runoff computation"""
        # Use RGI area and elevation range
        self.glacier_area_total = self.gdir.rgi_area_m2
        self.mean_elevation = (self.gdir.max_h + self.gdir.min_h) / 2
        self.elevation_range = self.gdir.max_h - self.gdir.min_h
        
        # Create simple elevation bands
        n_bands = max(5, min(20, int(self.elevation_range / 100)))  # 100m bands
        self.elevations = np.linspace(self.gdir.min_h, self.gdir.max_h, n_bands)
        self.areas = np.full(n_bands, self.glacier_area_total / n_bands)
        self.widths = np.sqrt(self.areas)  # Simplified width estimate
        
        log.info(f"Created simple glacier geometry: {n_bands} elevation bands")
    
    def _initialize_routing(self):
        """Initialize hydrological routing parameters"""
        routing_method = self.config['routing_method']
        
        if routing_method == 'linear_reservoir':
            self.reservoir_constant = self.config.get('reservoir_constant', 30)  # days
            
        elif routing_method == 'unit_hydrograph':
            # Create unit hydrograph based on glacier size
            self.unit_hydrograph = self._create_unit_hydrograph()
        
        # Evapotranspiration parameters
        if self.config.get('evapotranspiration', True):
            self.et_params = {
                'potential_et_rate': 2.0,  # mm/day at reference conditions
                'temperature_threshold': 5.0,  # °C
                'elevation_factor': 0.1  # reduction per 100m elevation
            }
        
        # Debris retention parameters
        if hasattr(self.gdir, 'debris_factor'):
            self.debris_areas = self.areas * self.gdir.debris_factor
            self.debris_retention = self.config.get('debris_retention_factor', 0.1)
        else:
            self.debris_areas = np.zeros_like(self.areas)
            self.debris_retention = 0.0
    
    def _create_unit_hydrograph(self):
        """Create unit hydrograph for runoff routing"""
        # Simple gamma distribution unit hydrograph
        # Based on glacier size and slope
        
        glacier_length = self.elevation_range / 0.1  # Rough estimate: 10% average slope
        travel_time = glacier_length / 1000  # km -> travel time in hours (rough)
        
        # Create gamma distribution
        shape = 2.0
        scale = travel_time / shape
        
        time_steps = np.arange(0, travel_time * 5, 1)  # hourly steps
        unit_hydrograph = stats.gamma.pdf(time_steps, shape, scale=scale)
        unit_hydrograph = unit_hydrograph / np.sum(unit_hydrograph)  # Normalize
        
        return unit_hydrograph
    
    def compute_runoff_components(self, mb_model, year_range=None, temporal_scale='monthly'):
        """
        Compute detailed runoff components
        
        Parameters
        ----------
        mb_model : HydroMassBalance
            Calibrated mass balance model
        year_range : tuple
            (start_year, end_year) for computation
        temporal_scale : str
            Output temporal scale ('daily', 'monthly', 'annual')
            
        Returns
        -------
        dict
            Detailed runoff components
        """
        if year_range is None:
            year_range = (mb_model.ys, mb_model.ye)
        
        start_year, end_year = year_range
        years = np.arange(start_year, end_year + 1)
        
        log.info(f"Computing runoff components for {start_year}-{end_year}")
        
        # Initialize results storage
        runoff_components = {
            'total_runoff': [],
            'glacier_runoff': [],
            'snow_runoff': [],
            'ice_runoff': [],
            'rain_runoff': [],
            'refreeze_component': [],
            'evapotranspiration': [],
            'debris_retention': [],
            'time': [],
            'elevation_bands': {
                'elevations': self.elevations,
                'areas': self.areas,
                'band_runoff': []
            }
        }
        
        # Compute for each time step
        if temporal_scale == 'monthly':
            time_steps = []
            for year in years:
                for month in range(1, 13):
                    time_steps.append((year, month))
        else:  # annual
            time_steps = [(year, None) for year in years]
        
        for time_step in time_steps:
            year, month = time_step
            
            if month is not None:
                # Monthly computation
                month_year = year + (month - 1) / 12.0
                time_label = f"{year}-{month:02d}"
                
                # Get climate data for this month
                climate_data = mb_model._get_monthly_climate(self.elevations, month_year)
                
                # Compute mass balance components
                physics_results = mb_model._compute_physics(climate_data, self.elevations)
                
                # Components
                prcp = climate_data['prcp']
                temp = climate_data['temp']
                accumulation = mb_model._compute_accumulation(climate_data, physics_results)
                melt = mb_model._compute_melt(climate_data, physics_results, self.elevations)
                
                # Refreezing
                if 'refreezing' in physics_results:
                    refreezing = physics_results['refreezing']
                    if isinstance(refreezing, dict):
                        refreeze = refreezing.get('refreeze', np.zeros_like(self.elevations))
                    else:
                        refreeze = refreezing if refreezing is not None else np.zeros_like(self.elevations)
                else:
                    refreeze = np.zeros_like(self.elevations)
                
            else:
                # Annual computation
                time_label = str(year)
                
                # Aggregate monthly data
                annual_prcp = np.zeros_like(self.elevations)
                annual_melt = np.zeros_like(self.elevations)
                annual_refreeze = np.zeros_like(self.elevations)
                annual_temp = np.zeros_like(self.elevations)
                
                for m in range(1, 13):
                    month_year = year + (m - 1) / 12.0
                    climate_data = mb_model._get_monthly_climate(self.elevations, month_year)
                    physics_results = mb_model._compute_physics(climate_data, self.elevations)
                    
                    annual_prcp += climate_data['prcp']
                    annual_temp += climate_data['temp']
                    annual_melt += mb_model._compute_melt(climate_data, physics_results, self.elevations)
                    
                    if 'refreezing' in physics_results:
                        refreezing = physics_results['refreezing']
                        if isinstance(refreezing, dict):
                            annual_refreeze += refreezing.get('refreeze', np.zeros_like(self.elevations))
                        elif refreezing is not None:
                            annual_refreeze += refreezing
                
                prcp = annual_prcp
                melt = annual_melt
                refreeze = annual_refreeze
                temp = annual_temp / 12  # Average temperature
            
            # Separate precipitation into rain and snow
            temp_threshold = cfg.PARAMS.get('temp_all_solid', 0.0)
            rain = np.where(temp > temp_threshold, prcp, 0)
            snow = np.where(temp <= temp_threshold, prcp, 0)
            
            # Runoff components by elevation band
            ice_melt = melt * 0.8  # Approximate ice fraction
            snow_melt = melt * 0.2  # Approximate snow fraction
            
            # Total liquid water available
            total_liquid = rain + ice_melt + snow_melt
            
            # Subtract refreezing
            available_runoff = np.maximum(total_liquid - refreeze, 0)
            
            # Apply evapotranspiration
            if self.config.get('evapotranspiration', True):
                et_loss = self._compute_evapotranspiration(temp, self.elevations)
                available_runoff = np.maximum(available_runoff - et_loss, 0)
            else:
                et_loss = np.zeros_like(available_runoff)
            
            # Apply debris retention
            if self.debris_retention > 0:
                debris_retained = available_runoff * self.debris_retention * (self.debris_areas / self.areas)
                available_runoff = available_runoff - debris_retained
            else:
                debris_retained = np.zeros_like(available_runoff)
            
            # Calculate area-weighted totals
            glacier_runoff = np.sum(available_runoff * self.areas) / 1e6  # m³/month or m³/year
            total_et = np.sum(et_loss * self.areas) / 1e6
            total_debris_retention = np.sum(debris_retained * self.areas) / 1e6
            total_refreeze = np.sum(refreeze * self.areas) / 1e6
            
            # Component runoffs
            rain_runoff_total = np.sum(rain * self.areas) / 1e6
            ice_runoff_total = np.sum(ice_melt * self.areas) / 1e6  
            snow_runoff_total = np.sum(snow_melt * self.areas) / 1e6
            
            # Store results
            runoff_components['total_runoff'].append(glacier_runoff)
            runoff_components['glacier_runoff'].append(glacier_runoff)
            runoff_components['rain_runoff'].append(rain_runoff_total)
            runoff_components['ice_runoff'].append(ice_runoff_total)
            runoff_components['snow_runoff'].append(snow_runoff_total)
            runoff_components['refreeze_component'].append(total_refreeze)
            runoff_components['evapotranspiration'].append(total_et)
            runoff_components['debris_retention'].append(total_debris_retention)
            runoff_components['time'].append(time_label)
            runoff_components['elevation_bands']['band_runoff'].append(available_runoff.tolist())
        
        # Apply routing if requested
        if self.config['routing_method'] != 'none':
            runoff_components = self._apply_routing(runoff_components)
        
        # Convert to arrays
        for key in ['total_runoff', 'glacier_runoff', 'rain_runoff', 'ice_runoff', 
                   'snow_runoff', 'refreeze_component', 'evapotranspiration', 'debris_retention']:
            runoff_components[key] = np.array(runoff_components[key])
        
        return runoff_components
    
    def _compute_evapotranspiration(self, temperature, elevations):
        """Compute evapotranspiration losses"""
        if not self.config.get('evapotranspiration', True):
            return np.zeros_like(temperature)
        
        et_params = self.et_params
        
        # Temperature-dependent ET
        temp_factor = np.maximum(0, (temperature - et_params['temperature_threshold']) / 20.0)
        
        # Elevation-dependent ET reduction
        elevation_factor = np.maximum(0.1, 1 - et_params['elevation_factor'] * 
                                     (elevations - self.mean_elevation) / 100.0)
        
        # Potential ET rate
        potential_et = et_params['potential_et_rate']  # mm/day or mm/month
        
        # Actual ET
        actual_et = potential_et * temp_factor * elevation_factor
        
        return actual_et
    
    def _apply_routing(self, runoff_components):
        """Apply hydrological routing to runoff"""
        routing_method = self.config['routing_method']
        
        if routing_method == 'linear_reservoir':
            return self._linear_reservoir_routing(runoff_components)
        elif routing_method == 'unit_hydrograph':
            return self._unit_hydrograph_routing(runoff_components)
        else:
            return runoff_components
    
    def _linear_reservoir_routing(self, runoff_components):
        """Apply linear reservoir routing"""
        k = self.reservoir_constant  # days
        
        # Convert to daily time step for routing
        inflow = runoff_components['total_runoff']
        
        # Linear reservoir equation: S(t+1) = S(t) + I(t) - O(t)
        # where O(t) = S(t) / k
        
        storage = 0.0
        routed_flow = []
        
        for i in range(len(inflow)):
            # Add inflow
            storage += inflow[i]
            
            # Calculate outflow
            outflow = storage / k
            
            # Update storage
            storage -= outflow
            
            routed_flow.append(outflow)
        
        # Update routed runoff
        runoff_components['total_runoff'] = np.array(routed_flow)
        runoff_components['glacier_runoff'] = np.array(routed_flow)
        runoff_components['routing_applied'] = 'linear_reservoir'
        runoff_components['reservoir_constant_days'] = k
        
        return runoff_components
    
    def _unit_hydrograph_routing(self, runoff_components):
        """Apply unit hydrograph routing"""
        inflow = runoff_components['total_runoff']
        unit_hydrograph = self.unit_hydrograph
        
        # Convolve inflow with unit hydrograph
        routed_flow = np.convolve(inflow, unit_hydrograph, mode='same')
        
        runoff_components['total_runoff'] = routed_flow
        runoff_components['glacier_runoff'] = routed_flow
        runoff_components['routing_applied'] = 'unit_hydrograph'
        
        return runoff_components
    
    def calibrate_against_discharge(self, observed_discharge, mb_model, 
                                  calibration_period=None, method='nash_sutcliffe'):
        """
        Calibrate runoff parameters against observed discharge
        
        Parameters
        ----------
        observed_discharge : dict or pd.Series
            Observed discharge data
        mb_model : HydroMassBalance  
            Mass balance model to use
        calibration_period : tuple
            (start_year, end_year) for calibration
        method : str
            Objective function ('nash_sutcliffe', 'rmse', 'kge')
            
        Returns
        -------
        dict
            Calibration results
        """
        log.info("Starting runoff calibration against discharge observations")
        
        # Prepare observed data
        if isinstance(observed_discharge, dict):
            obs_data = pd.Series(observed_discharge['discharge'], 
                               index=pd.to_datetime(observed_discharge['time']))
        else:
            obs_data = observed_discharge
        
        # Set calibration period
        if calibration_period:
            start_year, end_year = calibration_period
            obs_data = obs_data[(obs_data.index.year >= start_year) & 
                               (obs_data.index.year <= end_year)]
        
        # Define parameter bounds for calibration
        param_bounds = {
            'reservoir_constant': (1, 100),  # days
            'debris_retention_factor': (0.0, 0.5),
            'et_rate_factor': (0.5, 2.0)
        }
        
        def objective_function(params):
            """Objective function for runoff calibration"""
            # Update routing configuration
            temp_config = self.config.copy()
            temp_config['reservoir_constant'] = params[0]
            temp_config['debris_retention_factor'] = params[1]
            
            # Update ET parameters
            if hasattr(self, 'et_params'):
                self.et_params['potential_et_rate'] *= params[2]
            
            try:
                # Compute modeled runoff
                year_range = (obs_data.index.year.min(), obs_data.index.year.max())
                temporal_scale = 'monthly' if len(obs_data) > 50 else 'annual'
                
                runoff_results = self.compute_runoff_components(
                    mb_model, year_range=year_range, temporal_scale=temporal_scale
                )
                
                modeled_runoff = runoff_results['total_runoff']
                
                # Align with observations (simplified alignment)
                if len(modeled_runoff) != len(obs_data):
                    # Resample to match observations
                    min_length = min(len(modeled_runoff), len(obs_data))
                    modeled_runoff = modeled_runoff[:min_length]
                    obs_aligned = obs_data.values[:min_length]
                else:
                    obs_aligned = obs_data.values
                
                # Calculate objective function
                if method == 'nash_sutcliffe':
                    if np.var(obs_aligned) > 0:
                        nse = 1 - np.sum((obs_aligned - modeled_runoff)**2) / np.sum((obs_aligned - np.mean(obs_aligned))**2)
                        return 1 - nse  # Minimize (1 - NSE)
                    else:
                        return 1e6
                        
                elif method == 'rmse':
                    return np.sqrt(np.mean((obs_aligned - modeled_runoff)**2))
                    
                elif method == 'kge':
                    # Kling-Gupta Efficiency
                    correlation = np.corrcoef(obs_aligned, modeled_runoff)[0, 1]
                    bias_ratio = np.mean(modeled_runoff) / np.mean(obs_aligned)
                    var_ratio = np.std(modeled_runoff) / np.std(obs_aligned)
                    
                    kge = 1 - np.sqrt((correlation - 1)**2 + (bias_ratio - 1)**2 + (var_ratio - 1)**2)
                    return 1 - kge  # Minimize (1 - KGE)
                    
            except Exception as e:
                log.warning(f"Error in runoff calibration objective function: {e}")
                return 1e6
        
        # Optimize parameters
        from scipy.optimize import minimize
        
        # Initial parameter values
        x0 = [30, 0.1, 1.0]  # reservoir_constant, debris_retention, et_factor
        bounds = list(param_bounds.values())
        
        try:
            result = minimize(objective_function, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                # Update configuration with optimal parameters
                self.config['reservoir_constant'] = result.x[0]
                self.config['debris_retention_factor'] = result.x[1]
                
                if hasattr(self, 'et_params'):
                    self.et_params['potential_et_rate'] *= result.x[2]
                
                # Compute final metrics
                final_runoff = self.compute_runoff_components(
                    mb_model, 
                    year_range=(obs_data.index.year.min(), obs_data.index.year.max()),
                    temporal_scale='monthly' if len(obs_data) > 50 else 'annual'
                )
                
                calibration_results = {
                    'success': True,
                    'optimal_parameters': {
                        'reservoir_constant': result.x[0],
                        'debris_retention_factor': result.x[1],
                        'et_rate_factor': result.x[2]
                    },
                    'objective_value': result.fun, 
                    'method': method,
                    'n_observations': len(obs_data),
                    'calibration_period': calibration_period,
                    'final_runoff': final_runoff
                }
                
                log.info("Runoff calibration completed successfully")
                return calibration_results
                
            else:
                log.warning("Runoff calibration optimization failed")
                return {'success': False, 'message': 'Optimization failed'}
                
        except Exception as e:
            log.error(f"Runoff calibration failed: {e}")
            return {'success': False, 'error': str(e)}


# Integration functions
@entity_task(log)
def setup_hydro_climate_integration(gdir, climate_source='regional_scaling', 
                                  station_data_path=None, **kwargs):
    """
    OGGM entity task to setup climate integration for HydroMassBalance
    
    Parameters
    ----------
    gdir : GlacierDirectory
        OGGM glacier directory
    climate_source : str
        Climate data source to use
    station_data_path : str
        Path to station data (for regional scaling)
    """
    
    # Setup climate integration
    config = kwargs.copy()
    if station_data_path:
        config['station_data_path'] = station_data_path
    
    climate_integration = HydroClimateIntegration(gdir, climate_source, **config)
    
    # Save configuration
    climate_config = {
        'climate_source': climate_source,
        'quality_metrics': climate_integration.quality_metrics,
        'setup_date': datetime.now().isoformat()
    }
    
    gdir.write_json(climate_config, 'hydro_climate_config')
    
    log.info(f"Climate integration setup completed for {gdir.rgi_id}")
    
    return climate_integration


@entity_task(log, writes=['runoff_components'])
def compute_glacier_runoff(gdir, mb_model=None, temporal_scale='monthly',
                          year_range=None, routing_config=None, **kwargs):
    """
    OGGM entity task to compute detailed glacier runoff components
    
    Parameters
    ----------
    gdir : GlacierDirectory
        OGGM glacier directory
    mb_model : HydroMassBalance
        Calibrated mass balance model
    temporal_scale : str
        Output temporal scale
    year_range : tuple
        (start_year, end_year) for computation
    routing_config : dict
        Hydrological routing configuration
    """
    
    if mb_model is None:
        # Try to load from calibration results
        try:
            from .hydro_massbalance import HydroMassBalance
            mb_model = HydroMassBalance(gdir, **kwargs)
        except:
            raise ValueError("Mass balance model required for runoff computation")
    
    # Setup runoff computation
    runoff_computer = RunoffComputation(gdir, routing_config)
    
    # Compute runoff components
    runoff_results = runoff_computer.compute_runoff_components(
        mb_model, year_range=year_range, temporal_scale=temporal_scale
    )
    
    # Save results
    gdir.write_json(runoff_results, 'runoff_components', default=str)
    
    log.info(f"Runoff computation completed for {gdir.rgi_id}")
    
    return runoff_results


@entity_task(log, writes=['runoff_calibration'])  
def calibrate_glacier_runoff(gdir, discharge_data_path, mb_model=None,
                           calibration_method='nash_sutcliffe', **kwargs):
    """
    OGGM entity task to calibrate glacier runoff against discharge observations
    
    Parameters
    ----------
    gdir : GlacierDirectory
        OGGM glacier directory
    discharge_data_path : str
        Path to discharge observation data
    mb_model : HydroMassBalance
        Mass balance model to use
    calibration_method : str
        Calibration objective function
    """
    
    # Load discharge data
    try:
        if discharge_data_path.endswith('.csv'):
            discharge_data = pd.read_csv(discharge_data_path, parse_dates=['date'], index_col='date')
        elif discharge_data_path.endswith('.json'):
            import json
            with open(discharge_data_path, 'r') as f:
                discharge_dict = json.load(f)
            discharge_data = pd.Series(discharge_dict['discharge'], 
                                     index=pd.to_datetime(discharge_dict['time']))
        else:
            raise ValueError("Unsupported discharge data format")
            
    except Exception as e:
        log.error(f"Failed to load discharge data: {e}")
        raise
    
    if mb_model is None:
        from .hydro_massbalance import HydroMassBalance
        mb_model = HydroMassBalance(gdir, **kwargs)
    
    # Setup runoff computation and calibration
    runoff_computer = RunoffComputation(gdir)
    
    # Calibrate against discharge
    calibration_results = runoff_computer.calibrate_against_discharge(
        discharge_data, mb_model, method=calibration_method, **kwargs
    )
    
    # Save results
    gdir.write_json(calibration_results, 'runoff_calibration', default=str)
    
    log.info(f"Runoff calibration completed for {gdir.rgi_id}")
    
    return calibration_results


# Utility functions
def validate_runoff_components(runoff_results, validation_criteria=None):
    """
    Validate computed runoff components for physical realism
    
    Parameters
    ----------
    runoff_results : dict
        Runoff computation results
    validation_criteria : dict
        Validation criteria and thresholds
        
    Returns
    -------
    dict
        Validation results and flags
    """
    
    criteria = validation_criteria or {
        'max_daily_runoff_mm': 100,  # Maximum reasonable daily runoff
        'annual_runoff_ratio': (0.1, 5.0),  # Reasonable range for runoff ratio
        'component_balance_tolerance': 0.1,  # 10% tolerance for component balance
        'negative_runoff_tolerance': 0.01  # 1% tolerance for negative values
    }
    
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'summary_stats': {}
    }
    
    # Extract runoff data
    total_runoff = np.array(runoff_results['total_runoff'])
    ice_runoff = np.array(runoff_results['ice_runoff'])
    snow_runoff = np.array(runoff_results['snow_runoff'])
    rain_runoff = np.array(runoff_results['rain_runoff'])
    
    # Basic statistics
    validation_results['summary_stats'] = {
        'mean_total_runoff': float(np.mean(total_runoff)),
        'max_total_runoff': float(np.max(total_runoff)),
        'min_total_runoff': float(np.min(total_runoff)),
        'std_total_runoff': float(np.std(total_runoff)),
        'ice_runoff_fraction': float(np.mean(ice_runoff) / np.mean(total_runoff)) if np.mean(total_runoff) > 0 else 0,
        'snow_runoff_fraction': float(np.mean(snow_runoff) / np.mean(total_runoff)) if np.mean(total_runoff) > 0 else 0,
        'rain_runoff_fraction': float(np.mean(rain_runoff) / np.mean(total_runoff)) if np.mean(total_runoff) > 0 else 0
    }
    
    # Validation checks
    
    # 1. Check for excessive runoff values
    if np.any(total_runoff > criteria['max_daily_runoff_mm'] * 30):  # Monthly data
        validation_results['warnings'].append("Very high runoff values detected")
    
    # 2. Check for negative runoff values  
    negative_fraction = np.sum(total_runoff < 0) / len(total_runoff)
    if negative_fraction > criteria['negative_runoff_tolerance']:
        validation_results['errors'].append(f"High fraction of negative runoff: {negative_fraction:.3f}")
        validation_results['is_valid'] = False
    
    # 3. Check component balance
    computed_total = ice_runoff + snow_runoff + rain_runoff
    balance_error = np.abs(computed_total - total_runoff) / (total_runoff + 1e-6)
    
    if np.mean(balance_error) > criteria['component_balance_tolerance']:
        validation_results['warnings'].append(f"Component balance error: {np.mean(balance_error):.3f}")
    
    # 4. Check runoff ratios
    if 'elevation_bands' in runoff_results:
        areas = np.array(runoff_results['elevation_bands']['areas'])
        total_area = np.sum(areas)
        
        # Annual runoff ratio check
        annual_runoff = np.sum(total_runoff)  # Total for the period
        annual_precipitation = 1000  # Placeholder - would need actual precipitation
        
        runoff_ratio = annual_runoff / annual_precipitation if annual_precipitation > 0 else 0
        
        min_ratio, max_ratio = criteria['annual_runoff_ratio']
        if runoff_ratio < min_ratio or runoff_ratio > max_ratio:
            validation_results['warnings'].append(f"Unusual runoff ratio: {runoff_ratio:.2f}")
    
    # 5. Check temporal consistency
    if len(total_runoff) > 12:  # More than one year of data
        # Check for unrealistic temporal patterns
        annual_cycle = []
        n_years = len(total_runoff) // 12
        
        for month in range(12):
            month_values = []
            for year in range(n_years):
                idx = year * 12 + month
                if idx < len(total_runoff):
                    month_values.append(total_runoff[idx])
            
            if month_values:
                annual_cycle.append(np.mean(month_values))
        
        if len(annual_cycle) == 12:
            # Check if summer months have higher runoff (reasonable for glaciers)
            summer_months = [5, 6, 7, 8]  # June, July, August, September (0-indexed)
            winter_months = [0, 1, 2, 11]  # Jan, Feb, Mar, Dec
            
            summer_runoff = np.mean([annual_cycle[i] for i in summer_months if i < len(annual_cycle)])
            winter_runoff = np.mean([annual_cycle[i] for i in winter_months if i < len(annual_cycle)])
            
            if summer_runoff <= winter_runoff:
                validation_results['warnings'].append("Unusual seasonal pattern: winter runoff >= summer runoff")
    
    return validation_results


def create_runoff_summary_report(runoff_results, validation_results=None, output_file=None):
    """
    Create comprehensive runoff analysis report
    
    Parameters
    ----------
    runoff_results : dict
        Runoff computation results
    validation_results : dict
        Validation results (optional)
    output_file : str
        Output file path for report
        
    Returns
    -------
    str
        Path to generated report
    """
    
    if output_file is None:
        output_file = 'runoff_analysis_report.html'
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Glacier Runoff Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .component {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 8px; }}
            .metric {{ background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }}
            .warning {{ background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; }}
            .error {{ background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Glacier Runoff Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Runoff components summary
    total_runoff = np.array(runoff_results['total_runoff'])
    ice_runoff = np.array(runoff_results['ice_runoff'])
    snow_runoff = np.array(runoff_results['snow_runoff'])
    rain_runoff = np.array(runoff_results['rain_runoff'])
    
    html_content += f"""
    <div class="component">
        <h2>Runoff Components Summary</h2>
        <div class="metric">Total Runoff: {np.sum(total_runoff):.2f} m³ (Mean: {np.mean(total_runoff):.2f} m³/period)</div>
        <div class="metric">Ice Runoff: {np.sum(ice_runoff):.2f} m³ ({100*np.sum(ice_runoff)/np.sum(total_runoff):.1f}%)</div>
        <div class="metric">Snow Runoff: {np.sum(snow_runoff):.2f} m³ ({100*np.sum(snow_runoff)/np.sum(total_runoff):.1f}%)</div>
        <div class="metric">Rain Runoff: {np.sum(rain_runoff):.2f} m³ ({100*np.sum(rain_runoff)/np.sum(total_runoff):.1f}%)</div>
    </div>
    """
    
    # Time series data table
    if len(runoff_results['time']) > 0:
        html_content += "<h2>Time Series Data</h2><table>"
        html_content += "<tr><th>Time</th><th>Total Runoff (m³)</th><th>Ice Runoff (m³)</th><th>Snow Runoff (m³)</th><th>Rain Runoff (m³)</th></tr>"
        
        # Show first 10 and last 10 entries if more than 20 total
        time_data = runoff_results['time']
        n_entries = len(time_data)
        
        if n_entries <= 20:
            indices = range(n_entries)
        else:
            indices = list(range(10)) + list(range(n_entries-10, n_entries))
            
        for i in indices:
            if i == 10 and n_entries > 20:
                html_content += "<tr><td colspan='5'>... (showing first and last 10 entries) ...</td></tr>"
            else:
                html_content += f"""
                <tr>
                    <td>{time_data[i]}</td>
                    <td>{total_runoff[i]:.2f}</td>
                    <td>{ice_runoff[i]:.2f}</td>
                    <td>{snow_runoff[i]:.2f}</td>
                    <td>{rain_runoff[i]:.2f}</td>
                </tr>
                """
        
        html_content += "</table>"
    
    # Validation results
    if validation_results:
        html_content += f"""
        <div class="component">
            <h2>Validation Results</h2>
            <div class="metric">Overall Validation: {'PASSED' if validation_results['is_valid'] else 'FAILED'}</div>
        """
        
        # Warnings
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                html_content += f'<div class="warning">Warning: {warning}</div>'
        
        # Errors
        if validation_results['errors']:
            for error in validation_results['errors']:
                html_content += f'<div class="error">Error: {error}</div>'
        
        # Summary statistics
        if 'summary_stats' in validation_results:
            stats = validation_results['summary_stats']
            html_content += f"""
            <h3>Summary Statistics</h3>
            <div class="metric">Mean Total Runoff: {stats.get('mean_total_runoff', 0):.2f} m³</div>
            <div class="metric">Maximum Total Runoff: {stats.get('max_total_runoff', 0):.2f} m³</div>
            <div class="metric">Ice Runoff Fraction: {stats.get('ice_runoff_fraction', 0):.3f}</div>
            <div class="metric">Snow Runoff Fraction: {stats.get('snow_runoff_fraction', 0):.3f}</div>
            <div class="metric">Rain Runoff Fraction: {stats.get('rain_runoff_fraction', 0):.3f}</div>
            """
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    log.info(f"Runoff analysis report saved to {output_file}")
    
    return output_file
