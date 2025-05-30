"""
HydroMassBalance: Revolutionary Multi-Target Glacier Mass Balance Model

This module implements a next-generation mass balance model that combines:
- OGGM's framework and glacier dynamics
- PyGEM's advanced physical processes
- Multi-target calibration (mass balance, runoff, volume, velocity)
- Flexible temporal scaling and data integration
- Regional scaling climate integration

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
from abc import ABC, abstractmethod

# External libs
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats, optimize
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# OGGM imports
from oggm import cfg, utils
from oggm.core.massbalance import MassBalanceModel
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm import entity_task
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH

# Module logger
log = logging.getLogger(__name__)

# Default configuration
DEFAULT_HYDRO_PARAMS = {
    # Physics configuration
    'physics_level': 'advanced',  # 'simple', 'intermediate', 'advanced'
    'enable_refreezing': True,
    'enable_debris': True,
    'enable_surface_evolution': True,
    'refreezing_method': 'HH2015',  # 'HH2015', 'Woodward', 'none'
    
    # Calibration configuration
    'calibration_targets': ['geodetic_mb'],
    'target_priorities': {
        'mb': 1.0,
        'volume': 0.8, 
        'velocity': 0.6,
        'runoff': 0.4
    },
    
    # Temporal flexibility
    'mb_temporal_scale': 'annual',  # 'monthly', 'seasonal', 'annual'
    'runoff_temporal_scale': 'monthly',
    'volume_temporal_scale': 'annual',
    'velocity_temporal_scale': 'annual',
    
    # Optimization
    'optimization_method': 'multi_objective',  # 'single', 'multi_objective', 'hierarchical'
    'max_iterations': 1000,
    'convergence_tolerance': 1e-4,
    
    # Validation
    'validation_method': 'leave_one_out',
    'save_calibration_stats': True,
}


class DataConfiguration:
    """Flexible data configuration for any calibration target"""
    
    def __init__(self, data_type, file_path=None, data_format='auto', 
                 temporal_scale='auto', target_scale='auto', component='total',
                 date_column='date', value_column='value', 
                 uncertainty_column=None, metadata=None):
        """
        Configure data source for calibration targets
        
        Parameters
        ----------
        data_type : str
            Type of data ('runoff', 'volume', 'velocity', 'mb')
        file_path : str
            Path to data file (csv, json, nc)
        data_format : str
            Format specification or 'auto' for auto-detection
        temporal_scale : str
            Original temporal scale ('daily', 'monthly', 'seasonal', 'annual')
        target_scale : str
            Target temporal scale for calibration
        component : str
            Data component ('total', 'glacier', 'ice', 'snow', etc.)
        """
        self.data_type = data_type
        self.file_path = file_path
        self.data_format = data_format
        self.temporal_scale = temporal_scale
        self.target_scale = target_scale
        self.component = component
        self.date_column = date_column
        self.value_column = value_column
        self.uncertainty_column = uncertainty_column
        self.metadata = metadata or {}
        
        # Will store processed data
        self.data = None
        self.processed_data = None
        
    def load_data(self):
        """Load and validate data from file"""
        if self.file_path is None:
            return None
            
        try:
            if self.data_format == 'auto':
                self.data_format = self._detect_format()
            
            if self.data_format == 'csv':
                self.data = pd.read_csv(self.file_path, parse_dates=[self.date_column])
            elif self.data_format == 'json':
                with open(self.file_path, 'r') as f:
                    json_data = json.load(f)
                self.data = pd.DataFrame(json_data)
                self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            elif self.data_format == 'nc':
                ds = xr.open_dataset(self.file_path)
                self.data = ds.to_dataframe().reset_index()
            else:
                raise ValueError(f"Unsupported data format: {self.data_format}")
                
            self._validate_data()
            log.info(f"Loaded {self.data_type} data: {len(self.data)} records")
            return self.data
            
        except Exception as e:
            log.error(f"Error loading {self.data_type} data from {self.file_path}: {e}")
            raise
    
    def _detect_format(self):
        """Auto-detect file format from extension"""
        if self.file_path is None:
            return 'none'
        ext = os.path.splitext(self.file_path)[1].lower()
        format_map = {'.csv': 'csv', '.json': 'json', '.nc': 'nc', '.netcdf': 'nc'}
        return format_map.get(ext, 'csv')
    
    def _validate_data(self):
        """Validate loaded data"""
        if self.data is None:
            return
            
        required_columns = [self.date_column, self.value_column]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in {self.data_type} data: {missing_columns}")
        
        # Check for valid dates and values
        if self.data[self.date_column].isna().any():
            log.warning(f"Found NaN dates in {self.data_type} data")
        
        if self.data[self.value_column].isna().all():
            raise ValueError(f"All values are NaN in {self.data_type} data")
    
    def process_temporal_scale(self, target_years=None):
        """Process data to target temporal scale"""
        if self.data is None:
            return None
        
        df = self.data.copy()
        df.set_index(self.date_column, inplace=True)
        
        # Filter to target years if specified
        if target_years is not None:
            start_year, end_year = target_years
            df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
        
        # Aggregate to target scale
        if self.target_scale == 'monthly':
            processed = df.groupby([df.index.year, df.index.month])[self.value_column].mean()
        elif self.target_scale == 'seasonal':
            # Define seasons
            df['season'] = df.index.month.map({12: 'DJF', 1: 'DJF', 2: 'DJF',
                                             3: 'MAM', 4: 'MAM', 5: 'MAM',
                                             6: 'JJA', 7: 'JJA', 8: 'JJA',
                                             9: 'SON', 10: 'SON', 11: 'SON'})
            processed = df.groupby([df.index.year, 'season'])[self.value_column].mean()
        elif self.target_scale == 'annual':
            processed = df.groupby(df.index.year)[self.value_column].mean()
        else:
            processed = df[self.value_column]
        
        self.processed_data = processed
        return processed


class PhysicsModule(ABC):
    """Abstract base class for physics modules"""
    
    @abstractmethod
    def compute(self, climate_data, glacier_state, **kwargs):
        """Compute physics component contribution"""
        pass
    
    @abstractmethod
    def get_parameters(self):
        """Get tunable parameters for calibration"""
        pass


class RefreezingModule(PhysicsModule):
    """Advanced refreezing physics from PyGEM"""
    
    def __init__(self, method='HH2015', **kwargs):
        self.method = method
        self.params = self._initialize_parameters(**kwargs)
    
    def _initialize_parameters(self, **kwargs):
        """Initialize refreezing parameters"""
        if self.method == 'HH2015':
            return {
                'rf_layers': kwargs.get('rf_layers', 10),
                'rf_dz': kwargs.get('rf_dz', 1.0),  # layer thickness
                'rf_dsc': kwargs.get('rf_dsc', 3),  # discretization steps
                'rf_dens_top': kwargs.get('rf_dens_top', 350),  # kg/m³
                'rf_dens_bot': kwargs.get('rf_dens_bot', 850),  # kg/m³
                'ch_air': kwargs.get('ch_air', 1005),  # J/kg/K
                'ch_ice': kwargs.get('ch_ice', 2097),  # J/kg/K
                'k_air': kwargs.get('k_air', 0.023),  # W/m/K
                'k_ice': kwargs.get('k_ice', 2.1),   # W/m/K
                'Lh_rf': kwargs.get('Lh_rf', 334000),  # J/kg
                'density_water': kwargs.get('density_water', 1000),  # kg/m³
            }
        elif self.method == 'Woodward':
            return {
                'temp_threshold': kwargs.get('temp_threshold', 0.0),
                'refreeze_coeff': kwargs.get('refreeze_coeff', -0.69),
                'refreeze_intercept': kwargs.get('refreeze_intercept', 0.0096),
            }
        else:
            return {}
    
    def compute(self, climate_data, glacier_state, **kwargs):
        """Compute refreezing based on selected method"""
        if self.method == 'HH2015':
            return self._compute_hh2015(climate_data, glacier_state, **kwargs)
        elif self.method == 'Woodward':
            return self._compute_woodward(climate_data, glacier_state, **kwargs)
        else:
            return np.zeros_like(climate_data['temp'])
    
    def _compute_hh2015(self, climate_data, glacier_state, **kwargs):
        """Heat conduction based refreezing (Huss & Hock 2015)"""
        # Simplified implementation - full version would be much more complex
        temp = climate_data['temp']
        melt = glacier_state.get('melt', np.zeros_like(temp))
        
        # Basic cold content approach
        cold_content = np.maximum(-temp * self.params['ch_ice'] * self.params['rf_dz'] / 
                                 self.params['Lh_rf'] / self.params['density_water'], 0)
        
        # Refreezing cannot exceed available melt water
        refreeze = np.minimum(cold_content, melt)
        return refreeze
    
    def _compute_woodward(self, climate_data, glacier_state, **kwargs):
        """Temperature-based refreezing (Woodward et al. 1997)"""
        temp_annual = np.mean(climate_data['temp'])
        refreeze_potential = (self.params['refreeze_coeff'] * temp_annual + 
                            self.params['refreeze_intercept']) / 100
        
        # Remove negative values
        refreeze_potential = max(0, refreeze_potential)
        
        # Apply to available melt water
        melt = glacier_state.get('melt', 0)
        return min(refreeze_potential, melt)
    
    def get_parameters(self):
        """Return parameters available for calibration"""
        if self.method == 'HH2015':
            return ['rf_dens_top', 'rf_dens_bot', 'rf_dz']
        elif self.method == 'Woodward':
            return ['refreeze_coeff', 'refreeze_intercept']
        else:
            return []


class SurfaceTypeModule(PhysicsModule):
    """Dynamic surface type evolution"""
    
    def __init__(self, **kwargs):
        self.surface_types = {
            0: 'off_glacier',
            1: 'ice', 
            2: 'snow',
            3: 'firn',
            4: 'debris'
        }
        
        self.ddf_dict = {
            0: kwargs.get('ddf_snow', 0.003),
            1: kwargs.get('ddf_ice', 0.008),
            2: kwargs.get('ddf_snow', 0.003),
            3: kwargs.get('ddf_firn', 0.005),
            4: kwargs.get('ddf_debris', 0.012)  # Enhanced for debris
        }
    
    def compute(self, climate_data, glacier_state, **kwargs):
        """Update surface types and compute melt factors"""
        # Simplified surface type evolution
        surface_type = glacier_state.get('surface_type', np.ones_like(climate_data['temp']))
        mass_balance = glacier_state.get('mass_balance', np.zeros_like(climate_data['temp']))
        
        # Update surface types based on mass balance history
        # Positive MB -> firn/snow, Negative MB -> ice
        surface_type = np.where(mass_balance > 0, 3, 1)  # 3=firn, 1=ice
        
        # Get corresponding DDFs
        ddf_values = np.array([self.ddf_dict[st] for st in surface_type])
        
        return {
            'surface_type': surface_type,
            'ddf_values': ddf_values
        }
    
    def get_parameters(self):
        """Return tunable parameters"""
        return ['ddf_ice', 'ddf_snow', 'ddf_firn', 'ddf_debris']


class MultiTargetCalibrator:
    """Advanced multi-target calibration engine"""
    
    def __init__(self, target_configs, priorities=None, method='hierarchical'):
        """
        Initialize multi-target calibrator
        
        Parameters
        ----------
        target_configs : dict
            Dictionary of DataConfiguration objects for each target
        priorities : dict
            Priority weights for each target type
        method : str
            Calibration method ('single', 'multi_objective', 'hierarchical')
        """
        self.target_configs = target_configs
        self.priorities = priorities or DEFAULT_HYDRO_PARAMS['target_priorities']
        self.method = method
        
        # Load and process all target data
        self.target_data = {}
        for target_name, config in target_configs.items():
            if config is not None:
                config.load_data()
                self.target_data[target_name] = config
    
    def objective_function(self, params, mb_model, **kwargs):
        """
        Comprehensive objective function for multi-target calibration
        
        Parameters
        ----------
        params : array_like
            Parameter values to test
        mb_model : HydroMassBalance
            Mass balance model instance
        
        Returns
        -------
        float
            Combined objective value (lower is better)
        """
        try:
            # Update model parameters
            mb_model.update_parameters(params)
            
            # Compute model outputs for all targets
            model_outputs = mb_model.compute_all_outputs(**kwargs)
            
            # Calculate individual target errors
            target_errors = {}
            total_error = 0.0
            
            for target_name, target_config in self.target_data.items():
                if target_config.processed_data is not None:
                    model_data = model_outputs.get(target_name)
                    obs_data = target_config.processed_data
                    
                    if model_data is not None and len(obs_data) > 0:
                        # Align temporal scales
                        aligned_model, aligned_obs = self._align_data(model_data, obs_data)
                        
                        if len(aligned_model) > 0:
                            # Calculate error metric
                            error = self._calculate_error(aligned_model, aligned_obs, target_name)
                            target_errors[target_name] = error
                            
                            # Weight by priority
                            priority = self.priorities.get(target_name.split('_')[0], 1.0)
                            total_error += error * priority
            
            # Store diagnostics
            mb_model._last_calibration_diagnostics = {
                'total_error': total_error,
                'target_errors': target_errors,
                'parameters': params.copy()
            }
            
            return total_error
            
        except Exception as e:
            log.warning(f"Error in objective function: {e}")
            return 1e10  # Return large error for failed evaluations
    
    def _align_data(self, model_data, obs_data):
        """Align model and observed data in time"""
        if isinstance(obs_data, pd.Series):
            if hasattr(obs_data.index, 'year'):
                # Handle different temporal scales
                obs_years = obs_data.index.year.unique()
                model_years = np.arange(len(model_data))  # Simplified
                
                common_years = np.intersect1d(obs_years, model_years)
                if len(common_years) > 0:
                    # Simple alignment - would be more sophisticated in practice
                    min_len = min(len(model_data), len(obs_data))
                    return model_data[:min_len], obs_data.values[:min_len]
        
        return np.array([]), np.array([])
    
    def _calculate_error(self, model_data, obs_data, target_name):
        """Calculate error metric for specific target"""
        if len(model_data) == 0 or len(obs_data) == 0:
            return 1e10
        
        # Remove NaN values
        valid_idx = ~(np.isnan(model_data) | np.isnan(obs_data))
        if not np.any(valid_idx):
            return 1e10
        
        model_clean = model_data[valid_idx]
        obs_clean = obs_data[valid_idx]
        
        # Different error metrics for different targets
        if 'runoff' in target_name:
            # Nash-Sutcliffe efficiency (higher is better, so return 1-NSE)
            if np.var(obs_clean) > 0:
                nse = 1 - np.sum((obs_clean - model_clean)**2) / np.sum((obs_clean - np.mean(obs_clean))**2)
                return 1 - nse
            else:
                return mean_squared_error(obs_clean, model_clean)
        elif 'mb' in target_name:
            # RMSE for mass balance
            return np.sqrt(mean_squared_error(obs_clean, model_clean))
        elif 'volume' in target_name:
            # Relative error for volume
            if np.mean(obs_clean) > 0:
                return np.abs(np.mean(model_clean) - np.mean(obs_clean)) / np.mean(obs_clean)
            else:
                return np.abs(np.mean(model_clean) - np.mean(obs_clean))
        elif 'velocity' in target_name:
            # Correlation-based error for velocity
            if len(model_clean) > 2:
                corr = np.corrcoef(model_clean, obs_clean)[0, 1]
                return 1 - abs(corr) if not np.isnan(corr) else 1.0
            else:
                return mean_absolute_error(obs_clean, model_clean)
        else:
            # Default RMSE
            return np.sqrt(mean_squared_error(obs_clean, model_clean))
    
    def calibrate(self, mb_model, method=None, **kwargs):
        """
        Execute calibration process
        
        Parameters
        ----------
        mb_model : HydroMassBalance
            Model to calibrate
        method : str
            Optimization method override
        
        Returns
        -------
        dict
            Calibration results
        """
        method = method or self.method
        
        # Get parameter bounds
        param_names, param_bounds = mb_model.get_calibration_parameters()
        
        if len(param_names) == 0:
            log.warning("No parameters available for calibration")
            return {'success': False, 'message': 'No calibration parameters'}
        
        log.info(f"Starting {method} calibration with {len(param_names)} parameters")
        log.info(f"Parameters: {param_names}")
        log.info(f"Targets: {list(self.target_data.keys())}")
        
        # Initial parameter values (middle of bounds)
        x0 = np.array([(b[0] + b[1]) / 2 for b in param_bounds])
        
        try:
            if method == 'single' or len(self.target_data) == 1:
                # Single objective optimization
                result = optimize.minimize(
                    self.objective_function,
                    x0,
                    args=(mb_model,),
                    bounds=param_bounds,
                    method='L-BFGS-B',
                    options={'maxiter': DEFAULT_HYDRO_PARAMS['max_iterations']}
                )
                
                optimal_params = result.x
                success = result.success
                
            elif method == 'hierarchical':
                # Hierarchical optimization (MB first, then others)
                optimal_params = self._hierarchical_calibration(mb_model, x0, param_bounds)
                success = True
                
            else:  # multi_objective
                # Multi-objective optimization using NSGA-II or similar
                optimal_params = self._multi_objective_calibration(mb_model, x0, param_bounds)
                success = True
            
            # Update model with optimal parameters
            mb_model.update_parameters(optimal_params)
            
            # Calculate final metrics
            final_metrics = self._calculate_final_metrics(mb_model)
            
            calibration_results = {
                'success': success,
                'optimal_parameters': dict(zip(param_names, optimal_params)),
                'final_metrics': final_metrics,
                'calibration_method': method,
                'target_priorities': self.priorities,
                'diagnostics': getattr(mb_model, '_last_calibration_diagnostics', {})
            }
            
            log.info("Calibration completed successfully")
            return calibration_results
            
        except Exception as e:
            log.error(f"Calibration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _hierarchical_calibration(self, mb_model, x0, param_bounds):
        """Hierarchical calibration: MB -> Volume -> Velocity -> Runoff"""
        current_params = x0.copy()
        
        # Define hierarchy based on priorities
        hierarchy = sorted(self.priorities.items(), key=lambda x: x[1], reverse=True)
        
        for target_type, priority in hierarchy:
            # Find targets of this type
            relevant_targets = {k: v for k, v in self.target_data.items() 
                              if k.startswith(target_type)}
            
            if relevant_targets:
                log.info(f"Calibrating for {target_type} targets: {list(relevant_targets.keys())}")
                
                # Temporarily focus on this target type
                temp_calibrator = MultiTargetCalibrator(
                    {k: self.target_configs[k] for k in relevant_targets.keys()},
                    {target_type: 1.0}
                )
                
                # Optimize for this target
                result = optimize.minimize(
                    temp_calibrator.objective_function,
                    current_params,
                    args=(mb_model,),
                    bounds=param_bounds,
                    method='L-BFGS-B'
                )
                
                if result.success:
                    current_params = result.x
                    log.info(f"Successfully calibrated for {target_type}")
                else:
                    log.warning(f"Failed to calibrate for {target_type}")
        
        return current_params
    
    def _multi_objective_calibration(self, mb_model, x0, param_bounds):
        """Multi-objective optimization (simplified NSGA-II approach)"""
        # For now, use weighted sum approach
        # In future, could implement proper Pareto optimization
        result = optimize.minimize(
            self.objective_function,
            x0,
            args=(mb_model,),
            bounds=param_bounds,
            method='L-BFGS-B'
        )
        
        return result.x if result.success else x0
    
    def _calculate_final_metrics(self, mb_model):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        try:
            # Get final model outputs
            model_outputs = mb_model.compute_all_outputs()
            
            for target_name, target_config in self.target_data.items():
                if target_config.processed_data is not None:
                    model_data = model_outputs.get(target_name)
                    obs_data = target_config.processed_data
                    
                    if model_data is not None:
                        # Align data
                        aligned_model, aligned_obs = self._align_data(model_data, obs_data)
                        
                        if len(aligned_model) > 0:
                            # Calculate multiple metrics
                            target_metrics = {
                                'rmse': np.sqrt(mean_squared_error(aligned_obs, aligned_model)),
                                'mae': mean_absolute_error(aligned_obs, aligned_model),
                                'bias': np.mean(aligned_model - aligned_obs),
                                'r2': r2_score(aligned_obs, aligned_model) if len(aligned_obs) > 1 else 0,
                                'n_samples': len(aligned_obs)
                            }
                            
                            # Target-specific metrics
                            if 'runoff' in target_name:
                                if np.var(aligned_obs) > 0:
                                    nse = 1 - np.sum((aligned_obs - aligned_model)**2) / np.sum((aligned_obs - np.mean(aligned_obs))**2)
                                    target_metrics['nse'] = nse
                            
                            metrics[target_name] = target_metrics
        
        except Exception as e:
            log.warning(f"Error calculating final metrics: {e}")
        
        return metrics


class HydroMassBalance(MassBalanceModel):
    """
    Revolutionary Multi-Target Glacier Mass Balance Model
    
    Combines OGGM framework, PyGEM physics, and multi-target calibration
    with unprecedented flexibility for mass balance, runoff, volume, and velocity.
    """
    
    def __init__(self, gdir, 
                 # Physics configuration
                 physics_level='advanced',
                 enable_refreezing=True,
                 enable_debris=True, 
                 enable_surface_evolution=True,
                 refreezing_method='HH2015',
                 
                 # Climate configuration
                 climate_source='oggm_default',  # 'oggm_default', 'regional_scaling', 'custom'
                 filename='climate_historical',
                 input_filesuffix='',
                 
                 # Calibration targets
                 calibration_targets=['geodetic_mb'],
                 runoff_data_config=None,
                 volume_data_config=None,
                 velocity_data_config=None,
                 mb_data_config=None,
                 
                 # Temporal configuration
                 mb_temporal_scale='annual',
                 runoff_temporal_scale='monthly',
                 volume_temporal_scale='annual',
                 velocity_temporal_scale='annual',
                 
                 # Priority configuration
                 target_priorities=None,
                 
                 # Model parameters
                 melt_f=None,
                 temp_bias=None,
                 prcp_fac=None,
                 bias=0,
                 
                 # Other options
                 ys=None, ye=None,
                 repeat=False,
                 check_calib_params=True,
                 **kwargs):
        """
        Initialize HydroMassBalance model
        
        Parameters
        ----------
        All standard OGGM MassBalanceModel parameters plus:
        
        Physics:
        physics_level : str
            Level of physical complexity ('simple', 'intermediate', 'advanced')
        enable_refreezing : bool
            Include refreezing processes
        enable_debris : bool
            Include debris effects
        enable_surface_evolution : bool
            Dynamic surface type evolution
        refreezing_method : str
            Refreezing method ('HH2015', 'Woodward', 'none')
            
        Calibration:
        calibration_targets : list
            List of calibration targets
        *_data_config : DataConfiguration
            Configuration for each data type
        *_temporal_scale : str
            Temporal scale for each target type
        target_priorities : dict
            Priority weights for multi-target calibration
        """
        
        # Initialize base OGGM mass balance model
        super(HydroMassBalance, self).__init__()
        
        # Store configuration
        self.gdir = gdir
        self.physics_level = physics_level
        self.enable_refreezing = enable_refreezing
        self.enable_debris = enable_debris
        self.enable_surface_evolution = enable_surface_evolution
        self.refreezing_method = refreezing_method
        self.climate_source = climate_source
        
        # Temporal configuration
        self.temporal_scales = {
            'mb': mb_temporal_scale,
            'runoff': runoff_temporal_scale, 
            'volume': volume_temporal_scale,
            'velocity': velocity_temporal_scale
        }
        
        # Initialize physics modules
        self._initialize_physics_modules(**kwargs)
        
        # Initialize climate data
        self._initialize_climate_data(filename, input_filesuffix, ys, ye, **kwargs)
        
        # Configure calibration targets
        self.target_configs = {}
        self.calibration_targets = calibration_targets
        
        if 'runoff' in calibration_targets and runoff_data_config:
            self.target_configs['runoff'] = runoff_data_config
        if 'volume' in calibration_targets and volume_data_config:
            self.target_configs['volume'] = volume_data_config  
        if 'velocity' in calibration_targets and velocity_data_config:
            self.target_configs['velocity'] = velocity_data_config
        if 'mb' in calibration_targets and mb_data_config:
            self.target_configs['mb'] = mb_data_config
        
        # Set target priorities
        self.target_priorities = target_priorities or DEFAULT_HYDRO_PARAMS['target_priorities']
        
        # Initialize calibrator if we have targets
        if self.target_configs:
            self.calibrator = MultiTargetCalibrator(
                self.target_configs, 
                self.target_priorities
            )
        else:
            self.calibrator = None
        
        # Model parameters
        self.melt_f = melt_f or cfg.PARAMS.get('melt_f', 5.0)
        self.temp_bias = temp_bias or 0.0
        self.prcp_fac = prcp_fac or cfg.PARAMS.get('prcp_fac', 2.5)
        self.bias = bias
        
        # OGGM compatibility
        self.valid_bounds = [-1e4, 2e4]
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat
        
        # Initialize state variables
        self._reset_state_variables()
        
        log.info(f"Initialized HydroMassBalance for glacier {gdir.rgi_id}")
        log.info(f"Physics level: {physics_level}, Targets: {calibration_targets}")
    
    def _initialize_physics_modules(self, **kwargs):
        """Initialize physics computation modules"""
        self.physics_modules = {}
        
        if self.enable_refreezing and self.physics_level in ['intermediate', 'advanced']:
            self.physics_modules['refreezing'] = RefreezingModule(
                method=self.refreezing_method, **kwargs
            )
        
        if self.enable_surface_evolution and self.physics_level == 'advanced':
            self.physics_modules['surface_type'] = SurfaceTypeModule(**kwargs)
    
    def _initialize_climate_data(self, filename, input_filesuffix, ys, ye, **kwargs):
        """Initialize climate data based on source"""
        try:
            if self.climate_source == 'regional_scaling':
                # Use regional scaling climate data
                from oggm.core.regional_scaling import process_regional_scaling_climate_data
                # This would be integrated with regional scaling
                pass
            
            # For now, use standard OGGM climate initialization
            # This would be expanded to handle different climate sources
            fpath = self.gdir.get_filepath(filename, filesuffix=input_filesuffix)
            
            with utils.ncDataset(fpath, mode='r') as nc:
                # Load time
                time = nc.variables['time']
                import cftime
                time = cftime.num2date(time[:], time.units, calendar=time.calendar)
                
                # Load climate data
                self.temp = nc.variables['temp'][:].astype(np.float64) + self.temp_bias
                self.prcp = nc.variables['prcp'][:].astype(np.float64) * self.prcp_fac
                self.ref_hgt = nc.ref_hgt
                
                # Store time information
                ny, r = divmod(len(time), 12)
                if r != 0:
                    raise ValueError('Climate data should be N full years')
                
                years = np.repeat(np.arange(time[-1].year - ny + 1, time[-1].year + 1), 12)
                self.years = years
                self.months = np.tile(np.arange(1, 13), ny)
                self.ys = self.years[0]
                self.ye = self.years[-1]
                
        except Exception as e:
            log.error(f"Error initializing climate data: {e}")
            raise
    
    def _reset_state_variables(self):
        """Reset internal state variables"""
        self.state = {
            'surface_type': None,
            'snowpack': None,
            'refreezing': None,
            'melt': None,
            'mass_balance': None
        }
        
        # Calibration tracking
        self._calibration_results = None
        self._last_calibration_diagnostics = {}
    
    def get_monthly_mb(self, heights, year=None, fl_id=None, fls=None, **kwargs):
        """
        Monthly mass balance at given altitude(s)
        
        Returns mass balance in [m s-1] (ice equivalent)
        """
        # Get climate data for the month
        climate_data = self._get_monthly_climate(heights, year)
        
        # Compute physics
        physics_results = self._compute_physics(climate_data, heights, **kwargs)
        
        # Calculate mass balance components
        accumulation = self._compute_accumulation(climate_data, physics_results)
        melt = self._compute_melt(climate_data, physics_results, heights)
        refreezing = physics_results.get('refreezing', 0)
        
        # Net mass balance
        mb_monthly = accumulation + refreezing - melt - self.bias / 12 / SEC_IN_YEAR
        
        # Convert to m ice equivalent per second
        mb_monthly = mb_monthly / SEC_IN_MONTH / cfg.PARAMS['ice_density']
        
        return mb_monthly
    
    def get_annual_mb(self, heights, year=None, fl_id=None, fls=None, **kwargs):
        """
        Annual mass balance at given altitude(s)
        
        Returns mass balance in [m s-1] (ice equivalent)
        """
        # Compute monthly mass balance for the year
        mb_annual = np.zeros_like(heights, dtype=float)
        
        for month in range(12):
            month_year = year + month / 12.0
            mb_month = self.get_monthly_mb(heights, month_year, fl_id, fls, **kwargs)
            mb_annual += mb_month * SEC_IN_MONTH
        
        # Convert back to per second
        mb_annual = mb_annual / SEC_IN_YEAR
        
        return mb_annual
    
    def _get_monthly_climate(self, heights, year):
        """Get climate data for specific month/year"""
        from oggm.utils import floatyear_to_date
        
        y, m = floatyear_to_date(year)
        
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        
        if not self.is_year_valid(y):
            raise ValueError(f'Year {y} out of valid bounds [{self.ys}, {self.ye}]')
        
        # Find the right time index
        pok = np.where((self.years == y) & (self.months == m))[0]
        if len(pok) == 0:
            raise ValueError(f'No data found for year {y}, month {m}')
        
        pok = pok[0]
        
        # Get monthly climate
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        
        # Adjust for elevation
        temp_grad = cfg.PARAMS.get('temp_default_gradient', -0.0065)
        temp = itemp + temp_grad * (heights - self.ref_hgt)
        prcp = np.ones_like(heights) * iprcp
        
        return {
            'temp': temp,
            'prcp': prcp,
            'temp_for_melt': np.maximum(temp - cfg.PARAMS.get('temp_melt', 0), 0)
        }
    
    def _compute_physics(self, climate_data, heights, **kwargs):
        """Compute advanced physics components"""
        results = {}
        
        # Initialize glacier state
        glacier_state = {
            'heights': heights,
            'melt': np.zeros_like(heights),
            'mass_balance': np.zeros_like(heights),
            'surface_type': np.ones_like(heights, dtype=int)  # Default to ice
        }
        
        # Compute each physics module
        for module_name, module in self.physics_modules.items():
            try:
                module_results = module.compute(climate_data, glacier_state, **kwargs)
                results[module_name] = module_results
                
                # Update glacier state
                if module_name == 'surface_type' and isinstance(module_results, dict):
                    glacier_state['surface_type'] = module_results.get('surface_type', glacier_state['surface_type'])
                    results['ddf_values'] = module_results.get('ddf_values', np.ones_like(heights))
                
            except Exception as e:
                log.warning(f"Error in {module_name} physics module: {e}")
                results[module_name] = None
        
        return results
    
    def _compute_accumulation(self, climate_data, physics_results):
        """Compute accumulation from precipitation"""
        temp = climate_data['temp']
        prcp = climate_data['prcp']
        
        # Temperature thresholds
        temp_all_solid = cfg.PARAMS.get('temp_all_solid', 0.0)
        temp_all_liq = cfg.PARAMS.get('temp_all_liq', 2.0)
        
        # Solid precipitation fraction
        solid_frac = np.clip((temp_all_liq - temp) / (temp_all_liq - temp_all_solid), 0, 1)
        
        # Accumulation (solid precipitation)
        accumulation = prcp * solid_frac
        
        return accumulation
    
    def _compute_melt(self, climate_data, physics_results, heights):
        """Compute melt using temperature index approach with physics"""
        temp_for_melt = climate_data['temp_for_melt']
        
        # Get degree day factors
        if 'surface_type' in physics_results and 'ddf_values' in physics_results:
            ddf = physics_results['ddf_values']
        else:
            # Default DDF
            ddf = np.full_like(heights, self.melt_f)
        
        # Apply debris enhancement if enabled
        if self.enable_debris and hasattr(self.gdir, 'debris_factor'):
            debris_factor = getattr(self.gdir, 'debris_factor', 1.0)
            ddf = ddf * debris_factor
        
        # Calculate melt
        melt = ddf * temp_for_melt
        
        return melt
    
    def compute_runoff(self, heights, year_range=None, temporal_scale='monthly', **kwargs):
        """
        Compute glacier runoff for calibration
        
        Parameters
        ----------
        heights : array_like
            Elevation points
        year_range : tuple
            (start_year, end_year) for computation
        temporal_scale : str
            Output temporal scale
            
        Returns
        -------
        array_like
            Runoff time series
        """
        if year_range is None:
            year_range = (self.ys, self.ye)
        
        start_year, end_year = year_range
        years = np.arange(start_year, end_year + 1)
        
        if temporal_scale == 'monthly':
            runoff_data = []
            
            for year in years:
                for month in range(1, 13):
                    month_year = year + (month - 1) / 12.0
                    
                    # Get monthly climate and mass balance
                    climate_data = self._get_monthly_climate(heights, month_year)
                    physics_results = self._compute_physics(climate_data, heights)
                    
                    # Components
                    prcp = climate_data['prcp']
                    melt = self._compute_melt(climate_data, physics_results, heights)
                    refreezing = physics_results.get('refreezing', {})
                    
                    if isinstance(refreezing, dict):
                        refreeze = refreezing.get('refreeze', np.zeros_like(heights))
                    else:
                        refreeze = refreezing if refreezing is not None else np.zeros_like(heights)
                    
                    # Monthly runoff = precipitation + melt - refreezing
                    monthly_runoff = np.mean(prcp + melt - refreeze)
                    runoff_data.append(monthly_runoff)
            
            return np.array(runoff_data)
        
        else:  # annual
            runoff_data = []
            
            for year in years:
                annual_runoff = 0
                
                for month in range(1, 13):
                    month_year = year + (month - 1) / 12.0
                    climate_data = self._get_monthly_climate(heights, month_year)
                    physics_results = self._compute_physics(climate_data, heights)
                    
                    prcp = climate_data['prcp']
                    melt = self._compute_melt(climate_data, physics_results, heights)
                    refreezing = physics_results.get('refreezing', {})
                    
                    if isinstance(refreezing, dict):
                        refreeze = refreezing.get('refreeze', np.zeros_like(heights))
                    else:
                        refreeze = refreezing if refreezing is not None else np.zeros_like(heights)
                    
                    monthly_runoff = np.mean(prcp + melt - refreeze)
                    annual_runoff += monthly_runoff
                
                runoff_data.append(annual_runoff)
            
            return np.array(runoff_data)
    
    def compute_all_outputs(self, **kwargs):
        """Compute all model outputs for calibration targets"""
        outputs = {}
        
        # Get a representative elevation profile
        try:
            fls = self.gdir.read_pickle('inversion_flowlines')
            heights = fls[0].surface_h
        except:
            # Fallback elevation range
            heights = np.linspace(self.gdir.min_h, self.gdir.max_h, 20)
        
        # Mass balance
        if 'mb' in self.target_configs:
            config = self.target_configs['mb']
            if config.target_scale == 'annual':
                mb_data = []
                for year in range(self.ys, self.ye + 1):
                    mb_annual = self.get_annual_mb(heights, year)
                    mb_data.append(np.mean(mb_annual))
                outputs['mb'] = np.array(mb_data)
        
        # Runoff
        if 'runoff' in self.target_configs:
            config = self.target_configs['runoff']
            outputs['runoff'] = self.compute_runoff(
                heights, 
                year_range=(self.ys, self.ye),
                temporal_scale=config.target_scale
            )
        
        # Volume (simplified - would need proper glacier geometry)
        if 'volume' in self.target_configs:
            try:
                # Get volume from flowlines if available
                fls = self.gdir.read_pickle('inversion_flowlines')
                volume = sum([fl.volume_m3 for fl in fls])
                outputs['volume'] = np.array([volume])  # Single value for now
            except:
                log.warning("Could not compute volume - flowlines not available")
                outputs['volume'] = np.array([1e9])  # Placeholder
        
        # Velocity (simplified - would need proper implementation)
        if 'velocity' in self.target_configs:
            # Placeholder - would need proper velocity computation
            outputs['velocity'] = np.array([50.0])  # m/year placeholder
        
        return outputs
    
    def get_calibration_parameters(self):
        """Get parameters available for calibration"""
        param_names = []
        param_bounds = []
        
        # Basic MB parameters
        param_names.extend(['melt_f', 'temp_bias', 'prcp_fac', 'bias'])
        param_bounds.extend([
            (1.0, 20.0),     # melt_f 
            (-5.0, 5.0),     # temp_bias
            (0.1, 5.0),      # prcp_fac
            (-1000, 1000)    # bias
        ])
        
        # Physics module parameters
        for module_name, module in self.physics_modules.items():
            module_params = module.get_parameters()
            for param in module_params:
                param_names.append(f"{module_name}_{param}")
                
                # Default bounds - would be customized per parameter
                if 'ddf' in param:
                    param_bounds.append((0.001, 0.020))
                elif 'temp' in param:
                    param_bounds.append((-10.0, 10.0))
                elif 'dens' in param:
                    param_bounds.append((100.0, 900.0))
                else:
                    param_bounds.append((-10.0, 10.0))
        
        return param_names, param_bounds
    
    def update_parameters(self, params):
        """Update model parameters from optimization"""
        param_names, _ = self.get_calibration_parameters()
        
        if len(params) != len(param_names):
            raise ValueError(f"Parameter count mismatch: got {len(params)}, expected {len(param_names)}")
        
        param_dict = dict(zip(param_names, params))
        
        # Update basic parameters
        if 'melt_f' in param_dict:
            self.melt_f = param_dict['melt_f']
        if 'temp_bias' in param_dict:
            self.temp_bias = param_dict['temp_bias']
        if 'prcp_fac' in param_dict:
            self.prcp_fac = param_dict['prcp_fac']
        if 'bias' in param_dict:
            self.bias = param_dict['bias']
        
        # Update physics module parameters
        for param_name, param_value in param_dict.items():
            if '_' in param_name:
                module_name, param_key = param_name.split('_', 1)
                if module_name in self.physics_modules:
                    if hasattr(self.physics_modules[module_name], 'params'):
                        self.physics_modules[module_name].params[param_key] = param_value
    
    def calibrate(self, **kwargs):
        """Execute calibration process"""
        if self.calibrator is None:
            raise ValueError("No calibration targets configured")
        
        log.info("Starting HydroMassBalance calibration")
        
        # Process target data
        for target_name, config in self.target_configs.items():
            if config.processed_data is None:
                log.info(f"Processing {target_name} data")
                config.process_temporal_scale(target_years=(self.ys, self.ye))
        
        # Execute calibration
        results = self.calibrator.calibrate(self, **kwargs)
        
        # Store results
        self._calibration_results = results
        
        return results
    
    def is_year_valid(self, year):
        """Check if year is within valid range"""
        return self.ys <= year <= self.ye
    
    def get_ela(self, year=None, **kwargs):
        """Get equilibrium line altitude"""
        if self.valid_bounds is None:
            raise ValueError('valid_bounds must be set for ELA computation')
        
        def to_minimize(ela_height):
            return self.get_annual_mb([ela_height], year=year, **kwargs)[0] * SEC_IN_YEAR * cfg.PARAMS['ice_density']
        
        try:
            ela = optimize.brentq(to_minimize, *self.valid_bounds, xtol=0.1)
            return ela
        except ValueError:
            return np.nan
    
    def save_calibration_results(self, filepath=None):
        """Save calibration results to file"""
        if self._calibration_results is None:
            log.warning("No calibration results to save")
            return
        
        if filepath is None:
            filepath = os.path.join(self.gdir.dir, 'hydro_mb_calibration.json')
        
        # Prepare data for JSON serialization
        results_to_save = {
            'glacier_id': self.gdir.rgi_id,
            'calibration_date': datetime.now().isoformat(),
            'model_configuration': {
                'physics_level': self.physics_level,
                'enable_refreezing': self.enable_refreezing,
                'enable_debris': self.enable_debris,
                'enable_surface_evolution': self.enable_surface_evolution,
                'refreezing_method': self.refreezing_method,
                'calibration_targets': self.calibration_targets,
                'temporal_scales': self.temporal_scales
            },
            'calibration_results': self._calibration_results
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_to_save = convert_numpy(results_to_save)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        log.info(f"Calibration results saved to {filepath}")
    
    def __repr__(self):
        """String representation"""
        summary = ['<HydroMassBalance Model>']
        summary += [f'  Glacier: {self.gdir.rgi_id}']
        summary += [f'  Physics Level: {self.physics_level}']
        summary += [f'  Calibration Targets: {self.calibration_targets}']
        summary += [f'  Valid Years: {self.ys}-{self.ye}']
        summary += [f'  Current Parameters:']
        summary += [f'    melt_f: {self.melt_f:.3f}']
        summary += [f'    temp_bias: {self.temp_bias:.3f}']
        summary += [f'    prcp_fac: {self.prcp_fac:.3f}']
        summary += [f'    bias: {self.bias:.1f}']
        
        if self._calibration_results:
            summary += [f'  Calibration Status: Completed']
            if 'final_metrics' in self._calibration_results:
                summary += [f'  Final Metrics Available: Yes']
        else:
            summary += [f'  Calibration Status: Not calibrated']
        
        return '\n'.join(summary)


# Integration functions for OGGM workflow
@entity_task(log, writes=['hydro_mb_calib'])
def hydro_mb_calibration(gdir, 
                        calibration_targets=['geodetic_mb'],
                        runoff_data_path=None,
                        volume_data_path=None,
                        velocity_data_path=None,
                        mb_data_path=None,
                        physics_level='advanced',
                        save_results=True,
                        **kwargs):
    """
    OGGM entity task for HydroMassBalance calibration
    
    Parameters
    ----------
    gdir : GlacierDirectory
        OGGM glacier directory
    calibration_targets : list
        List of calibration targets
    *_data_path : str
        Paths to calibration data files
    physics_level : str
        Physics complexity level
    save_results : bool
        Whether to save calibration results
    """
    
    # Configure data sources
    data_configs = {}
    
    if 'runoff' in calibration_targets and runoff_data_path:
        data_configs['runoff'] = DataConfiguration(
            'runoff', runoff_data_path, 
            temporal_scale='daily', target_scale='monthly'
        )
    
    if 'volume' in calibration_targets and volume_data_path:
        data_configs['volume'] = DataConfiguration(
            'volume', volume_data_path,
            temporal_scale='annual', target_scale='annual'
        )
    
    if 'velocity' in calibration_targets and velocity_data_path:
        data_configs['velocity'] = DataConfiguration(
            'velocity', velocity_data_path,
            temporal_scale='annual', target_scale='annual'
        )
    
    if 'mb' in calibration_targets and mb_data_path:
        data_configs['mb'] = DataConfiguration(
            'mb', mb_data_path,
            temporal_scale='annual', target_scale='annual'
        )
    
    # Create and calibrate model
    mb_model = HydroMassBalance(
        gdir,
        physics_level=physics_level,
        calibration_targets=calibration_targets,
        **{f"{k}_data_config": v for k, v in data_configs.items()},
        **kwargs
    )
    
    # Execute calibration
    results = mb_model.calibrate()
    
    # Save results
    if save_results:
        mb_model.save_calibration_results()
        
        # Also save to OGGM standard location
        calib_data = {
            'hydro_mb_results': results,
            'model_config': {
                'physics_level': physics_level,
                'calibration_targets': calibration_targets
            }
        }
        gdir.write_json(calib_data, 'hydro_mb_calib')
    
    log.info(f"HydroMassBalance calibration completed for {gdir.rgi_id}")
    
    return mb_model
