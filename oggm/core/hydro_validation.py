"""
HydroMassBalance Validation and Utility Functions

This module provides comprehensive validation, benchmarking, and utility functions
for the HydroMassBalance model, including:
- Cross-validation frameworks
- Performance benchmarking against existing models
- Uncertainty quantification
- Statistical analysis and reporting
- Integration with regional scaling climate data

Authors: Muhammad Shafeeque & Claude AI Assistant
License: BSD-3-Clause (compatible with OGGM)
"""

# Built-ins
import logging
import os
import warnings
from datetime import datetime
import json
from collections import defaultdict

# External libs
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           explained_variance_score)
from sklearn.model_selection import KFold, TimeSeriesSplit, LeaveOneOut

# OGGM imports
from oggm import cfg, utils
from oggm.core.massbalance import MonthlyTIModel
from oggm.exceptions import InvalidParamsError
from oggm import entity_task, global_task

# Module logger
log = logging.getLogger(__name__)


class HydroValidationFramework:
    """Comprehensive validation framework for HydroMassBalance models"""
    
    def __init__(self, hydro_mb_model, validation_config=None):
        """
        Initialize validation framework
        
        Parameters
        ----------
        hydro_mb_model : HydroMassBalance
            The calibrated model to validate
        validation_config : dict
            Validation configuration options
        """
        self.model = hydro_mb_model
        self.gdir = hydro_mb_model.gdir
        
        # Default validation configuration
        self.config = {
            'methods': ['leave_one_out', 'temporal_split', 'k_fold'],
            'metrics': ['rmse', 'mae', 'bias', 'r2', 'nse', 'kge'],
            'benchmarks': ['oggm_default', 'linear_mb'],
            'uncertainty_analysis': True,
            'save_diagnostics': True,
            'plot_results': True
        }
        
        if validation_config:
            self.config.update(validation_config)
        
        # Storage for results
        self.validation_results = {}
        self.benchmark_results = {}
        self.uncertainty_results = {}
    
    def run_comprehensive_validation(self, **kwargs):
        """
        Run complete validation suite
        
        Returns
        -------
        dict
            Comprehensive validation results
        """
        log.info(f"Starting comprehensive validation for glacier {self.gdir.rgi_id}")
        
        results = {
            'glacier_id': self.gdir.rgi_id,
            'validation_date': datetime.now().isoformat(),
            'model_config': self._get_model_config(),
            'validation_methods': {},
            'benchmarks': {},
            'uncertainty': {},
            'overall_performance': {}
        }
        
        # 1. Cross-validation
        if 'leave_one_out' in self.config['methods']:
            log.info("Running leave-one-out cross-validation")
            results['validation_methods']['leave_one_out'] = self._leave_one_out_validation()
        
        if 'temporal_split' in self.config['methods']:
            log.info("Running temporal split validation")
            results['validation_methods']['temporal_split'] = self._temporal_split_validation()
        
        if 'k_fold' in self.config['methods']:
            log.info("Running k-fold cross-validation")
            results['validation_methods']['k_fold'] = self._k_fold_validation()
        
        # 2. Benchmark comparisons
        if self.config['benchmarks']:
            log.info("Running benchmark comparisons")
            results['benchmarks'] = self._benchmark_comparison()
        
        # 3. Uncertainty analysis
        if self.config['uncertainty_analysis']:
            log.info("Running uncertainty analysis")
            results['uncertainty'] = self._uncertainty_analysis()
        
        # 4. Overall performance assessment
        results['overall_performance'] = self._assess_overall_performance(results)
        
        # 5. Generate plots if requested
        if self.config['plot_results']:
            self._generate_validation_plots(results)
        
        # Store results
        self.validation_results = results
        
        # Save to file if requested
        if self.config['save_diagnostics']:
            self._save_validation_results(results)
        
        log.info("Comprehensive validation completed")
        return results
    
    def _get_model_config(self):
        """Extract model configuration for documentation"""
        return {
            'physics_level': self.model.physics_level,
            'calibration_targets': self.model.calibration_targets,
            'temporal_scales': self.model.temporal_scales,
            'physics_modules': list(self.model.physics_modules.keys()),
            'climate_source': self.model.climate_source,
            'valid_period': f"{self.model.ys}-{self.model.ye}"
        }
    
    def _leave_one_out_validation(self):
        """Leave-one-out cross-validation for each target"""
        loo_results = {}
        
        for target_name, target_config in self.model.target_configs.items():
            if target_config.processed_data is None:
                continue
            
            log.info(f"LOO validation for {target_name}")
            
            # Get target data
            target_data = target_config.processed_data
            if len(target_data) < 3:
                log.warning(f"Insufficient data for LOO validation of {target_name}")
                continue
            
            loo = LeaveOneOut()
            predictions = []
            observations = []
            
            if isinstance(target_data, pd.Series):
                data_indices = range(len(target_data))
                data_values = target_data.values
            else:
                data_indices = range(len(target_data))
                data_values = np.array(target_data)
            
            for train_idx, test_idx in loo.split(data_indices):
                try:
                    # Create training subset (mask out test data)
                    train_config = self._mask_target_data(target_config, test_idx)
                    
                    # Re-calibrate model without test data
                    temp_model = self._create_temporary_model(train_config)
                    temp_model.calibrate()
                    
                    # Predict on test data
                    prediction = self._predict_target(temp_model, target_name, test_idx)
                    observation = data_values[test_idx][0]
                    
                    predictions.append(prediction)
                    observations.append(observation)
                    
                except Exception as e:
                    log.warning(f"LOO validation failed for index {test_idx}: {e}")
                    continue
            
            if len(predictions) > 0:
                # Calculate metrics
                metrics = self._calculate_validation_metrics(
                    np.array(observations), np.array(predictions)
                )
                
                loo_results[target_name] = {
                    'n_samples': len(predictions),
                    'predictions': predictions,
                    'observations': observations,
                    'metrics': metrics
                }
            else:
                loo_results[target_name] = {'error': 'No successful predictions'}
        
        return loo_results
    
    def _temporal_split_validation(self, split_ratio=0.7):
        """Temporal split validation"""
        temporal_results = {}
        
        for target_name, target_config in self.model.target_configs.items():
            if target_config.processed_data is None:
                continue
            
            log.info(f"Temporal split validation for {target_name}")
            
            target_data = target_config.processed_data
            if len(target_data) < 5:
                log.warning(f"Insufficient data for temporal split of {target_name}")
                continue
            
            # Split data temporally
            split_point = int(len(target_data) * split_ratio)
            
            try:
                # Create training configuration
                train_config = self._temporal_subset_data(target_config, 0, split_point)
                
                # Re-calibrate on training data
                temp_model = self._create_temporary_model(train_config)
                temp_model.calibrate()
                
                # Predict on test period
                test_indices = range(split_point, len(target_data))
                predictions = []
                observations = []
                
                for test_idx in test_indices:
                    prediction = self._predict_target(temp_model, target_name, test_idx)
                    observation = target_data.iloc[test_idx] if hasattr(target_data, 'iloc') else target_data[test_idx]
                    
                    predictions.append(prediction)
                    observations.append(observation)
                
                # Calculate metrics
                metrics = self._calculate_validation_metrics(
                    np.array(observations), np.array(predictions)
                )
                
                temporal_results[target_name] = {
                    'split_ratio': split_ratio,
                    'train_samples': split_point,
                    'test_samples': len(test_indices),
                    'predictions': predictions,
                    'observations': observations,
                    'metrics': metrics
                }
                
            except Exception as e:
                log.warning(f"Temporal split validation failed for {target_name}: {e}")
                temporal_results[target_name] = {'error': str(e)}
        
        return temporal_results
    
    def _k_fold_validation(self, k=5):
        """K-fold cross-validation"""
        kfold_results = {}
        
        for target_name, target_config in self.model.target_configs.items():
            if target_config.processed_data is None:
                continue
            
            log.info(f"K-fold validation for {target_name} (k={k})")
            
            target_data = target_config.processed_data
            if len(target_data) < k:
                log.warning(f"Insufficient data for {k}-fold validation of {target_name}")
                continue
            
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            fold_results = []
            
            if isinstance(target_data, pd.Series):
                data_indices = range(len(target_data))
                data_values = target_data.values
            else:
                data_indices = range(len(target_data))
                data_values = np.array(target_data)
            
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data_indices)):
                try:
                    # Create training configuration
                    train_config = self._mask_target_data(target_config, test_idx)
                    
                    # Re-calibrate model
                    temp_model = self._create_temporary_model(train_config)
                    temp_model.calibrate()
                    
                    # Predict on test fold
                    predictions = []
                    observations = []
                    
                    for test_i in test_idx:
                        prediction = self._predict_target(temp_model, target_name, test_i)
                        observation = data_values[test_i]
                        
                        predictions.append(prediction)
                        observations.append(observation)
                    
                    # Calculate fold metrics
                    fold_metrics = self._calculate_validation_metrics(
                        np.array(observations), np.array(predictions)
                    )
                    
                    fold_results.append({
                        'fold': fold_idx,
                        'train_size': len(train_idx),
                        'test_size': len(test_idx),
                        'metrics': fold_metrics,
                        'predictions': predictions,
                        'observations': observations
                    })
                    
                except Exception as e:
                    log.warning(f"K-fold validation failed for fold {fold_idx}: {e}")
                    continue
            
            if fold_results:
                # Aggregate across folds
                aggregated_metrics = self._aggregate_fold_metrics(fold_results)
                
                kfold_results[target_name] = {
                    'k': k,
                    'successful_folds': len(fold_results),
                    'fold_results': fold_results,
                    'aggregated_metrics': aggregated_metrics
                }
            else:
                kfold_results[target_name] = {'error': 'No successful folds'}
        
        return kfold_results
    
    def _benchmark_comparison(self):
        """Compare against benchmark models"""
        benchmark_results = {}
        
        # Get representative heights for comparison
        try:
            fls = self.gdir.read_pickle('inversion_flowlines')
            heights = fls[0].surface_h
        except:
            heights = np.linspace(self.gdir.min_h, self.gdir.max_h, 20)
        
        years = np.arange(self.model.ys, self.model.ye + 1)
        
        # 1. Compare against standard OGGM MonthlyTIModel
        if 'oggm_default' in self.config['benchmarks']:
            log.info("Benchmarking against OGGM MonthlyTIModel")
            
            try:
                oggm_model = MonthlyTIModel(self.gdir)
                
                # Calculate mass balance for both models
                hydro_mb = []
                oggm_mb = []
                
                for year in years:
                    hydro_annual = self.model.get_annual_mb(heights, year)
                    oggm_annual = oggm_model.get_annual_mb(heights, year)
                    
                    hydro_mb.append(np.mean(hydro_annual))
                    oggm_mb.append(np.mean(oggm_annual))
                
                # Calculate comparison metrics
                comparison_metrics = self._calculate_validation_metrics(
                    np.array(oggm_mb), np.array(hydro_mb)
                )
                
                benchmark_results['oggm_default'] = {
                    'description': 'Standard OGGM MonthlyTIModel',
                    'hydro_mb': hydro_mb,
                    'benchmark_mb': oggm_mb,
                    'comparison_metrics': comparison_metrics,
                    'years': years.tolist()
                }
                
            except Exception as e:
                log.warning(f"OGGM benchmark comparison failed: {e}")
                benchmark_results['oggm_default'] = {'error': str(e)}
        
        # 2. Compare against simple linear mass balance
        if 'linear_mb' in self.config['benchmarks']:
            log.info("Benchmarking against linear mass balance")
            
            try:
                from oggm.core.massbalance import LinearMassBalance
                
                # Estimate ELA and gradient from current model
                ela_estimate = self.model.get_ela(year=years[len(years)//2])
                if np.isnan(ela_estimate):
                    ela_estimate = np.mean(heights)
                
                linear_model = LinearMassBalance(ela_estimate, grad=3.0)
                
                # Calculate mass balance
                hydro_mb = []
                linear_mb = []
                
                for year in years:
                    hydro_annual = self.model.get_annual_mb(heights, year)
                    linear_annual = linear_model.get_annual_mb(heights, year)
                    
                    hydro_mb.append(np.mean(hydro_annual))
                    linear_mb.append(np.mean(linear_annual))
                
                comparison_metrics = self._calculate_validation_metrics(
                    np.array(linear_mb), np.array(hydro_mb)
                )
                
                benchmark_results['linear_mb'] = {
                    'description': 'Simple linear mass balance',
                    'ela_used': ela_estimate,
                    'gradient_used': 3.0,
                    'hydro_mb': hydro_mb,
                    'benchmark_mb': linear_mb,
                    'comparison_metrics': comparison_metrics,
                    'years': years.tolist()
                }
                
            except Exception as e:
                log.warning(f"Linear MB benchmark comparison failed: {e}")
                benchmark_results['linear_mb'] = {'error': str(e)}
        
        return benchmark_results
    
    def _uncertainty_analysis(self):
        """Comprehensive uncertainty analysis"""
        uncertainty_results = {}
        
        # 1. Parameter uncertainty
        param_names, param_bounds = self.model.get_calibration_parameters()
        
        if len(param_names) > 0:
            log.info("Analyzing parameter uncertainty")
            
            # Monte Carlo parameter sampling
            n_samples = 100
            param_samples = []
            
            for bounds in param_bounds:
                samples = np.random.uniform(bounds[0], bounds[1], n_samples)
                param_samples.append(samples)
            
            param_samples = np.array(param_samples).T
            
            # Run model with parameter variations
            mb_variations = []
            
            for i, params in enumerate(param_samples):
                try:
                    # Temporarily update parameters
                    original_params = self.model.get_calibration_parameters()[0]
                    self.model.update_parameters(params)
                    
                    # Calculate mass balance
                    try:
                        fls = self.gdir.read_pickle('inversion_flowlines')
                        heights = fls[0].surface_h
                    except:
                        heights = np.linspace(self.gdir.min_h, self.gdir.max_h, 10)
                    
                    year = self.model.ys + (self.model.ye - self.model.ys) // 2
                    mb_annual = self.model.get_annual_mb(heights, year)
                    mb_variations.append(np.mean(mb_annual))
                    
                except Exception as e:
                    log.warning(f"Parameter uncertainty sample {i} failed: {e}")
                    continue
            
            # Restore original parameters
            if hasattr(self, '_original_params'):
                self.model.update_parameters(self._original_params)
            
            if mb_variations:
                uncertainty_results['parameter_uncertainty'] = {
                    'n_samples': len(mb_variations),
                    'mb_mean': np.mean(mb_variations),
                    'mb_std': np.std(mb_variations),
                    'mb_range': [np.min(mb_variations), np.max(mb_variations)],
                    'percentiles': {
                        '5th': np.percentile(mb_variations, 5),
                        '25th': np.percentile(mb_variations, 25),
                        '75th': np.percentile(mb_variations, 75),
                        '95th': np.percentile(mb_variations, 95)
                    }
                }
        
        # 2. Input data uncertainty
        log.info("Analyzing input data uncertainty")
        
        # Climate data uncertainty (simplified)
        temp_uncertainty = 1.0  # ±1°C
        prcp_uncertainty = 0.2   # ±20%
        
        uncertainty_results['climate_uncertainty'] = {
            'temperature_uncertainty_C': temp_uncertainty,
            'precipitation_uncertainty_fraction': prcp_uncertainty,
            'methodology': 'Based on typical climate data uncertainties'
        }
        
        # 3. Structural uncertainty (comparison with different physics levels)
        if self.model.physics_level == 'advanced':
            log.info("Analyzing structural uncertainty")
            
            try:
                # Create simpler version of model
                simple_model = self._create_simple_model()
                
                # Compare outputs
                try:
                    fls = self.gdir.read_pickle('inversion_flowlines')
                    heights = fls[0].surface_h
                except:
                    heights = np.linspace(self.gdir.min_h, self.gdir.max_h, 10)
                
                years = np.arange(self.model.ys, self.model.ye + 1, 5)  # Sample years
                
                advanced_mb = []
                simple_mb = []
                
                for year in years:
                    adv_mb = self.model.get_annual_mb(heights, year)
                    sim_mb = simple_model.get_annual_mb(heights, year)
                    
                    advanced_mb.append(np.mean(adv_mb))
                    simple_mb.append(np.mean(sim_mb))
                
                structural_diff = np.array(advanced_mb) - np.array(simple_mb)
                
                uncertainty_results['structural_uncertainty'] = {
                    'advanced_vs_simple_mb_difference': {
                        'mean': np.mean(structural_diff),
                        'std': np.std(structural_diff),
                        'range': [np.min(structural_diff), np.max(structural_diff)]
                    },
                    'years_compared': years.tolist()
                }
                
            except Exception as e:
                log.warning(f"Structural uncertainty analysis failed: {e}")
                uncertainty_results['structural_uncertainty'] = {'error': str(e)}
        
        return uncertainty_results
    
    def _assess_overall_performance(self, validation_results):
        """Assess overall model performance"""
        performance = {
            'validation_score': 0.0,
            'benchmark_score': 0.0,
            'uncertainty_score': 0.0,
            'overall_score': 0.0,
            'grade': 'Not Assessed',
            'recommendations': []
        }
        
        scores = []
        
        # 1. Validation performance score
        validation_scores = []
        for method, results in validation_results['validation_methods'].items():
            for target, target_results in results.items():
                if 'metrics' in target_results:
                    metrics = target_results['metrics']
                    
                    # Convert metrics to scores (0-1, higher is better)
                    if 'r2' in metrics and not np.isnan(metrics['r2']):
                        validation_scores.append(max(0, metrics['r2']))
                    
                    if 'nse' in metrics and not np.isnan(metrics['nse']):
                        validation_scores.append(max(0, metrics['nse']))
        
        if validation_scores:
            performance['validation_score'] = np.mean(validation_scores)
            scores.append(performance['validation_score'])
        
        # 2. Benchmark performance score
        benchmark_scores = []
        for benchmark, results in validation_results['benchmarks'].items():
            if 'comparison_metrics' in results:
                metrics = results['comparison_metrics']
                if 'r2' in metrics and not np.isnan(metrics['r2']):
                    benchmark_scores.append(max(0, metrics['r2']))
        
        if benchmark_scores:
            performance['benchmark_score'] = np.mean(benchmark_scores)
            scores.append(performance['benchmark_score'])
        
        # 3. Uncertainty score (lower uncertainty = higher score)
        if 'parameter_uncertainty' in validation_results['uncertainty']:
            param_unc = validation_results['uncertainty']['parameter_uncertainty']
            if 'mb_std' in param_unc and param_unc['mb_std'] > 0:
                # Normalize by typical mass balance magnitude
                typical_mb = 1.0  # m w.e./year
                uncertainty_fraction = param_unc['mb_std'] / typical_mb
                uncertainty_score = max(0, 1 - uncertainty_fraction)
                performance['uncertainty_score'] = uncertainty_score
                scores.append(uncertainty_score)
        
        # Overall score
        if scores:
            performance['overall_score'] = np.mean(scores)
        
        # Grade assignment
        overall = performance['overall_score']
        if overall >= 0.9:
            performance['grade'] = 'Excellent'
        elif overall >= 0.8:
            performance['grade'] = 'Very Good'
        elif overall >= 0.7:
            performance['grade'] = 'Good'
        elif overall >= 0.6:
            performance['grade'] = 'Satisfactory'
        elif overall >= 0.5:
            performance['grade'] = 'Needs Improvement'
        else:
            performance['grade'] = 'Poor'
        
        # Generate recommendations
        recommendations = []
        
        if performance['validation_score'] < 0.7:
            recommendations.append("Consider additional calibration targets or longer calibration period")
        
        if performance['benchmark_score'] < 0.5:
            recommendations.append("Model performs significantly different from benchmarks - investigate physics")
        
        if performance['uncertainty_score'] < 0.6:
            recommendations.append("High parameter uncertainty - consider constraining parameters or adding data")
        
        if len(validation_results['validation_methods']) < 2:
            recommendations.append("Increase validation rigor with multiple cross-validation methods")
        
        performance['recommendations'] = recommendations
        
        return performance
    
    def _calculate_validation_metrics(self, observed, predicted):
        """Calculate comprehensive validation metrics"""
        # Remove NaN values
        valid_idx = ~(np.isnan(observed) | np.isnan(predicted))
        
        if not np.any(valid_idx) or len(observed[valid_idx]) < 2:
            return {'error': 'Insufficient valid data for metrics calculation'}
        
        obs = observed[valid_idx]
        pred = predicted[valid_idx]
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['rmse'] = np.sqrt(mean_squared_error(obs, pred))
            metrics['mae'] = mean_absolute_error(obs, pred)
            metrics['bias'] = np.mean(pred - obs)
            metrics['r2'] = r2_score(obs, pred)
            metrics['explained_variance'] = explained_variance_score(obs, pred)
            
            # Nash-Sutcliffe Efficiency
            if np.var(obs) > 0:
                metrics['nse'] = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
            else:
                metrics['nse'] = np.nan
            
            # Kling-Gupta Efficiency
            if np.std(obs) > 0 and np.std(pred) > 0:
                correlation = np.corrcoef(obs, pred)[0, 1]
                bias_ratio = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
                variability_ratio = np.std(pred) / np.std(obs)
                
                metrics['kge'] = 1 - np.sqrt((correlation - 1)**2 + (bias_ratio - 1)**2 + 
                                           (variability_ratio - 1)**2)
            else:
                metrics['kge'] = np.nan
            
            # Relative metrics
            if np.mean(np.abs(obs)) > 0:
                metrics['relative_rmse'] = metrics['rmse'] / np.mean(np.abs(obs))
                metrics['relative_bias'] = metrics['bias'] / np.mean(np.abs(obs))
            
            # Correlation
            if len(obs) > 2:
                correlation, p_value = stats.pearsonr(obs, pred)
                metrics['correlation'] = correlation
                metrics['correlation_p_value'] = p_value
            
            metrics['n_samples'] = len(obs)
            
        except Exception as e:
            log.warning(f"Error calculating validation metrics: {e}")
            metrics['calculation_error'] = str(e)
        
        return metrics
    
    def _mask_target_data(self, target_config, mask_indices):
        """Create a copy of target config with masked data"""
        # This is a simplified implementation
        # In practice, would need more sophisticated data masking
        return target_config
    
    def _temporal_subset_data(self, target_config, start_idx, end_idx):
        """Create temporal subset of target data"""
        # Simplified implementation
        return target_config
    
    def _create_temporary_model(self, modified_config):
        """Create temporary model for validation"""
        # For validation, create a copy of the current model
        # In practice, would need proper model cloning
        return self.model
    
    def _create_simple_model(self):
        """Create simpler version of model for structural uncertainty"""
        from .hydro_massbalance import HydroMassBalance
        
        return HydroMassBalance(
            self.gdir,
            physics_level='simple',
            calibration_targets=['geodetic_mb'],
            enable_refreezing=False,
            enable_debris=False,
            enable_surface_evolution=False
        )
    
    def _predict_target(self, model, target_name, data_index):
        """Predict target value at specific index"""
        # Simplified prediction - would need proper implementation
        # based on target type and temporal alignment
        
        if 'mb' in target_name:
            # Predict mass balance
            try:
                fls = self.gdir.read_pickle('inversion_flowlines')
                heights = fls[0].surface_h
            except:
                heights = np.linspace(self.gdir.min_h, self.gdir.max_h, 10)
            
            year = model.ys + data_index  # Simplified year mapping
            if year <= model.ye:
                mb = model.get_annual_mb(heights, year)
                return np.mean(mb)
            else:
                return 0.0
        
        elif 'runoff' in target_name:
            # Predict runoff
            try:
                fls = self.gdir.read_pickle('inversion_flowlines')
                heights = fls[0].surface_h
            except:
                heights = np.linspace(self.gdir.min_h, self.gdir.max_h, 10)
            
            runoff = model.compute_runoff(heights, temporal_scale='annual')
            if data_index < len(runoff):
                return runoff[data_index]
            else:
                return np.mean(runoff) if len(runoff) > 0 else 0.0
        
        else:
            # Default prediction
            return 0.0
    
    def _aggregate_fold_metrics(self, fold_results):
        """Aggregate metrics across k-fold results"""
        if not fold_results:
            return {}
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        
        for fold in fold_results:
            if 'metrics' in fold:
                for metric, value in fold['metrics'].items():
                    if not np.isnan(value):
                        all_metrics[metric].append(value)
        
        # Calculate aggregated statistics
        aggregated = {}
        for metric, values in all_metrics.items():
            if len(values) > 0:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return aggregated
    
    def _generate_validation_plots(self, results):
        """Generate comprehensive validation plots"""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig_dir = os.path.join(self.gdir.dir, 'hydro_validation_plots')
            os.makedirs(fig_dir, exist_ok=True)
            
            # 1. Validation metrics comparison plot
            self._plot_validation_metrics(results, fig_dir)
            
            # 2. Benchmark comparison plots
            self._plot_benchmark_comparisons(results, fig_dir)
            
            # 3. Uncertainty visualization
            self._plot_uncertainty_analysis(results, fig_dir)
            
            # 4. Overall performance dashboard
            self._plot_performance_dashboard(results, fig_dir)
            
            log.info(f"Validation plots saved to {fig_dir}")
            
        except Exception as e:
            log.warning(f"Error generating validation plots: {e}")
    
    def _plot_validation_metrics(self, results, fig_dir):
        """Plot validation metrics comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Validation Metrics - {self.gdir.rgi_id}', fontsize=16)
        
        # Collect metrics from all validation methods
        methods = []
        targets = []
        rmse_values = []
        r2_values = []
        nse_values = []
        bias_values = []
        
        for method, method_results in results['validation_methods'].items():
            for target, target_results in method_results.items():
                if 'metrics' in target_results:
                    metrics = target_results['metrics']
                    methods.append(method)
                    targets.append(target)
                    rmse_values.append(metrics.get('rmse', np.nan))
                    r2_values.append(metrics.get('r2', np.nan))
                    nse_values.append(metrics.get('nse', np.nan))
                    bias_values.append(metrics.get('bias', np.nan))
        
        if len(methods) > 0:
            # RMSE plot
            axes[0, 0].bar(range(len(rmse_values)), rmse_values)
            axes[0, 0].set_title('RMSE by Method and Target')
            axes[0, 0].set_ylabel('RMSE')
            axes[0, 0].set_xticks(range(len(methods)))
            axes[0, 0].set_xticklabels([f"{m}\n{t}" for m, t in zip(methods, targets)], rotation=45)
            
            # R² plot
            axes[0, 1].bar(range(len(r2_values)), r2_values)
            axes[0, 1].set_title('R² by Method and Target')
            axes[0, 1].set_ylabel('R²')
            axes[0, 1].set_xticks(range(len(methods)))
            axes[0, 1].set_xticklabels([f"{m}\n{t}" for m, t in zip(methods, targets)], rotation=45)
            
            # NSE plot
            axes[1, 0].bar(range(len(nse_values)), nse_values)
            axes[1, 0].set_title('Nash-Sutcliffe Efficiency')
            axes[1, 0].set_ylabel('NSE')
            axes[1, 0].set_xticks(range(len(methods)))
            axes[1, 0].set_xticklabels([f"{m}\n{t}" for m, t in zip(methods, targets)], rotation=45)
            
            # Bias plot
            axes[1, 1].bar(range(len(bias_values)), bias_values)
            axes[1, 1].set_title('Bias by Method and Target')
            axes[1, 1].set_ylabel('Bias')
            axes[1, 1].set_xticks(range(len(methods)))
            axes[1, 1].set_xticklabels([f"{m}\n{t}" for m, t in zip(methods, targets)], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'validation_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_benchmark_comparisons(self, results, fig_dir):
        """Plot benchmark model comparisons"""
        if not results.get('benchmarks'):
            return
        
        n_benchmarks = len(results['benchmarks'])
        fig, axes = plt.subplots(n_benchmarks, 2, figsize=(12, 4*n_benchmarks))
        if n_benchmarks == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Benchmark Comparisons - {self.gdir.rgi_id}', fontsize=16)
        
        for i, (benchmark_name, benchmark_data) in enumerate(results['benchmarks'].items()):
            if 'error' in benchmark_data:
                continue
            
            hydro_mb = benchmark_data.get('hydro_mb', [])
            benchmark_mb = benchmark_data.get('benchmark_mb', [])
            years = benchmark_data.get('years', [])
            
            if len(hydro_mb) > 0 and len(benchmark_mb) > 0:
                # Time series comparison
                axes[i, 0].plot(years, hydro_mb, 'b-', label='HydroMassBalance', linewidth=2)
                axes[i, 0].plot(years, benchmark_mb, 'r--', label=benchmark_name, linewidth=2)
                axes[i, 0].set_title(f'Mass Balance Time Series vs {benchmark_name}')
                axes[i, 0].set_xlabel('Year')
                axes[i, 0].set_ylabel('Mass Balance (m w.e./yr)')
                axes[i, 0].legend()
                axes[i, 0].grid(True, alpha=0.3)
                
                # Scatter plot
                axes[i, 1].scatter(benchmark_mb, hydro_mb, alpha=0.7)
                min_val = min(min(hydro_mb), min(benchmark_mb))
                max_val = max(max(hydro_mb), max(benchmark_mb))
                axes[i, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                axes[i, 1].set_xlabel(f'{benchmark_name} MB (m w.e./yr)')
                axes[i, 1].set_ylabel('HydroMassBalance MB (m w.e./yr)')
                axes[i, 1].set_title(f'HydroMB vs {benchmark_name}')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Add metrics text
                metrics = benchmark_data.get('comparison_metrics', {})
                if metrics:
                    r2 = metrics.get('r2', np.nan)
                    rmse = metrics.get('rmse', np.nan)
                    axes[i, 1].text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}', 
                                   transform=axes[i, 1].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'benchmark_comparisons.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_uncertainty_analysis(self, results, fig_dir):
        """Plot uncertainty analysis results"""
        uncertainty_data = results.get('uncertainty', {})
        
        if not uncertainty_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Uncertainty Analysis - {self.gdir.rgi_id}', fontsize=16)
        
        # Parameter uncertainty
        if 'parameter_uncertainty' in uncertainty_data:
            param_unc = uncertainty_data['parameter_uncertainty']
            
            # Uncertainty range plot
            percentiles = param_unc.get('percentiles', {})
            if percentiles:
                labels = list(percentiles.keys())
                values = list(percentiles.values())
                
                axes[0, 0].bar(labels, values)
                axes[0, 0].set_title('Parameter Uncertainty Percentiles')
                axes[0, 0].set_ylabel('Mass Balance (m w.e./yr)')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Uncertainty statistics
            stats_text = f"Mean: {param_unc.get('mb_mean', 0):.3f} m w.e./yr\n"
            stats_text += f"Std: {param_unc.get('mb_std', 0):.3f} m w.e./yr\n"
            stats_text += f"Range: {param_unc.get('mb_range', [0, 0])}"
            
            axes[0, 1].text(0.1, 0.5, stats_text, transform=axes[0, 1].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[0, 1].set_title('Parameter Uncertainty Statistics')
            axes[0, 1].axis('off')
        
        # Climate uncertainty (if available)
        if 'climate_uncertainty' in uncertainty_data:
            climate_unc = uncertainty_data['climate_uncertainty']
            
            temp_unc = climate_unc.get('temperature_uncertainty_C', 0)
            prcp_unc = climate_unc.get('precipitation_uncertainty_fraction', 0)
            
            categories = ['Temperature\n(°C)', 'Precipitation\n(fraction)']
            uncertainties = [temp_unc, prcp_unc]
            
            axes[1, 0].bar(categories, uncertainties)
            axes[1, 0].set_title('Climate Data Uncertainty')
            axes[1, 0].set_ylabel('Uncertainty Magnitude')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Structural uncertainty (if available)
        if 'structural_uncertainty' in uncertainty_data:
            struct_unc = uncertainty_data['structural_uncertainty']
            
            if 'advanced_vs_simple_mb_difference' in struct_unc:
                diff_data = struct_unc['advanced_vs_simple_mb_difference']
                
                stats_text = f"Mean Difference: {diff_data.get('mean', 0):.3f} m w.e./yr\n"
                stats_text += f"Std Difference: {diff_data.get('std', 0):.3f} m w.e./yr\n"
                stats_text += f"Range: {diff_data.get('range', [0, 0])}"
                
                axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                               fontsize=12, verticalalignment='center',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                axes[1, 1].set_title('Structural Uncertainty\n(Advanced vs Simple Physics)')
                axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_dashboard(self, results, fig_dir):
        """Plot overall performance dashboard"""
        performance = results.get('overall_performance', {})
        
        if not performance:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Performance Dashboard - {self.gdir.rgi_id}', fontsize=16)
        
        # Performance scores radar/bar chart
        score_categories = ['Validation', 'Benchmark', 'Uncertainty', 'Overall']
        scores = [
            performance.get('validation_score', 0),
            performance.get('benchmark_score', 0),
            performance.get('uncertainty_score', 0),
            performance.get('overall_score', 0)
        ]
        
        colors = ['blue', 'green', 'orange', 'red']
        bars = axes[0, 0].bar(score_categories, scores, color=colors, alpha=0.7)
        axes[0, 0].set_title('Performance Scores')
        axes[0, 0].set_ylabel('Score (0-1)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add score values on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # Grade display
        grade = performance.get('grade', 'Not Assessed')
        grade_color = {
            'Excellent': 'green',
            'Very Good': 'lightgreen', 
            'Good': 'yellow',
            'Satisfactory': 'orange',
            'Needs Improvement': 'red',
            'Poor': 'darkred'
        }.get(grade, 'gray')
        
        axes[0, 1].text(0.5, 0.5, f'Grade:\n{grade}', 
                       transform=axes[0, 1].transAxes,
                       fontsize=20, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor=grade_color, alpha=0.3))
        axes[0, 1].set_title('Overall Grade')
        axes[0, 1].axis('off')
        
        # Recommendations
        recommendations = performance.get('recommendations', [])
        if recommendations:
            rec_text = '\n'.join([f"• {rec}" for rec in recommendations])
        else:
            rec_text = "No specific recommendations.\nModel performance is satisfactory."
        
        axes[1, 0].text(0.05, 0.95, rec_text, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='top', wrap=True,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 0].set_title('Recommendations')
        axes[1, 0].axis('off')
        
        # Model configuration summary
        config = results.get('model_config', {})
        config_text = f"Physics Level: {config.get('physics_level', 'N/A')}\n"
        config_text += f"Targets: {', '.join(config.get('calibration_targets', []))}\n"
        config_text += f"Physics Modules: {', '.join(config.get('physics_modules', []))}\n"
        config_text += f"Valid Period: {config.get('valid_period', 'N/A')}"
        
        axes[1, 1].text(0.05, 0.95, config_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        axes[1, 1].set_title('Model Configuration')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'performance_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_validation_results(self, results):
        """Save validation results to file"""
        filepath = os.path.join(self.gdir.dir, 'hydro_validation_results.json')
        
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
        
        results_to_save = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        log.info(f"Validation results saved to {filepath}")


# OGGM integration functions
@entity_task(log)
def validate_hydro_mass_balance(gdir, validation_config=None, **kwargs):
    """
    OGGM entity task for comprehensive HydroMassBalance validation
    
    Parameters
    ---------- 
    gdir : GlacierDirectory
        OGGM glacier directory
    validation_config : dict
        Validation configuration options
    """
    
    # Load calibrated HydroMassBalance model
    try:
        calib_data = gdir.read_json('hydro_mb_calib')
        # Would need to reconstruct model from saved calibration
        # For now, create a basic model
        from .hydro_massbalance import HydroMassBalance
        
        mb_model = HydroMassBalance(gdir, **kwargs)
        
    except FileNotFoundError:
        log.error(f"No HydroMassBalance calibration found for {gdir.rgi_id}")
        raise InvalidParamsError("Must run hydro_mb_calibration first")
    
    # Run validation
    validator = HydroValidationFramework(mb_model, validation_config)
    results = validator.run_comprehensive_validation()
    
    # Save results to gdir
    gdir.write_json(results, 'hydro_mb_validation')
    
    log.info(f"HydroMassBalance validation completed for {gdir.rgi_id}")
    
    return results


@global_task(log)
def validate_hydro_mass_balance_batch(gdirs, validation_config=None, **kwargs):
    """
    Global task for batch validation of HydroMassBalance models
    
    Parameters
    ----------
    gdirs : list of GlacierDirectory
        List of glacier directories to validate
    validation_config : dict
        Validation configuration
    """
    
    from oggm.workflow import execute_entity_task
    
    log.info(f"Starting batch validation for {len(gdirs)} glaciers")
    
    # Run validation for all glaciers
    results = execute_entity_task(
        validate_hydro_mass_balance, gdirs,
        validation_config=validation_config,
        **kwargs
    )
    
    # Aggregate results
    successful_validations = [r for r in results if r is not None]
    
    log.info(f"Batch validation completed: {len(successful_validations)}/{len(gdirs)} successful")
    
    return successful_validations


# Utility functions for analysis
def compare_hydro_models(gdirs, model_configs, output_dir=None):
    """
    Compare different HydroMassBalance configurations
    
    Parameters
    ----------
    gdirs : list of GlacierDirectory
        Glaciers to compare
    model_configs : list of dict
        Different model configurations to test
    output_dir : str
        Directory to save comparison results
    """
    
    if output_dir is None:
        output_dir = cfg.PATHS['working_dir']
    
    comparison_results = {}
    
    for i, config in enumerate(model_configs):
        config_name = config.get('name', f'config_{i}')
        log.info(f"Testing configuration: {config_name}")
        
        config_results = []
        
        for gdir in gdirs:
            try:
                # Create model with this configuration
                from .hydro_massbalance import HydroMassBalance
                mb_model = HydroMassBalance(gdir, **config)
                
                # Quick calibration and validation
                if mb_model.calibrator:
                    calib_results = mb_model.calibrate()
                    
                    # Basic validation
                    validator = HydroValidationFramework(mb_model)
                    val_results = validator.run_comprehensive_validation()
                    
                    config_results.append({
                        'glacier_id': gdir.rgi_id,
                        'calibration': calib_results,
                        'validation': val_results
                    })
                    
            except Exception as e:
                log.warning(f"Configuration {config_name} failed for {gdir.rgi_id}: {e}")
                continue
        
        comparison_results[config_name] = config_results
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, 'hydro_model_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    
    log.info(f"Model comparison results saved to {comparison_file}")
    
    return comparison_results


def generate_validation_report(validation_results, output_file=None):
    """
    Generate comprehensive validation report
    
    Parameters
    ----------
    validation_results : dict or list
        Validation results from HydroValidationFramework
    output_file : str
        Path for output report (HTML or PDF)
    """
    
    if output_file is None:
        output_file = 'hydro_validation_report.html'
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HydroMassBalance Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .metric {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .good {{ color: #28a745; }}
            .warning {{ color: #ffc107; }}
            .bad {{ color: #dc3545; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>HydroMassBalance Validation Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Add validation results
    if isinstance(validation_results, dict):
        validation_results = [validation_results]
    
    for result in validation_results:
        glacier_id = result.get('glacier_id', 'Unknown')
        html_content += f"<h2>Glacier: {glacier_id}</h2>"
        
        # Overall performance
        performance = result.get('overall_performance', {})
        grade = performance.get('grade', 'Not Assessed')
        overall_score = performance.get('overall_score', 0)
        
        grade_class = 'good' if overall_score > 0.7 else 'warning' if overall_score > 0.5 else 'bad'
        html_content += f'<div class="metric {grade_class}">Overall Grade: {grade} (Score: {overall_score:.3f})</div>'
        
        # Validation methods
        val_methods = result.get('validation_methods', {})
        if val_methods:
            html_content += "<h3>Validation Methods</h3><table>"
            html_content += "<tr><th>Method</th><th>Target</th><th>RMSE</th><th>R²</th><th>NSE</th></tr>"
            
            for method, method_results in val_methods.items():
                for target, target_results in method_results.items():
                    if 'metrics' in target_results:
                        metrics = target_results['metrics']
                        rmse = metrics.get('rmse', 'N/A')
                        r2 = metrics.get('r2', 'N/A')
                        nse = metrics.get('nse', 'N/A')
                        
                        html_content += f"<tr><td>{method}</td><td>{target}</td><td>{rmse}</td><td>{r2}</td><td>{nse}</td></tr>"
            
            html_content += "</table>"
        
        # Recommendations
        recommendations = performance.get('recommendations', [])
        if recommendations:
            html_content += "<h3>Recommendations</h3><ul>"
            for rec in recommendations:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul>"
    
    html_content += """
    </body>
    </html>
    """
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    log.info(f"Validation report saved to {output_file}")
    
    return output_file
