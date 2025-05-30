"""HydroMassBalance workflow functions for OGGM - parallel module approach."""

# Built ins
import logging
import os
import shutil
from collections.abc import Sequence
# External libs
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize as optimization
import json

# Locals
import oggm
from oggm import cfg, tasks, utils
from oggm.core import centerlines, flowline, climate, gis
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.utils import global_task
from oggm.workflow import execute_entity_task, gis_prepro_tasks, climate_tasks

# HydroMassBalance imports - handle import gracefully
try:
    from oggm.core.hydro_massbalance import HydroMassBalance, hydro_mb_calibration
    from oggm.core.hydro_validation import validate_hydro_mass_balance
    from oggm.core.hydro_climate import compute_glacier_runoff
    _have_hydro = True
except ImportError:
    _have_hydro = False

# MPI
try:
    import oggm.mpi as ogmpi
    _have_ogmpi = True
except ImportError:
    _have_ogmpi = False
    ogmpi = None

# Module logger
log = logging.getLogger(__name__)

# Multiprocessing Pool
_mp_manager = None
_mp_pool = None


def _init_pool_globals(_cfg_contents, global_lock):
    cfg.unpack_config(_cfg_contents)
    utils.lock = global_lock


@global_task(log)
def hydro_mass_balance_workflow(gdirs, 
                               station_data_path=None,
                               runoff_data_path=None,
                               era5_data_path=None,
                               physics_level='advanced',
                               calibration_targets=None,
                               climate_source='regional_scaling',
                               y0=1979, y1=2019,
                               output_filesuffix='',
                               overwrite_gdir=False,
                               override_missing=None,
                               validation=True,
                               save_results=True):
    """
    Complete HydroMassBalance workflow for multiple glaciers.
    
    This is the main workflow function that implements the full HydroMassBalance
    processing chain including multi-target calibration, advanced physics, and
    comprehensive validation.
    
    Parameters
    ----------
    gdirs : list of GlacierDirectory
        List of glacier directories to process
    station_data_path : str, optional
        Path to meteorological station data file (CSV format)
        Used for regional scaling climate processing
    runoff_data_path : str, optional
        Path to directory containing runoff/discharge measurements
        Files should be CSV with columns: date, discharge_m3s
    era5_data_path : str, optional
        Path to ERA5 reanalysis data file (NetCDF format)
    physics_level : str
        Level of physics complexity:
        - 'simple': basic temperature-index mass balance
        - 'intermediate': enhanced physics with refreezing
        - 'advanced': full physics including debris, surface evolution, enhanced refreezing
    calibration_targets : list
        List of calibration targets from ['geodetic_mb', 'runoff', 'volume', 'velocity']
        Default: ['geodetic_mb']
    climate_source : str
        Climate data processing method:
        - 'regional_scaling': use station data with physical downscaling
        - 'oggm_default': standard OGGM climate processing
        - 'w5e5': W5E5 reanalysis data
        - 'era5': ERA5 reanalysis data
    y0, y1 : int
        Start and end years for climate data processing
    output_filesuffix : str
        Suffix to append to output files for identification
    overwrite_gdir : bool
        Whether to overwrite existing glacier directory data
    override_missing : bool or None
        Override missing data handling in mass balance calibration
    validation : bool
        Run comprehensive validation using HydroValidationFramework
    save_results : bool
        Save detailed results to files
        
    Returns
    -------
    results : dict
        Comprehensive workflow results containing:
        - 'gdirs': processed glacier directories
        - 'n_glaciers': number of glaciers processed
        - 'mb_models': list of calibrated HydroMassBalance models
        - 'runoff_results': runoff computation results
        - 'validation_results': validation framework results
        - 'glacier_statistics': detailed statistics DataFrame
        - 'workflow_summary': summary of completed steps and results
    """
    
    if calibration_targets is None:
        calibration_targets = ['geodetic_mb']
    
    log.info('=== Starting HydroMassBalance Workflow ===')
    log.info('Processing {} glaciers with physics_level={}, climate_source={}'
             .format(len(gdirs), physics_level, climate_source))
    log.info('Calibration targets: {}'.format(calibration_targets))
    log.info('Station data: {}'.format(station_data_path))
    log.info('Runoff data: {}'.format(runoff_data_path))
    
    # Initialize results structure
    results = {
        'gdirs': gdirs,
        'n_glaciers': len(gdirs),
        'workflow_steps': [],
        'parameters': {
            'station_data_path': station_data_path,
            'runoff_data_path': runoff_data_path,
            'era5_data_path': era5_data_path,
            'physics_level': physics_level,
            'calibration_targets': calibration_targets,
            'climate_source': climate_source,
            'y0': y0, 'y1': y1,
            'output_filesuffix': output_filesuffix
        },
        'errors': [],
        'warnings': []
    }
    
    # Step 1: Validate input data and setup
    log.info('Step 1: Input validation and setup')
    
    if station_data_path and not os.path.exists(station_data_path):
        error_msg = f'Station data file not found: {station_data_path}'
        log.error(error_msg)
        raise InvalidParamsError(error_msg)
    
    if runoff_data_path and not os.path.exists(runoff_data_path):
        log.warning(f'Runoff data directory not found: {runoff_data_path}')
        if 'runoff' in calibration_targets:
            calibration_targets.remove('runoff')
            results['warnings'].append('Removed runoff from calibration targets due to missing data')
    
    # Validate that we have necessary preprocessed data
    try:
        test_file = gdirs[0].get_filepath('centerlines')
        if not os.path.exists(test_file):
            log.info('Running GIS preprocessing tasks')
            gis_prepro_tasks(gdirs)
    except Exception as e:
        log.warning(f'GIS preprocessing check failed: {e}')
        # Try to run preprocessing anyway
        try:
            gis_prepro_tasks(gdirs)
        except Exception as e2:
            results['errors'].append(f'GIS preprocessing failed: {e2}')
    
    results['workflow_steps'].append('input_validation_setup')
    
    # Step 2: Climate data processing using regional scaling
    log.info('Step 2: Climate data processing')
    
    if climate_source == 'regional_scaling' and station_data_path:
        try:
            # Use regional scaling climate processing
            # This would be implemented in a separate module
            from oggm.workflow import regional_scaling_tasks
            
            regional_scaling_tasks(
                gdirs,
                station_data_path=station_data_path,
                era5_data_path=era5_data_path,
                y0=y0, y1=y1,
                output_filesuffix=output_filesuffix,
                overwrite_gdir=overwrite_gdir,
                override_missing=override_missing,
                compute_physical_params=(physics_level == 'advanced'),
                save_qc=True
            )
            results['workflow_steps'].append('regional_scaling_climate')
            log.info('✓ Regional scaling climate processing completed')
            
        except Exception as e:
            log.error(f'Regional scaling failed: {e}')
            results['errors'].append(f'Regional scaling: {e}')
            # Fallback to standard climate processing
            log.info('Falling back to standard climate processing')
            climate_tasks(gdirs, overwrite_gdir=overwrite_gdir, 
                         override_missing=override_missing)
            results['workflow_steps'].append('standard_climate_fallback')
    else:
        # Standard climate processing based on source
        log.info(f'Running standard climate processing with source: {climate_source}')
        climate_tasks(gdirs, overwrite_gdir=overwrite_gdir, 
                     override_missing=override_missing)
        results['workflow_steps'].append('standard_climate')
    
    # Step 3: Advanced multi-target mass balance calibration
    log.info('Step 3: Multi-target mass balance calibration')
    
    try:
        # Import HydroMassBalance components
        from oggm.core.hydro_massbalance import hydro_mb_calibration, DataConfiguration
        
        # Setup data configurations for calibration targets
        data_config_kwargs = {}
        
        # Configure runoff data
        if 'runoff' in calibration_targets and runoff_data_path and os.path.exists(runoff_data_path):
            discharge_files = [f for f in os.listdir(runoff_data_path) 
                             if f.endswith(('.csv', '.json'))]
            if discharge_files:
                runoff_file = os.path.join(runoff_data_path, discharge_files[0])
                data_config_kwargs['runoff_data_path'] = runoff_file
                log.info(f'Configured runoff calibration with: {runoff_file}')
        
        # Configure volume data (uses OGGM internal consensus data)
        if 'volume' in calibration_targets:
            data_config_kwargs['volume_data_path'] = None  # Use internal OGGM data
            log.info('Configured volume calibration using OGGM consensus data')
        
        # Run HydroMassBalance calibration
        mb_models = execute_entity_task(
            hydro_mb_calibration, gdirs,
            calibration_targets=calibration_targets,
            physics_level=physics_level,
            save_results=save_results,
            **data_config_kwargs
        )
        
        results['mb_models'] = mb_models
        results['workflow_steps'].append('hydro_mb_calibration')
        log.info('✓ HydroMassBalance calibration completed')
        
    except ImportError as e:
        log.warning(f'HydroMassBalance components not available: {e}')
        # Fallback to standard OGGM calibration
        log.info('Using standard OGGM mass balance calibration')
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                          override_missing=override_missing,
                          overwrite_gdir=overwrite_gdir)
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs)
        results['workflow_steps'].append('standard_mb_calibration')
        
    except Exception as e:
        log.error(f'HydroMassBalance calibration failed: {e}')
        results['errors'].append(f'MB calibration: {e}')
        # Fallback to standard calibration
        execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs,
                          override_missing=override_missing,
                          overwrite_gdir=overwrite_gdir)
        execute_entity_task(tasks.apparent_mb_from_any_mb, gdirs)
        results['workflow_steps'].append('mb_calibration_fallback')
    
    # Step 4: Ice thickness inversion (required for volume calibration)
    if 'volume' in calibration_targets:
        log.info('Step 4: Ice thickness inversion')
        try:
            from oggm.workflow import inversion_tasks, calibrate_inversion_from_consensus
            
            inversion_tasks(gdirs, add_to_log_file=True)
            
            # Advanced inversion calibration if consensus data available
            try:
                volume_stats = calibrate_inversion_from_consensus(
                    gdirs, ignore_missing=True, error_on_mismatch=False,
                    add_to_log_file=True
                )
                results['volume_stats'] = volume_stats
                results['workflow_steps'].append('consensus_volume_calibration')
                log.info('✓ Consensus volume calibration completed')
            except Exception as e:
                log.warning(f'Consensus volume calibration failed: {e}')
                results['warnings'].append(f'Consensus calibration: {e}')
                results['workflow_steps'].append('basic_inversion')
            
        except Exception as e:
            log.error(f'Ice thickness inversion failed: {e}')
            results['errors'].append(f'Inversion: {e}')
    
    # Step 5: Advanced runoff computation and calibration
    if 'runoff' in calibration_targets or runoff_data_path:
        log.info('Step 5: Advanced runoff computation')
        try:
            # Import RunoffComputation components
            from oggm.core.hydro_climate import compute_glacier_runoff, calibrate_glacier_runoff
            
            # Compute runoff using RunoffComputation class
            runoff_results = execute_entity_task(
                compute_glacier_runoff, gdirs,
                temporal_scale='monthly',
                year_range=(y0, y1),
                routing_config={
                    'routing_method': 'linear_reservoir',
                    'reservoir_constant': 30,
                    'evapotranspiration': True,
                    'groundwater_component': True
                }
            )
            
            results['runoff_results'] = runoff_results
            results['workflow_steps'].append('advanced_runoff_computation')
            log.info('✓ Advanced runoff computation completed')
            
            # Calibrate runoff against discharge observations if available
            if runoff_data_path and os.path.exists(runoff_data_path):
                discharge_files = [f for f in os.listdir(runoff_data_path) 
                                 if f.endswith(('.csv', '.json'))]
                if discharge_files:
                    discharge_file = os.path.join(runoff_data_path, discharge_files[0])
                    try:
                        calibration_results = execute_entity_task(
                            calibrate_glacier_runoff, gdirs,
                            discharge_data_path=discharge_file,
                            calibration_method='nash_sutcliffe'
                        )
                        
                        results['runoff_calibration'] = calibration_results
                        results['workflow_steps'].append('runoff_calibration')
                        log.info('✓ Runoff calibration against observations completed')
                        
                    except Exception as e:
                        log.warning(f'Runoff calibration failed: {e}')
                        results['warnings'].append(f'Runoff calibration: {e}')
            
        except ImportError as e:
            log.warning(f'Advanced runoff computation not available: {e}')
            results['warnings'].append('Advanced runoff computation not available')
            
        except Exception as e:
            log.error(f'Runoff computation failed: {e}')
            results['errors'].append(f'Runoff computation: {e}')
    
    # Step 6: Comprehensive validation using HydroValidationFramework
    if validation:
        log.info('Step 6: Comprehensive validation')
        try:
            from oggm.core.hydro_validation import validate_hydro_mass_balance
            
            validation_config = {
                'methods': ['leave_one_out', 'temporal_split', 'k_fold'],
                'metrics': ['rmse', 'mae', 'bias', 'r2', 'nse', 'kge'],
                'benchmarks': ['oggm_default', 'linear_mb'],
                'uncertainty_analysis': True,
                'save_diagnostics': save_results,
                'plot_results': save_results
            }
            
            validation_results = execute_entity_task(
                validate_hydro_mass_balance, gdirs,
                validation_config=validation_config
            )
            
            results['validation_results'] = validation_results
            results['workflow_steps'].append('comprehensive_validation')
            log.info('✓ Comprehensive validation completed')
            
        except ImportError as e:
            log.warning(f'Validation framework not available: {e}')
            results['warnings'].append('Validation framework not available')
            
        except Exception as e:
            log.error(f'Validation failed: {e}')
            results['errors'].append(f'Validation: {e}')
    
    # Step 7: Collect comprehensive statistics
    log.info('Step 7: Collecting comprehensive statistics')
    
    glacier_statistics = []
    for i, gdir in enumerate(gdirs):
        stats = {
            'rgi_id': gdir.rgi_id,
            'name': getattr(gdir, 'name', gdir.rgi_id),
            'area_km2': gdir.rgi_area_km2,
            'cenlon': gdir.cenlon,
            'cenlat': gdir.cenlat,
            'min_h': getattr(gdir, 'min_h', np.nan),
            'max_h': getattr(gdir, 'max_h', np.nan)
        }
        
        # Climate statistics from regional scaling
        try:
            climate_file = gdir.get_filepath('climate_historical', 
                                           filesuffix=output_filesuffix)
            if os.path.exists(climate_file):
                with utils.ncDataset(climate_file, mode='r') as nc:
                    stats['mean_temp_c'] = float(nc.variables['temp'][:].mean())
                    stats['mean_prcp_mmyr'] = float(nc.variables['prcp'][:].mean() * 12)
                    stats['temp_range_c'] = float(nc.variables['temp'][:].max() - 
                                                nc.variables['temp'][:].min())
        except Exception:
            stats.update({'mean_temp_c': np.nan, 'mean_prcp_mmyr': np.nan, 
                         'temp_range_c': np.nan})
          # Mass balance calibration results
        if ('mb_models' in results and 
            results['mb_models'] is not None and 
            i < len(results['mb_models'])):
            try:
                mb_model = results['mb_models'][i]
                if mb_model is not None and hasattr(mb_model, '_calibration_results'):
                    calib_results = mb_model._calibration_results
                    if calib_results and calib_results.get('success'):
                        stats['calibration_success'] = True
                        optimal_params = calib_results.get('optimal_parameters', {})
                        stats.update({f'param_{k}': v for k, v in optimal_params.items()})
                        
                        # Final metrics
                        final_metrics = calib_results.get('final_metrics', {})
                        for target, metrics in final_metrics.items():
                            if isinstance(metrics, dict):
                                stats[f'{target}_rmse'] = metrics.get('rmse', np.nan)
                                stats[f'{target}_r2'] = metrics.get('r2', np.nan)
                                stats[f'{target}_nse'] = metrics.get('nse', np.nan)
                    else:
                        stats['calibration_success'] = False
            except Exception:
                stats['calibration_success'] = np.nan
          # Runoff statistics
        if ('runoff_results' in results and 
            results['runoff_results'] is not None and 
            i < len(results['runoff_results'])):
            try:
                runoff_data = results['runoff_results'][i]
                if isinstance(runoff_data, dict):
                    total_runoff = runoff_data.get('total_runoff', [])
                    if len(total_runoff) > 0:
                        stats['annual_runoff_m3'] = np.sum(total_runoff)
                        stats['mean_monthly_runoff_m3'] = np.mean(total_runoff)
                        stats['runoff_coefficient'] = (
                            stats['annual_runoff_m3'] / 
                            (stats.get('mean_prcp_mmyr', 1000) * gdir.rgi_area_m2 / 1000)
                        )
            except Exception:
                pass
          # Ice volume statistics
        if 'volume' in calibration_targets:
            try:
                vol_result = execute_entity_task(tasks.get_inversion_volume, [gdir])
                if vol_result and len(vol_result) > 0 and vol_result[0] is not None:
                    vol_m3 = vol_result[0]
                    stats['volume_km3'] = vol_m3 * 1e-9
                    stats['thickness_m'] = vol_m3 / gdir.rgi_area_m2
                else:
                    stats.update({'volume_km3': np.nan, 'thickness_m': np.nan})
            except Exception:
                stats.update({'volume_km3': np.nan, 'thickness_m': np.nan})
          # Validation statistics
        if ('validation_results' in results and 
            results['validation_results'] is not None and 
            i < len(results['validation_results'])):
            try:
                val_result = results['validation_results'][i]
                if isinstance(val_result, dict):
                    overall_perf = val_result.get('overall_performance', {})
                    stats['validation_grade'] = overall_perf.get('grade', 'Not Assessed')
                    stats['validation_score'] = overall_perf.get('overall_score', np.nan)
            except Exception:
                pass
        
        glacier_statistics.append(stats)
    
    results['glacier_statistics'] = pd.DataFrame(glacier_statistics)
    results['workflow_steps'].append('statistics_collection')
    
    # Workflow summary
    total_area = results['glacier_statistics']['area_km2'].sum()
    
    results['workflow_summary'] = {
        'completed_steps': len(results['workflow_steps']),
        'step_list': results['workflow_steps'],
        'total_area_km2': total_area,
        'calibration_targets_used': calibration_targets,
        'climate_source_used': climate_source,
        'physics_level_used': physics_level,
        'n_errors': len(results['errors']),
        'n_warnings': len(results['warnings'])
    }
    
    # Add volume statistics if available
    if 'volume' in calibration_targets:
        valid_volumes = results['glacier_statistics']['volume_km3'].dropna()
        if len(valid_volumes) > 0:
            results['workflow_summary']['total_volume_km3'] = valid_volumes.sum()
            results['workflow_summary']['mean_thickness_m'] = (
                results['glacier_statistics']['thickness_m'].mean()
            )
    
    # Add runoff statistics if available
    if 'annual_runoff_m3' in results['glacier_statistics'].columns:
        valid_runoff = results['glacier_statistics']['annual_runoff_m3'].dropna()
        if len(valid_runoff) > 0:
            results['workflow_summary']['total_annual_runoff_m3'] = valid_runoff.sum()
            results['workflow_summary']['mean_runoff_coefficient'] = (
                results['glacier_statistics']['runoff_coefficient'].mean()
            )
    
    # Save comprehensive results if requested
    if save_results:
        output_dir = os.path.join(cfg.PATHS['working_dir'], 'hydro_workflow_results')
        utils.mkdir(output_dir)
        
        # Save glacier statistics
        stats_file = os.path.join(output_dir, f'glacier_statistics{output_filesuffix}.csv')
        results['glacier_statistics'].to_csv(stats_file, index=False)
        
        # Save workflow summary
        summary_file = os.path.join(output_dir, f'workflow_summary{output_filesuffix}.json')
        with open(summary_file, 'w') as f:
            json.dump(results['workflow_summary'], f, indent=2, default=str)
        
        # Save detailed results (excluding gdirs for JSON compatibility)
        detailed_results = {k: v for k, v in results.items() if k != 'gdirs'}
        detailed_file = os.path.join(output_dir, f'detailed_results{output_filesuffix}.json')
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        log.info(f'Results saved to {output_dir}')
        results['workflow_steps'].append('results_saved')
    
    # Final summary
    log.info('=== HydroMassBalance Workflow Completed ===')
    log.info('Processed {} glaciers, total area: {:.1f} km²'
             .format(results['n_glaciers'], total_area))
    log.info('Completed {} workflow steps: {}'
             .format(len(results['workflow_steps']), 
                    ', '.join(results['workflow_steps'])))
    
    if results['errors']:
        log.info('Encountered {} errors: {}'
                 .format(len(results['errors']), '; '.join(results['errors'])))
    
    if results['warnings']:
        log.info('Encountered {} warnings: {}'
                 .format(len(results['warnings']), '; '.join(results['warnings'])))
    
    if 'total_volume_km3' in results['workflow_summary']:
        log.info('Total ice volume: {:.2f} km³, mean thickness: {:.1f} m'
                 .format(results['workflow_summary']['total_volume_km3'],
                        results['workflow_summary']['mean_thickness_m']))
    
    if 'total_annual_runoff_m3' in results['workflow_summary']:
        log.info('Total annual runoff: {:.2e} m³, mean runoff coefficient: {:.3f}'
                 .format(results['workflow_summary']['total_annual_runoff_m3'],
                        results['workflow_summary']['mean_runoff_coefficient']))
    
    return results


@global_task(log)
def hydro_mass_balance_quickstart(rgi_id, 
                                 physics_level='advanced',
                                 calibration_targets=None,
                                 from_prepro_level=3,
                                 **kwargs):
    """
    One-line quickstart for HydroMassBalance workflow.
    
    This function provides the simplest way to run a complete 
    HydroMassBalance analysis on a single glacier.
    
    Parameters
    ----------
    rgi_id : str
        RGI glacier identifier (e.g., 'RGI60-11.00897')
    physics_level : str
        Physics complexity level: 'simple', 'intermediate', 'advanced'
    calibration_targets : list
        Calibration targets: ['geodetic_mb', 'runoff', 'volume', 'velocity']
    from_prepro_level : int
        OGGM preprocessing level to start from
    **kwargs
        Additional parameters for workflow
        
    Returns
    -------
    dict
        Complete analysis results
    """
    if not _have_hydro:
        raise ImportError("HydroMassBalance modules not available. "
                         "Please ensure hydro modules are properly installed.")
    
    # Set defaults
    if calibration_targets is None:
        calibration_targets = ['geodetic_mb']
    
    log.info(f'HydroMassBalance quickstart for glacier: {rgi_id}')
    
    try:
        # Initialize glacier directory
        from oggm import workflow
        cfg.initialize()
        
        gdirs = workflow.init_glacier_directories([rgi_id], 
                                                 from_prepro_level=from_prepro_level)
        gdir = gdirs[0]
        
        log.info(f'Initialized glacier directory for {rgi_id}')
        
        # Run complete workflow
        results = hydro_mass_balance_workflow(
            gdirs,
            physics_level=physics_level,
            calibration_targets=calibration_targets,
            save_results=True,
            **kwargs
        )
        
        # Add quickstart specific information
        results['rgi_id'] = rgi_id
        results['glacier_name'] = gdir.name
        results['glacier_area_km2'] = gdir.rgi_area_km2
        results['quickstart'] = True
        
        log.info(f'HydroMassBalance quickstart completed for {rgi_id}')
        
        return results
        
    except Exception as e:
        log.error(f'HydroMassBalance quickstart failed for {rgi_id}: {str(e)}')
        raise

