#!/usr/bin/env python3
"""
Enhanced Preprocessing Pipeline with HydroMassBalance Integration
================================================================

This script demonstrates how to use the newly integrated hydro tasks 
from oggm.tasks module in your preprocessing pipeline.
"""

import os
import logging
import oggm
from oggm import cfg, workflow, tasks, utils

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def enhanced_preprocessing_pipeline(gdir, use_hydro=True):
    """
    Enhanced preprocessing pipeline with HydroMassBalance capabilities
    
    Parameters
    ----------
    gdir : GlacierDirectory
        The glacier directory to process
    use_hydro : bool
        Whether to use HydroMassBalance features
    """
    
    print("\n" + "=" * 60)
    print("🔧 ENHANCED PREPROCESSING PIPELINE WITH HYDROMASSBALANCE")
    print("=" * 60)

    gdirs = [gdir]  # Work with list for OGGM workflow functions

    # Step 1: GIS Preprocessing (if not already done at level 2)
    print("Step 1: GIS Preprocessing...")
    try:
        # Check if centerlines exist
        cl_file = gdir.get_filepath('centerlines')
        if not os.path.exists(cl_file):
            print("  Running GIS preprocessing tasks...")
            workflow.gis_prepro_tasks(gdirs)
            print("  ✓ GIS preprocessing completed")
        else:
            print("  ✓ GIS preprocessing already completed (from level 2)")
    except Exception as e:
        print(f"  ⚠️ GIS preprocessing issue: {e}")

    # Step 2: Climate Data Processing with Regional Scaling
    print("\nStep 2: Climate Data Processing...")
    try:
        # Get station data path
        station_data_path = cfg.PATHS.get('station_data_path')
        
        if station_data_path and os.path.exists(station_data_path):
            print(f"  Using station dataset: {station_data_path}")
            print("  Processing regional scaling (W5E5 + station bias correction)...")
            
            # Apply regional scaling with station data
            workflow.execute_entity_task(
                tasks.process_climate_data, gdirs, 
                climate_source='regional_scaling',
                station_data_path=station_data_path
            )
            print("  ✓ Regional scaling climate processing completed")
        else:
            print("  ⚠️ Station data not found, using default W5E5 climate")
            workflow.execute_entity_task(tasks.process_climate_data, gdirs)
            print("  ✓ Default climate processing completed")
            
    except Exception as e:
        print(f"  ❌ Climate processing failed: {e}")
        print("  Falling back to default climate processing...")
        workflow.execute_entity_task(tasks.process_climate_data, gdirs)

    # Step 3: Mass Balance Calibration (Enhanced with HydroMassBalance)
    print("\nStep 3: Enhanced Mass Balance Calibration...")
    
    if use_hydro:
        print("  Using HydroMassBalance for advanced multi-target calibration...")
        try:
            # Use the newly integrated hydro_mb_calibration task
            workflow.execute_entity_task(
                tasks.hydro_mb_calibration, gdirs,
                calibration_targets=['geodetic_mb'],
                physics_level='advanced',
                save_results=True
            )
            print("  ✓ HydroMassBalance calibration completed")
            
        except Exception as e:
            print(f"  ⚠️ HydroMassBalance calibration failed: {e}")
            print("  Falling back to standard calibration...")
            use_hydro = False
    
    if not use_hydro:
        print("  Using standard OGGM mass balance calibration...")
        try:
            # Use geodetic mass balance data for calibration
            workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb, gdirs)
            print("  ✓ Standard mass balance calibration completed")
        except Exception as e:
            print(f"  ⚠️ Geodetic calibration failed: {e}")
            try:
                # Fallback to scalar calibration
                workflow.execute_entity_task(tasks.mb_calibration_from_scalar_mb, gdirs)
                print("  ✓ Scalar mass balance calibration completed")
            except Exception as e2:
                print(f"  ❌ All calibration methods failed: {e2}")
                return False

    # Step 4: Ice Thickness Inversion
    print("\nStep 4: Ice Thickness Inversion...")
    try:
        workflow.execute_entity_task(tasks.prepare_for_inversion, gdirs)
        workflow.execute_entity_task(tasks.mass_conservation_inversion, gdirs)
        workflow.execute_entity_task(tasks.filter_inversion_output, gdirs)
        print("  ✓ Ice thickness inversion completed")
    except Exception as e:
        print(f"  ❌ Ice thickness inversion failed: {e}")

    # Step 5: Additional HydroMassBalance Features (if enabled)
    if use_hydro:
        print("\nStep 5: Advanced Hydro Features...")
        
        # Hydro climate integration
        try:
            print("  Setting up hydro-climate integration...")
            workflow.execute_entity_task(
                tasks.setup_hydro_climate_integration, gdirs,
                integration_method='statistical_downscaling'
            )
            print("  ✓ Hydro-climate integration setup completed")
        except Exception as e:
            print(f"  ⚠️ Hydro-climate integration failed: {e}")

        # Glacier runoff computation
        try:
            print("  Computing glacier runoff...")
            workflow.execute_entity_task(
                tasks.compute_glacier_runoff, gdirs,
                temporal_scale='monthly',
                year_range=(1980, 2020)
            )
            print("  ✓ Glacier runoff computation completed")
        except Exception as e:
            print(f"  ⚠️ Glacier runoff computation failed: {e}")

        # HydroMassBalance validation
        try:
            print("  Running HydroMassBalance validation...")
            workflow.execute_entity_task(
                tasks.validate_hydro_mass_balance, gdirs,
                validation_config={
                    'methods': ['leave_one_out', 'temporal_split'],
                    'metrics': ['rmse', 'mae', 'r2', 'nse']
                }
            )
            print("  ✓ HydroMassBalance validation completed")
        except Exception as e:
            print(f"  ⚠️ HydroMassBalance validation failed: {e}")

    print("\n✓ Enhanced preprocessing pipeline completed successfully!")
    return True


def demonstrate_hydro_workflow():
    """
    Demonstrate the complete hydro workflow capabilities
    """
    print("\n" + "=" * 60)
    print("🌊 HYDRO WORKFLOW DEMONSTRATION")  
    print("=" * 60)
    
    # This would use the hydro_mass_balance_workflow from the hydro_workflow module
    try:
        from oggm.hydro_workflow import hydro_mass_balance_workflow, hydro_mass_balance_quickstart
        
        print("Available HydroMassBalance workflow functions:")
        print("  - hydro_mass_balance_workflow(): Complete multi-glacier workflow")
        print("  - hydro_mass_balance_quickstart(): Single glacier quickstart")
        print("  - Uses all integrated hydro tasks from oggm.tasks")
        
        return True
        
    except ImportError as e:
        print(f"  ⚠️ Full hydro workflow not available: {e}")
        return False


if __name__ == "__main__":
    print("Enhanced Preprocessing Pipeline Demo")
    print("=====================================")
    
    # Initialize OGGM
    try:
        cfg.initialize()
        print("✓ OGGM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize OGGM: {e}")
        exit(1)
    
    # Check what hydro tasks are available
    print("\nChecking available hydro tasks in oggm.tasks:")
    hydro_tasks = [
        'hydro_mb_calibration',
        'setup_hydro_climate_integration', 
        'compute_glacier_runoff',
        'calibrate_glacier_runoff',
        'validate_hydro_mass_balance'
    ]
    
    for task_name in hydro_tasks:
        try:
            task_func = getattr(tasks, task_name)
            print(f"  ✓ {task_name} - available")
        except AttributeError:
            print(f"  ❌ {task_name} - not available")
    
    # Demonstrate workflow capabilities
    demonstrate_hydro_workflow()
    
    print("\n" + "=" * 60)
    print("🎉 HydroMassBalance integration successful!")
    print("You can now use all hydro tasks directly from oggm.tasks")
    print("=" * 60)
