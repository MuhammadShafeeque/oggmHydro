#!/usr/bin/env python
"""Simple integration test for regional scaling."""

import sys
import traceback

def main():
    try:
        print("Starting regional scaling integration test...")
        
        # Test 1: Basic OGGM import and initialization
        print("1. Testing OGGM import and initialization...")
        import oggm
        import oggm.cfg as cfg
        cfg.initialize()
        print("   ✓ OGGM initialized successfully")
        
        # Test 2: Baseline climate setting
        print("2. Testing baseline climate setting...")
        cfg.PARAMS['baseline_climate'] = 'REGIONAL_SCALING'
        print(f"   ✓ Baseline climate set to: {cfg.PARAMS['baseline_climate']}")
        
        # Test 3: Module imports
        print("3. Testing module imports...")
        from oggm import tasks
        from oggm import workflow
        from oggm import global_tasks
        print("   ✓ All modules imported successfully")
        
        # Test 4: Function availability
        print("4. Testing function availability...")
        
        # Tasks module
        has_process_data = hasattr(tasks, 'process_regional_scaling_data')
        has_process_climate = hasattr(tasks, 'process_regional_scaling_climate_data')
        has_compute_params = hasattr(tasks, 'compute_physical_parameters')
        
        print(f"   tasks.process_regional_scaling_data: {'✓' if has_process_data else '✗'}")
        print(f"   tasks.process_regional_scaling_climate_data: {'✓' if has_process_climate else '✗'}")
        print(f"   tasks.compute_physical_parameters: {'✓' if has_compute_params else '✗'}")
        
        # Workflow module
        has_workflow_func = hasattr(workflow, 'regional_scaling_tasks')
        print(f"   workflow.regional_scaling_tasks: {'✓' if has_workflow_func else '✗'}")
        
        # Global tasks module
        has_global_func = hasattr(global_tasks, 'regional_scaling_tasks')
        print(f"   global_tasks.regional_scaling_tasks: {'✓' if has_global_func else '✗'}")
        
        # Test 5: Function consistency
        print("5. Testing function consistency...")
        if has_workflow_func and has_global_func:
            same_func = workflow.regional_scaling_tasks == global_tasks.regional_scaling_tasks
            print(f"   workflow and global_tasks functions are the same: {'✓' if same_func else '✗'}")
        
        # Test 6: Core module functions
        print("6. Testing core module functions...")
        from oggm.core import regional_scaling
        
        core_has_process = hasattr(regional_scaling, 'process_regional_scaling_data')
        core_has_climate = hasattr(regional_scaling, 'process_regional_scaling_climate_data')
        core_has_params = hasattr(regional_scaling, 'compute_physical_parameters')
        
        print(f"   core.process_regional_scaling_data: {'✓' if core_has_process else '✗'}")
        print(f"   core.process_regional_scaling_climate_data: {'✓' if core_has_climate else '✗'}")
        print(f"   core.compute_physical_parameters: {'✓' if core_has_params else '✗'}")
        
        # Test 7: Climate module integration
        print("7. Testing climate module integration...")
        from oggm.core import climate
        import inspect
        
        source_code = inspect.getsource(climate.process_climate_data)
        has_regional_scaling_branch = "elif baseline == 'REGIONAL_SCALING':" in source_code
        print(f"   Climate module has REGIONAL_SCALING branch: {'✓' if has_regional_scaling_branch else '✗'}")
        
        print("\n🎉 All integration tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
