#!/usr/bin/env python3

import sys
sys.path.insert(0, '/workspace')

print("Testing OGGM regional scaling integration...")

# Test basic import
try:
    import oggm
    print("✓ OGGM imported successfully")
except Exception as e:
    print(f"✗ OGGM import failed: {e}")
    sys.exit(1)

# Test tasks module
try:
    from oggm import tasks
    print("✓ OGGM tasks module imported")
    
    # Check if our functions are available
    funcs = ['process_regional_scaling_data', 'process_regional_scaling_climate_data', 'compute_physical_parameters']
    for func in funcs:
        available = hasattr(tasks, func)
        print(f"  - {func}: {'✓' if available else '✗'}")
        
except Exception as e:
    print(f"✗ Tasks import failed: {e}")
    import traceback
    traceback.print_exc()

# Test workflow module
try:
    from oggm import workflow
    print("✓ OGGM workflow module imported")
    
    # Check if our function is available
    if hasattr(workflow, 'regional_scaling_tasks'):
        print("  - regional_scaling_tasks: ✓")
    else:
        print("  - regional_scaling_tasks: ✗")
        
except Exception as e:
    print(f"✗ Workflow import failed: {e}")
    import traceback
    traceback.print_exc()

# Test global_tasks module
try:
    from oggm import global_tasks
    print("✓ OGGM global_tasks module imported")
    
    # Check if our function is available
    if hasattr(global_tasks, 'regional_scaling_tasks'):
        print("  - regional_scaling_tasks: ✓")
    else:
        print("  - regional_scaling_tasks: ✗")
        
except Exception as e:
    print(f"✗ Global tasks import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nIntegration test completed!")
