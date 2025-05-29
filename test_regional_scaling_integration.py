#!/usr/bin/env python3
"""Test script to verify regional scaling module integration."""

def test_regional_scaling_integration():
    """Test that regional scaling module is properly integrated."""
    
    print("Testing OGGM regional scaling integration...")
    
    # Test 1: Import OGGM tasks
    try:
        import oggm.tasks as tasks
        print("✓ OGGM tasks module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import OGGM tasks: {e}")
        return False
    
    # Test 2: Check if regional scaling tasks are available
    has_process_data = hasattr(tasks, 'process_regional_scaling_data')
    has_process_climate = hasattr(tasks, 'process_regional_scaling_climate_data')
    
    print(f"✓ process_regional_scaling_data available: {has_process_data}")
    print(f"✓ process_regional_scaling_climate_data available: {has_process_climate}")
    
    if not (has_process_data and has_process_climate):
        print("✗ Regional scaling tasks not found in oggm.tasks")
        return False
    
    # Test 3: Try to access the functions
    try:
        func1 = tasks.process_regional_scaling_data
        func2 = tasks.process_regional_scaling_climate_data
        print("✓ Regional scaling functions accessible")
    except AttributeError as e:
        print(f"✗ Cannot access regional scaling functions: {e}")
        return False
    
    # Test 4: Check function signatures
    import inspect
    sig1 = inspect.signature(func1)
    sig2 = inspect.signature(func2)
    
    print(f"✓ process_regional_scaling_data signature: {sig1}")
    print(f"✓ process_regional_scaling_climate_data signature: {sig2}")
    
    # Test 5: Direct import test
    try:
        from oggm.core.regional_scaling import (
            process_regional_scaling_data,
            process_regional_scaling_climate_data,
            StationDataReader,
            PhysicalDownscaler,
            ClimateValidator
        )
        print("✓ Direct import of regional scaling components successful")
    except ImportError as e:
        print(f"✗ Direct import failed: {e}")
        return False
    
    print("\n🎉 Regional scaling module is successfully integrated!")
    return True

if __name__ == "__main__":
    test_regional_scaling_integration()
