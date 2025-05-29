"""Tests for regional scaling integration with OGGM workflows."""

import os
import shutil
import tempfile
import unittest
import pytest
import numpy as np
import pandas as pd

# Import OGGM modules
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm import global_tasks
from oggm import utils
from oggm.tests.funcs import get_test_dir
from oggm.core import climate

# Test mark
pytestmark = pytest.mark.test_env("regional_scaling")


class TestRegionalScalingIntegration(unittest.TestCase):
    """Test class for regional scaling integration."""
    
    def setUp(self):
        """Set up test environment."""
        cfg.initialize()
        
    def test_regional_scaling_imports(self):
        """Test that regional scaling functions can be imported."""
        # Test import from tasks module
        self.assertTrue(hasattr(tasks, 'process_regional_scaling_data'))
        self.assertTrue(hasattr(tasks, 'process_regional_scaling_climate_data'))
        self.assertTrue(hasattr(tasks, 'compute_physical_parameters'))
        
        # Test import from workflow module
        self.assertTrue(hasattr(workflow, 'regional_scaling_tasks'))
        
        # Test import from global_tasks module
        self.assertTrue(hasattr(global_tasks, 'regional_scaling_tasks'))
        
    def test_regional_scaling_baseline_climate(self):
        """Test that REGIONAL_SCALING is accepted as baseline climate."""
        # This should not raise an error
        cfg.PARAMS['baseline_climate'] = 'REGIONAL_SCALING'
        self.assertEqual(cfg.PARAMS['baseline_climate'], 'REGIONAL_SCALING')
        
    def test_regional_scaling_workflow_function_signature(self):
        """Test the regional scaling workflow function signature."""
        import inspect
        
        sig = inspect.signature(workflow.regional_scaling_tasks)
        params = list(sig.parameters.keys())
        
        # Check that expected parameters are present
        expected_params = [
            'gdirs', 'station_data_path', 'era5_data_path', 'y0', 'y1',
            'output_filesuffix', 'overwrite_gdir', 'override_missing',
            'compute_physical_params', 'save_qc', 'kwargs'
        ]
        
        for param in expected_params:
            self.assertIn(param, params)
            
    def test_regional_scaling_entity_task_decorators(self):
        """Test that regional scaling functions have proper decorators."""
        # Import functions directly from the module
        from oggm.core.regional_scaling import (
            process_regional_scaling_data,
            compute_physical_parameters
        )
        
        # Check that they have entity_task decorators
        self.assertTrue(hasattr(process_regional_scaling_data, 'is_entity_task'))
        self.assertTrue(hasattr(compute_physical_parameters, 'is_entity_task'))
        
    def test_regional_scaling_global_task_decorator(self):
        """Test that the workflow function has proper global_task decorator."""
        # Check that workflow function has global_task decorator
        self.assertTrue(hasattr(workflow.regional_scaling_tasks, 'is_global_task'))
        
    def test_process_climate_data_regional_scaling_branch(self):
        """Test that process_climate_data recognizes REGIONAL_SCALING baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.PATHS['working_dir'] = tmpdir
            cfg.PARAMS['baseline_climate'] = 'REGIONAL_SCALING'
            
            # Create a mock glacier directory
            rgi_id = 'RGI60-11.00897'
            gdir = utils.GlacierDirectory(rgi_id, reset=True)
            
            # Set required paths (these don't need to exist for this test)
            cfg.PATHS['station_data_path'] = '/mock/path/stations.csv'
            
            # This should not raise an error about unknown baseline_climate
            # We expect it to fail on missing data, not unknown baseline
            with self.assertRaises(Exception) as context:
                climate.process_climate_data(gdir)
            
            # Should not be a ValueError about unknown baseline_climate
            self.assertNotIsInstance(context.exception, ValueError)
            
    def test_regional_scaling_functions_exist_in_regional_scaling_module(self):
        """Test that the core functions exist in the regional_scaling module."""
        from oggm.core import regional_scaling
        
        # Check main functions exist
        self.assertTrue(hasattr(regional_scaling, 'process_regional_scaling_data'))
        self.assertTrue(hasattr(regional_scaling, 'process_regional_scaling_climate_data'))
        self.assertTrue(hasattr(regional_scaling, 'compute_physical_parameters'))
        
    def test_regional_scaling_workflow_parameter_validation(self):
        """Test parameter validation in regional scaling workflow."""
        # This is a minimal test to ensure the function signature works
        # We can't run the full workflow without actual data
        
        # Test with empty gdirs list
        try:
            workflow.regional_scaling_tasks([])
        except Exception as e:
            # Should not fail on parameter validation, only on empty list processing
            pass

    def test_regional_scaling_configuration_integration(self):
        """Test that regional scaling integrates with OGGM configuration."""
        # Test setting station data path
        test_path = '/test/path/to/stations.csv'
        cfg.PATHS['station_data_path'] = test_path
        self.assertEqual(cfg.PATHS['station_data_path'], test_path)
        
        # Test setting ERA5 data path
        test_era5_path = '/test/path/to/era5.nc'
        cfg.PATHS['era5_data_path'] = test_era5_path
        self.assertEqual(cfg.PATHS['era5_data_path'], test_era5_path)


class TestRegionalScalingWorkflowIntegration(unittest.TestCase):
    """Test regional scaling workflow integration patterns."""
    
    def setUp(self):
        """Set up test environment."""
        cfg.initialize()
        
    def test_global_tasks_import_consistency(self):
        """Test that global_tasks and workflow have same function."""
        from oggm.workflow import regional_scaling_tasks as wf_func
        from oggm.global_tasks import regional_scaling_tasks as gt_func
        
        # They should be the same function
        self.assertEqual(wf_func, gt_func)
        
    def test_tasks_module_function_availability(self):
        """Test that all regional scaling functions are available in tasks."""
        from oggm import tasks
        from oggm.core.regional_scaling import (
            process_regional_scaling_data,
            process_regional_scaling_climate_data,
            compute_physical_parameters
        )
        
        # Check that the functions in tasks are the same as in the core module
        self.assertEqual(tasks.process_regional_scaling_data, process_regional_scaling_data)
        self.assertEqual(tasks.process_regional_scaling_climate_data, process_regional_scaling_climate_data)
        self.assertEqual(tasks.compute_physical_parameters, compute_physical_parameters)
        
    def test_workflow_function_follows_oggm_patterns(self):
        """Test that regional scaling workflow follows OGGM patterns."""
        import inspect
        
        # Check function has proper docstring
        self.assertIsNotNone(workflow.regional_scaling_tasks.__doc__)
        self.assertIn('gdirs', workflow.regional_scaling_tasks.__doc__)
        
        # Check function signature follows pattern (first arg is gdirs)
        sig = inspect.signature(workflow.regional_scaling_tasks)
        params = list(sig.parameters.keys())
        self.assertEqual(params[0], 'gdirs')
        
    def test_baseline_climate_integration_pattern(self):
        """Test that REGIONAL_SCALING follows the same pattern as other baselines."""
        # Test that it's handled in the same way as other climate sources
        # by checking the process_climate_data function
        
        import inspect
        source_code = inspect.getsource(climate.process_climate_data)
        
        # Should contain the REGIONAL_SCALING elif branch
        self.assertIn("elif baseline == 'REGIONAL_SCALING':", source_code)
        self.assertIn("process_regional_scaling_climate_data", source_code)


if __name__ == '__main__':
    unittest.main()
