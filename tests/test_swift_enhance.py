#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 16:05:17 2025

@author: brennan
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import subprocess

from pathlib import Path
from dmsr import DMSRGenerator
from swift_tools import generate_mock_snapshots


PARAMETER_TEMPLATE = """\
model_dir        = '{model_dir}'
data_dir         = '{data_dir}'
snapshot_pattern = '{snap_pattern}'
output_suffix    = '_sr_test' 
output_dir       = '{output_dir}'
seed             = 7 
use_compression  = false
"""


class TestSwiftEnhance(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Use `SHOW_TEST_OUTPUT=1 python -m unittest` to show test std output.
        cls.show_output = os.getenv('SHOW_TEST_OUTPUT', '0') == '1'
    
    
    @classmethod
    def tearDownClass(cls):
        pycache = Path('__pycache__')
        if pycache.exists():
            shutil.rmtree(pycache)
    
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir='.')
        self.test_path = Path(self.temp_dir.name)
        
        # Define internal paths
        self.parameter_file = self.test_path / 'parameters.toml'
        self.model_dir      = self.test_path / 'mock_model'
        self.data_dir       = self.test_path / 'lr_snapshots'
        self.output_dir     = self.test_path / 'sr_snapshots'
        self.snap_pattern   = 'run*/LR_snapshots/*' 
        
        self.data_dir.mkdir()
    
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    
    def create_mock_model(self):
        """Creates a dummy DMSRGenerator and saves it to model_dir."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model = DMSRGenerator(
            grid_size = 20,
            input_channels = 6,
            base_channels = 8,
            crop_size=2,
            scale_factor=2,
            style_size=1,
            nn_distance=False,
        )
        model.save(self.model_dir)
        
        # Create the normalization file.
        metadata = {
            'LR_Coordinates_std'  : 1.0,
            'LR_Coordinates_mean' : 0.0,
            'HR_Coordinates_std'  : 1.0,
            'HR_Coordinates_mean' : 0.0,
            'LR_Velocities_std'   : 1.0,
            'LR_Velocities_mean'  : 0.0,
            'HR_Velocities_std'   : 1.0,
            'HR_Velocities_mean'  : 0.0,
        }
        with open(self.model_dir / 'metadata.json', 'w') as file:
            json.dump(metadata, file)
    
    
    def run_script(self, script_path, args):
        """Helper to run scripts and return the result."""
        command = [sys.executable, script_path] + args
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_path,
            env=env
        )
        
        if self.show_output and result.stdout:
            name = os.path.basename(script_path)
            print(f"\n--- {name} STDOUT ---\n", result.stdout)
            
        return result
    
    
    def test_swift_enhance_execution(self):
        """
        Tests that the enhancement script runs and produces upscaled files.
        """
        
        # 1. Arrange: Create mock model and mock HDF5 snapshots
        self.create_mock_model()
        
        # Generate 2 LR snapshots to test the globbing loop
        generate_mock_snapshots(
            snapshot_dir=self.data_dir,
            num_mock_runs=1,
            num_snapshots=1,
            LR_size=32,
            HR_size=32
        )
        
        # Create mock parameter file
        parameter_file = PARAMETER_TEMPLATE.format(
            model_dir    = self.model_dir.as_posix(),
            data_dir     = self.data_dir.as_posix(),
            output_dir   = self.output_dir.as_posix(),
            snap_pattern = self.snap_pattern
        )
        with open(self.parameter_file, 'w') as f:
            f.write(parameter_file)
        
        # 2. Act: Run the script
        # Using a relative path to the script from the temp_dir
        script_path = '../../scripts/swift_enhance.py'
        
        args = [
            f'--parameter_file={self.parameter_file.as_posix()}',
        ]
        
        result = self.run_script(script_path, args)
        
        # 3. Assert
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        # Verify output files exist with the correct suffix
        sr_files = list(self.output_dir.glob('*_sr_test.hdf5'))
        self.assertEqual(
            len(sr_files), 1,
            msg="No enhanced snapshots found."
        )