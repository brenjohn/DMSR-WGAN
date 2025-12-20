#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 16:05:17 2025

@author: brennan
"""

import os
import sys
import shutil
import tempfile
import unittest
import subprocess
import numpy as np

from pathlib import Path
from dmsr import DMSRGenerator
from swift_tools import generate_mock_snapshots


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
        self.model_dir = self.test_path / 'mock_model'
        self.data_dir = self.test_path / 'lr_snapshots'
        self.output_dir = self.test_path / 'sr_snapshots'
        
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
        np.save(self.model_dir / 'normalisation.npy', {
            'LR_Coordinates_std'  : np.float64(1.0),
            'LR_Coordinates_mean' : np.float64(0.0),
            'HR_Coordinates_std'  : np.float64(1.0),
            'HR_Coordinates_mean' : np.float64(0.0),
            'LR_Velocities_std'   : np.float64(1.0),
            'LR_Velocities_mean'  : np.float64(0.0),
            'HR_Velocities_std'   : np.float64(1.0),
            'HR_Velocities_mean'  : np.float64(0.0),
        })
    
    
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
        # Note: adjust snapshot_dir search if generate_mock_snapshots 
        # creates a 'run1' subfolder.
        
        # 2. Act: Run the script
        # Using a relative path to the script from the temp_dir
        script_path = '../../scripts/swift_enhance.py'
        
        args = [
            f'--model_dir={self.model_dir.as_posix()}',
            f'--data_dir={self.data_dir.as_posix()}',
            '--snapshot_pattern=run0/LR_snapshots/snap_*.hdf5',
            '--output_suffix=_sr_test',
            f'--output_dir={self.output_dir.as_posix()}',
            '--seed=21'
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