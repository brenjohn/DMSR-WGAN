#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 16:41:49 2025

@author: brennan
"""

import os
import sys
import shutil
import tempfile
import unittest
import subprocess

from pathlib import Path
from swift_tools import generate_mock_snapshots


CONFIG_TEMPLATE = """\
[base]
output_dir = "{output_dir}"
data_dir = "{snapshot_dir}"
stride = 1
include_velocities = true

# --- Low Resolution (LR) Patch Arguments ---
[LR_patch_args]
snapshot_glob = "run[1-2]/LR_snapshots/snap_*.hdf5"
inner_size = {LR_size}
padding = 2

# --- High Resolution (HR) Patch Arguments ---
[HR_patch_args]
snapshot_glob = "run[1-2]/HR_snapshots/snap_*.hdf5"
inner_size = {HR_size}
padding = 0
"""


class TestDatasetPreparation(unittest.TestCase):
    
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
        self.test_output_dir = Path(self.temp_dir.name)
    
    
    def tearDown(self):
        self.temp_dir.cleanup()
    
    
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
            cwd=self.test_output_dir,
            env=env
        )
        
        if self.show_output and result.stdout:
            name = os.path.basename(script_path)
            print(f"\n--- {name} STDOUT ---\n", result.stdout)
            
        return result
    
    
    def prepare_mock_data(self):
        """Shared setup logic for mock data and config."""
        self.LR_size, self.HR_size = 32, 64
        self.snapshot_dir = self.test_output_dir / 'snapshots'
        self.output_dir = self.test_output_dir / 'dataset'
        self.config_path = self.test_output_dir / 'dataset_preparation.toml'

        generate_mock_snapshots(
            snapshot_dir=self.snapshot_dir,
            num_mock_runs=2,
            num_snapshots=2,
            LR_size=self.LR_size,
            HR_size=self.HR_size
        )

        config_content = CONFIG_TEMPLATE.format(
            output_dir=self.output_dir.as_posix(),
            snapshot_dir=self.snapshot_dir.as_posix(),
            LR_size=self.LR_size // 2,
            HR_size=self.HR_size // 2
        )
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    
    def test_preparation(self):
        """Tests the end-to-end data pipeline sequentially."""
        self.prepare_mock_data()

        # 1. Test patches preparation
        results_prep = self.run_script(
            '../../scripts/dataset_preparation.py', 
            [f'--config_file={self.config_path}']
        )
        self.assertEqual(results_prep.returncode, 0, msg=results_prep.stderr)
        self.assertTrue(
            list(self.output_dir.glob('patches/*.h5')), 
            "No patches generated"
        )

        # 2. Test statistics preparation
        results_stats = self.run_script(
            '../../scripts/dataset_statistics.py', 
            [f'--dataset_dir={self.output_dir}', '--include_velocities']
        )
        self.assertEqual(results_stats.returncode, 0, msg=results_stats.stderr)
        self.assertTrue((self.output_dir / 'summary_stats.npy').exists())

        # 3. Test spectra preparation
        results_spec = self.run_script(
            '../../scripts/dataset_spectra.py', 
            [f'--dataset_dir={self.output_dir}']
        )
        self.assertEqual(results_spec.returncode, 0, msg=results_spec.stderr)