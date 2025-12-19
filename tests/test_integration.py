#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:33:33 2025

@author: brennan
"""

import os
import sys
import shutil
import tempfile
import unittest
import subprocess

from pathlib import Path
from dmsr.data_tools import generate_mock_dataset


class TestTrainingIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Use `SHOW_TEST_OUTPUT=1 python -m unittest` to show test std output.
        cls.show_output = os.getenv('SHOW_TEST_OUTPUT', '0') == '1'
        
        # Default arguments for mock data generation.
        cls.mock_train_data_args = {
            'num_patches'        : 16,
            'lr_grid_size'       : 20,
            'hr_grid_size'       : 32,
            'lr_padding'         : 2,
            'hr_padding'         : 0,
            'include_velocities' : True, 
            'include_scales'     : False,
            'include_spectra'    : False
        }
        cls.mock_valid_data_args = {
            'num_patches'        : 4,
            'lr_grid_size'       : 20,
            'hr_grid_size'       : 32,
            'lr_padding'         : 2,
            'hr_padding'         : 0,
            'include_velocities' : True, 
            'include_scales'     : False,
            'include_spectra'    : True
        }
    
    
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
    
    
    def _run(self, parameter_file, output_dir, distributed=False):
        command = [
            sys.executable,
            '../../scripts/dmsr_train.py', 
            f'--parameter_file={parameter_file}'
        ]
        if distributed:
            command = [
                'torchrun', 
                '--nproc_per_node', '1',
                '--rdzv_backend', 'c10d',
                '--rdzv_endpoint', 'localhost:29500'
            ] + command[1:]
        
        subprocess_env = os.environ.copy()
        subprocess_env['MKL_THREADING_LAYER'] = 'GNU'
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_output_dir,
            env=subprocess_env
        )
        
        if self.show_output:
            print(f"\n--- {output_dir} test STDOUT ---\n", result.stdout)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        output_dir = self.test_output_dir / output_dir
        checkpoints_path = output_dir / 'checkpoints'
        self.assertTrue(checkpoints_path.exists(), "No checkpoints created")
        
        samples_path = output_dir / 'samples_monitor_1'
        self.assertTrue(samples_path.exists(), "No samples generated")
        
        
    #=========================================================================#
    #                          Integration Tests
    #=========================================================================#
    
    def test_velocity_critic(self):
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'train_data', 
            **self.mock_train_data_args
        )
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'valid_data', 
            **self.mock_valid_data_args
        )
        
        parameter_file = '../training_examples/velocity_critic.toml'
        output_dir = 'velocity_run'
        self._run(parameter_file, output_dir)
        
    
    
    def test_original_critic(self):
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'train_data', 
            **self.mock_train_data_args | {'include_velocities' : False}
        )
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'valid_data', 
            **self.mock_valid_data_args | {'include_velocities' : False}
        )
        
        parameter_file = '../training_examples/original_critic.toml'
        output_dir = 'original_run'
        self._run(parameter_file, output_dir)
        
       
        
    def test_style(self):
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'train_data', 
            **self.mock_train_data_args | {'include_scales' : True}
        )
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'valid_data', 
            **self.mock_valid_data_args | {'include_scales' : True}
        )
        
        parameter_file = '../training_examples/style_critic.toml'
        output_dir = 'style_run'
        self._run(parameter_file, output_dir)
        
        parameter_file = '../training_examples/style_critic_restart.toml'
        output_dir = 'style_run_restart'
        self._run(parameter_file, output_dir)
    
    
    
    def test_nn_distance(self):
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'train_data', 
            **self.mock_train_data_args | {
                'include_scales' : True, 'hr_padding' : 1, 'hr_grid_size' : 34
            }
        )
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'valid_data', 
            **self.mock_valid_data_args | {
                'include_scales' : True, 'hr_padding' : 1, 'hr_grid_size' : 34
            }
        )
        
        parameter_file = '../training_examples/nn_distance_critic.toml'
        output_dir = 'nn_distance_run'
        self._run(parameter_file, output_dir)
        
        
    def test_ddp_training(self):
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'train_data', 
            **self.mock_train_data_args | {'include_scales' : True}
        )
        generate_mock_dataset(
            data_dir = self.test_output_dir / 'valid_data', 
            **self.mock_valid_data_args | {'include_scales' : True}
        )
        
        parameter_file = '../training_examples/distributed.toml'
        output_dir = 'distributed_run'
        self._run(parameter_file, output_dir, distributed=True)