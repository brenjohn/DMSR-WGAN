#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:33:33 2025

@author: brennan
"""

import os
import shutil
import tempfile
import unittest
import subprocess

from pathlib import Path


class TestTrainingIntegrationTest(unittest.TestCase):
    
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
        
        
    #=========================================================================#
    #                          Integration Tests
    #=========================================================================#
    
    def test_velocity_critic(self):
        result = subprocess.run(
            ['python', '../training_examples/training_velocity_critic.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_output_dir
        )
        
        if self.show_output:
            print("\n--- Velocity test STDOUT ---\n", result.stdout)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        checkpoints_path = self.test_output_dir / 'velocity_run/checkpoints'
        self.assertTrue(checkpoints_path.exists(), "No checkpoints created")
        
        samples_path = self.test_output_dir / 'velocity_run/samples'
        self.assertTrue(samples_path.exists(), "No samples generated")
        
    
    def test_original_critic(self):
        result = subprocess.run(
            ['python', '../training_examples/training_original_critic.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_output_dir
        )
        
        if self.show_output:
            print("\n--- Original test STDOUT ---\n", result.stdout)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        checkpoints_path = self.test_output_dir / 'test_run/checkpoints'
        self.assertTrue(checkpoints_path.exists(), "No checkpoints created")
        
        samples_path = self.test_output_dir / 'test_run/samples'
        self.assertTrue(samples_path.exists(), "No samples generated")
    
    
    def test_density_critic(self):
        result = subprocess.run(
            ['python', '../training_examples/training_density_critic.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_output_dir
        )
        
        if self.show_output:
            print("\n--- Density test STDOUT ---\n", result.stdout)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        checkpoints_path = self.test_output_dir / 'test_run/checkpoints'
        self.assertTrue(checkpoints_path.exists(), "No checkpoints created")
        
        samples_path = self.test_output_dir / 'test_run/samples'
        self.assertTrue(samples_path.exists(), "No samples generated")
        
        
    def test_style(self):
        result = subprocess.run(
            ['python', '../training_examples/style_training.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_output_dir
        )
        
        if self.show_output:
            print("\n--- style test STDOUT ---\n", result.stdout)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        checkpoints_path = self.test_output_dir / 'test_run/checkpoints'
        self.assertTrue(checkpoints_path.exists(), "No checkpoints created")
        
        samples_path = self.test_output_dir / 'test_run/samples'
        self.assertTrue(samples_path.exists(), "No samples generated")
        
        
    def test_datastream(self):
        result = subprocess.run(
            ['python', '../training_examples/style_datastream.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_output_dir
        )
        
        if self.show_output:
            print("\n--- Datastream test STDOUT ---\n", result.stdout)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        checkpoints_path = self.test_output_dir / 'test_run/checkpoints'
        self.assertTrue(checkpoints_path.exists(), "No checkpoints created")
        
        samples_path = self.test_output_dir / 'test_run/samples_1'
        self.assertTrue(samples_path.exists(), "No samples generated")
    
    
    def test_nn_distance(self):
        result = subprocess.run(
            ['python', '../training_examples/training_nn_distance_critic.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.test_output_dir
        )
        
        if self.show_output:
            print("\n--- NN distance test STDOUT ---\n", result.stdout)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        
        checkpoints_path = self.test_output_dir / 'test_run/checkpoints'
        self.assertTrue(checkpoints_path.exists(), "No checkpoints created")
        
        samples_path = self.test_output_dir / 'test_run/samples_1'
        self.assertTrue(samples_path.exists(), "No samples generated")
