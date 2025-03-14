#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:27:20 2025

@author: brennan
"""

import os
import numpy as np

from .monitor import Monitor


class SamplesMonitor(Monitor):
    """A monitor class for saving super resolution samples created by the
    generator during training.
    """
    
    def __init__(
            self,
            generator, 
            lr_sample, 
            hr_sample,
            device,
            style = None,
            samples_dir = './data/samples/'
        ):
        
        self.lr_sample = lr_sample
        self.hr_sample = hr_sample
        self.style = style
        
        self.device = device
        self.generator = generator
        batch_size = lr_sample.shape[0]
        z = generator.sample_latent_space(batch_size, device)
        self.z = [(z0.cpu(), z1.cpu()) for z0, z1 in z]
        
        self.samples_dir = samples_dir
        os.makedirs(self.samples_dir, exist_ok=True)
        np.save(self.samples_dir + 'lr_sample.npy', lr_sample.numpy())
        np.save(self.samples_dir + 'hr_sample.npy', hr_sample.numpy())
        
        
    def post_epoch_processing(self, epoch):
        # Move data to the device and use the generator to create fake data.
        lr_sample = self.lr_sample.to(self.device)
        z = [(z0.to(self.device), z1.to(self.device)) for z0, z1 in self.z]
        style = None 
        if self.style is not None:
            style = self.style.to(self.device)
        sr_sample = self.generator(lr_sample, z, style)
        
        # Move the fake data to the cpu and save.
        sr_sample = sr_sample.detach().cpu()
        filename = self.samples_dir + f'sr_sample_{epoch:04}.npy'
        np.save(filename, sr_sample.numpy())