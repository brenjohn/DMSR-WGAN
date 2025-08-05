#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:24:30 2025

@author: brennan
"""

from torch.nn import MSELoss
from .monitor import Monitor


class SupervisedMonitor(Monitor):
    """A monitor class for tracking the mean squared error loss during
    supervised training of the generator model.
    """
    
    def __init__(self, generator, validation_set, device):
        self.generator = generator
        self.device = device
        self.validation_set = validation_set
        self.generator_valid_losses = []
        self.generator_valid_epochs = []
        self.mse_loss = MSELoss()
    
    
    def post_epoch_processing(self, epoch):
    
        total_loss = 0
        for lr_batch, hr_batch in self.validation_set:
            # Move data to the device.
            lr_batch = lr_batch.to(self.device)
            hr_batch = hr_batch.to(self.device)
            
            # Use the generator to create fake data.
            z = self.generator.sample_latent_space(1, self.device)
            sr_batch = self.generator(lr_batch, z)
            
            # Compute the loss.
            loss = self.mse_loss(sr_batch, hr_batch)
            total_loss += loss.item()
        
        total_loss /= len(self.validation_set)
        self.generator_valid_losses.append(total_loss)
        self.generator_valid_epochs.append(epoch)