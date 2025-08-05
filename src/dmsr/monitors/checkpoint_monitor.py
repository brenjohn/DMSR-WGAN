#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:31:18 2025

@author: brennan
"""

from .monitor import Monitor


class CheckpointMonitor(Monitor):
    """A monitor class for saving a checkpoint for the current model at the
    end of each epoch. Only the current model is saved, previous checkpoints
    are over written.
    """
    
    def __init__(self, gan, checkpoint_dir):
        self.gan = gan
        self.checkpoint_dir = checkpoint_dir
    
    
    def post_epoch_processing(self, epoch):
        checkpoint_name = 'current_model/'
        self.gan.save(self.checkpoint_dir + checkpoint_name)