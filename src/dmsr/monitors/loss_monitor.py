#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:26:04 2025

@author: brennan
"""

import numpy as np

from .monitor import Monitor


class LossMonitor(Monitor):
    """A monitor class for tracking the loss values of the generator and critic
    models during WGAN training.
    """
    
    def __init__(self, data_dir='./data/'):
        self.data_dir = data_dir
        
        self.critic_loss       = []
        self.critic_batches    = []
        self.generator_loss    = []
        self.generator_batches = []
        self.gradient_penalty  = []
        self.grad_pen_batches  = []
        
        self.average_critic_loss = 0
        self.average_generator_loss = 0
        
        
    def post_batch_processing(self, epoch, batch, batch_counter, losses):
        """
        """
        if 'critic_loss' in losses:
            critic_loss = losses['critic_loss']
            self.average_critic_loss += critic_loss
            
            if not isinstance(critic_loss, int):
                self.critic_loss.append(critic_loss)
                self.critic_batches.append(batch_counter)
        
        if 'generator_loss' in losses:
            generator_loss = losses['generator_loss']
            self.average_generator_loss += generator_loss
            
            if not isinstance(generator_loss, int):
                self.generator_loss.append(generator_loss)
                self.generator_batches.append(batch_counter)
        
        if 'gradient_penalty' in losses:
            gradient_penalty = losses['gradient_penalty']
            
            if not isinstance(gradient_penalty, int):
                self.gradient_penalty.append(gradient_penalty)
                self.grad_pen_batches.append(batch_counter)
          
        report = ''
        if not (self.average_critic_loss == 0):
            report += f"[C loss: {self.average_critic_loss/(batch+1):.8f}]"

        if not (self.average_generator_loss == 0):
            report += f"[G loss: {self.average_generator_loss/(batch+1):.8f}]"
        
        return report
                
                
    def post_epoch_processing(self, epoch):
        self.save_losses()
        self.average_critic_loss = 0
        self.average_generator_loss = 0
    
    
    def save_losses(self):
        filename = self.data_dir + 'losses.npz'
        np.savez(filename, **{
            'critic_loss'       : self.critic_loss,
            'critic_batches'    : self.critic_batches,
            'generator_loss'    : self.generator_loss,
            'generator_batches' : self.generator_batches,
            'gradient_penalty'  : self.gradient_penalty,
            'grad_pen_batches'  : self.grad_pen_batches
        })