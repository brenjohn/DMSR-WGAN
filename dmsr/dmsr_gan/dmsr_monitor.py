#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:31:22 2024

@author: brennan
"""

import os
import time
import numpy as np


class DMSRMonitor():
    
    def __init__(
            self, 
            generator,
            report_rate,
            lr_sample, 
            lr_box_size, 
            hr_sample, 
            hr_box_size,
            device
        ):
        
        batch_size = lr_sample.shape[0]
        
        self.generator = generator
        self.z = generator.sample_latent_space(batch_size, device)
        
        self.lr_sample = lr_sample.to(device)
        self.hr_sample = hr_sample.to(device)
        self.lr_box_size = lr_box_size
        self.hr_box_size = hr_box_size
        
        self.report_rate = report_rate
        self.samples_dir = './data/samples/'
        os.makedirs(self.samples_dir, exist_ok=True)
        np.save(self.samples_dir + 'lr_sample.npy', lr_sample.numpy())
        np.save(self.samples_dir + 'hr_sample.npy', hr_sample.numpy())
        
        self.critic_loss       = []
        self.critic_batches    = []
        self.generator_loss    = []
        self.generator_batches = []
        self.gradient_penalty  = []
        self.grad_pen_batches  = []
        
        self.average_critic_loss = 0
        self.average_generator_loss = 0
        
        
    def end_of_epoch(self, epoch):
        epoch_time = time.time() - self.epoch_start_time
        print(f"[Epoch {epoch} took: {epoch_time:.4f} sec]")
        
        self.save_generator_sample(epoch)
        self.save_losses()
        self.average_critic_loss = 0
        self.average_generator_loss = 0
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()
        
    
    def save_generator_sample(self, epoch):
        sr_sample = self.generator(self.lr_sample, self.z)
        sr_sample = sr_sample.detach().cpu()
        filename = self.samples_dir + f'sr_sample_{epoch:04}.npy'
        np.save(filename, sr_sample.numpy())
      
        
    def end_of_batch(self, epoch, batch, batch_counter, losses):
        
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
        
        self.batch_report(epoch, batch)
            
    
    def init_monitoring(self, num_epochs, num_batches):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.batch_start_time = time.time()
        self.epoch_start_time = time.time()
    
    
    def batch_report(self, epoch, batch):
        """Report some satistics for the last few batch updates.
        """
        if (batch > 0 and batch % self.report_rate == 0):
            time_curr = time.time()
            time_prev = self.batch_start_time
            average_batch_time = (time_curr - time_prev) / self.report_rate
            
            report  = f"[Epoch {epoch:04}/{self.num_epochs}]"
            report += f"[Batch {batch:03}/{self.num_batches}]"
            report += f"[time per batch: {average_batch_time*1000:.4f} ms]"
            
            if not (self.average_critic_loss == 0):
                report += f"[C loss: {self.average_critic_loss/batch:.8f}]"
            
            if not (self.average_generator_loss == 0):
                report += f"[G loss: {self.average_generator_loss/batch:.8f}]"
            
            print(report)
            self.batch_start_time = time.time()
    
    
    def save_losses(self):
        filename = 'losses.npz'
        np.savez(filename, **{
            'critic_loss'       : self.critic_loss,
            'critic_batches'    : self.critic_batches,
            'generator_loss'    : self.generator_loss,
            'generator_batches' : self.generator_batches,
            'gradient_penalty'  : self.gradient_penalty,
            'grad_pen_batches'  : self.grad_pen_batches
        })