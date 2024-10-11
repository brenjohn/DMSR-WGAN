#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:31:22 2024

@author: brennan
"""

import os
import time
import torch
import numpy as np

from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from ..analysis import displacement_power_spectrum


#=============================================================================#
#                               Monitor Manager
#=============================================================================#

class MonitorManager():
    
    def __init__(self, report_rate, device):
        self.device = device
        self.report_rate = report_rate
        
    
    def set_monitors(self, monitors):
        self.monitors = monitors
        
    
    def init_monitoring(self, num_epochs, num_batches):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.batch_start_time = time.time()
        self.epoch_start_time = time.time()
    
    
    def end_of_epoch(self, epoch):
        epoch_time = time.time() - self.epoch_start_time
        print(f"[Epoch {epoch} took: {epoch_time:.4f} sec]")
        post_processing_start_time = time.time()
        
        for monitor in self.monitors.values():
            monitor.post_epoch_processing(epoch)
        
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()
        post_processing_time = time.time() - post_processing_start_time
        print(f"[Post-processing took: {post_processing_time:.4f} sec]")
    
        
    def end_of_batch(self, epoch, batch, batch_counter, losses):
        monitor_report = ''
        
        for monitor in self.monitors.values():
            monitor_report += monitor.post_batch_processing(
                epoch, batch, batch_counter, losses
            )
        
        self.batch_report(epoch, batch, monitor_report)
    
    
    def batch_report(self, epoch, batch, monitor_report):
        """Report some satistics for the last few batch updates.
        """
        if (batch > 0 and batch % self.report_rate == 0):
            time_curr = time.time()
            time_prev = self.batch_start_time
            average_batch_time = (time_curr - time_prev) / self.report_rate
            
            report  = f"[Epoch {epoch:04}/{self.num_epochs}]"
            report += f"[Batch {batch:03}/{self.num_batches}]"
            report += f"[time per batch: {average_batch_time*1000:.4f} ms]"
            report += monitor_report
            
            print(report)
            self.batch_start_time = time.time()
    


#=============================================================================#
#                                 Monitors
#=============================================================================#

class BaseMonitor():
    
    def post_epoch_processing(self, epoch):
        pass
    
    def post_batch_processing(self, epoch, batch, batch_counter, losses):
        return ''



class SupervisedMonitor(BaseMonitor):
    
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
        


class LossMonitor(BaseMonitor):
    
    def __init__(self):
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
        filename = 'losses.npz'
        np.savez(filename, **{
            'critic_loss'       : self.critic_loss,
            'critic_batches'    : self.critic_batches,
            'generator_loss'    : self.generator_loss,
            'generator_batches' : self.generator_batches,
            'gradient_penalty'  : self.gradient_penalty,
            'grad_pen_batches'  : self.grad_pen_batches
        })
        
        
  
class SamplesMonitor(BaseMonitor):
    
    def __init__(
            self,
            generator, 
            lr_sample, 
            hr_sample, 
            lr_box_size, 
            hr_box_size,
            device
        ):
        
        self.lr_sample = lr_sample
        self.hr_sample = hr_sample
        self.lr_box_size = lr_box_size
        self.hr_box_size = hr_box_size
        
        self.device = device
        self.generator = generator
        batch_size = lr_sample.shape[0]
        # TODO: Manage the  creation and movement of the latent space variable
        # in a cleaner way.
        z = generator.sample_latent_space(batch_size, device)
        self.z = [(z0.cpu(), z1.cpu()) for z0, z1 in z]
        
        self.samples_dir = './data/samples/'
        os.makedirs(self.samples_dir, exist_ok=True)
        np.save(self.samples_dir + 'lr_sample.npy', lr_sample.numpy())
        np.save(self.samples_dir + 'hr_sample.npy', hr_sample.numpy())
        
        
    def post_epoch_processing(self, epoch):
        # Move data to the device and use the generator to create fake data.
        lr_sample = self.lr_sample.to(self.device)
        z = [(z0.to(self.device), z1.to(self.device)) for z0, z1 in self.z]
        sr_sample = self.generator(lr_sample, z)
        
        # Move the fake data to the cpu and save.
        sr_sample = sr_sample.detach().cpu()
        filename = self.samples_dir + f'sr_sample_{epoch:04}.npy'
        np.save(filename, sr_sample.numpy())
        
        
        
class UpscaleMonitor(BaseMonitor):
    
    def __init__(
            self, 
            generator, 
            realisations, 
            device, 
        ):
        self.generator = generator
        self.realisations = realisations
        self.device = device
        self.sup_norm = []
    
        
    def set_data_set(
            self, 
            lr_data, 
            hr_data, 
            particle_mass, 
            box_size, 
            grid_size
        ):
        """
        """
        self.lr_data = lr_data
        self.mass = particle_mass
        self.box_size = box_size
        self.grid_size = grid_size
        
        hr_spectra = self.get_spectra(hr_data)
        self.hr_spectra = hr_spectra
        
        
    def get_spectra(self, data):
        """
        """
        spectra = []
        for sample in data:
            sample = sample[None, ...]
            spectrum = displacement_power_spectrum(
                sample, self.mass, self.box_size, self.grid_size
            )
            spectra.append(spectrum[1])
        
        return torch.stack(spectra)
    
        
    def post_epoch_processing(self, epoch):
        """
        """
        sup_norm = 0
        
        for lr_sample, hr_spectrum in zip(self.lr_data, self.hr_spectra):
            lr_sample = lr_sample.to(self.device)
            lr_sample = torch.unsqueeze(lr_sample, dim=0)
            hr_spectrum = hr_spectrum.to(self.device)
            
            z = self.generator.sample_latent_space(1, self.device)
            sr_sample = self.generator(lr_sample, z)
            sr_sample = sr_sample.detach()
            
            sr_ks, sr_spectrum, sr_uncertainty = displacement_power_spectrum(
                sr_sample, self.mass, self.box_size, self.grid_size
            )
            
            metric = self.spectral_metric(sr_spectrum, hr_spectrum)
            sup_norm = max(sup_norm, metric.item())
            
        self.sup_norm.append(sup_norm)
        
        
    def spectral_metric(self, spectrum_a, spectrum_b):
        return torch.max(torch.abs(spectrum_a - spectrum_b))



class CheckpointMonitor(BaseMonitor):
    
    def __init__(self, gan, checkpoint_dir):
        self.gan = gan
        self.checkpoint_dir = checkpoint_dir
    
    
    def post_epoch_processing(self, epoch):
        checkpoint_name = 'current_model/'
        self.gan.save(self.checkpoint_dir + checkpoint_name)