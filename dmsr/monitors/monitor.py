#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:31:22 2024

@author: brennan

This file defines the a collection of Monitor classes which can be used to
track various quantities or perform various tasks at different stages during
training of a DMSR-WGAN. A Monitor Manager class is also defined and is used
to handle monitor objects during training.
"""

import os
import torch
import numpy as np

from torch.nn import MSELoss
from ..analysis import displacement_power_spectrum


class BaseMonitor():
    
    def post_epoch_processing(self, epoch):
        pass
    
    def post_batch_processing(self, epoch, batch, batch_counter, losses):
        return ''



class SupervisedMonitor(BaseMonitor):
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
        


class LossMonitor(BaseMonitor):
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
        
        
  
class SamplesMonitor(BaseMonitor):
    """A monitor class for saving super resolution samples created by the
    generator during training.
    """
    
    def __init__(
            self,
            generator, 
            lr_sample, 
            hr_sample,
            device,
            samples_dir = './data/samples/'
        ):
        
        self.lr_sample = lr_sample
        self.hr_sample = hr_sample
        
        self.device = device
        self.generator = generator
        batch_size = lr_sample.shape[0]
        # TODO: Manage the creation and movement of the latent space variable
        # in a cleaner way.
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
        sr_sample = self.generator(lr_sample, z)
        
        # Move the fake data to the cpu and save.
        sr_sample = sr_sample.detach().cpu()
        filename = self.samples_dir + f'sr_sample_{epoch:04}.npy'
        np.save(filename, sr_sample.numpy())
        
        
        
class UpscaleMonitor(BaseMonitor):
    """A monitor class for tracking the uniform metric, or Chebyshev distance,
    between the power spectra of real high resolution data and fake super
    resolution data created by the generator.
    
    This monitor also saves a checkpoint for the best value for the uniform 
    metric found during training.
    """
    
    def __init__(
            self, 
            gan, 
            realisations, 
            device,
            checkpoint_dir
        ):
        self.gan = gan
        self.generator = gan.generator
        self.realisations = realisations
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        self.current_best_uniform_metric = float('inf')
        self.uniform_metric_history = []
        self.current_best_l2_metric = float('inf')
        self.l2_metric_history = []
    
        
    def set_data_set(
            self, 
            lr_data, 
            hr_data, 
            particle_mass, 
            box_size, 
            grid_size
        ):
        """Set the dataset to be used for computing the uniform metric.
        """
        self.lr_data    = lr_data
        self.mass       = particle_mass
        self.box_size   = box_size
        self.grid_size  = grid_size
        self.hr_spectra = self.get_spectra(hr_data)
        
        
    def get_spectra(self, displacements):
        """Return a tensor containing the power spectra of the given 
        displacement data.
        """
        spectra = []
        for sample in displacements:
            sample = sample[None, 0:3, ...]
            spectrum = displacement_power_spectrum(
                sample, self.mass, self.box_size, self.grid_size
            )
            spectra.append(spectrum[1])
        
        return torch.stack(spectra)
    
    
    def post_epoch_processing(self, epoch):
        """Compute the uniform metric between the power spectra of real high
        resolution data and fake super resolution data generated by the current
        model.
        """
        uniform_metric = 0
        l2_metric = 0
        
        for lr_sample, hr_spectrum in zip(self.lr_data, self.hr_spectra):
            # Move the low resolution displacement data and corresponding high
            # resolution power spectrum to the device.
            lr_sample = lr_sample.to(self.device)
            lr_sample = torch.unsqueeze(lr_sample, dim=0)
            hr_spectrum = hr_spectrum.to(self.device)
            
            # Generate fake super resolution data with the current generator.
            z = self.generator.sample_latent_space(1, self.device)
            sr_sample = self.generator(lr_sample, z)
            sr_sample = sr_sample.detach()
            
            # Get the power spectrum of the fake data.
            sr_sample = sr_sample[:, 0:3, ...]
            sr_ks, sr_spectrum, sr_uncertainty = displacement_power_spectrum(
                sr_sample, self.mass, self.box_size, self.grid_size
            )
            
            # Compute the uniform metric between the real and fake power 
            # spectrum.
            metric = self.uniform_metric(sr_spectrum, hr_spectrum)
            uniform_metric = max(uniform_metric, metric.item())
            
            # Compute the L2 metric between the real and fake spectra.
            metric = self.l2_metric(sr_spectrum, hr_spectrum, sr_ks)
            l2_metric += metric.item()
        
        l2_metric /= len(self.lr_data)
        
        # Save the new metric values.
        self.uniform_metric_history.append(uniform_metric)
        self.l2_metric_history.append(l2_metric)
        filename = self.checkpoint_dir + 'power_spectra_metrics.npz'
        np.savez(filename, **{
            'uniform_metric' : self.uniform_metric_history,
            'l2_metric' : self.l2_metric_history
        })
        
        # If the current metric is better than the best one found so far then
        # replace the best checkpoint with a checkpoint of the current model.
        if uniform_metric < self.current_best_uniform_metric:
            self.current_best_uniform_metric = uniform_metric
            
            print(f"[Current best uniform metric {uniform_metric:.4f}]")
            checkpoint_name = 'best_uniform_model'
            checkpoint_name += f'_epoch={epoch}'
            checkpoint_name += f'_{uniform_metric:.4f}/'
            self.gan.save(self.checkpoint_dir + checkpoint_name)
            
        if l2_metric < self.current_best_l2_metric:
            self.current_best_l2_metric = l2_metric
            
            print(f"[Current best l2 metric {l2_metric:.4f}]")
            checkpoint_name = 'best_l2_model'
            checkpoint_name += f'_epoch={epoch}'
            checkpoint_name += f'_{l2_metric:.4f}/'
            self.gan.save(self.checkpoint_dir + checkpoint_name)
    
        
    def uniform_metric(self, spectrum_a, spectrum_b):
        return torch.max(torch.abs(spectrum_a - spectrum_b))
    
    
    def l2_metric(self, spectrum_a, spectrum_b, ks):
        return torch.trapezoid((spectrum_a - spectrum_b) ** 2, ks)**0.5



class CheckpointMonitor(BaseMonitor):
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