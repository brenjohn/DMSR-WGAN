#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:34:12 2024

@author: brennan

This file define the DMSR-WGAN class.
"""

import torch.distributed as dist

from pathlib import Path
from torch import optim
from torch import save, load
from torch.nn import MSELoss
from torch.nn.functional import interpolate
from torch.nn.parallel import DistributedDataParallel as DDP

from ..data_tools.resize import crop
from .critic import DMSRCritic
from .generator import DMSRGenerator
from .serial_container import SerialContainer


class DMSRWGAN:
    """The DMSR-WGAN model.
    
    The name abreviates the Dark Matter, Super Resolution, Wasserstein 
    Generative Adversarial Neural Networks.
    """
    
    def __init__(
            self, 
            generator, 
            critic, 
            device, 
            gradient_penalty_rate=16,
            distributed=False
        ):
        self.device = device
        self.gradient_penalty_rate = gradient_penalty_rate
        self.distributed = distributed
        self.batch_counter = 0
        self.scale_factor = generator.scale_factor
        self.mse_loss = MSELoss()
        
        if distributed:
            self.generator = DDP(generator, device_ids=[device])
            self.critic = DDP(critic, device_ids=[device])
        else:
            self.generator = SerialContainer(generator)
            self.critic = SerialContainer(critic)
        
        self.compute_crop_sizes()
        
        
    def compute_crop_sizes(self):
        """
        To condition the critic model on lr data, the lr data needs to be
        upscaled using linear interpolation. On top of this, the data needs to
        be cropped before and after the interpolation to remove excess cells in
        the data. This method computes the crop sizes for these operations
        using the input sizes of the generator and critic models.
        """
        lr_size = self.generator.module.grid_size
        hr_size = self.critic.module.input_size
        hr_size += 2 * self.critic.module.use_nn_distance_features
        scale = self.scale_factor
        
        # Calculate the crop size for the lr data.
        self.lr_crop_size = (lr_size - hr_size // scale) // 2
        
        # Calculate the final crop size for the upscaled lr data.
        self.hr_crop_size = 2 * (lr_size - 2 * self.lr_crop_size) - hr_size
        self.hr_crop_size //=2
        
        
    def set_dataset(
            self, 
            dataloader, 
            batch_size, 
            box_size,
        ):
        self.data = dataloader
        self.box_size = box_size
        self.batch_size = batch_size
        
        
    def set_optimizers(self, lr_G=0.000001, lr_C=0.000002, b1=0.0, b2=0.99):
        self.optimizer_g = optim.Adam(
            self.generator.parameters(), lr=lr_G, betas=(b1, b2)
        )
        self.optimizer_c = optim.Adam(
            self.critic.parameters(), lr=lr_C, betas=(b1, b2)
        )
        
    
    def set_monitor(self, monitor):
        self.monitor = monitor
        
    
    def train(self, num_epochs, train_step=None):
        """Train the DMSR-WGAN for the given number of epochs.
        """
        if train_step is None:
            train_step = self.train_step
        
        self.monitor.init_monitoring(num_epochs, len(self.data))
        
        for epoch in range(num_epochs):
            if self.distributed:
                self.data.sampler.set_epoch(epoch)
                dist.barrier()
            
            for batch_num, batch in enumerate(self.data):
                losses = train_step(*batch)
                
                # End of batch processing.
                self.monitor.end_of_batch(
                    epoch, batch_num, self.batch_counter, losses
                )
                self.batch_counter += 1
                    
            # End of epoch processing.
            self.monitor.end_of_epoch(epoch)
            
        if self.distributed:
            dist.barrier()
    
        
    #=========================================================================#
    #                        Supervised learning
    #=========================================================================#
                
    def generator_supervised_step(self, lr_batch, hr_batch, style=None):
        # Move data to the device.
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        if style is not None:
            style = style.to(self.device)
        
        # Use the generator to create fake data.
        batch_size, device = self.batch_size, self.device
        z = self.generator.module.sample_latent_space(batch_size, device)
        sr_batch = self.generator(lr_batch, z, style)
        
        # Compute the loss and update the generator parameters.
        self.optimizer_g.zero_grad()
        loss = self.mse_loss(sr_batch, hr_batch)
        loss.backward()
        self.optimizer_g.step()
        
        losses = {'generator_loss' : loss.item()}
        return losses
    
    
    def critic_supervised_step(self, lr_batch, hr_batch, style=None):
        # Move data to the device.
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        if style is not None:
            style = style.to(self.device)
        
        # Prepare upscaled data
        us_batch = crop(lr_batch, self.lr_crop_size)
        us_batch = interpolate(
            us_batch, scale_factor=self.scale_factor, mode='trilinear'
        )
        us_batch = crop(us_batch, self.hr_crop_size).detach()
        
        # Compute the loss and update the generator parameters.
        losses = self.critic_train_step(lr_batch, hr_batch, us_batch, style)
        return losses
        
        
    #=========================================================================#
    #                           WGAN learning
    #=========================================================================#
            
    def train_step(self, lr_batch, hr_batch, style=None):
        """Train step for the WGAN.
        """
        # Move data to the device.
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        if style is not None:
            style = style.to(self.device)
        
        # Prepare upscaled data
        us_batch = crop(lr_batch, self.lr_crop_size)
        us_batch = interpolate(
            us_batch, scale_factor=self.scale_factor, mode='trilinear'
        )
        us_batch = crop(us_batch, self.hr_crop_size).detach()
        
        # Train the critic and generator models.
        critic_losses = self.critic_train_step(
            lr_batch, hr_batch, us_batch, style
        )
        generator_losses = self.generator_train_step(
            lr_batch, us_batch, style
        )
        return critic_losses | generator_losses
    
    
    def critic_train_step(self, lr_batch, hr_batch, us_batch, style=None):
        """Train step for the critic.
        """
        self.optimizer_c.zero_grad()
        batch_size, device = self.batch_size, self.device
        
        # Create fake data using the generator.
        z = self.generator.module.sample_latent_space(batch_size, device)
        sr_batch = self.generator(lr_batch, z, style)
        fake_data = self.critic.module.prepare_batch(
            sr_batch, us_batch, self.box_size
        )
        fake_data = tuple(tensor.detach() for tensor in fake_data)
        
        # Prepare real data.
        real_data = self.critic.module.prepare_batch(
            hr_batch, us_batch, self.box_size
        )
        real_data = tuple(tensor.detach() for tensor in real_data)
        
        # Use the critic to score the real and fake data and compute the loss.
        real_scores = self.critic(*real_data, style)
        fake_scores = self.critic(*fake_data, style)
        critic_loss = fake_scores.mean() - real_scores.mean()
        losses = {'critic_loss' : critic_loss.item()}
        
        # Add the gradient penalty term to the loss.
        if self.batch_counter % self.gradient_penalty_rate == 0:
            gradient_penalty = self.critic.module.gradient_penalty(
                batch_size, *real_data, *fake_data, style, device
            )
            losses['gradient_penalty'] = gradient_penalty.item()
        else:
            gradient_penalty = 0
            losses['gradient_penalty'] = gradient_penalty
        
        # Update the critic parameters.
        total_critic_loss = critic_loss + gradient_penalty
        total_critic_loss.backward()
        self.optimizer_c.step()
        
        return losses
    
    
    def generator_train_step(self, lr_batch, us_batch, style=None):
        """Train step for the generator.
        """
        self.optimizer_g.zero_grad()
        batch_size, device = self.batch_size, self.device
        
        # Use the generator to create fake data.
        z = self.generator.module.sample_latent_space(batch_size, device)
        sr_batch = self.generator(lr_batch, z, style)
        fake_data = self.critic.module.prepare_batch(
            sr_batch, us_batch, self.box_size
        )
        
        # Use the critic to score the generated data.
        fake_scores = self.critic(*fake_data, style)
        generator_loss = -fake_scores.mean()
        
        # Update the generator parameters.
        generator_loss.backward()
        self.optimizer_g.step()
        
        losses = {'generator_loss' : generator_loss.item()}
        return losses
    
    
    #=========================================================================#
    #                         Saving and Loading
    #=========================================================================#

    def save(self, model_dir=Path('./data/model/')):
        """Save the model
        
        Note: data attributes to are note saved. These should be set by the
        set_dataset method.
        """
        if self.distributed:
            if not (dist.get_rank() == 0):
                return
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the models.
        self.critic.module.save(model_dir)
        self.generator.module.save(model_dir)
        
        # Save optimizer states
        optimizer_states = {
            'optimizer_c': self.optimizer_c.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict()
        }
        save(optimizer_states, model_dir / 'optimizers.pth')
        
        # Save other attributes
        attributes = {
            'batch_counter': self.batch_counter,
            'gradient_penalty_rate': self.gradient_penalty_rate
        }
        save(attributes, model_dir / 'attributes.pth')
        
    
    @classmethod
    def load(cls, model_dir, device, distributed = False):
        """Load a saved model
        """
        # Load the generator and critic models. Then create a WGAN instance.
        generator = DMSRGenerator.load(model_dir, device)
        critic = DMSRCritic.load(model_dir, device)
        gan = cls(generator, critic, device, distributed=distributed)
        
        # Load optimizers.
        optimizers = load(
            model_dir / 'optimizers.pth', 
            map_location=device, 
            weights_only=False
        )
        gan.set_optimizers()
        gan.optimizer_c.load_state_dict(optimizers['optimizer_c'])
        gan.optimizer_g.load_state_dict(optimizers['optimizer_g'])
        
        # Load any additional attributes that were saved.
        attributes = load(
            model_dir / 'attributes.pth', 
            map_location=device,
            weights_only=False
        )
        vars(gan).update(attributes)
        
        return gan