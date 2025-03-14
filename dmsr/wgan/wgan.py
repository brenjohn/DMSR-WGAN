#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:34:12 2024

@author: brennan

This file define the DMSR-WGAN class.
"""

import os

from torch import save, load
from torch.nn import MSELoss
from torch.nn.functional import interpolate

from ..field_operations.resize import crop


class DMSRWGAN:
    """The DMSR-WGAN model.
    
    The name abreviates the Dark Matter, Super Resolution, Wasserstein 
    Generative Adversarial Neural Networks.
    """
    
    def __init__(self, generator, critic, device, gradient_penalty_rate=16):
        self.generator = generator
        self.critic = critic
        self.device = device
        self.gradient_penalty_rate = gradient_penalty_rate
        self.batch_counter = 0
        self.scale_factor = generator.scale_factor
        self.mse_loss = MSELoss()
        
        
    def set_dataset(
            self, 
            dataloader, 
            batch_size, 
            box_size,
        ):
        self.data = dataloader
        self.box_size = box_size
        self.batch_size = batch_size
        
        
    def set_optimizer(self, optimizer_c, optimizer_g):
        self.optimizer_c = optimizer_c
        self.optimizer_g = optimizer_g
        
    
    def set_monitor(self, monitor):
        self.monitor = monitor
        
    
    def train(self, num_epochs, train_step=None):
        """Train the DMSR-WGAN for the given number of epochs.
        """
        if train_step is None:
            train_step = self.train_step
        
        self.monitor.init_monitoring(num_epochs, len(self.data))
        
        for epoch in range(num_epochs):
            for batch_num, batch in enumerate(self.data):
                losses = train_step(*batch)
                
                # End of batch processing.
                self.monitor.end_of_batch(
                    epoch, batch_num, self.batch_counter, losses
                )
                self.batch_counter += 1
                    
            # End of epoch processing.
            self.monitor.end_of_epoch(epoch)
    
        
    #=========================================================================#
    #                        Supervised learning
    #=========================================================================#
                
    def generator_supervised_step(self, lr_batch, hr_batch, style=None):
        # Move data to the device.
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        
        # Use the generator to create fake data.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
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
        
        # Prepare upscaled data
        us_batch = crop(lr_batch, self.generator.padding)
        us_batch = interpolate(
            us_batch, scale_factor=self.scale_factor, mode='trilinear'
        ).detach()
        
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
        us_batch = crop(lr_batch, self.generator.padding)
        us_batch = interpolate(
            us_batch, scale_factor=self.scale_factor, mode='trilinear'
        ).detach()
        
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
        
        # Create fake data using the generator.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
        sr_batch = self.generator(lr_batch, z, style)
        fake_data = self.critic.prepare_batch(
            sr_batch, us_batch, self.box_size
        )
        fake_data = tuple(tensor.detach() for tensor in fake_data)
        
        # Prepare real data.
        real_data = self.critic.prepare_batch(
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
            gradient_penalty = self.critic.gradient_penalty(
                self.batch_size, *real_data, *fake_data, style, self.device
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
        
        # Use the generator to create fake data.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
        sr_batch = self.generator(lr_batch, z, style)
        fake_data = self.critic.prepare_batch(
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

    def save(self, model_dir='./data/model/'):
        """Save the model
        
        Note: data attributes to are note saved. These should be set by the
        set_dataset method.
        """
        os.makedirs(model_dir, exist_ok=True)

        save(self.critic, model_dir + 'critic.pth')
        save(self.generator, model_dir + 'generator.pth')

        optimizer_states = {
            'optimizer_c' : self.optimizer_c.state_dict(),
            'optimizer_g' : self.optimizer_g.state_dict()
        }
        save(optimizer_states, model_dir + 'optimizers.pth')
        
        attributes = {
            'batch_counter'         : self.batch_counter,
            'gradient_penalty_rate' : self.gradient_penalty_rate
        }
        save(attributes, model_dir + 'attributes.pth')
        
        
    def load(self, model_dir):
        """Load a saved model
        """
        self.critic = load(model_dir + 'critic.pth', weights_only=False)
        self.generator = load(model_dir + 'generator.pth', weights_only=False)
        
        # Here we create new instances of the optimizers to update the weights
        # of the new models just created.
        optimizer_type = type(self.optimizer_c)
        self.optimizer_c = optimizer_type(self.critic.parameters())
        
        optimizer_type = type(self.optimizer_g)
        self.optimizer_g = optimizer_type(self.generator.parameters())
        
        # Now load the saved state of the optimizers.
        optimizers = load(model_dir + 'optimizers.pth', weights_only=False)
        self.optimizer_c.load_state_dict(optimizers['optimizer_c'])
        self.optimizer_g.load_state_dict(optimizers['optimizer_g'])
        
        # Load any additional attributes that were saved.
        attributes = load(model_dir + 'attributes.pth', weights_only=False)
        vars(self).update(attributes)