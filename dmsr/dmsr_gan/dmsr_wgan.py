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
    
    def __init__(self, generator, critic, device):
        self.generator = generator
        self.critic = critic
        self.device = device
        self.batch_counter = 0
        self.mse_loss = MSELoss()
        
        
    def set_dataset(
            self, 
            dataloader, 
            batch_size, 
            box_size, 
            lr_padding, 
            scale_factor
        ):
        self.data = dataloader
        self.box_size = box_size
        self.lr_padding = lr_padding
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        
        
    def set_optimizer(self, optimizer_c, optimizer_g):
        self.optimizer_c = optimizer_c
        self.optimizer_g = optimizer_g
        
    
    def set_monitor(self, monitor):
        self.monitor = monitor
        
    
    def train(self, num_epochs, train_step=None):
        """Train the DMSR-WGAN for the given number of epochs.
        """
        if not train_step:
            train_step = self.train_step
        
        self.monitor.init_monitoring(num_epochs, len(self.data))
        
        for epoch in range(num_epochs):
            for batch, (lr_batch, hr_batch) in enumerate(self.data):
                losses = train_step(lr_batch, hr_batch)
                
                # End of batch processing.
                self.monitor.end_of_batch(
                    epoch, batch, self.batch_counter, losses
                )
                self.batch_counter += 1
                    
            # End of epoch processing.
            self.monitor.end_of_epoch(epoch)
    
        
    #=========================================================================#
    #                        Supervised learning
    #=========================================================================#
                
    def generator_supervised_step(self, lr_batch, hr_batch):
        # Move data to the device.
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        
        # Use the generator to create fake data.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
        sr_batch = self.generator(lr_batch, z)
        
        # Compute the loss and update the generator parameters.
        self.optimizer_g.zero_grad()
        loss = self.mse_loss(sr_batch, hr_batch)
        loss.backward()
        self.optimizer_g.step()
        
        losses = {'generator_loss' : loss.item()}
        return losses
    
    
    def critic_supervised_step(self, lr_batch, hr_batch):
        # Move data to the device.
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        
        # Prepare upscaled data
        us_batch = crop(lr_batch, self.lr_padding)
        us_batch = interpolate(
            us_batch, scale_factor=self.scale_factor, mode='trilinear'
        ).detach()
        
        # Compute the loss and update the generator parameters.
        losses = self.critic_train_step(lr_batch, hr_batch, us_batch)
        return losses
        
        
    #=========================================================================#
    #                           WGAN learning
    #=========================================================================#
            
    def train_step(self, lr_batch, hr_batch):
        """Train step for the WGAN.
        """
        # Move data to the device.
        lr_batch = lr_batch.to(self.device)
        hr_batch = hr_batch.to(self.device)
        
        # Prepare upscaled data
        us_batch = crop(lr_batch, self.lr_padding)
        us_batch = interpolate(
            us_batch, scale_factor=self.scale_factor, mode='trilinear'
        ).detach()
        
        # TODO: Maybe add random augmentation to data before generator step.
        # Train the critic and generator models.
        critic_losses = self.critic_train_step(lr_batch, hr_batch, us_batch)
        generator_losses = self.generator_train_step(lr_batch, us_batch)
        return critic_losses | generator_losses
    
    
    def critic_train_step(self, lr_batch, hr_batch, us_batch):
        """Train step for the critic.
        """
        self.optimizer_c.zero_grad()
        
        # Create fake data using the generator.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
        sr_batch = self.generator(lr_batch, z)
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
        real_scores = self.critic(*real_data)
        fake_scores = self.critic(*fake_data)
        critic_loss = fake_scores.mean() - real_scores.mean()
        losses = {'critic_loss' : critic_loss.item()}
        
        # Add the gradient penalty term to the loss.
        if self.batch_counter % 16 == 0:
            gradient_penalty = self.critic.gradient_penalty(
                self.batch_size, *real_data, *fake_data, self.device
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
    
    
    def generator_train_step(self, lr_batch, us_batch):
        """Train step for the generator.
        """
        self.optimizer_g.zero_grad()
        
        # Use the generator to create fake data.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
        sr_batch = self.generator(lr_batch, z)
        fake_data = self.critic.prepare_batch(
            sr_batch, us_batch, self.box_size
        )
        
        # Use the critic to score the generated data.
        fake_scores = self.critic(*fake_data)
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
            'batch_counter' : self.batch_counter
        }
        save(attributes, model_dir + 'attributes.pth')
        
        
    def load(self, checkpoint_dir):
        """Load a saved model
        """
        self.critic = load(checkpoint_dir + 'critic.pth')
        self.generator = load(checkpoint_dir + 'generator.pth')
        
        # Here we create new instances of the optimizers to update the weights
        # of the new models just created.
        optimizer_type = type(self.optimizer_c)
        self.optimizer_c = optimizer_type(self.critic.parameters())
        
        optimizer_type = type(self.optimizer_g)
        self.optimizer_g = optimizer_type(self.generator.parameters())
        
        # Now load the saved state of the optimizers.
        optimizer_states = load(checkpoint_dir + 'optimizers.pth')
        self.optimizer_c.load_state_dict(optimizer_states['optimizer_c'])
        self.optimizer_g.load_state_dict(optimizer_states['optimizer_g'])
        
        # Load any additional attributes that were saved.
        attributes = load(checkpoint_dir + 'attributes.pth')
        vars(self).update(attributes)