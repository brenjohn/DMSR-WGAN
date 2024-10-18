#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:34:12 2024

@author: brennan
"""

import os

from torch import concat, rand, autograd, save, load
from torch.nn import MSELoss
from torch.nn.functional import interpolate

from dmsr.field_operations.resize import crop
from dmsr.field_operations.conversion import cic_density_field


class DMSRWGAN:
    """
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
        """Train the DMSRWGAN for the given number of epochs.
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
        )
        us_density = cic_density_field(us_batch, self.box_size)
        us_batch = concat((us_density, us_batch), dim=1).detach()
        
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
        density_size = self.critic.density_size
        us_batch = crop(lr_batch, self.lr_padding)
        us_batch = interpolate(
            us_batch, scale_factor=self.scale_factor, mode='trilinear'
        )
        us_density = cic_density_field(us_batch, self.box_size, density_size)
        us_batch = (us_batch.detach(), us_density.detach())
        
        # TODO: Maybe add random augmentation to data before generator step.
        # Train the critic and generator models.
        critic_losses = self.critic_train_step(lr_batch, hr_batch, us_batch)
        generator_losses = self.generator_train_step(lr_batch, us_batch)
        return critic_losses | generator_losses
    
    
    def critic_train_step(self, lr_batch, hr_batch, us_batch):
        """Train step for the critic.
        """
        self.optimizer_c.zero_grad()
        us_batch, us_density = us_batch
        density_size = self.critic.density_size
        
        # Create fake data using the generator.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
        sr_batch = self.generator(lr_batch, z)
        sr_density = cic_density_field(sr_batch, self.box_size, density_size)
        fake_batch = concat((sr_batch, us_batch), dim=1).detach()
        fake_density = concat((sr_density, us_density), dim=1).detach()
        fake_data = (fake_batch, fake_density)
        
        # Prepare real data.
        hr_density = cic_density_field(hr_batch, self.box_size, density_size)
        real_batch = concat((hr_batch, us_batch), dim=1).detach()
        real_density = concat((hr_density, us_density), dim=1).detach()
        real_data = (real_batch, real_density)
        
        # Use the critic to score the real and fake data and compute the loss.
        real_scores = self.critic(*real_data)
        fake_scores = self.critic(*fake_data)
        critic_loss = fake_scores.mean() - real_scores.mean()
        losses = {'critic_loss' : critic_loss.item()}
        
        # Add the gradient penalty term to the loss.
        if self.batch_counter % 16 == 0:
            gradient_penalty = self.gradient_penalty(real_data, fake_data)
            losses['gradient_penalty'] = gradient_penalty.item()
        else:
            gradient_penalty = 0
            losses['gradient_penalty'] = gradient_penalty
        
        # Update the critic parameters.
        total_critic_loss = critic_loss + gradient_penalty
        total_critic_loss.backward()
        self.optimizer_c.step()
        
        return losses
    
    
    def gradient_penalty(self, real_data, fake_data, weight=10):
        """Calculate the gradient penalty for WGAN critic.
        """
        # Create data by interploating real and fake data and random amount.
        alpha = rand(self.batch_size, device=self.device)
        alpha = alpha.reshape(self.batch_size, *(1, 1, 1, 1))
        real_displacements, real_density = real_data
        fake_displacements, fake_density = fake_data
        displacements  = real_displacements * alpha
        displacements += fake_displacements * (1 - alpha)
        density  = real_density * alpha
        density += fake_density * (1 - alpha)
        
        # Score the new data with the critic model and get its derivative.
        score = self.critic(
            displacements.requires_grad_(True), density.requires_grad_(True)
        )
        score = score.sum()
        displacement_grad, density_grad = autograd.grad(
            score,
            (displacements, density),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )
        
        # Compute the gradient penalty term.
        density_grad = density_grad.flatten(start_dim=1)
        displacement_grad = displacement_grad.flatten(start_dim=1)
        grad = concat((displacement_grad, density_grad), dim=1)
        penalty = (weight * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
            + 0 * score  # hack to trigger DDP allreduce hooks
        )
        return penalty
    
    
    def generator_train_step(self, lr_batch, us_batch):
        """Train step for the generator.
        """
        self.optimizer_g.zero_grad()
        us_batch, us_density = us_batch
        density_size = self.critic.density_size
        
        # Use the generator to create fake data.
        z = self.generator.sample_latent_space(self.batch_size, self.device)
        sr_batch = self.generator(lr_batch, z)
        sr_density = cic_density_field(sr_batch, self.box_size, density_size)
        fake_batch = concat((sr_batch, us_batch), dim=1)
        fake_density = concat((sr_density, us_density), dim=1)
        fake_data = (fake_batch, fake_density)
        
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
        """
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
        self.critic = load(checkpoint_dir + 'critic.pth')
        self.generator = load(checkpoint_dir + 'generator.pth')
        
        optimizer_states = load(checkpoint_dir + 'optimizers.pth')
        self.optimizer_c.load_state_dict(optimizer_states['optimizer_c'])
        self.optimizer_g.load_state_dict(optimizer_states['optimizer_g'])
        
        attributes = load(checkpoint_dir + 'attributes.pth')
        vars(self).update(attributes)