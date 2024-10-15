#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:15:52 2024

@author: brennan
"""
import sys
sys.path.append("..")
sys.path.append("../..")

import time
import torch

from dmsr.dmsr_gan.dmsr_critic import DMSRCritic


batch_size = 3
density_size = 2 * 32 + 4
displacement_size = 32
density_channels = 16
displacement_channels = 16
critic = DMSRCritic(density_size, displacement_size, density_channels, displacement_channels)

#%%
disp = torch.randn((batch_size, 6, displacement_size, displacement_size, displacement_size))
dens = torch.randn((batch_size, 2, density_size, density_size, density_size))

initial_time = time.time()
score = critic(disp, dens)
final_time = time.time()

print("Score for fake data is", score)
print("Prediction took", final_time-initial_time, "seconds")