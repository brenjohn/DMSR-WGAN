#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:15:52 2024

@author: brennan
"""

import time
import torch

from dmsr.dmsr_gan.dmsr_critic import DMSRCritic


batch_size = 3
critic_size = 32
critic_channels = 128
critic = DMSRCritic(critic_size, critic_channels)

#%%
fake_data = torch.randn((batch_size, 8, critic_size, critic_size, critic_size))

score = critic(fake_data)

print("Score for fake data is", score)