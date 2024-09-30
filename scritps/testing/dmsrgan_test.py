#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:14:03 2024

@author: brennan
"""

import sys
sys.path.append("..")
sys.path.append("../..")

import os
import tensorflow as tf

# Enable detailed logging of retracing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.get_logger().setLevel('INFO')

import tensorflow.keras as keras
import numpy as np
import time

from dmsr.models.dmsr_gan.dmsr_gan import build_dmsrgan


#%% Create the GAN.
gan_args = { 
    'LR_grid_size'       : int(20),
    'scale_factor'       : 2,
    'HR_box_size'        : 1.0,
    'generator_channels' : 256,
    'critic_channels'    : 32
}

gan = build_dmsrgan(**gan_args)


#%%
gan_training_args = {
    'critic_optimizer'    : keras.optimizers.Adam(
        learning_rate=0.00002, beta_1=0.0, beta_2=0.99 , weight_decay=0.000001
    ),
    'generator_optimizer' : keras.optimizers.Adam(
        learning_rate=0.00001, beta_1=0.0, beta_2=0.99
    ),
    'critic_steps' : 2,
    'gp_weight'    : 10.0,
    'gp_rate'      : 1,
}

gan.compile(**gan_training_args)


#%%
LR_size = gan.generator.input[0].shape[-1]
HR_size = gan.generator.output.shape[-1]

LR_data = tf.random.normal((1, 3, LR_size, LR_size, LR_size))
US_data = tf.random.normal((1, 4, HR_size, HR_size, HR_size))
HR_data = tf.random.normal((1, 3, HR_size, HR_size, HR_size))

#%%
ti = time.time()
gan.critic_train_step(LR_data, US_data, HR_data)
time_critic_step = time.time() - ti
print('Critic step took :', time_critic_step)


#%%
ti = time.time()
gan.generator_train_step(LR_data, US_data)
time_generator_step = time.time() - ti
print('Generator step took :', time_generator_step)


#%%
ti = time.time()
gan.train_step((LR_data, HR_data))
time_gan_step = time.time() - ti
print('GAN step took :', time_gan_step)