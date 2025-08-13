#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:31:22 2024

@author: brennan

This file defines a base Monitor classes from which other monitor classes can
inherit from. It provides base implementations of methods every monitor class
is expected to have.

Monitor classes can be used to track various quantities or perform various 
tasks at different stages during training of a DMSR-WGAN.
"""

class Monitor():
    
    def post_epoch_processing(self, epoch):
        pass
    
    def post_batch_processing(self, epoch, batch, batch_counter, losses):
        return ''