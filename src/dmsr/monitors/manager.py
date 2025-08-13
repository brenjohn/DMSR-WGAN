#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:25:54 2025

@author: brennan

This file defines the a Monitor Manager class which is used to handle monitor 
objects during training.
"""

import time


class MonitorManager():
    """A class to manage monitor objects.
    
    The Monitor Manager class stores and calls Monitor objects during DMSR-WGAN
    training at appropriate times.
    
    Monitor objects are stored in a monitors dictionary. During DMSR training,
    at the end of a batch update the `post_batch_processing` method of each 
    monitor object is called. Similarly, at the end of each epoch, the
    `post_epoch_processing` method of each monitor is called by the monitor
    manager.
    
    Any messages returned by the `post_batch_processing` calls are passed to a
    batch report method which prints them along with some information regarding
    batch/epoch number and timings. At the end of each epoch, the monitor
    manager also prints some timing information regarding the epoch and epoch
    post processing.
    """
    
    def __init__(self, report_rate, device):
        self.device = device
        self.report_rate = report_rate
        
    
    def set_monitors(self, monitors):
        self.monitors = monitors
        
    
    def init_monitoring(self, num_epochs, num_batches):
        """Initializes values for variables used for timing batches and epochs.
        """
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.batch_start_time = time.time()
        self.epoch_start_time = time.time()
    
    
    def end_of_epoch(self, epoch):
        """Calls the `post_epoch_processing` method of each monitor.
        """
        epoch_time = time.time() - self.epoch_start_time
        print(f"[Epoch {epoch} took: {epoch_time:.4f} sec]")
        post_processing_start_time = time.time()
        
        for monitor in self.monitors.values():
            monitor.post_epoch_processing(epoch)
        
        self.epoch_start_time = time.time()
        self.batch_start_time = time.time()
        post_processing_time = time.time() - post_processing_start_time
        print(
            f"[Epoch post-processing took: {post_processing_time:.4f} sec]",
            flush=True
        )
    
        
    def end_of_batch(self, epoch, batch, batch_counter, losses):
        """Calls the `post_batch_processing` method of each monitor.
        """
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