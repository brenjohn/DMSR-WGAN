#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:31:25 2024

@author: john

This script uses a specified dmsr generator to enhance the dark-matter data in
a low-resolution swift snapshot.
"""

import time
import argparse
import numpy as np

from pathlib import Path
from dmsr.wgan import DMSRGenerator
from swift_tools.enhance import enhance


def main(
        model_dir: Path, 
        data_dir: Path, 
        snapshot_pattern: str, 
        output_suffix: str
    ):
    """
    Enhance a Swift snapshots using a DMSR generator model.
    
    Args:
        model_dir (Path): The directory containing the dmsr model files.
        data_dir (Path): The directory containing low-resolution snapshots.
        snapshot_pattern (str): A glob pattern for low-resolution snapshots.
        output_suffix (str): A suffix to add to enhanced snapshot filenames.
    """
    device = "cpu"
    print(f"Using device: {device}")
    
    # Load the generator model and sample its latent space.
    generator = DMSRGenerator.load(model_dir, device)
    z = generator.sample_latent_space(1, device)
    
    # Load any scaling parameters if they exist.
    scale_path = model_dir / "normalisation.npy"
    scale_params = None
    if scale_path.exists():
        scale_params = np.load(scale_path, allow_pickle=True).item()
        scale_params = {k : v.item() for k, v in scale_params.items()}
    
    # Get paths to low-resolution snapshots.
    lr_snapshots = np.sort(list(data_dir.glob(snapshot_pattern)))
    
    for lr_snapshot in lr_snapshots:
        ti = time.time()
        print('Upscaling', lr_snapshot)
        sr_snapshot = lr_snapshot.parent
        sr_snapshot /= f"{lr_snapshot.stem}{output_suffix}{lr_snapshot.suffix}"
        
        # Enhance the low-resolution snapshot
        enhance(lr_snapshot, sr_snapshot, generator, z, scale_params, device)
        print(f'Upscaling took {time.time() - ti}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Enhance a Swift snapshot using a PyTorch model."
    )
    
    parser.add_argument(
        '--model_dir', 
        type=Path, 
        default='./dmsr_model/',
        help="Path to the directory containing the generator model."
    )
    parser.add_argument(
        '--data_dir', 
        type=Path, 
        default='./swift_snapshots/',
        help="Path to the directory containing the low-resolution snapshots."
    )
    parser.add_argument(
        '--snapshot_pattern', 
        type=str, 
        default='snap_*.hdf5',
        help="Glob pattern for low-resolution snapshots (e.g., 'snap_*.hdf5')."
    )
    parser.add_argument(
        '--output_suffix', 
        type=str, 
        default='_sr',
        help="Suffix to add to the output filename (e.g., '_sr')."
    )
    
    args = parser.parse_args()
    main(
        args.model_dir, 
        args.data_dir, 
        args.snapshot_pattern, 
        args.output_suffix
    )