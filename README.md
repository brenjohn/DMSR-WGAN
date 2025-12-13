# DMSR-WGAN: Dark-Matter Super-Resolution WGAN

This repository contains code and resources aimed at enhancing the particle resolution of dark matter-only simulations using a Wasserstein Generative Adversarial Network (WGAN). 
This work has applications in astrophysics and cosmology, particularly for improving simulation fidelity in large-scale structure studies.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Usage](#usage)
3. [Reproducing Results](#reproducing-results)
<!-- 3. [Dependencies](#installation) -->

---

## Project Overview

The **DMSR-WGAN** uses adversarial training to super-resolve dark matter simulations. It generates high-resolution particle distributions from low-resolution inputs. This approach enables:
- Enhanced particle resolution.
- Improved substructure identification.
- Reduced computational cost compared to high-resolution simulations.

For more details, refer to: 
[On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ).

---

## Usage

Parameters for creating datasets and training a model should be set in parameter toml files. Example parameter files can be found in the example directory.

To create training and validation datasets, it is assumed a collection or LR-HR swift snapshots exist in a directory whose path is specified in a toml file. The following command will create a training dataset and place it in a directory specified in the used toml file. The same command can be used to create a validation set.
```
python ../scripts/dataset_preperation.py --config_file=dataset_config_train.toml --num_procs=4
```

The following commands will compute summary statistics for the training set (used for rescaling data for the WGAN) and compute power spectra for patches (used by the spectrum monitor for validation).
```
python ../scripts/dataset_statistics.py --dataset_dir=./data/dmsr_train/ --include_velocities=True
python ../scripts/dataset_spectra.py --dataset_dir=./data/dmsr_valid/
```

A training job can be launched with the following command
```
python ../scripts/dmsr_train.py --parameter_file=./test_training_parameters.toml
```

Once a model is trained, it can be used to upscale a swift snapshot with a command like the following
```
python ../scripts/swift_enhance.py --model_dir=./sr_model/current_model/ --data_dir=../data/swift_sims/run1/064/ --snapshot_pattern=snap_000[0-8].hdf5 --output_suffix=_sr --output_dir=./sr_snapshots --seed=7
```

---

## Reproducing Results
For scripts and steps to reproduce results from [On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ), 
please switch to the `on-wgan-super-resolution` branch.
