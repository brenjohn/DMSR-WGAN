# DMSR-WGAN: Dark-Matter Super-Resolution WGAN

This repository contains code and resources aimed at enhancing the particle resolution of dark matter-only simulations using a Wasserstein Generative Adversarial Network (WGAN). 
This work has applications in astrophysics and cosmology, particularly for improving simulation fidelity in large-scale structure studies.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Results](#results)
<!-- 3. [Dependencies](#installation) -->
<!-- 4. [Usage](#usage) -->

---

## Project Overview

The **DMSR-WGAN** uses adversarial training to super-resolve dark matter simulations. It generates high-resolution particle distributions from low-resolution inputs. This approach enables:
- Enhanced particle resolution.
- Improved substructure identification.
- Reduced computational cost compared to high-resolution simulations.

For more details, refer to: 
[On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ).

---

## Results
For scripts and steps to reproduce results from [On the Use of WGANs for Super Resolution in Dark-Matter Simulations](https://arxiv.org/abs/2501.13056 ), 
please switch to the `on-wgan-super-resolution` branch.

The WGAN model used to produce for [Machine-Learned Super-Resolution Reproduces Halo Assembly and Early Galaxy Populations](https://arxiv.org/abs/2501.13056 ) can be found in `scripts/swift_upscaling/models/model_particle_density/`. Scripts for used to train the model can be found in `scripts/training/` while scripts and commands used to enhance a snapshot using a trained model can be found in `scripts/swift_upscaling/`.
