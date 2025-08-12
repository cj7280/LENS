## Setting up the evironment
To set up environment, run the following commands in terminal on the login node:

```bash
CONDA_OVERRIDE_CUDA="12.0" conda create --name env_jrl python=3.10 jax==0.4.23 "jaxlib==0.4.23=cuda120*" flax==0.7.4 numpy==1.26.4 scipy==1.12 matplotlib imageio pytorch -c conda-forge -c nvidia 

pip install tyro wandb==0.17.9 wandb_osh==1.2.2  brax==0.10.1 mediapy==1.2.2 scipy==1.12.0 pyyaml h5py torch
```
or run the following which directly creates the environment from the yaml:
```bash
conda env create -f environment.yml
```

A todo is to clean up the the environment and get rid of brax. 

## File organization

All main training files live in `src` folder and `training.py` file.

The `dataset_creation` folder contains code to generate harmonically-coupled beads, render these beads, simulate 2D Complex Ginzburg-Landau dynamics, and simulate a driven bead in a periodic potential.