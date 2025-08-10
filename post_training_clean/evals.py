"""
evals.py gets the eigenvalues of the A and B matrices. 

See Fig. S6 in the SI for more details. 

Used to check consistently-learned representations/subspaces up to permutations and rotations 
when phi(x) = Px + b.
"""

import os
import pickle
import json
import argparse
import logging
import glob

import numpy as np
import jax
from matplotlib import pyplot as plt

# Highest float precision for numerical accuracy.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
jax.config.update("jax_default_matmul_precision", "highest")

################################################################################
# Parameter handling
################################################################################

def get_parameters_by_index(config, index):
    """Get combination of params from config based on index."""
    keys = [key for key in config.keys() if key != "directory"]
    sizes = [len(config[key]) for key in keys]
    total_combinations = np.prod(sizes)
    if index >= total_combinations:
        raise IndexError("Index out of bounds for the given configuration.")
    params = {}

    # Divmod to get parameter values.
    for key in keys:
        size = len(config[key])
        index, value_index = divmod(index, size)
        params[key] = config[key][value_index]
    return params

################################################################################
# Model loading functions
################################################################################

def load_model_params(config_dir, parent_dir, dataset_filepath, render_flag=False):
    """Load model parameters from checkpoint files."""
    # Load model params
    run_filepath = parent_dir + '/' + config_dir
    config_file = run_filepath + '/args.json'

    # Gather both checkpoint types: ones with and without '_vars'
    ckpt_files = glob.glob(run_filepath + '/ckpt/step_*.pkl')
    ckpt_files_vars = [f for f in ckpt_files if '_vars' in os.path.basename(f)]
    ckpt_files_nonvars = [f for f in ckpt_files if '_vars' not in os.path.basename(f)]
    if ckpt_files_vars and ckpt_files_nonvars:
        # Load the latest version from each category
        pkl_path_vars = max(ckpt_files_vars, key=os.path.getctime)
        pkl_path_nonvars = max(ckpt_files_nonvars, key=os.path.getctime)
        model_params_vars = pickle.load(open(pkl_path_vars, "rb"))
        model_params_nonvars = pickle.load(open(pkl_path_nonvars, "rb"))

        # Combine the dictionaries (non-vars updated with vars).
        model_params = {**model_params_nonvars, **model_params_vars}
    else:
        raise FileNotFoundError("No valid checkpoint files found in the directory.")
    
    return model_params

def extract_eigenvalues_from_model(model_params):
    """
    Extract A and B eigenvalues from loaded model parameters.

    Parameters:
        model_params (dict): Saved model params.

    Returns:
        A_evals (dict): A eigenvalues.
        B_evals (dict): B eigenvalues.
    """
    A_evals = {}
    B_evals = {}
    
    # Get A and B matrices from saved model params.
    if 'A' in model_params and 'B' in model_params:
        A = model_params['A']
        B = model_params['B']
        
        # Calculate eigenvalues
        A_evals_tmp = np.linalg.eigvals(A)
        B_evals_tmp = np.linalg.eigvals(B)
        
        # Take absolute value of eigenvalues
        A_evals_tmp = np.abs(A_evals_tmp)
        B_evals_tmp = np.abs(B_evals_tmp)
        
        num_evals = A_evals_tmp.shape[0]
        if num_evals not in A_evals:
            A_evals[num_evals] = []
            B_evals[num_evals] = []

        A_evals[num_evals].append(A_evals_tmp)
        B_evals[num_evals].append(B_evals_tmp)
        
        logging.info(f"Extracted eigenvalues for matrix size {num_evals}")
    else:
        logging.warning("A and B matrices not found in saved model params!")
    
    return A_evals, B_evals

################################################################################
# Eigenvalue plotting functions
################################################################################

def plot_eigenvalues(A_evals, B_evals, save_dir):
    """Plot and save eigenvalue figures and data."""
    
    # Sort eigenvalues for each key.
    for eval_key in A_evals.keys():
        A_evals[eval_key] = np.array(A_evals[eval_key])
        B_evals[eval_key] = np.array(B_evals[eval_key])
        A_evals[eval_key] = np.sort(A_evals[eval_key], axis=-1)
        B_evals[eval_key] = np.sort(B_evals[eval_key], axis=-1)
        logging.info(f"Shape of A_evals[{eval_key}]: {A_evals[eval_key].shape}")
        logging.info(f"Shape of B_evals[{eval_key}]: {B_evals[eval_key].shape}")

    cmap = plt.get_cmap('viridis')  
    colors = cmap(np.linspace(0, 1, len(A_evals)))

    # Plot A eigenvalues from largest to smallest eigenvalue.
    # x axis: ordered eigenvalue index.
    # y axis: eigenvalue.
    fig, ax = plt.subplots(figsize=(8,6))
    A_evals_plots_dict = {}
    for (eval_key, color) in zip(A_evals.keys(), colors):
        data = np.sort(np.array(A_evals[eval_key]), axis=-1)    # [num_seeds, num_eigs]
        mean = data.mean(axis=0)
        std  = data.std(axis=0)
        A_evals_plots_dict[eval_key] = {'mean': mean.tolist(), 'std': std.tolist()}
        x    = np.arange(mean.shape[0])

        ax.errorbar(
            x, mean, yerr=std,
            label=f'M={eval_key}',
            color=color,
            capsize=3,   # little caps on the error bars
        )

    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'A Evals')
    ax.legend(title='M=', loc='best')
    ax.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'A_evals_combined.png')
    fig.savefig(out_path)
    plt.close(fig)

    # Plot B eigenvalues.
    # x axis: ordered eigenvalue index.
    # y axis: eigenvalue.
    fig, ax = plt.subplots(figsize=(8,6))
    B_evals_plots_dict = {}
    for (eval_key, color) in zip(B_evals.keys(), colors):
        data = np.sort(np.array(B_evals[eval_key]), axis=-1)
        mean = data.mean(axis=0)
        std  = data.std(axis=0)
        B_evals_plots_dict[eval_key] = {'mean': mean.tolist(), 'std': std.tolist()}
        x    = np.arange(mean.shape[0])

        ax.errorbar(
            x, mean, yerr=std,
            label=f'M={eval_key}',
            color=color,
            capsize=3,
        )

    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(f'B Evals')
    ax.legend(title='M=', loc='best')   
    ax.grid(True)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'B_evals_combined.png')
    fig.savefig(out_path)
    plt.close(fig)

    # Save eigenvalue data to jsons for later re-plotting.
    with open(os.path.join(save_dir, f'A_evals_plots_dict.json'), 'w') as f:
        json.dump(A_evals_plots_dict, f)
    with open(os.path.join(save_dir, f'B_evals_plots_dict.json'), 'w') as f:
        json.dump(B_evals_plots_dict, f)
    
    logging.info(f"Eigenvalue plots and data saved to {save_dir}")

################################################################################
# Main execution
################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process logs from a run directory.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--current_folder', type=str, required=True,)
    args = parser.parse_args()

    # Load model params saved during training. 
    # See src/networks.py for definitions of the saved model params. 
    model_params = load_model_params(args.config_file, args.current_folder)

    # Extract eigenvalues.
    A_evals, B_evals = extract_eigenvalues_from_model(model_params)

    # Plot eigenvalues.
    plot_eigenvalues(A_evals, B_evals, args.current_folder)


