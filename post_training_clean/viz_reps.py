"""
viz_reps.py saves the learned representations and vector fields as CSVs. 

Loads in trained model params and pulls in corresponding saved datasets.

Copyright Catherine Ji, 2025. 
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import jax.numpy as jnp
import jax
import yaml

# Imports from epr_estimator
from ..utils import update_args_json
from ..src.networks import make_cl_networks
from ..dataset_creation.utils import generate_dataset, generate_dataset_render
from post_training_utils import normalize_data, load_model_params, load_config, create_cgl_gifs

jax.config.update("jax_default_matmul_precision", "highest")

################################################################################
# Functions for computing EPRs
################################################################################

def dict_epr_learned_AB(x, y, phi_x, phi_y, A, B):
    """
    Compute representations, flows, and localEPRs for the learned representations and vector fields.

    Parameters:
        x: Input data.
        y: Input data.
        phi_x: Learned representation of x.
        phi_y: Learned representation of y.
        A: Learned matrix A.
        B: Learned matrix B.

    Returns:
        epr_dictionary (dict): Dictionary containing the learned representations, flows, and local EPRs.
    """
    # Compute logit terms.
    psi_x = jnp.dot(phi_x, A.T)
    psi_y = jnp.dot(phi_y, A.T)  
    eta_x = jnp.dot(phi_x, B.T)
    eta_y = jnp.dot(phi_y, B.T)

    # Compute logits for EPR calculation.
    logits_1 = jnp.sum(phi_x * psi_y, axis=1)
    logits_2 = jnp.sum(phi_y * psi_x, axis=1)
    logits_3 = jnp.sum(phi_x * eta_x, axis=1)
    logits_4 = jnp.sum(phi_y * eta_y, axis=1)
    logits_traj = (logits_1 - logits_2) + (logits_3 - logits_4)
    
    # Create dictionary.
    return {
        'states': (x + y) / 2,
        'phi_x': phi_x,
        'rep_vec_field': (phi_y - phi_x) / 2,  # /2 because arrow drawn from midpoint of phi_x and phi_y.
        'state_field': (y - x) / 2,  # /2 because arrow drawn from midpoint of x and y.
        'reps': (phi_x + phi_y) / 2, 
        'epr': logits_traj, 
    }

################################################################################
# Data loading functions
################################################################################

def load_data(config_dir, parent_dir, dataset_filepath, render_flag=False, viz_cgl_flag=False):
    """
    Load model params and corresponding dataset.

    Parameters:
        config_dir (str): Config directory containing the config file.
        parent_dir (str): Parent directory containing the config directory.
        dataset_filepath (str): Path to the dataset file.
        render_flag (bool): Whether to render the dataset.
        viz_cgl_flag (bool): Whether to visualize the dataset.

    Returns:
        tuple: (model_params, train_dataset, val_dataset, config_namedtuple)
    """
    run_filepath = parent_dir + '/' + config_dir
    config_file = run_filepath + '/args.json'

    # Load model parameters
    model_params = load_model_params(config_dir, parent_dir)
    
    # Load dataset
    if render_flag:
        train_dataset, val_dataset, epr = generate_dataset(dataset_filepath, on_device=0)
        train_dataset = {'x': jnp.squeeze(train_dataset['x']), 'y': jnp.squeeze(train_dataset['y'])}
        val_dataset = {'x': jnp.squeeze(val_dataset['x']), 'y': jnp.squeeze(val_dataset['y'])}
    else:
        train_dataset, val_dataset, epr = generate_dataset_render(dataset_filepath, on_device=0)
        train_dataset = {'x': jnp.squeeze(train_dataset['x']), 'y': jnp.squeeze(train_dataset['y'])}
        val_dataset = {'x': jnp.squeeze(val_dataset['x']), 'y': jnp.squeeze(val_dataset['y'])}
    
    logging.info(f"Shape of dataset_eval: {val_dataset['x'].shape}")
    logging.info(f"Shape of dataset_train: {train_dataset['x'].shape}")

    # Create CGL GIFs if requested
    if viz_cgl_flag:
        create_cgl_gifs(val_dataset, run_filepath)

    # Load and process configuration
    config_namedtuple = load_config(config_file)
    
    return model_params, train_dataset, val_dataset, config_namedtuple

################################################################################
# Main execution
################################################################################

def main():
    """Main execution function."""
    # Read yaml file with log file paths
    with open(args.current_folder + '/' + args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Get configuration parameters.
    run_dir = config['run_dirs_to_analyze']
    directory = config['directory']
    configs_to_plot = config.get('configs_to_plot', [])
    render_flag = config['render_flag']
    viz_cgl_flag = config['viz_cgl_flag']
    condition_dict = config['condition_dict']

    # Update args.json files with config_dict (dataset params)
    config_dict_path = os.path.join(directory, run_dir, 'config_dict.json')
    update_args_json(config_dict_path)

    with open(config_dict_path, 'r') as f:
        config_dict = json.load(f)
        # Set config_folder_names to be all keys with word config in them
        config_folder_names = [key for key in config_dict.keys() 
                     if 'config' in key and
                     all(config_dict[key].get(cond_key) == cond_val 
                     for cond_key, cond_val in condition_dict.items())]

    logging.info(f"Before filter: {config_folder_names}")
    if configs_to_plot and len(configs_to_plot) > 0:
        allowed_configs = set(configs_to_plot)
        config_folder_names = [folder for folder in config_folder_names if folder in allowed_configs]
    logging.info(f"After filter: {config_folder_names}")

    for config_folder in config_folder_names:
        # Initialize filepath
        dataset_filepath = config_dict[config_folder]["datasets_to_analyze"]

        # Load model params and data
        parent_dir = os.path.join(directory, run_dir)
        model_params, dataset_train, dataset_eval, config_namedtuple = load_data(
            config_folder, parent_dir, dataset_filepath, render_flag=render_flag, viz_cgl_flag=viz_cgl_flag
        )

        # Normalize data
        mean, std = normalize_data(config_namedtuple, dataset_train, dataset_eval)
        del dataset_train  # Free memory

        logging.info(f"Mean: {mean}")
        logging.info(f"Std: {std}")

        x = dataset_eval['x']
        y = dataset_eval['y']

        # Update config_namedtuple with mean and std
        config_namedtuple = config_namedtuple._replace(mean=mean, std=std)
        
        # Create network and apply
        cl_network = make_cl_networks(config=config_namedtuple)
        logging.info("Network created successfully")
        
        model_dict_of_collections = {'params': model_params}
        reps = cl_network.apply(model_dict_of_collections, x, y, mutable=["matrices"])
        
        # Get learned matrices
        matrices = model_params['matrices']
        A_rep = matrices['A_asym']
        B_rep = matrices['B_sym']

        # Get learned dictionary
        if config_namedtuple.repr_fn == 'linear_local_subspaces':
            output, _ = reps
            phi_x, phi_y, A, B = output    
            logging.info(f"Shape of phi_x: {phi_x.shape}")
            logging.info(f"Shape of phi_y: {phi_y.shape}")
            logging.info(f"Shape of A: {A.shape}")
            logging.info(f"Shape of B: {B.shape}")

            dict_epr_learned = dict_epr_learned_AB(x, y, phi_x, phi_y, A_rep, B_rep)
            logging.info(f"Shape of phi_x: {phi_x.shape}")
            logging.info(f"First 10 entries of phi_x: {phi_x[:10]}")

            # Save representations as CSV files
            for key, value in dict_epr_learned.items():
                if key in ['reps', 'rep_vec_field', 'phi_x']:
                    output_path = os.path.join(directory, run_dir, config_folder, f"{key}_learned.csv")
                    np.savetxt(output_path, value, delimiter=",", fmt="%.6f")
                    logging.info(f"Saved {key} to {output_path}")
                else:
                    logging.info(f"Skipping {key}")
        else:
            raise ValueError("Not supported")
            
    logging.info("Finished post-training analysis")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process logs from a run directory.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to the YAML configuration file.')
    parser.add_argument('--current_folder', type=str, required=True,)
    
    args = parser.parse_args()

    main()




