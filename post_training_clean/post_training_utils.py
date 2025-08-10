"""
post_training_utils.py contains functions for post-training analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
import glob
import re
import json
from collections import namedtuple
import os
import imageio
from matplotlib import pyplot as plt

def normalize_data(config_namedtuple, train_dataset, val_dataset):
    """
    Normalize datasets based on configuration. 
    There are a lot of intermediate calculations done here from bug-fixing. 
    Note that there are no assumptions about the effective distributions or sizes of the training and validation datasets
    for x_t (x) or x_{t+1} (y). 
    
    Parameters:
        config_namedtuple: Configuration object containing normalization settings
        train_dataset: Training dataset dictionary with 'x' and 'y' keys
        val_dataset: Validation dataset dictionary with 'x' and 'y' keys
        
    Returns:
        mean: Normalization mean
        std: Normalization standard deviation
    """
    mean = 0.0
    std = 1.0
    
    if config_namedtuple.normalize_dataset == "fully_observed":
        with jax.default_device(jax.devices('cpu')[0]):
            # compute means
            n_x_train = train_dataset['x'].shape[0]
            n_x_val = val_dataset['x'].shape[0]
            n_y_train = train_dataset['y'].shape[0]
            n_y_val = val_dataset['y'].shape[0]

            sum_x_train = jnp.sum(train_dataset['x'], axis=0)
            sum_x_val = jnp.sum(val_dataset['x'], axis=0)
            sum_y_train = jnp.sum(train_dataset['y'], axis=0)
            sum_y_val = jnp.sum(val_dataset['y'], axis=0)

            # Force computation of squared terms serially
            sum_x_train_sq = jnp.sum(train_dataset['x']**2, axis=0)
            sum_x_val_sq = jnp.sum(val_dataset['x']**2, axis=0)
            sum_y_train_sq = jnp.sum(train_dataset['y']**2, axis=0)
            sum_y_val_sq = jnp.sum(val_dataset['y']**2, axis=0)
            
            mean_x_squared = (sum_x_train_sq + sum_x_val_sq) / (n_x_train + n_x_val)
            mean_y_squared = (sum_y_train_sq + sum_y_val_sq) / (n_y_train + n_y_val)
            mean_x = (sum_x_train + sum_x_val) / (n_x_train + n_x_val)
            mean_y = (sum_y_train + sum_y_val) / (n_y_train + n_y_val)

            var_x = jnp.maximum(mean_x_squared - mean_x**2, 0) # to handle numerical stability
            var_y = jnp.maximum(mean_y_squared - mean_y**2, 0) # to handle numerical stability

            pooled_var = ((n_x_train + n_x_val) * var_x + (n_y_train + n_y_val) * var_y) / ((n_x_train + n_x_val) + (n_y_train + n_y_val))
            std = jnp.sqrt(pooled_var)
            mean = (sum_x_train + sum_y_train + sum_x_val + sum_y_val) / (n_x_train + n_x_val + n_y_train + n_y_val)
    elif config_namedtuple.normalize_dataset == "partially_observed":
         with jax.default_device(jax.devices('cpu')[0]):
            # compute means
            n_x_train = float(train_dataset['x'].shape[0] * train_dataset['x'].shape[1] * train_dataset['x'].shape[2])
            n_x_val = float(val_dataset['x'].shape[0] * val_dataset['x'].shape[1] * val_dataset['x'].shape[2])
            n_y_train = float(train_dataset['y'].shape[0] * train_dataset['y'].shape[1] * train_dataset['y'].shape[2])
            n_y_val = float(val_dataset['y'].shape[0] * val_dataset['y'].shape[1] * val_dataset['y'].shape[2])

            if val_dataset['x'].dtype == jnp.uint8:
                train_dataset_calc = train_dataset.copy()
                val_dataset_calc = val_dataset.copy()

                train_dataset_calc['x'] = train_dataset['x'].astype(np.float16) / 255.0
                val_dataset_calc['x'] = val_dataset['x'].astype(np.float16) / 255.0
                train_dataset_calc['y'] = train_dataset['y'].astype(np.float16) / 255.0
                val_dataset_calc['y'] = val_dataset['y'].astype(np.float16) / 255.0
            else:
                train_dataset_calc = train_dataset.copy()
                val_dataset_calc = val_dataset.copy()

            # Taking means and stds across *all pixels* for each channel 
            sum_x_train = np.sum(train_dataset_calc['x'], axis=(0,1,2), dtype=np.float32)
            sum_x_val = np.sum(val_dataset_calc['x'], axis=(0,1,2), dtype=np.float32)
            sum_y_train = np.sum(train_dataset_calc['y'], axis=(0,1,2), dtype=np.float32)
            sum_y_val = np.sum(val_dataset_calc['y'], axis=(0,1,2), dtype=np.float32)

            mean_x = (sum_x_train + sum_x_val) / (n_x_train + n_x_val)
            mean_y = (sum_y_train + sum_y_val) / (n_y_train + n_y_val)

            sum_x_train_sq = np.sum(train_dataset_calc['x']**2, axis=(0,1,2), dtype=np.float32)
            sum_x_val_sq = np.sum(val_dataset_calc['x']**2, axis=(0,1,2), dtype=np.float32)
            sum_y_train_sq = np.sum(train_dataset_calc['y']**2, axis=(0,1,2), dtype=np.float32)
            sum_y_val_sq = np.sum(val_dataset_calc['y']**2, axis=(0,1,2), dtype=np.float32)
            
            mean_x_squared = (sum_x_train_sq + sum_x_val_sq) / (n_x_train + n_x_val)
            mean_y_squared = (sum_y_train_sq + sum_y_val_sq) / (n_y_train + n_y_val)
            
            mean_x = (sum_x_train + sum_x_val) / (n_x_train + n_x_val)
            mean_y = (sum_y_train + sum_y_val) / (n_y_train + n_y_val)
            
            var_x = np.maximum(mean_x_squared - mean_x**2, 0) # to handle numerical stability
            var_y = np.maximum(mean_y_squared - mean_y**2, 0) # to handle numerical stability, later add on fp precision

            pooled_var = ((n_x_train + n_x_val) * var_x + (n_y_train + n_y_val) * var_y) / ((n_x_train + n_x_val) + (n_y_train + n_y_val))
            std = np.sqrt(pooled_var)  
            mean = (sum_x_train + sum_y_train + sum_x_val + sum_y_val) / (n_x_train + n_x_val + n_y_train + n_y_val)

    return mean, std

def numeric_key(path):
    """
    Helper function to get run number from filename.
    Used to determine the checkpoint file to read from w/ largest number of gradient steps.

    Parameters:
        path (str): Path to the file.

    Returns:
        int: The first run of digits in the filename.
    """
    base = os.path.basename(path)
    m = re.search(r'(\d+)', base)
    return int(m.group(1)) if m else -1

def load_model_params(config_dir, parent_dir):
    """
    Load model params from checkpoint files.

    Parameters:
        config_dir (str): Directory containing the config file.
        parent_dir (str): Parent directory containing the config file.

    Returns:
        model_params (dict): Model parameters.
    """
    run_filepath = parent_dir + '/' + config_dir
    
    # Gather both ckpt types: w/ and w/o '_vars'
    ckpt_files = glob.glob(run_filepath + '/ckpt/step_*.pkl')
    ckpt_files_vars = [f for f in ckpt_files if '_vars' in os.path.basename(f)]
    ckpt_files_nonvars = [f for f in ckpt_files if '_vars' not in os.path.basename(f)]
    
    if ckpt_files_vars and ckpt_files_nonvars:
        # Load the latest ckpts from directory. 
        pkl_path_vars = max(ckpt_files_vars, key=numeric_key)
        pkl_path_nonvars = max(ckpt_files_nonvars, key=numeric_key)
        logging.info(f"Loading model params from {pkl_path_vars} and {pkl_path_nonvars}")
        model_params_vars = pickle.load(open(pkl_path_vars, "rb"))
        model_params_nonvars = pickle.load(open(pkl_path_nonvars, "rb"))
        # Combine the dictionaries.
        model_params = {**model_params_nonvars, **model_params_vars}
    else:
        raise FileNotFoundError("No valid checkpoint files found in the directory.")
    
    return model_params

def convert_config_types(config_dict):
    """
    Convert string numeric values in config dictionary to proper types.

    Parameters:
        config_dict (dict): Configuration dictionary.

    Returns:
        dict: Configuration dictionary with proper types.

    TODO: have parser function do this + check types.
    """
    converted = {}
    for key, value in config_dict.items():
        if isinstance(value, str):
            # Special handling for list-like strings (e.g., "[64, 128, 256]")
            if value.startswith('[') and value.endswith(']'):
                try:
                    # Use eval to parse list strings safely
                    parsed_list = eval(value)
                    if isinstance(parsed_list, (list, tuple)):
                        converted[key] = parsed_list
                    else:
                        converted[key] = value
                except:
                    converted[key] = value
            # Comma-separated strings (e.g., "32,64,128" ==> [32, 64, 128])
            elif ',' in value:
                try:
                    # Split by comma and convert each element to int
                    parsed_list = [int(x.strip()) for x in value.split(',')]
                    converted[key] = parsed_list
                except ValueError:
                    # If conversion fails, keep as string
                    converted[key] = value
            else:
                # Try to convert to int first, then float
                try:
                    converted[key] = int(value)
                except ValueError:
                    try:
                        converted[key] = float(value)
                    except ValueError:
                        converted[key] = value  # Keep as string
        elif isinstance(value, dict):
            converted[key] = convert_config_types(value)
        else:
            converted[key] = value
    return converted

def load_config(config_file_path):
    """
    Load and process config from JSON file.

    Parameters:
        config_file_path (str): Path to the configuration file.

    Returns:
        namedtuple: Configuration object.
    """
    config = json.load(open(config_file_path))
    config = convert_config_types(config)
    config['mean'] = 0.0
    config['std'] = 1.0
    config['convert_to_sin_cos'] = False
    
    config_namedtuple = namedtuple('x', config.keys())(*config.values())
    return config_namedtuple

def create_cgl_gifs(val_dataset, run_filepath, num_frames=10000):
    """
    Create GIFs of CGL simulation frames. 

    Parameters:
        val_dataset: Validation dataset containing CGL frames.
        run_filepath (str): Path to save the GIFs.
        num_frames (int): Number of frames to process.
    """
    gif_frames = []
    gif_frames_sine = []
    gif_frames_cosine = []

    # Get the cyclic colormap.
    cmap = plt.get_cmap('hsv')
    frames = np.array(val_dataset['x'][:num_frames])

    for i in range(num_frames):
        frame = frames[i]  # shape: (L, L, 2)
        # Compute phase from cosine and sine components.
        phase = np.arctan2(frame[..., 1], frame[..., 0])
        phase_norm = (phase + np.pi) / (2 * np.pi)

        # Flatten, apply the colormap, and reshape.
        color_phase = cmap(phase_norm.ravel()).reshape(phase_norm.shape + (4,))[:, :, :3]

        # Convert to uint8 for imageio (values in [0, 255])
        phase_uint8 = np.ascontiguousarray((color_phase * 255).astype(np.uint8))
        gif_frames.append(phase_uint8)
        sine_uint8 = np.ascontiguousarray(((frame[..., 1] + 1) / 2 * 255).astype(np.uint8))
        cosine_uint8 = np.ascontiguousarray(((frame[..., 0] + 1) / 2 * 255).astype(np.uint8))
        gif_frames_sine.append(sine_uint8)
        gif_frames_cosine.append(cosine_uint8)

    run_dir = run_filepath + '/'
    gif_path = os.path.join(run_dir, 'first_' + str(num_frames) + '_frames_sine.gif')
    imageio.mimsave(gif_path, gif_frames_sine, duration=10 / num_frames)

    gif_path = os.path.join(run_dir, 'first_' + str(num_frames) + '_frames_cosine.gif')
    imageio.mimsave(gif_path, gif_frames_cosine, duration=10 / num_frames)

    gif_path = os.path.join(run_dir, 'first_' + str(num_frames) + '_frames_phase.gif')
    imageio.mimsave(gif_path, gif_frames, duration=10 / num_frames)
    
    logging.info(f"Created CGL GIFs in {run_dir}") 