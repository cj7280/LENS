"""
sweep.py generates hyperparameter combinations and runs the training loop from training.py.

Functionality is the same as wandb sweep and expects the same config file format,
but with a few tweaks. For offline compute nodes. Supports grid search.

Code based off of the wandb sweep functionality for offline compute nodes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import argparse
import itertools
import subprocess
import yaml
import sys
import logging
import os
import json
import fcntl
import time
import random

from utils import create_parser

# Set up logging format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def generate_hyperparameter_combinations(
    parameters: Dict[str, Any], 
    method: str, 
    data_folder: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generate hyperparameter combinations for grid search.
    
    Args:
        parameters: Dictionary of parameter configurations with 'values' or 'min'/'max' keys
        method: Search method ('grid' currently supported)
        data_folder: Optional folder containing data files to iterate over
        
    Returns:
        List of hyperparameter dictionaries for each combination
        
    Raises:
        ValueError: If parameter configuration is invalid
        NotImplementedError: If search method is not implemented
    """
    if method == 'grid':
        # Generate all hyperparameter combinations
        param_grid = {}
        for param, config in parameters.items():
            if 'values' in config:
                param_grid[param] = config['values']
            elif 'min' in config and 'max' in config:
                if config.get('distribution') == 'int_uniform':
                    param_grid[param] = list(range(config['min'], config['max'] + 1))
                else:
                    raise ValueError(f"Invalid distribution type '{param}': {config.get('distribution')}")
            else:
                raise ValueError(f"Invalid param config for: {param}")
        
        keys, values = zip(*param_grid.items())
        base_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # If data_folder is provided, add the hdf5 data files to the hyperparameter combinations
        # rather than specifying all filenames in the folder
        if data_folder:
            data_files = [
                os.path.join(data_folder, f) 
                for f in os.listdir(data_folder) 
                if os.path.isfile(os.path.join(data_folder, f))
            ]
            hyperparameter_combinations = []
            
            for base_combination in base_combinations:
                for data_file in data_files:
                    # Only add if hdf5 file
                    if not data_file.endswith('.h5'):
                        continue
                    
                    combination = base_combination.copy()
                    combination['datasets_to_analyze'] = data_file

                    # Add any 'x=y' argument in the filename to the args object
                    # This is important so the saved args files contain all hyperparameters,
                    # both training and simulation hyperparameters for analysis
                    # TODO: just pull in the config_simulation.yml file instead of reading
                    # simulation hyperparameters from the filename title itself
                    filename = os.path.basename(data_file)
                    filename_args = filename.split('_')
                    
                    for arg in filename_args:
                        if '=' in arg:
                            key, value = arg.split('=')
                            
                            # If ends in .h5 (i.e. last 'x=y' in filename), remove the .h5
                            if value.endswith('.h5'):
                                value = value[:-3]

                            key = "data_" + key
                            combination[key] = value
                    
                    hyperparameter_combinations.append(combination)
            
            return hyperparameter_combinations
        else:
            return base_combinations
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

def construct_command(
    command_template: List[str], 
    hyperparams: Dict[str, Any]
) -> List[str]:
    """Construct command to run training.py with hyperparameter substitutions.
    
    Args:
        command_template: List of command elements with ${param_name} placeholders
        hyperparams: Dictionary of hyperparameters to substitute
        
    Returns:
        List of command elements with substitutions applied
        
    Raises:
        ValueError: If parameter not found in hyperparameters
    """
    command = []
    data_folder_to_analyze = hyperparams.get('data_folder_to_analyze', None)

    i = 0
    while i < len(command_template):
        item = command_template[i]

        # If current item is substitution placeholder ${param_name}
        if isinstance(item, str) and item.startswith("${") and item.endswith("}"):
            param_name = item[2:-1]
            if param_name not in hyperparams:
                raise ValueError(f"Parameter '{param_name}' not found in hyperparameters.")
            command.append(str(hyperparams[param_name]))
            i += 1
            continue

        # Check if next item is substitution placeholder
        if (i + 1) < len(command_template):
            next_item = command_template[i + 1]
            if isinstance(next_item, str) and next_item.startswith("${") and next_item.endswith("}"):
                param_name = next_item[2:-1]
                if param_name not in hyperparams:
                    raise ValueError(f"Parameter '{param_name}' not found in hyperparameters.")
                param_value = hyperparams[param_name]

                # Boolean flags
                if isinstance(param_value, bool):
                    # Include flag only if True
                    if param_value:
                        command.append(item)
                    # Skip flag *and* substitution placeholder
                    i += 2
                    continue

                # Substitution for non-booleans
                if param_name == "datasets_to_analyze" and data_folder_to_analyze is not None:
                    i += 2
                    continue

                command.append(item)
                command.append(str(param_value))
                i += 2
                continue

        # Append the current item as is
        command.append(item)
        i += 1

    # Append datasets_to_analyze if provided via the hyperparameters when data folder is in use
    if data_folder_to_analyze is not None and "datasets_to_analyze" in hyperparams:
        ds_arg = hyperparams.get("datasets_to_analyze")
        if ds_arg:
            command.extend(["--datasets_to_analyze", ds_arg])
    
    return command

def load_config(config_file: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file.
    
    Args:
        config_file: Path to YAML configuration file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error(f"{config_file} not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing {config_file}: {e}")
        raise

def save_config_files(
    config: Dict[str, Any], 
    run_dir: str, 
    data_folder: Optional[str] = None
) -> None:
    """Save configuration files to run directory.
    
    Args:
        config: Training configuration dictionary
        run_dir: Run directory path
        data_folder: Optional data folder path for simulation config
    """
    # Dump YAML training config into run_dir
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config_training.yml'), 'w') as file:
        yaml.dump(config, file)

    # Dump YAML simulation config from data_folder into run_dir if provided
    if data_folder:
        try:
            sim_config = yaml.safe_load(open(os.path.join(data_folder, 'config.yml')))
            yaml.safe_dump(sim_config, open(os.path.join(run_dir, 'config_sim.yml'), 'w'), sort_keys=False)
        except (FileNotFoundError, yaml.YAMLError) as e:
            logging.warning(f"Could not load simulation config from {data_folder}: {e}")

def update_config_dict_with_retry(
    config_file_path: str, 
    config_key: str, 
    combined_config: Dict[str, Any], 
    max_retries: int = 10
) -> bool:
    """Update config dictionary file with retry logic for file locking.
    
    Args:
        config_file_path: Path to config dictionary file
        config_key: Key for the configuration
        combined_config: Configuration dictionary to save
        max_retries: Maximum number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Open file with write permissions, create if doesn't exist
            with open(config_file_path, 'a+') as file:
                fcntl.flock(file.fileno(), fcntl.LOCK_EX)
                
                # Read existing content
                file.seek(0)
                content = file.read()
                
                if content.strip():
                    existing_config = json.loads(content)
                else:
                    existing_config = {}
                
                # Update with new config
                existing_config[config_key] = combined_config
                
                # Write updated content
                file.seek(0)
                file.truncate()
                json.dump(existing_config, file, indent=4)
            
            return True  # Success
            
        except (json.JSONDecodeError, OSError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                logging.error(f"Failed to update config file after {max_retries} attempts: {e}")
                return False
            
            # Random backoff to reduce collision
            time.sleep(random.uniform(0.1, 0.5))
    
    return False

def run_hyperparameter_sweep(
    config: Dict[str, Any],
    hyperparameter_combinations: List[Dict[str, Any]],
    command_template: List[str],
    run_dir_slurm: str,
    job_index: Optional[int] = None
) -> None:
    """Run hyperparameter sweep with specified configurations.
    
    Args:
        config: Configuration dictionary
        hyperparameter_combinations: List of hyperparameter dictionaries
        command_template: Command template for training
        run_dir_slurm: SLURM run directory path
        job_index: Optional job index to run only specific configuration
    """
    logging.info("Starting sweep.")
    
    for idx, hyperparams in enumerate(hyperparameter_combinations, 1):
        logging.info(f"Running configuration {idx}/{len(hyperparameter_combinations)}: {hyperparams}")

        # Construct command
        try:
            command = construct_command(command_template, hyperparams)
        except ValueError as e:
            logging.error(f"Error constructing command: {e}")
            continue

        # Create directory for each config
        run_dir_name = f"config_{job_index}" if job_index is not None else f"config_{idx}"    
        run_dir = os.path.join(run_dir_slurm, run_dir_name)
        os.makedirs(run_dir, exist_ok=True)
        command.extend(["--run_dir", run_dir])

        logging.info(f"Executing command: {' '.join(command)}")

        try:
            result = subprocess.run(command, check=True)
            logging.info(f"Configuration {idx} completed successfully.")

            # Combine args that are not swept over and args that are swept over
            parser_silent = create_parser()
            args_silent = parser_silent.parse_args(command[3:])
            base_args = vars(args_silent)
            combined_config = {**base_args, **hyperparams}

            # File locking to prevent race conditions
            config_file_path = os.path.join(run_dir_slurm, 'config_dict.json')
            config_key = f"config_{job_index if job_index is not None else idx}"
            
            success = update_config_dict_with_retry(config_file_path, config_key, combined_config)
            if not success:
                logging.error(f"Failed to update config file for configuration {idx}")
                    
        except subprocess.CalledProcessError as e:
            logging.error(f"Configuration {idx} failed with error: {e}")
            continue

    logging.info("All experiments completed.")

def main() -> None:
    """Main function to run hyperparameter sweep."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Sweep hyperparameters.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to YAML config file.')
    parser.add_argument('--job_index', type=int, default=None,
                        help='If provided (1-indexed), run only the hyperparameter combination at this index.')
    parser.add_argument('--slurm_id', type=int, default=None, help='SLURM job ID for run directory.')
    args = parser.parse_args()

    logging.info(f"Load config {args.config_file}.")
    
    # Load configuration
    try:
        config = load_config(args.config_file)
    except (FileNotFoundError, yaml.YAMLError):
        sys.exit(1)

    # Get method, parameters, and command template from config
    method = config.get('method', 'grid')
    parameters = config.get('parameters', {})
    command_template = config.get('command', [])
    data_folder = parameters.get('data_folder_to_analyze', {}).get('values', [None])[0]

    # Save configuration files
    run_dir = os.path.join('runs', f"{args.slurm_id}")
    save_config_files(config, run_dir, data_folder)
    
    # Generate hyperparameter combinations
    try:
        hyperparameter_combinations = generate_hyperparameter_combinations(parameters, method, data_folder)
    except (ValueError, NotImplementedError) as e:
        logging.error(f"Error generating hyperparameter combinations: {e}")
        sys.exit(1)

    # Check total number of configs
    total_configs = len(hyperparameter_combinations)
    logging.info(f"Total number of configurations: {total_configs}")

    # Create run directory for SLURM
    run_dir_slurm = os.path.join('runs', f"{args.slurm_id}")
    os.makedirs(run_dir_slurm, exist_ok=True)

    # Save hyperparameter combinations to file
    with open(os.path.join(run_dir_slurm, 'hyperparameters.json'), 'w') as file:
        json.dump(hyperparameter_combinations, file, indent=4)

    # Check if job index is valid
    if args.job_index is not None:
        if args.job_index < 1 or args.job_index > total_configs:
            logging.error(f"Invalid job index {args.job_index}. Must be between 1 and {total_configs}.")
            sys.exit(1)
        hyperparameter_combinations = [hyperparameter_combinations[args.job_index - 1]]
        logging.info(f"Running only configuration {args.job_index} due to job index selection.")

    # Run hyperparameter sweep
    run_hyperparameter_sweep(
        config=config,
        hyperparameter_combinations=hyperparameter_combinations,
        command_template=command_template,
        run_dir_slurm=run_dir_slurm,
        job_index=args.job_index
    )

if __name__ == "__main__":
    main()
