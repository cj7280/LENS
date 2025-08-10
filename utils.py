"""
utils.py contains utility functions for the training script.

Code based off of JAXGCRL codebase (Bortkiewicz, 2025).

Copyright Catherine Ji, 2025. 
"""
import argparse
from collections import namedtuple
from datetime import datetime

from matplotlib import pyplot as plt
import wandb

import logging
import sys
import os

def create_parser():
    """
    Creates the parser for the training script of all hyperparameters.

    Returns:
        parser: The parser for the training script.
    """
    parser = argparse.ArgumentParser(description="Training script arguments")
    parser.add_argument("--exp_name", type=str, default="test", help="Name of the wandb experiment")
    parser.add_argument("--group_name", type=str, default="test", help="Name of the wandb group of experiment")
    parser.add_argument("--project_name", type=str, default="crl", help="Name of the wandb project of experiment")
    parser.add_argument("--num_timesteps", type=int, default=1000000, help="Number of training timesteps")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to wandb")
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--contrastive_loss_fn', type=str, default='forward_backward')
    parser.add_argument('--energy_fn', type=str, default='dot')
    parser.add_argument('--use_ln', type=int, default=0, help="Which normalization")
    parser.add_argument("--h_dim", type=int, default=256, help="Width of hidden layers")
    parser.add_argument('--repr_dim', type=int, default=64, help="Dimension of the representation")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the encoder")
    parser.add_argument('--run_dir', type=str, default="runs", help="Directory to save the run")  
    parser.add_argument('--datasets_to_analyze', type=str, default="beads_simulation.h5", help="Datasets to analyze")
    parser.add_argument('--data_folder_to_analyze', type=str, default="beads_simulation", help="Folder to analyze")
    parser.add_argument('--precision', type=str, default="float32", help="Precision for matmul operations")
    parser.add_argument('--repr_fn', type=str, default="nonlinear", help="Representation function to use")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    parser.add_argument('--fits_in_vram', default=False, action='store_true', help="Whether the dataset fits in VRAM")
    parser.add_argument('--use_cnn', default=False, action='store_true', help="Whether to use a CNN for the encoder")
    parser.add_argument('--normalize_dataset', type=str, default="none", help="Normalization on dataset")
    parser.add_argument('--cnn_layer_sizes', type=str, default="32,64", help="CNN layer sizes")
    parser.add_argument('--net_shape', type=str, default="max_pool", help="Whether to flatten or max pool the CNN output")
    parser.add_argument('--coord_scale', type=float, default=1.0, help="Scale of the coordinate")
    parser.add_argument('--adamw', default=False, action='store_true', help="Whether to use AdamW")
    parser.add_argument('--separation', type=int, default=1, help="Separation between pairs")
    parser.add_argument('--linear_network', default=False, action='store_true', help="Whether to use a linear network")
    parser.add_argument('--frac_of_data', type=float, default=1.0, help="Frac of data to train over")
    parser.add_argument('--use_bias', default=True, action='store_true', help="Whether to use bias")
    parser.add_argument('--render_load', default=False, action='store_true', help="Whether to load render")
    parser.add_argument('--image_normalize', default=False, action='store_true', help="Whether to normalize image by 255.0")
    parser.add_argument('--phase_normalize', default=False, action='store_true', help="Whether to do phase normalize")
    parser.add_argument('--use_ortho_P', default=False, action='store_true', help="Whether to use ortho P")
    parser.add_argument('--use_sine_activation', default=False, action='store_true', help="Whether to use sine activation")
    return parser

# Metrics recorder for wandb.
class MetricsRecorder:
    """
    Records the metrics for the training script.
    """
    def __init__(self, num_timesteps):
        self.x_data = []
        self.y_data = {}
        self.y_data_err = {}
        self.times = [datetime.now()]

        self.max_x, self.min_x = num_timesteps * 1.1, 0

    def record(self, num_steps, metrics):
        self.times.append(datetime.now())
        self.x_data.append(num_steps)

        for key, value in metrics.items():
            if key not in self.y_data:
                self.y_data[key] = []
                self.y_data_err[key] = []

            self.y_data[key].append(value)
            self.y_data_err[key].append(metrics.get(f"{key}_std", 0))

    def log_wandb(self):
        data_to_log = {}
        for key, value in self.y_data.items():
            data_to_log[key] = value[-1]
        data_to_log["step"] = self.x_data[-1]
        wandb.log(data_to_log, step=self.x_data[-1])

    def plot_progress(self):
        num_plots = len(self.y_data)
        num_rows = (num_plots + 1) // 2  # Calculate number of rows needed for 2 columns

        fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

        for idx, (key, y_values) in enumerate(self.y_data.items()):
            row = idx // 2
            col = idx % 2

            axs[row, col].set_xlim(self.min_x, self.max_x)
            axs[row, col].set_xlabel("# environment steps")
            axs[row, col].set_ylabel(key)
            axs[row, col].errorbar(self.x_data, y_values, yerr=self.y_data_err[key])
            axs[row, col].set_title(f"{key}: {y_values[-1]:.3f}")

        # Hide any empty subplots
        for idx in range(num_plots, num_rows * 2):
            row = idx // 2
            col = idx % 2
            axs[row, col].axis("off")
        plt.tight_layout()
        plt.show()

    def print_progress(self):
        for idx, (key, y_values) in enumerate(self.y_data.items()):
            print(f"step: {self.x_data[-1]}, {key}: {y_values[-1]:.3f} +/- {self.y_data_err[key][-1]:.3f}")

    def print_times(self):
        print(f"time to jit: {self.times[1] - self.times[0]}")
        print(f"time to train: {self.times[-1] - self.times[1]}")

# To log to both stdout and a log.txt file in config folders.
def setup_logging(run_dir):
    """
    Sets up logging to both stdout and a log.txt file in the run directory.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_file_path = os.path.join(run_dir, 'logs.txt')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger