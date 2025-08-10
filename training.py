"""
training.py parses in configuration file and runs the training loop from train_file_og.py.

Note: most configs are not used in the paper, but are included for completeness.
Full list of possible configurations in sweep.py file. 

Code based off of JAXGCRL codebase (Bortkiewicz, 2025).

Copyright Catherine Ji, 2025. 
"""
import os
import json
import wandb
from collections import namedtuple
import jax 
from brax.io import model  
import jax.numpy as jnp
import re

from dataset_creation.utils import generate_dataset, generate_dataset_render
from utils import create_parser, MetricsRecorder, setup_logging
from src import train_file_og
from post_training_clean.post_training_utils import normalize_data

def main(args): 
    # Set jax matmul precision to args.precision.
    # Set to "highest" (fp32) for all paper results.
    jax.config.update("jax_default_matmul_precision", args.precision)

    # Create directories for configs and checkpoints.
    # Config directory name from sweep.py.
    run_dir = args.run_dir
    dataset_filename = args.datasets_to_analyze

    # Create checkpoint directory.
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Setup logging.
    logger = setup_logging(run_dir)
    logger.info("Logging initialized.")

    # Save experiment args to json file within config folder. 
    args_path = os.path.join(run_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved arguments to {args_path}.")

    # Set on_device to 0 for small datasets that fit in VRAM.
    # Set on_device to 1 for large datasets that don't fit in VRAM.
    on_device = 1 if args.fits_in_vram == True else 0

    # Prep datasets.
    # Two types of loading functions depending on dataset size.
    if args.render_load:
        """
        generate_dataset_render is used to initialize larger datasets/datasets that are
        stored as a contiguous array in time (only x_t) to save disk space.

        generate_dataset is used to initialize smaller datasets/datasets that are
        stored as x_{t} and x_{t+1} pairs. 

        Same data for both/same training and validation sets for both, only difference is disk space + VRAM considerations.
        """
        train_dataset, val_dataset, true_epr = generate_dataset_render(dataset_filename, separation = args.separation, num_samples_percentage = args.frac_of_data, on_device=on_device)
    else:
        train_dataset, val_dataset, true_epr = generate_dataset(dataset_filename, num_samples_percentage = args.frac_of_data, on_device = on_device)

    # Get seed number from dataset filename which is used for parameter initialization
    # and batch RNG in training loop.
    match = re.search(r'seed=(\d+)', dataset_filename)
    if match:
        seed_number = int(match.group(1))
        args.seed = seed_number
    else:
        raise ValueError(f"Could not get seet from filename: {dataset_filename}")

    # Save true EPR/frame in a separate file.
    true_epr_file = os.path.join(run_dir, 'true_epr.json')
    with open(true_epr_file, 'w') as f:
        true_epr_float = float(true_epr.item())

        # Normalize by separation, which is the frames between x_t and x_{t+1} pairs for training. 
        # Set to 1 for all experiments in paper. 
        true_epr = true_epr_float * args.separation
        json.dump({'true_epr': true_epr_float}, f)
    logger.info(f"Saved true EPR to {true_epr_file}")

    # Will be overwritten if normalization is used, and won't affect normalization otherwise
    mean = jnp.zeros(1)  
    std = jnp.ones(1)  

    # Get mean and std of datasets for on-the-fly normalization.
    # Very clunky -- should be done in the dataset creation and loading scripts
    # but left as a TODO for now. 

    #########################################################################################
    # Normalization of inputs.
    # Normalization generally did not lead to significant changes in performance, 
    # so omitted in main paper results. Included for completeness. 
    #########################################################################################
    mean, std = normalize_data(args, train_dataset, val_dataset)

    # Check mean and std values.
    logger.info(f"Mean: {mean}")
    logger.info(f"Std: {std}") 

    #########################################################################################
    # Initializing all hyperparameters and calling training loop from train_file_og.py.
    #########################################################################################

    # Create config namedtuple.
    Config = namedtuple(
        "Config",
        "repr_dim num_layers repr_fn use_ln batch_size h_dim true_epr contrastive_loss_fn lr energy_fn seed precision patience fits_in_vram use_cnn normalize_dataset cnn_layer_sizes net_shape coord_scale adamw separation linear_network frac_of_data use_bias image_normalize phase_normalize use_ortho_P use_sine_activation mean std",
    )

    # Convert layer sizes from e.g. "32,64,128" to e.g. [32, 64, 128] for CNN initialization in networks.py. 
    args.cnn_layer_sizes = [int(size) for size in args.cnn_layer_sizes.split(",")]

    config = Config(
        repr_dim=args.repr_dim, # Representation dimension.   
        num_layers=args.num_layers, # Number of layers in the network.
        repr_fn=args.repr_fn, # Representation function.
        use_ln=args.use_ln, # Layer normalization.
        batch_size=args.batch_size, # Batch size.
        h_dim=args.h_dim, # Hidden dimension width.
        true_epr=true_epr, # True EPR, set to 1.0 for CGL w/ no analytical baseline. 
        contrastive_loss_fn=args.contrastive_loss_fn, # Contrastive loss function.
        lr=args.lr, # Learning rate.
        energy_fn=args.energy_fn, # Energy function.
        seed=args.seed, # Seed for parameter initialization and batch RNG, matches that of dataset creation seed.
        precision=args.precision, # JAX precision. Set to highest for all paper results.
        patience=args.patience, # Patience for early stopping.
        fits_in_vram=args.fits_in_vram, # Whether the dataset fits in VRAM. 
        use_cnn=args.use_cnn, # Whether to use a CNN.
        normalize_dataset=args.normalize_dataset, # What type of normalization on the inputs.
        cnn_layer_sizes=args.cnn_layer_sizes, # CNN feature count for each convolution.
        net_shape=args.net_shape, # Whether to flatten or max_pool CNN outputs. 
        coord_scale=args.coord_scale, # Scaling on added coordinate features for CoordConv (Liu et al. 2018),
        # "An intriguing failing of convolutional neural networks and the CoordConv solution".
        adamw=args.adamw, # Whether to use AdamW instead of Adam optimizer. 
        separation=args.separation, # Frames between x_t and x_{t+1} pairs for training.
        linear_network=args.linear_network, # Whether to use a linear network (with the caveat of the use_sine_activation flag if set to true).
        frac_of_data=args.frac_of_data, # Fraction of data to use for training.
        use_bias=args.use_bias, # Whether to use a bias term in the network.
        image_normalize=args.image_normalize, # Whether to normalize the inputs by 255 to map to [0,1] range.
        phase_normalize=args.phase_normalize, # Whether to mod 2 pi inputs. 
        use_ortho_P=args.use_ortho_P, # Whether to use an orthogonal P matrix.
        use_sine_activation=args.use_sine_activation, # Whether to use a sine activation function after first dense layer for arbitrary architecture. 
        mean=mean, # Mean of the datasets.
        std=std, # Std of the datasets.
    )

    # Initialize metrics recorder.
    metrics_recorder = MetricsRecorder(args.num_timesteps)
    metrics_to_collect = [
        "train/loss", # Loss
        "train/epr_estimate", # EPR estimate on train set. 
        "train/true_epr", # True EPR.
        "train/epr_error", # EPR error on train set. 
        "train/A_asym_norm", # A_asym_norm on train set. 
        "train/B_sym_norm", # B_asym_norm on train set. 
        "eval/binary_accuracy", # Binary accuracy on validation set (how often can classifier tell fwd or bw in time correctly).
        "eval/x_mean", # Mean of x on validation set. 
        "eval/y_mean", # Mean of y on validation set. 
        "eval/phi_x_mean", # Mean of phi_x on validation set. 
        "eval/phi_y_mean", # Mean of phi_y on validation set. 
        "eval/loss", # Loss on validation set. 
        "eval/epr_estimate", # EPR estimate on validation set. 
        "eval/true_epr", # True EPR.
        "eval/epr_error", # EPR error on validation set. 
        "eval/best_loss", # Best loss on validation set. 
        "eval/best_epr_error", # Best EPR error on validation set. 
        "eval/epr_at_min", # EPR at minimum loss on validation set. 
        "eval/rep_dim_check" # Representation dimension double check.
        "eval/A_asym_norm",
        "eval/B_sym_norm",
        "eval/epr_at_min_loss",
        "eval/logits_1", 
        "eval/logits_2",
        "eval/logits_3", 
        "eval/logits_4"
    ]

    # Progress function to record and log metrics for wandb.
    def progress(num_steps, metrics):
        metrics_recorder.record(
            num_steps,
            {key: value for key, value in metrics.items() if key in metrics_to_collect},
        )
        metrics_recorder.log_wandb()

    # Call training loop.
    params, metrics = train_file_og.train(
        dataset=train_dataset,
        dataset_eval=val_dataset,
        progress_fn=progress,
        config=config,
        num_timesteps=args.num_timesteps,
        logger=logger,
        checkpoint_logdir=ckpt_dir
    )
    
    # Save final model parameters.
    model.save_params(os.path.join(ckpt_dir, 'final'), params)
    logger.info("Training done. Saved final model params.")

if __name__ == "__main__":
    # Parse arguments.
    parser = create_parser()
    args = parser.parse_args()

    # Initialize wandb.
    wandb.init(
        project=args.project_name,
        group=args.group_name, 
        name=args.exp_name,
        config=vars(args),
        mode="offline", 
    )

    # Run training loop.
    main(args)
    wandb.finish()
