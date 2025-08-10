"""
train_file_og.py contains the main training loop functions
called within training.py. 

Reads from losses.py for the loss function for gradient steps. 

Note: most configs are not used in the paper, but are included for completeness
if one wants to run the code with different configs. Hyperparameters are specified in paper appendix. 

Code based off of JAXGCRL codebase (Bortkiewicz, 2025).

Copyright Catherine Ji, 2025. 
"""
from typing import Any, Callable, Dict, Optional, Tuple, Union, NamedTuple, Generator

import numpy as np
import jax
import jax.numpy as jnp
import optax
import torch
import os
from torch.utils import data
from absl import logging

from brax.training.types import PRNGKey, Params, Metrics
from brax.io import model
import sys
import flax
import gc


from . import losses as cl_losses
from . import networks as cl_networks

_PM_AXIS_NAME = 'i'  # Axis name for pmap operations

# TrainingState for the learner. 
@flax.struct.dataclass
class TrainingState:
    gradient_steps: jnp.ndarray
    optimizer_state: optax.OptState
    variable_state: Any # TODO: fix
    params: Params

################################################################################
# Data loaders for datasets that are too large to fit in VRAM. 
# Lazily loads data onto GPU from disk. 
# Not used in paper, but included for completeness. 
################################################################################

# Collate function for the NumpyLoader class.
# Only used when datasts are too large to fit in VRAM. 
def numpy_collate(batch):
    # Return shape: (batch_size, x_dims), (batch_size, y_dims)
    collated_x, collated_y = data.default_collate(batch) 

    # Convert from torch tensor to np array.
    collated_x = collated_x.numpy()
    collated_y = collated_y.numpy()

    # Remove extra dimensions.
    collated_x = np.squeeze(collated_x)
    collated_y = np.squeeze(collated_y)

    return (
        collated_x,
        collated_y
    )

# NumpyDataset class for the NumpyLoader.
# Only used when datasts are too large to fit in VRAM. 
class NumpyDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset_x = dataset['x']
        self.dataset_y = dataset['y']
    
    def __len__(self):
        return len(self.dataset_x)
    
    # Load lazily which prevents OOM errors.
    def __getitem__(self, idx):
        x = self.dataset_x[idx]
        y = self.dataset_y[idx]
        return x, y 

# Data loader.
# Only used when datasts are too large to fit in VRAM. 
# Not used in paper, but included for completeness. 
class NumpyLoader(data.DataLoader):
  def __init__(self, 
               dataset, 
               batch_size=1024,
               shuffle=True, 
               num_workers=5, # Number of workers (threads) for data loading.
               pin_memory=True, 
               drop_last=False, 
               seed=None, 
               prefetch_factor=5 # Number of batches to prefetch.
            ):

    generator = torch.Generator()
    generator.manual_seed(seed)

    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        prefetch_factor=prefetch_factor,
    )

################################################################################
# Train functions.
################################################################################

def _init_training_state(
    key: PRNGKey,
    x: jnp.ndarray,
    y: jnp.ndarray,
    cl_network: Any,
    optimizer: optax.GradientTransformation,
) -> TrainingState:
    """
    Initializes the training state.

    Args:
        key: RNG key.
        x: x_t inputs. Expected to be a numpy array of shape (training samples, x_dim).
        y: x_{t+1} inputs. Expected to be a numpy array of shape (training samples, y_dim).
        cl_network: Contrastive learning network. Initialized in networks.py.
        optimizer: Optimizer.

    Returns:
        Training state.
    """

    # Initialize parameters and optimizer.
    # Variables are tracked metrics that are NOT updated in the gradient step. 
    # Variables are mutable. 
    logging.info(f"key: {key}")
    variables = cl_network.init({'params': key}, x=x, y=y)
    params = variables['params']
    variable_state = {k: variables[k] for k in variables if k != 'params'}

    # Track param count. 
    param_count = sum(x.size for x in jax.tree_leaves(params))
    activations, updated_variables = cl_network.apply(
        {'params': params, **variable_state}, x, y, mutable=['matrices']
    )

    # Update the mutable variables. 
    variable_state.update(updated_variables)
    activation_count = sum(x.size for x in jax.tree_leaves(activations))
    optimizer_state = optimizer.init(params)

    logging.info("Number of parameters: %d", param_count)
    logging.info("Number of activations: %d", activation_count)
    # Return initialized training state. 
    return TrainingState(
        optimizer_state=optimizer_state,
        variable_state=variable_state,
        params=params,
        gradient_steps=jnp.zeros(())
    )

def train(
    dataset: Dict[str, np.ndarray], # Training dataset.
    dataset_eval: Dict[str, np.ndarray], # Evaluation dataset.
    num_timesteps: int, # Number of gradient steps.
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None, # Progress function.
    checkpoint_logdir: Optional[str] = None, # Checkpoint directory.
    config: NamedTuple = None, # hyperparameters 
    logger: logging.ABSLLogger = logging.get_absl_logger(), # Logger.
):
    """
    Main train function on a single device. 

    Args:
        dataset: Training dataset. 
        NOTE: Expected to be a dictionary with keys 'x' and 'y', 
        where x and y are numpy arrays of shape (training samples, x_dim) and (training samples, y_dim) respectively
        and denote x_t and x_{t+1} respectively. 

        dataset_eval: Evaluation dataset. 
        NOTE: Expected to be a dictionary with keys 'x' and 'y', 
        where x and y are numpy arrays of shape (evaluation samples, x_dim) and (evaluation samples, y_dim) respectively
        and denote x_t and x_{t+1} respectively. 

        num_timesteps: Maximum number ofgradient steps.

        progress_fn: Progress function that updates the metrics in wandb.

        checkpoint_logdir: Checkpoint directory.

        config: Config namedtuple with all hyperparameters. Initialized in training.py. 

        logger: Logger.

    Returns:
        params: Final model parameters.
        metrics: Final metrics.
    """
    
    jax.config.update("jax_default_matmul_precision", config.precision)

    if config.fits_in_vram == 0:
        dataset_preload = NumpyDataset(dataset)
        dataset_eval_preload = NumpyDataset(dataset_eval)
        loader = NumpyLoader(
                    dataset=dataset_preload,
                    batch_size=config.batch_size,          
                    shuffle=True,
                    num_workers=5,         
                    pin_memory=True,         
                    seed=config.seed,
                    prefetch_factor=10,  # prefetch 2 batches per worker
                )
        eval_loader = NumpyLoader(
                    dataset=dataset_eval_preload,
                    batch_size=config.batch_size,          
                    shuffle=False,
                    num_workers=5,         
                    pin_memory=True,         # speeds up host to GPU transfers if using CUDA
                    seed=config.seed,
                    prefetch_factor=10, 
                )
    else:
        dataset_preload = dataset
        dataset_eval_preload = dataset_eval
        

    # Initialize RNG and track RNG to confirm correct seeding. 
    # The seed of the RNG matches the seed of the dataset creation. 
    seed = config.seed
    rng = jax.random.PRNGKey(seed)
    logger.info(f"rng: {seed}")
    num_training_steps_per_epoch = len(dataset['x']) // config.batch_size
    logger.info("Number of training steps per epoch: %d", num_training_steps_per_epoch)
    num_epochs = num_timesteps // num_training_steps_per_epoch
    logger.info("Number of epochs: %d", num_epochs)

    # Initialize the contrastive learning network and optimizer. 
    encoder_factory = cl_networks.make_cl_networks
    cl_network = encoder_factory(config=config)

    # Initialize hyperparameters. 
    lr = config.lr
    energy_fn = config.energy_fn
    contrastive_loss_fn = config.contrastive_loss_fn
    true_epr = config.true_epr

    # Initialize early stopping parameters. 
    patience = config.patience # For early stopping
    patience_count = 0
    min_delta = 0.0
    logger.info(f"true_epr: {true_epr}")

    # Initialize optimizer. 
    if config.adamw:
        optimizer = optax.adamw(learning_rate=lr, weight_decay=1e-4)
    else:
        optimizer = optax.adam(learning_rate=lr)
    
    # Initialize contrastive loss function.
    contrastive_loss = cl_losses.make_losses(
        config=config,
        contrastive_loss_fn=contrastive_loss_fn,
        energy_fn=energy_fn,
        true_epr=true_epr
    )
    
    # Gradient step function. 
    @jax.jit
    def update(params, x, y, optimizer_state, variable_state):
        def loss_fn(params):
            return contrastive_loss(params, variable_state, x, y)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)

        new_params = optax.apply_updates(params, updates)

        # Forward pass to update variables
        _, new_variables = cl_network.apply({'params': new_params, **variable_state}, x, y, mutable=['matrices'])
        new_variable_state = {'matrices': new_variables['matrices']}

        return new_params, new_optimizer_state, new_variable_state, metrics

    # Prep batches for training loop.
    # Need to this for fast training -- RNG can take up a lot of time. 
    # Returns a 2D array of shape (num_batches, batch_size).
    def prepare_batches(dataset_size: int, batch_size: int, key) -> jnp.ndarray:
        shuffled_indices = jax.random.permutation(key, dataset_size)
        num_batches = dataset_size // batch_size
        shuffled_indices = shuffled_indices[:num_batches * batch_size]
        return shuffled_indices.reshape((num_batches, batch_size))
    
    # Stochastic gradient descent step. 
    # Performs a single SGD step. 
    @jax.jit
    def sgd_step(training_state: TrainingState, batch_x: jnp.ndarray, batch_y: jnp.ndarray) -> Tuple[TrainingState, Metrics]:
        new_params, new_optimizer_state, new_variable_state, metrics = update(
            training_state.params,
            batch_x,
            batch_y,
            training_state.optimizer_state,
            training_state.variable_state
        )
        
        new_training_state = TrainingState(
            optimizer_state=new_optimizer_state,
            variable_state=new_variable_state,
            params=new_params,
            gradient_steps=training_state.gradient_steps + 1
        )

        return new_training_state, metrics

    # Training for one epoch.
    # Processes all batches.
    # Only use this function for small datasets that can fit in VRAM.
    @jax.jit
    def training_epoch_jit(training_state: TrainingState, batch_indices: jnp.ndarray, dataset_x: jnp.ndarray, dataset_y: jnp.ndarray) -> Tuple[TrainingState, Metrics]:
        def step_fn(carry, batch_idx):
            training_state, metrics_sum = carry
            batch_x = dataset_x[batch_idx]
            batch_y = dataset_y[batch_idx]
            new_state, metrics = sgd_step(training_state, batch_x, batch_y)
            metrics_sum = jax.tree_util.tree_map(lambda a, b: a + b, metrics_sum, metrics)
            return (new_state, metrics_sum), metrics

        # Initialize metrics_sum with zeros based on example_metrics structure.
        example_metrics = sgd_step(training_state, dataset_x[batch_indices[0]], dataset_y[batch_indices[0]])[1]
        metrics_sum = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), example_metrics)

        # Update training state and metrics over all batches.
        (final_state, total_metrics), _ = jax.lax.scan(step_fn, (training_state, metrics_sum), batch_indices)

        # Compute mean metrics over the epoch for logging. 
        num_steps = batch_indices.shape[0]
        mean_metrics = jax.tree_util.tree_map(lambda x: x / num_steps, total_metrics)

        return final_state, mean_metrics

    # Evaluation function. 
    # Only use this function for small datasets that can fit in VRAM.
    @jax.jit
    def run_evaluation(training_state: TrainingState, batch_indices: jnp.ndarray, dataset_x: jnp.ndarray, dataset_y: jnp.ndarray) -> Metrics:
        def eval_step(carry, batch_idx):
            batch_x = dataset_x[batch_idx]
            batch_y = dataset_y[batch_idx]
            _, metrics = contrastive_loss(training_state.params, training_state.variable_state, batch_x, batch_y)  # true_epr is embedded
            metrics_sum = jax.tree_util.tree_map(lambda a, b: a + b, carry, metrics)
            return metrics_sum, metrics

        # Initialize metrics_sum with zeros based on the first batch's metrics
        first_batch_idx = batch_indices[0]
        _, example_metrics = contrastive_loss(training_state.params, training_state.variable_state, dataset_x[first_batch_idx], dataset_y[first_batch_idx])
        metrics_sum = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), example_metrics)

        # Accumulate metrics over all batches
        metrics_sum, _ = jax.lax.scan(eval_step, metrics_sum, batch_indices)

        # Compute average metrics
        num_eval_batches = batch_indices.shape[0]
        eval_metrics = jax.tree_util.tree_map(lambda x: x / num_eval_batches, metrics_sum)
        
        return eval_metrics

    # Training for one epoch.
    # Processes all batches.
    # Only use this function for large datasets that CAN'T fit in VRAM.
    # Uses DataLoader class. 
    def training_epoch_jit_large(training_state: TrainingState, loader: NumpyLoader) -> Tuple[TrainingState, Metrics]:
        # Initialize metrics_sum with zeros based on example_metrics structure
        first_batch = next(iter(loader))
        batch_x_example, batch_y_example = first_batch
        final_state, example_metrics = sgd_step(training_state, batch_x_example, batch_y_example)
        metrics_sum = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), example_metrics)

        # Manually loop through/iterate over batches w/ lazy loading. 
        for batch in loader:
            batch_x, batch_y = batch
            final_state, metrics = sgd_step(final_state, batch_x, batch_y)
            metrics_sum = jax.tree_util.tree_map(lambda a, b: a + b, metrics_sum, metrics)

        # Compute mean metrics over the epoch for logging. 
        num_steps = len(loader)
        mean_metrics = jax.tree_util.tree_map(lambda x: x / num_steps, metrics_sum)

        return final_state, mean_metrics

    # Evaluation function. 
    # Only use this function for large datasets that CAN'T fit in VRAM.
    def run_evaluation_large(training_state: TrainingState, loader: NumpyLoader) -> Metrics:
        # Initialize metrics_sum with zeros based on example_metrics structure
        first_batch = next(iter(loader))
        batch_x_example, batch_y_example = first_batch
        _, example_metrics = contrastive_loss(training_state.params, training_state.variable_state, batch_x_example, batch_y_example)
        metrics_sum = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), example_metrics)

        for batch in loader:
            batch_x, batch_y = batch
            _, metrics = contrastive_loss(training_state.params, training_state.variable_state, batch_x, batch_y)
            metrics_sum = jax.tree_util.tree_map(lambda a, b: a + b, metrics_sum, metrics)

        num_steps = len(loader)
        mean_metrics = jax.tree_util.tree_map(lambda x: x / num_steps, metrics_sum)

        return mean_metrics
    
    # Training loop. 
    # fits_in_vram == 0 for large datasets that DON'T fit in VRAM.
    if config.fits_in_vram == False: 
        # Initialize training state with the first batch.
        init_key, train_key = jax.random.split(rng)
        logger.info('Num batches: %d', len(dataset_preload))
        first_batch = next(iter(loader))
        init_batch_x, init_batch_y = first_batch
        logger.info("init_batch_x shape: %s", init_batch_x.shape)
        logger.info("init_batch_y shape: %s", init_batch_y.shape)
        training_state = _init_training_state(init_key, x=init_batch_x, y=init_batch_y, cl_network=cl_network, optimizer=optimizer)

        # Initialize metrics.
        train_metrics = None
        final_metrics = {'train': {}, 'eval': {}}
        current_step = 0
        # Get RNG keys for batching of every epoch. 
        epoch_keys = jax.random.split(train_key, num_epochs)

        # Track early stopping.
        nth_last_eval_loss = [np.inf] * 5 
        min_loss = np.inf
        min_epr_error = np.inf
        epr_at_min = 1e-8 # Set minimum EPR to numbersmaller than float precision. Should always be positive in expectation in limit of infinite samples.
        old_step = 0

        # Loop over epochs.
        for epoch in range(num_epochs):
            logger.info("Epoch %d, Step %d", epoch, current_step)

            # Get key for epoch that controls batching RNG. 
            epoch_key = epoch_keys[epoch]

            # Run epoch.
            training_state, train_metrics = training_epoch_jit_large(
                training_state, 
                loader
            )
            
            # Save checkpoints.
            if checkpoint_logdir is not None:
                # Save new checkpoint.
                params = training_state.params
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                model.save_params(path, params)

                path_vars = f"{checkpoint_logdir}/step_{current_step}_vars.pkl"
                model.save_params(path_vars, training_state.variable_state)

                # Delete old checkpoint.
                if old_step > 0:
                    old_path = f"{checkpoint_logdir}/step_{old_step}.pkl"
                    os.remove(old_path)
                    old_path_vars = f"{checkpoint_logdir}/step_{old_step}_vars.pkl"
                    os.remove(old_path_vars)

            # Run evaluation.
            eval_metrics = run_evaluation_large(
                training_state,
                eval_loader
            )

            # Track min eval loss and EPR + EPR error at min eval loss.
            if eval_metrics['loss'] < min_loss:
                min_loss = eval_metrics['loss']
                min_epr_error = eval_metrics['epr_error']
                epr_at_min = eval_metrics['epr_estimate']
                logger.info(f"New best loss: {min_loss}, epr error: {min_epr_error}")

            eval_metrics['best_loss'] = min_loss
            eval_metrics['best_epr_error'] = min_epr_error
            eval_metrics['epr_at_min'] = epr_at_min

            # Update current step and log metrics.
            old_step = current_step
            current_step += num_training_steps_per_epoch
            logger.info("Training metrics at step %d: %s", current_step, train_metrics)
            logger.info("Evaluation metrics at step %d: %s", current_step, eval_metrics)
            logger.info("Best eval loss: %s, Best epr error: %s", min_loss, min_epr_error)

            # Update final metrics for logging.
            final_metrics = {}
            final_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
            final_metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})

            # Report progress.
            progress_fn(current_step, final_metrics)

            # Early stopping, only start counting after 5 epochs.
            if nth_last_eval_loss[0] - eval_metrics['loss'] < min_delta and epoch > 5:
                patience_count += 1
                if patience_count >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
            else:
                patience_count = 0

            # Update nth_last_eval_loss used for early stopping.
            nth_last_eval_loss.pop(0)
            nth_last_eval_loss.append(eval_metrics['loss'])
    else: # fits_in_vram == True for small datasets that fit in VRAM.
        # Initialize training state with the first batch.
        init_key, train_key = jax.random.split(rng)
        logger.info('Num batches: %d', len(dataset_preload['x']))
        initial_batch_indices = prepare_batches(len(dataset_preload['x']), config.batch_size, init_key)
        first_batch_idx = initial_batch_indices[0]
        init_batch_x = dataset_preload['x'][first_batch_idx]
        init_batch_y = dataset_preload['y'][first_batch_idx]

        print("init_batch_x shape: ", init_batch_x.shape)
        print("init_batch_y shape: ", init_batch_y.shape)
        logger.info("init_key: %s", init_key)
        logger.info("train_key: %s", train_key)
        training_state = _init_training_state(init_key, x=init_batch_x, y=init_batch_y, cl_network=cl_network, optimizer=optimizer)

        # Initialize metrics.
        train_metrics = None
        final_metrics = {'train': {}, 'eval': {}}
        current_step = 0
        epoch_keys = jax.random.split(train_key, num_epochs)
        logger.info("first epoch key: %s", epoch_keys[0])

        # Early stopping parameters.
        nth_last_eval_loss = [np.inf] * 5 # sure?
        min_loss = np.inf
        min_epr_error = np.inf
        epr_at_min = 1e-8
        old_step = 0

        # Loop over epochs.
        for epoch in range(num_epochs):
            logger.info("Epoch %d, Step %d", epoch, current_step)

            # Get key for epoch that controls batching RNG. 
            epoch_key = epoch_keys[epoch]
            batch_indices = prepare_batches(len(dataset_preload['x']), config.batch_size, epoch_key)

            # Check first batched elements of dataset_preload
            logger.info("first batched element of dataset_preload['x']: %s", dataset_preload['x'][batch_indices[0]])
            logger.info("first batched element of dataset_preload['y']: %s", dataset_preload['y'][batch_indices[0]])
            
            # Run epoch 
            training_state, train_metrics = training_epoch_jit(
                training_state, 
                batch_indices, 
                dataset_preload['x'],
                dataset_preload['y']
            )

            # Save checkpoints. 
            if checkpoint_logdir is not None:
                # Save new checkpoint
                params = training_state.params
                path = f"{checkpoint_logdir}/step_{current_step}.pkl"
                model.save_params(path, params)

                path_vars = f"{checkpoint_logdir}/step_{current_step}_vars.pkl"
                model.save_params(path_vars, training_state.variable_state)

                # Delete old checkpoint (for large models, this saves a lot of space).
                if old_step > 0:
                    old_path = f"{checkpoint_logdir}/step_{old_step}.pkl"
                    os.remove(old_path)
                    old_path_vars = f"{checkpoint_logdir}/step_{old_step}_vars.pkl"
                    os.remove(old_path_vars)

            # Run evaluation.
            eval_batch_indices = prepare_batches(len(dataset_eval_preload['x']), config.batch_size, epoch_key)
            eval_metrics = run_evaluation(
                training_state,
                eval_batch_indices,
                dataset_eval_preload['x'],
                dataset_eval_preload['y']
            )

            # Track min eval loss and EPR + EPR error at min eval loss.
            if eval_metrics['loss'] < min_loss:
                min_loss = eval_metrics['loss']
                min_epr_error = eval_metrics['epr_error']
                epr_at_min = eval_metrics['epr_estimate']
                logger.info(f"New best loss: {min_loss}, epr error: {min_epr_error}")

            eval_metrics['best_loss'] = min_loss
            eval_metrics['best_epr_error'] = min_epr_error
            eval_metrics['epr_at_min'] = epr_at_min

            # Update current step and log metrics
            old_step = current_step
            current_step += num_training_steps_per_epoch
            logger.info("Training metrics at step %d: %s", current_step, train_metrics)
            logger.info("Evaluation metrics at step %d: %s", current_step, eval_metrics)
            logger.info("Best eval loss: %s, Best epr error: %s", min_loss, min_epr_error)

            # Update final metrics for logging.
            final_metrics = {}
            final_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
            final_metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})

            # Report progress.
            progress_fn(current_step, final_metrics)

            # Early stopping, only start counting after 5 epochs
            if nth_last_eval_loss[0] - eval_metrics['loss'] < min_delta and epoch > 5:
                patience_count += 1
                if patience_count >= patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break
            else:
                patience_count = 0

            # Update nth_last_eval_loss used for early stopping.
            nth_last_eval_loss.pop(0)
            nth_last_eval_loss.append(eval_metrics['loss'])
        
    logger.info("Total training steps: %d", current_step)

    # Return final parameters and metrics.
    return training_state.params, final_metrics
