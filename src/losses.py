"""
losses.py contains the contrastive loss functions and metrics for training.
See networks.py for the representation functions and neural network architectures. 
Contrastive loss function used in train_file_og.py for gradient steps.

Note: most configs are not used in the paper, but are included for completeness
if one wants to run the code with different configs. Hyperparameters are specified in paper appendix. 

Code based off of JAXGCRL codebase (Bortkiewicz, 2025).
Copyright Catherine Ji, 2025. 
"""
from typing import Any, NamedTuple

from brax.training.types import Params
import jax
import jax.numpy as jnp

from . import networks as cl_networks
import logging

# Set up logging format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Makes the contrastive loss function that returns loss and metrics. 
def make_losses(
    config: NamedTuple, # Contains hyperparameters.
    contrastive_loss_fn: str, # Form of the loss function for comparisons between BCE and f-div methods. 
    energy_fn: str, # Energy function specification. Dot product is used in the paper.
    true_epr: jnp.float32 # True EPR value (if analytical solution exists, set to 1.0 for CGL w/ no analytical solution/dummy variable for CGL)
):
    # Setting matmul precision to fp32. 
    jax.config.update("jax_default_matmul_precision", config.precision)
    state_encoder = cl_networks.make_cl_networks(config)

    # Define the contrastive loss function
    def contrastive_loss(
        params: Params, 
        variable_state: Any,
        x, # x_t
        y  # x_{t+1}
    ):
        # Casts all inputs to fp32
        # (uint normalization handled separately in networks.py
        # with a uint normalization flag)
        x = x.astype(jnp.float32)
        y = y.astype(jnp.float32)
        
        # Apply learned representation functions to x_t and x_{t+1}
        if (config.repr_fn == 'linear_local' or config.repr_fn == 'linear_local_subspaces'):
            output, _ = state_encoder.apply({'params': params, **variable_state}, x, y, mutable='matrices')  
            phi_x, phi_y, A, B = output    
            psi_x = jnp.dot(phi_x, A.T)
            psi_y = jnp.dot(phi_y, A.T)  
            eta_x = jnp.dot(phi_x, B.T)
            eta_y = jnp.dot(phi_y, B.T)
        elif config.repr_fn == 'nonlinear':
            output, _ = state_encoder.apply({'params': params, **variable_state}, x, y, mutable='matrices')
            phi_x, psi_x, phi_y, psi_y = output
        elif config.repr_fn == 'mixed':
            output, _ = state_encoder.apply({'params': params, **variable_state}, x, y, mutable='matrices')
            logits_1, logits_2 = output

        # Different energy functions
        # Paper results all use dot product energy function
        if energy_fn == 'dot':
            logits_1 = jnp.sum(phi_x * psi_y, axis=1)
            logits_2 = jnp.sum(phi_y * psi_x, axis=1)
            logits_3 = jnp.sum(phi_x * eta_x, axis=1)
            logits_4 = jnp.sum(phi_y * eta_y, axis=1)
        elif energy_fn == 'l2':
            logits_1 = -jnp.linalg.norm(phi_x - psi_y, axis=-1)
            logits_2 = -jnp.linalg.norm(phi_y - psi_x, axis=-1)
            logits_3 = -jnp.linalg.norm(phi_x - eta_x, axis=-1)
            logits_4 = -jnp.linalg.norm(phi_y - eta_y, axis=-1)
        elif energy_fn == 'l1':
            logits_1 = -jnp.sum(jnp.abs(phi_x - psi_y), axis=-1)
            logits_2 = -jnp.sum(jnp.abs(phi_y - psi_x), axis=-1)
            logits_3 = -jnp.sum(jnp.abs(phi_x - eta_x), axis=-1)
            logits_4 = -jnp.sum(jnp.abs(phi_y - eta_y), axis=-1)
        else:
            raise ValueError(f"Energy function not implemented: {energy_fn}")

        # Compute metrics for logging
        @jax.jit
        def compute_metrics(logits_1, logits_2, logits_3, logits_4, phi_x, psi_x, phi_y, psi_y, eta_x, eta_y, A, B):
            phi_x_norm = jnp.linalg.norm(phi_x, axis=1)
            psi_x_norm = jnp.linalg.norm(psi_x, axis=1) 
            eta_x_norm = jnp.linalg.norm(eta_x, axis=1)

            binary_accuracy = jnp.mean((logits_1 + logits_3 > logits_2 + logits_4) == jnp.ones(x.shape[0], dtype=int))
            phi_mean = jnp.mean(phi_x_norm)
            psi_mean = jnp.mean(psi_x_norm)
            phi_std = jnp.std(phi_x_norm)
            psi_std = jnp.std(psi_x_norm)
            eta_mean = jnp.mean(eta_x_norm)
            eta_std = jnp.std(eta_x_norm)
            phi_x_mean = jnp.mean(phi_x)
            phi_y_mean = jnp.mean(phi_y)

            # Check for anti-symmetry and symmetry
            # of A and B respectively
            A_asym_norm = jnp.linalg.norm(A + A.T)
            B_sym_norm = jnp.linalg.norm(B - B.T)

            return binary_accuracy, phi_mean, phi_x_mean, phi_y_mean, psi_mean, phi_std, psi_std, eta_mean, eta_std, A_asym_norm, B_sym_norm

        # Implementation of sigmoid binary cross entropy with logits.
        # Identical to optax.sigmoid_binary_cross_entropy.
        # Included in code for completeness: this implementation is necessary for numerical stability.
        def log_sigmoid(x):
            return -jax.nn.softplus(-x)

        def sigmoid_binary_cross_entropy_with_logits(logits, labels):
            return -(labels * log_sigmoid(logits) + (1 - labels) * log_sigmoid(-logits))
        
        # Implementation of loss functions.
        
        # Forward-backward loss function is the loss function used in the paper.
        @jax.jit
        def forward_backward_loss(logits_1, logits_2, logits_3, logits_4):
            # Including both terms is more numerically stable,
            # but, with fp32 precision, these two empirically lead to the same training dynamics.
            loss_fb_AB = 0.5 * (sigmoid_binary_cross_entropy_with_logits((logits_1 + logits_3) - (logits_2 + logits_4), jnp.ones(x.shape[0], dtype=jnp.float32)) + 
                                        sigmoid_binary_cross_entropy_with_logits((logits_2 + logits_4) - (logits_1 + logits_3), jnp.zeros(x.shape[0], dtype=jnp.float32)))

            loss = jnp.nanmean(loss_fb_AB)
            return loss
        
        # F-divergence loss function.
        # (see Nowozin, 2016)
        # This is the loss function used in the NEEP method (Kim et al. 2020).
        # We note that the *parameterizations of the logits are different* 
        # between this method and i.e. NEEP, which is the basis of our paper's contribution.
        @jax.jit
        def f_div(h_1, h_2):
            j = h_1 - h_2

            # Clamp one side to prevent fp32 overflow
            j = jnp.maximum(j, -10)

            # -1 offset such that the loss corresponds to negative estimated EP in expectation.
            # (see Nowozin, 2016)
            loss = -(j - jnp.exp(-j)) - 1 
            loss_nanmean = jnp.nanmean(loss)
            return loss_nanmean
        
        # Calculating losses with above defined loss functions.
        if contrastive_loss_fn == 'forward_backward':
            loss = forward_backward_loss(logits_1, logits_2, logits_3, logits_4)
            epr_estimate = jnp.nanmean((logits_1 + logits_3) - (logits_2 + logits_4))
        elif contrastive_loss_fn == 'f_div':
            loss = f_div(logits_1 + logits_3, logits_2 + logits_4)
            epr_estimate = jnp.nanmean(logits_1 + logits_3 - (logits_2 + logits_4))
        else:
            raise ValueError(f"Contrastive loss function not implemented: {contrastive_loss_fn}")

        # Compute metrics for logging
        binary_accuracy, phi_mean, phi_x_mean, phi_y_mean, psi_mean, phi_std, psi_std, eta_mean, eta_std, A_asym_norm, B_sym_norm = compute_metrics(logits_1, logits_2, logits_3, logits_4, phi_x, psi_x, phi_y, psi_y, eta_x, eta_y, A, B)

        # Saving metrics to metrics dictionary.
        metrics = {
            "binary_accuracy": binary_accuracy, # Binary classification accuracy of model
            "phi_mean": phi_mean, # Mean norm of phi(x_t) value
            "phi_x_mean": phi_x_mean, # Mean phi(x_t) value
            "phi_y_mean": phi_y_mean, # Mean phi(x_{t+1}) value
            "psi_mean": psi_mean, # Mean psi(x_t) value
            "phi_std": phi_std, # Std of phi(x_t)
            "psi_std": psi_std, # Std of psi(x_t)
            "eta_mean": eta_mean, # Mean eta(x_t) value
            "eta_std": eta_std, # Std of eta(x_t) value
            "loss": loss, 
            "epr_estimate": epr_estimate,
            "true_epr": true_epr, # True EPR value (if analytical solution exists, set to 1.0 for CGL w/ no analytical solution / dummy variable for CGL)
            "epr_error": ((epr_estimate - true_epr) / true_epr), # Relative EPR error
            "rep_dim_check": phi_x.shape[1], # Check on representation dimension M
            "A_asym_norm": A_asym_norm, # Check that A is antisymmetric (should be 0)
            "B_sym_norm": B_sym_norm, # Check that B is symmetric (should be 0)
            "logits_1": jnp.mean(logits_1), # Mean logits_1 value
            "logits_2": jnp.mean(logits_2), # Mean logits_2 value
            "logits_3": jnp.mean(logits_3), # Mean logits_3 value
            "logits_4": jnp.mean(logits_4), # Mean logits_4 value
        }

        return loss, metrics
    
    return contrastive_loss