"""
CGL simulation. Uses ETD2RK stepping method for stiff PDEs from Cox & Matthew, 2002.

Copyright Catherine Ji, 2025. 
"""

import jax
import jax.numpy as jnp
import numpy as np
import h5py
from functools import partial
import logging
import argparse
import yaml
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
jax.config.update("jax_default_matmul_precision", "highest")

################################################################################
# Parameter handling
################################################################################

def get_parameters_by_index(config, index):
    """Get combination of params from config based on index from slurm job."""
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
# Simulation functions
################################################################################

def etdrk2rk_step(A_hat, E, M1, M2, beta, key):
    """One ETDRK2 step for stiff PDE following Cox & Matthew, 2002."""
    A = jnp.fft.ifft2(A_hat)
    NL1_hat = jnp.fft.fft2(-(1 + 1j*beta) * (jnp.abs(A)**2) * A)
    A_hat_mid = E * A_hat + M1 * NL1_hat
    A_mid = jnp.fft.ifft2(A_hat_mid)
    NL2_hat = jnp.fft.fft2(-(1 + 1j*beta) * (jnp.abs(A_mid)**2) * A_mid)
    A_hat_new = A_hat_mid + M2 * (NL2_hat - NL1_hat) # Cox & Matthew, 2002.

    return A_hat_new, key

@partial(jax.jit, static_argnums=(1, 5, 7))
def simulate_cgl_etd2rk_batch(A_hat, batch_steps, E, M1, M2, beta, key, save_every=1):
    """
    Simulate a batch of ETD2RK steps and save frames every save_every steps.
    ETD2RK is 2nd order RK method for stiff PDEs that splits the linear and nonlinear contributions (Cox & Matthew, 2002).
    
    Parameters:
        A_hat : array_like
            Initial Fourier-space field.
        batch_steps : int
            Number of steps. 
        E, M1, M2 : array_like
            Precomputed factors for the ETD2RK scheme. See Cox & Matthew, 2002.
        beta : float
            Nonlinear dispersion param.
        key : jax.random.PRNGKey
            JAX random key.
        save_every : int, optional
            Frequency of saving frames.
            
    Returns:
        A_hat_final : array_like
            Fourier-space field after stepping.
        saved_frames : array_like
            Saved frames from the simulation.
        key : jax.random.PRNGKey
            Updated random key.
    """
    num_saved_frames = batch_steps // save_every

    def step(carry, idx):
        A_hat, saved_frames, key = carry
        A_hat, key = etdrk2rk_step(A_hat, E, M1, M2, beta, key)
        
        # Save the frame every `save_every` steps.
        def save_frame(saved_frames):
            return saved_frames.at[idx // save_every].set(A_hat)
        
        saved_frames = jax.lax.cond(idx % save_every == 0,
                                    save_frame,
                                    lambda x: x,
                                    saved_frames)
        return (A_hat, saved_frames, key), None

    saved_frames_init = jnp.zeros((num_saved_frames,) + A_hat.shape, dtype=A_hat.dtype)
    (A_hat_final, saved_frames, key), _ = jax.lax.scan(step, (A_hat, saved_frames_init, key), jnp.arange(batch_steps))

    return A_hat_final, saved_frames, key

@partial(jax.jit, static_argnums=(1, 5))
def simulate_cgl_etd2rk_batch_no_frames(A_hat, batch_steps, E, M1, M2, beta, key):
    """
    Simulate a batch of ETD2RK steps and return only the final state.
    """
    def step(carry, _):
        A_hat, key = carry
        A_hat, key = etdrk2rk_step(A_hat, E, M1, M2, beta, key)
        return (A_hat, key), None

    (A_hat_final, key), _ = jax.lax.scan(step, (A_hat, key), None, length=batch_steps)
    return A_hat_final, key

################################################################################
# Main simulation function
################################################################################

def run_simulation(filename, batch_size, L, dt, total_steps, alpha, beta, seed, save_every, eps, save_amp):
    """
    Run CGL simulation using ETD2RK.

    Parameters:
        filename : str
            Name of saved file. 
        batch_size : int
            Number of steps per batch.
        L : int
            Grid size.
        dt : float
            Time step.
        total_steps : int
            Total number of steps.
        alpha : float
            Linear dispersion param.
        beta : float
            Nonlinear dispersion param.
        seed : int
            Seed for random number generator.
        save_every : int
            Save every save_every steps.
        eps : float
            Scale Gaussian noise to initialize field. Set to 1e-2. 
        save_amp : bool
            Whether to save amplitude in addition to phase information.

    Returns:
        filename : str
            Name of saved file.
        key : jax.random.PRNGKey
            Random number generator key.
    """
    # Initialize random key.
    key = jax.random.PRNGKey(seed)
    
    # Initialize complex field in position space.
    A0_real = jax.random.normal(key, (L, L)) * eps
    key, subkey = jax.random.split(key)
    A0_imag = jax.random.normal(subkey, (L, L)) * eps
    A = A0_real + 1j * A0_imag
    A_hat = jnp.fft.fft2(A)

    # Take FFT and pre-compute terms for ETD2RK. 
    kx = jnp.fft.fftfreq(L, d=1.0) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(L, d=1.0) * 2 * jnp.pi
    kx_grid, ky_grid = jnp.meshgrid(kx, ky)
    laplacian = -(kx_grid**2 + ky_grid**2) 
    L_lin = 1 + (1 + 1j * alpha) * laplacian # linear part of the Laplacian operator in fourier space. 
    E = jnp.exp(dt * L_lin)
    M1 = (E - 1) / L_lin # same as M1 in Cox & Matthew, 2002
    M2 = (E - 1 - L_lin * dt) / (L_lin**2 * dt) # same as M2 in Cox & Matthew, 2002 with lumped-in 1/dt factor.

    # Burn-in steps. Don't save frames.
    burn_in_steps = 2*10**6
    logging.info(f"Global burn-in: {burn_in_steps} steps")
    A_hat, key = simulate_cgl_etd2rk_batch_no_frames(A_hat, burn_in_steps, E, M1, M2, beta, key)

    num_saved_frames = total_steps // save_every
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with h5py.File(filename, 'w') as f:
        logging.info(f"Writing CGL frames to file: {filename}")
        if save_amp:
            shape_to_save = (num_saved_frames, L, L, 3) # extra for the saved amplitude
        else:
            shape_to_save = (num_saved_frames, L, L, 2)
        
        obs_dataset = f.create_dataset(
            'Observations', 
            shape=shape_to_save,
            dtype=np.float32
        )
        epr_dataset = f.create_dataset(
            'EPR', 
            shape=(1,), 
            dtype=np.float32
        )
        epr_dataset[0] = 1.0

        steps_done = 0
        frames_saved = 0
        while steps_done < total_steps:
            current_batch = min(batch_size, total_steps - steps_done)
            A_hat, batch_frames, key = simulate_cgl_etd2rk_batch(
                A_hat, current_batch, E, M1, M2, beta, key, save_every
            )
            batch_phases = jnp.fft.ifft2(batch_frames) 
            if save_amp:
                batch_save= jnp.stack([
                    jnp.cos(jnp.angle(batch_phases)),
                    jnp.sin(jnp.angle(batch_phases)), 
                    jnp.abs(batch_phases)
                ], axis=-1)
            else:
                batch_save = jnp.stack([
                    jnp.cos(jnp.angle(batch_phases)), 
                    jnp.sin(jnp.angle(batch_phases))
                ], axis=-1)

            # Convert to numpy.
            batch_save = np.array(batch_save)
            batch_len = batch_save.shape[0]
            obs_dataset[frames_saved:frames_saved + batch_len] = batch_save

            frames_saved += batch_len
            steps_done += current_batch
            logging.info(f"Saved {frames_saved} / {num_saved_frames} frames.")

        f.flush()

    return filename, key

################################################################################
# Main execution
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for CGL simulation."
    )
    parser.add_argument("--config_file", type=str, required=True, help="YAML config file (required).")
    parser.add_argument("--job_index", type=int, required=True, help="Job index for batch processing.")
    parser.add_argument("--slurm_id", type=int, required=True, help="SLURM job ID for batch processing.")
    args = parser.parse_args()

    # Load YAML config file. 
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Get params based on job index.
    params = get_parameters_by_index(config, args.job_index)

    # Extract parameters.
    L = params["grid_size"] # Grid size.
    dt = params["dt"] # Time step.
    total_steps = params["num_steps"] # Total number of steps.
    alpha = params["alpha"] # Linear dispersion param.
    beta = params["beta"] # Nonlinear dispersion param.
    seed = int(params["seed"]) # Seed for random number generator.
    save_every = int(params["save_every"]) # Save every save_every steps.
    eps = params["eps"] # Scale Gaussian noise to initialize field. Set to 1e-2. 
    save_amp = bool(params["save_amp"]) # Whether to save amplitude in addition to phase information.
    batch_size = params["batch_size"] # Number of steps per batch.
    
    # Create save directory.
    directory = config["directory"]
    directory = os.path.join(directory, f"{args.slurm_id}")
    os.makedirs(directory, exist_ok=True)
    
    with open(args.config_file, "r") as src:
        with open(directory + '/' + "config.yml", "w") as dst:
            dst.write(src.read())

    # Create filename.
    filename = os.path.join(
        directory,
        f"cgl_simulation_grid={params['grid_size']}_dt={params['dt']}_steps={params['num_steps']}_alpha={params['alpha']}_beta={params['beta']}_seed={params['seed']}_save_amp={params['save_amp']}.h5"
    )

    # Run simulation.
    run_simulation(filename, batch_size, L, dt, total_steps, alpha, beta, seed, save_every, eps, save_amp)
    logging.info(f"Finished CGL simulation. Frames saved in {filename}")
