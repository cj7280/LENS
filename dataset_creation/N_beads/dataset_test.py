import jax
import jax.numpy as jnp
import numpy as np
import h5py
from scipy.linalg import solve_continuous_lyapunov
from functools import partial
import logging
import argparse
import yaml
import os
from scipy.optimize import brentq

# Highest float precision for numerical accuracy. 
jax.config.update("jax_enable_x64", True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
jax.config.update("jax_default_matmul_precision", "highest")

################################################################################
# Parameter handling
################################################################################

def get_parameters_by_index(config, index):
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
# Matrix initialization and analytical EPR calculation.
################################################################################

def initialize_parameters(N, k, gamma, T, delta, mode="linear"):
    """Initialize matrices for beads system."""
    A_np = np.zeros((N, N))
    for i in range(N):
        A_np[i, i] = -2 * k / gamma 
        if i < N - 1:
            A_np[i, i + 1] = k / gamma
        if i > 0:
            A_np[i, i - 1] = k / gamma
    A = jnp.array(A_np, dtype=jnp.float64)

    if mode == "fourier":
        Tc = T - delta
        k = jnp.linspace(0, N - 1, N)
        Ts = T + (Tc - T) * jnp.cos(jnp.pi * k / ((N - 1) * 2)) # from cold to hot
    if mode == "alternate":
        Ts = jnp.array([T - delta, T] * (N // 2))
    if mode == "sine":
        Tc = T - delta
        k = jnp.linspace(0, N - 1, N)
        Ts = Tc + delta * jnp.sin(jnp.pi * k / (N - 1))
    elif mode == "linear":
        Ts = jnp.linspace(T - delta, T, N)

    # Diffusion matrix.
    F = jnp.sqrt(jnp.diag(2 * Ts[::-1]) / gamma) 
    D = (F @ F.T) / 2
    return A, F, D

def epr_from_mats(A, D, C, dt, dim=1):
    return np.trace(A @ np.linalg.solve(D, A @ C) - np.linalg.solve(C, D)) * dt * dim #NOTE the decoupled d.o.f.! 

def get_epr_analytical(A, D, dt, dim=1):
    C = solve_continuous_lyapunov(A, -2 * D)
    return epr_from_mats(A, D, C, dt, dim)

def get_covariance(A, D):
    C = solve_continuous_lyapunov(A, -2 * D)
    return C

################################################################################
# Simulation helper functions
################################################################################

@partial(jax.jit, static_argnums=(3,5,6))
def simulate_trajectory_batch(x0, key, A, dt, F, batch_steps, dim=1):
    """
    Simulate a batch of Euler-Maruyama steps.
    """
    def step(carry, _):
        x, key = carry
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, shape=(F.shape[0], dim))
        x = x + dt * A @ x + jnp.sqrt(dt) * F @ eps
        return (x, key), x

    (xf, _), xs = jax.lax.scan(step, (x0, key), None, length=batch_steps)
    return xf, xs

################################################################################
# Main simulation function
###############################################

def run_simulation(filename, num_episodes, max_episode_steps, A, D, C, dt, N, key, batch_size=256, dim=1):
    """
    Run simulation.

    Parameters:
        filename : str
            Name of saved file. 
        num_episodes : int
            Number of episodes.
        max_episode_steps : int
            Maximum number of steps per episode.
        A : jnp.ndarray
            Drift matrix.
        D : jnp.ndarray
            Diffusion matrix.
        C : jnp.ndarray
            Covariance matrix.
        dt : float
            Time step.
        N : int
            Number of beads.
        key : jnp.ndarray
            Random number generator key.
        dim : int
            Dimensionality of the system.
        batch_size : int
            Number of steps per batch.

    Returns:
        epr_analytical : float
            Analytical EPR.
        filename : str
            Name of saved file.
        key : jax.random.PRNGKey
            Random number generator key.
    """

    # Get analytical EPR.
    epr_analytical = get_epr_analytical(A, D, dt, dim=dim)
    logging.info(f"EPR: {epr_analytical}")

    # Global burn-in (once at the start).
    burn_in_steps = min(10**6, int(0.4 * max_episode_steps * num_episodes))
    logging.info(f"Global burn-in: {burn_in_steps} steps")

    # Initialize x0 before burn-in.
    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, shape=(N, dim), dtype = jnp.float64)
    current_x = x0

    # Run burn-in. 
    key, subkey = jax.random.split(key)
    current_x, _ = simulate_trajectory_batch(current_x, subkey, A, dt, F, burn_in_steps, dim=dim)

    # Save data to file.
    with h5py.File(filename, 'w') as f:
        logging.info(f"Writing to file: {filename}")

        # Save EPR. 
        f.create_dataset('EPR', data=np.array(epr_analytical, dtype=np.float64))

        # Single dataset for all episodes.
        total_observations = num_episodes * max_episode_steps
        obs_dataset = f.create_dataset(
            'Observations',
            shape=(total_observations, N, dim), 
            dtype=np.float64
        )

        # Save A, D, and C matrices
        f.create_dataset('A', data=np.array(A, dtype=np.float64)) # drift
        f.create_dataset('D', data=np.array(D, dtype=np.float64)) # diffusion
        f.create_dataset('C', data=np.array(C, dtype=np.float64)) # covariance

        frames_saved = 0
        episode_batches = max_episode_steps // batch_size
        episode_remainder = max_episode_steps % batch_size

        # Loop over episodes.
        for episode in range(num_episodes):
            logging.info(f"Starting episode {episode+1}/{num_episodes}")

            # For each episode, run exactly max_episode_steps steps in batches.
            key, subkey = jax.random.split(key)

            # Batches for this episode.
            for _ in range(episode_batches):
                current_key, subkey = jax.random.split(subkey)
                current_x, xs = simulate_trajectory_batch(current_x, current_key, A, dt, F, batch_size, dim=dim)

                # Store batch. 
                start_idx = frames_saved
                end_idx = frames_saved + batch_size
                obs_dataset[start_idx:end_idx, :, :] = xs  # (batch_size, N, dim)
                frames_saved = end_idx

            # Remainder steps.
            if episode_remainder > 0:
                current_key, subkey = jax.random.split(subkey)
                current_x, xs = simulate_trajectory_batch(current_x, current_key, A, dt, F, episode_remainder, dim=dim)

                start_idx = frames_saved
                end_idx = frames_saved + episode_remainder

                obs_dataset[start_idx:end_idx, :, :] = xs  # (episode_remainder, N)
                frames_saved = end_idx

            logging.info(f"Saved frames {frames_saved}/{total_observations}.")

        f.flush()

    return epr_analytical, filename, key

################################################################################
# Main execution
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None, help="YAML config")
    parser.add_argument("--job_index", type=int, default=None)
    parser.add_argument("--slurm_id", type=int, default=None)

    args = parser.parse_args()

    # Load YAML config file. 
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # Get params based on job index.
    params = get_parameters_by_index(config, args.job_index)

    # Initialize parameters.
    k = params["k"] # Spring constant
    gamma = params["gamma"] # Damping constant
    T = params["T"] # Temperature
    dt = params["dt"] # Time step
    delta = params["delta"] # Temperature difference
    mode = params["mode"] # Mode of temperature profile
    N = params["N"] # Number of beads
    key = jax.random.PRNGKey(params["seed"]) # Random number generator key
    num_episodes = params["num_episodes"] # Number of episodes
    max_episode_steps = params["max_episode_steps"] # Maximum number of steps per episode
    batch_size = params["batch_size"] # Batch size

    # Create save directory.
    directory = config["directory"]
    directory = os.path.join(directory, f"{args.slurm_id}")
    os.makedirs(directory, exist_ok=True)

    # Save config file to directory.
    with open(args.config_file, "r") as src:
        with open(directory + '/' + "config.yml", "w") as dst:
            dst.write(src.read())

    # Create filename.
    filename = os.path.join(
        directory,
        f"beads_N={params['N']}_delta={params['delta']}_steps={params['max_episode_steps'] * params['num_episodes']}_seed={params['seed']}_dt={params['dt']}_mode={params['mode']}_dataset.h5"
    )

    A, F, D = initialize_parameters(N, k, gamma, T, delta, mode=mode)

    # Get covariance.
    C = get_covariance(A, D)

    # Log the A, F, D matrices.
    A_np = np.array(A)
    F_np = np.array(F)
    D_np = np.array(D)
    np.set_printoptions(precision=2, suppress=True)
    logging.info(f"\nA: \n{A_np}")
    logging.info(f"\nF: \n{F_np}")
    logging.info(f"\nD: \n{D_np}")

    # Run simulation.
    logging.info(f'Running simulation: N: {params["N"]}, dt: {params["dt"]}, delta: {params["delta"]}, T: {params["T"]}, mode: {params["mode"]}, max_episode_steps: {params["max_episode_steps"] * params["num_episodes"]}, seed: {params["seed"]}')
    run_simulation(filename=filename, num_episodes=num_episodes, max_episode_steps=max_episode_steps, A=A, D=D, C=C, dt=dt, N=N, key=key, batch_size=batch_size)
    logging.info(f'Finished simulation: N: {params["N"]}, dt: {params["dt"]}, delta: {params["delta"]}, T: {params["T"]}, mode: {params["mode"]}, max_episode_steps: {params["max_episode_steps"] * params["num_episodes"]}, seed: {params["seed"]}')
