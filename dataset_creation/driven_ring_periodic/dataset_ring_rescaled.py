"""
Driven particle in periodic potential. Euler-Maruyama stepping method.  

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

# Highest float precision for numerical accuracy. 
# Note that saved data is in float32, but simulation is in float64. 
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
# Analytical EPR calculation
################################################################################

def compute_epr_analytical(U0, f, D, dt, num_points=1000, save_every=1):

    # Integral bounds.
    xs = np.linspace(0, 2 * np.pi, num_points)
    dx = 2 * np.pi / num_points

    Psi = (U0 * np.cos(xs) - f * xs) / D

    # Compute denom integral w/ trapezoid method. See appendix for derivation. 
    denom_vals = []
    for x in xs:
        zs = np.linspace(x, x + 2 * np.pi, num_points) # x is not just theta
        dz = 2 * np.pi / num_points
        Psi_z = (U0 * np.cos(zs) - f * zs) / D
        denom_vals.append(np.trapz(np.exp(Psi_z), dx=dz) / (2 * np.pi))
    denom_vals = np.array(denom_vals)

    # Normalization constant + stationary current.
    Z = np.trapz(np.exp(-Psi) * denom_vals, dx=dx)
    J = (1 - np.exp(-f * 2 * np.pi / D)) / Z

    # Get analytical EPR. 
    epr_exact = J * f * dt * save_every # Scale so EPR is per frame.

    return epr_exact

################################################################################
# Simulation helper functions
################################################################################

@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def simulate_trajectory_batch(x0, key, U0, f, D, dt, batch_steps):
    """
    Simulate batch of trajectories for a particle in a tilted periodic potential using Euler-maruyama.
    """
    def drift(x):
        return U0 * jnp.sin(x) + f

    def euler_maruyama_step(x, key):
        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, shape=x.shape) * jnp.sqrt(2 * D * dt)

        x_next = (x + drift(x) * dt + noise) 
        return x_next, key

    def step(carry, _):
        x, key = carry
        x, key = euler_maruyama_step(x, key)
        return (x, key), x

    (xf, _), xs = jax.lax.scan(step, (x0, key), None, length=batch_steps)
    return xf, xs

################################################################################
# Main simulation function
################################################################################

def run_simulation(filename, U0, kappa, D, dt, save_every, max_episode_steps, num_episodes, key, batch_size=2048):
    """
    Run simulation for a particle in a tilted periodic potential.

    Parameters:
        filename : str
            Name of saved file. 
        U0 : float
            Amplitude of potential.
        kappa : float
            Drive strength.
        D : float
            Diffusion constant.
        dt : float
            Time step.
        save_every : int
            Save every n steps.
        max_episode_steps : int
            Maximum number of steps per episode.
        num_episodes : int
            Number of episodes.
        key : jax.random.PRNGKey
            Random number generator key.
        batch_size : int
            Number of steps per batch.

    Returns:
        epr_analytical : float
            Analytical EPR.
        filename : str
            Name of saved file.
        key : jax.random.PRNGKey,
            Random number generator key.
    """

    # Get analytical EPR. 
    epr_analytical = compute_epr_analytical(U0, kappa, D, dt, save_every=save_every)
    logging.info(f"Analytical EPR: {epr_analytical}")

    # Burn-in.
    total_steps = max_episode_steps * num_episodes
    burn_in_steps = min(10**6, int(0.4 * total_steps))
    logging.info(f"Global burn-in: {burn_in_steps} steps")

    key, subkey = jax.random.split(key)
    x0 = jax.random.normal(subkey, shape=(1,), dtype=jnp.float32)
    current_x = x0

    key, subkey = jax.random.split(key)
    current_x, _ = simulate_trajectory_batch(current_x, subkey, U0, kappa, D, dt, burn_in_steps)

    num_saved = (total_steps + save_every - 1) // save_every

    with h5py.File(filename, 'w') as f:
        logging.info(f"Writing simulation data to file: {filename}")
        f.create_dataset('EPR', data=np.array(epr_analytical, dtype=np.float32))
        # Initialize saved dataset. 
        # Shape of dataset: (num_saved, 1)
        obs_dataset = f.create_dataset('Observations', shape=(num_saved, 1), dtype=np.float32)

        global_frame_counter = 0
        saved_frame_counter = 0
        episode_batches = max_episode_steps // batch_size
        episode_remainder = max_episode_steps % batch_size

        # Sim episodes. 
        # Episodes is just for progress tracking in simulation, nothing to do with ML training.
        for episode in range(num_episodes):
            key, subkey = jax.random.split(key)

            for b in range(episode_batches):
                key, subkey = jax.random.split(subkey)
                current_x, xs = simulate_trajectory_batch(current_x, key, U0, kappa, D, dt, batch_size)

                # Get save indices.
                offset = -global_frame_counter % save_every
                save_indices = np.arange(offset, batch_size, save_every)

                # Save xs.
                num_to_save = len(save_indices)
                obs_dataset[saved_frame_counter:saved_frame_counter + num_to_save, 0] = xs[save_indices, 0]

                saved_frame_counter += num_to_save
                global_frame_counter += batch_size

            # Remainder steps that don't fit into a batch.
            if episode_remainder > 0:
                key, subkey = jax.random.split(subkey)
                current_x, xs = simulate_trajectory_batch(current_x, key, U0, kappa, D, dt, episode_remainder)
                offset = -global_frame_counter % save_every
                save_indices = np.arange(offset, episode_remainder, save_every)

                num_to_save = len(save_indices)
                obs_dataset[saved_frame_counter:saved_frame_counter + num_to_save, 0] = xs[save_indices, 0]
                saved_frame_counter += num_to_save
                global_frame_counter += episode_remainder

            logging.info(f"Saved frames {saved_frame_counter}/{num_saved}.")

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
    num_episodes = params["num_episodes"] # Number of episodes
    max_episode_steps = params["max_episode_steps"] # Maximum number of steps per episode
    U0 = params["U0"] # Amplitude of potential
    kappa  = params["kappa"] # Drive strength
    D  = params["D"] # Diffusion constant
    dt = params["dt"] # Time step
    save_every = params["save_every"] # Save every n steps
    seed = params["seed"] # Seed for random number generator
    batch_size = params["batch_size"] # Batch size
    key = jax.random.PRNGKey(seed) # Random number generator key

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
        f"ring_simulation_U0={params['U0']}_kappa={params['kappa']}_D={params['D']}_steps={params['max_episode_steps'] * params['num_episodes']}_seed={params['seed']}_dt={params['dt']}_save_every={params['save_every']}.h5"
    )

    # Run simulation.
    epr_analytical, out_filename, key = run_simulation(filename=filename, U0=U0, kappa=kappa, D=D, dt=dt, save_every=save_every, max_episode_steps=max_episode_steps, num_episodes=num_episodes, key=key, batch_size=batch_size)
    logging.info(f"Finished periodic potential simulation for U0={params['U0']}, kappa={params['kappa']}, D={params['D']}, steps={params['max_episode_steps'] * params['num_episodes']}, seed={params['seed']}, dt={params['dt']}, save_every={params['save_every']}. Analytical EPR = {epr_analytical}, frames saved in {out_filename}")
