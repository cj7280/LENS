import jax
import jax.numpy as jnp
import numpy as np
import h5py

def generate_dataset(hdf5_filename, on_device=1, num_samples_percentage=1.0):
    """
    Generates datasets for training and validation. 
    For datasets saved as separate arrays of x_t and x_{t+1} pairs, the dataset is split into training and validation sets.
    Used for the N_beads_rendered dataset. 

    Parameters:
        hdf5_filename : str
            Name of saved file.
        on_device : bool
            Whether to store data on device.
        num_samples_percentage : float
            Percentage of total data to use for training+validation (useful for very large datasets). 
            Not used in paper. 

    Returns:
        train_dataset : dict
            Training dataset.
        val_dataset : dict
            Validation dataset.
        epr : float
            EPR of the dataset.
    """
    
    with h5py.File(hdf5_filename, 'r') as f:
        total_samples_test = f['data_test'].shape[0]
        total_samples_train = f['data_train'].shape[0]

        test = f['data_test'][:int(num_samples_percentage * total_samples_test)]
        train = f['data_train'][-int(num_samples_percentage * total_samples_train):]
        epr = f['EPR'][0]

        if on_device:
            train_dataset = dict(x=jnp.asarray(train[:, 0]), y=jnp.asarray(train[:, 1]))
            val_dataset = dict(x=jnp.asarray(test[:, 0]), y=jnp.asarray(test[:, 1]))
        else:
            train_dataset = dict(x=np.array(train[:, 0]), y=np.array(train[:, 1]))
            val_dataset = dict(x=np.array(test[:, 0]), y=np.array(test[:, 1]))

    return train_dataset, val_dataset, epr

def generate_dataset_render(hdf5_filename, on_device=1, num_samples_percentage=1.0, split_percentage=0.8, separation = 1, shift_displacements=False):
    """
    Generates datasets for training and validation. 
    For datasets saved as a contiguous array in time (only x_t), the dataset is split into training and validation sets.
    If shift_displacements is True, the dataset is shifted by a maximum displacement of the observations.
    Used for the N_beads, CGL, and driven_ring_periodic datasets.

    Parameters:
        hdf5_filename : str
            Name of saved file.
        on_device : bool
            Whether to store data on device.
        num_samples_percentage : float
            Percentage of contiguous total data to use for training+validation (useful for very large datasets). 
            Not used in paper. 
        split_percentage : float
            Percentage of data to use for training. Default is 80%. 
        separation : int
            Separation between pairs in the non-overlapping case.
        shift_displacements : bool
            Whether to shift the dataset by a maximum displacement of the observations.
            Only use for the N_beads dataset.

    Returns:
        train_dataset : dict
            Training dataset.
        val_dataset : dict
            Validation dataset.
        epr : float
            EPR of the dataset.
    """

    # Load entire datasets from file
    with h5py.File(hdf5_filename, 'r') as f:
        if "Observations" not in f:
            raise KeyError(f"'{hdf5_filename}' missing 'Observations'")
        # load only num_samples_percentage of the data
        observations = f['Observations'][:int(num_samples_percentage * f['Observations'].shape[0])]
        epr = f['EPR'][()]

    M = observations.shape[0]
    total_pairs = M - separation

    split_index = int(split_percentage * total_pairs)

    if on_device:
        if shift_displacements:
            max_displacement = jnp.max(np.abs(observations)) * 2
            num_beads = observations.shape[1]   
            max_displacement_array = jnp.arange(0, num_beads * max_displacement, max_displacement)
            max_displacement_array = max_displacement_array[:, None]  # add extra dimension to match observation dimensionality.
            train_dataset = dict(x=jnp.asarray(observations[:split_index-separation] + max_displacement_array), y=jnp.asarray(observations[separation:split_index] + max_displacement_array))
            val_dataset = dict(x=jnp.asarray(observations[split_index:-separation] + max_displacement_array), y=jnp.asarray(observations[split_index+separation:] + max_displacement_array))
        else:
            train_dataset = dict(x=jnp.asarray(observations[:split_index-separation]), y=jnp.asarray(observations[separation:split_index]))
            val_dataset = dict(x=jnp.asarray(observations[split_index:-separation]), y=jnp.asarray(observations[split_index+separation:]))
    else:
        if shift_displacements:
            max_displacement = jnp.max(np.abs(observations)) * 2 # to account that one bead is shifted to the left and one to the right
            num_beads = observations.shape[1]   
            max_displacement_array = jnp.arange(0, num_beads * max_displacement, max_displacement)
            max_displacement_array = max_displacement_array[:, None]  # add extra dimension to match observation dimensionality.
            train_dataset = dict(x=np.array(observations[:split_index-separation] + max_displacement_array), y=np.array(observations[separation:split_index] + max_displacement_array))
            val_dataset = dict(x=np.array(observations[split_index:-separation] + max_displacement_array), y=np.array(observations[split_index+separation:] + max_displacement_array))
        else:
            train_dataset = dict(x=np.array(observations[:split_index-separation]), y=np.array(observations[separation:split_index]))
            val_dataset = dict(x=np.array(observations[split_index:-separation]), y=np.array(observations[split_index+separation:]))
    
    return train_dataset, val_dataset, epr
