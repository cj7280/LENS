"""
networks.py contains the representation function and neural network architectures
used in the losses.py file for training. 

Code based off of JAXGCRL codebase (Bortkiewicz, 2025).

NOTE on RNG: The RNG behavior of this code is slightly different from the RNG that generated 
the paper results. In flax, without any additional bells and whistles,
the name of the parameters, modules, and submodules
affects the RNG (https://github.com/google/flax/issues/4367).

This discovery was made while cleaning the code. 

Previously, we used the name "resnet" for the ResNet *and* linear networks, 
rather than the correct/general "net" naming used here. 

We couldn't find an easy way to fix this without using the confusing name convention, 
so have left the code as-is. If one wishes to reproduce the paper results with the identical RNG, change 
every instance of "net" to "resnet" in this code. 

Copyright Catherine Ji, 2025. 
"""
from operator import is_
from typing import Sequence, Callable, Any

from attr import mutable

import jax
from flax import linen
import jax.numpy as jnp
import logging
from scipy.stats import ortho_group

# Set up logging format.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

class NeuralNet(linen.Module):
    layer_sizes: Sequence[int] # Width of the hidden layers for ResNet network. 
    hidden_size: int # Width of the hidden layers for linear network. 
    use_ln: int # 0/false: none, 1/true: LayerNorm
    repr_dim: int # Dimension of the representation.
    linear_network: bool # Whether or not to use a linear network, with possible exception of the sine activation for periodic potential case.
    use_bias: bool # Whether or not to use bias. Set to True for all cases but can be toggled. 
    use_sine_activation: bool # Whether or not to add on sine activation in the periodic potential case.

    """
    NeuralNet class for non-CNN experiments. 
    Functions of this class are NNs that take in x and output phi(x). 

    Args:
        layer_sizes: Width of the hidden layers for ResNet network.  
        hidden_size: Width of the hidden layers for linear network.
        use_ln: Normalization type. 0: none, 1: LayerNorm.
        repr_dim: Dimension of the representation (M).
        linear_network: Whether or not to use a linear network, with possible exception of the sine activation for periodic potential case.
        use_bias: Whether or not to use bias. Set to True for all cases but can be toggled. 
        use_sine_activation: Whether or not to add on sine activation in the periodic potential case.
    """

    @linen.compact
    def __call__(self, x):
        # Flatten input
        x = x.reshape((x.shape[0], -1))

        if self.linear_network: 
            x = linen.Dense(self.hidden_size, use_bias=self.use_bias)(x)
            # Network is no longer *fully*linear with the use_sine_activation=True.
            if self.use_sine_activation: 
                x = jnp.sin(x)
            x = linen.Dense(self.hidden_size, use_bias=self.use_bias)(x)
            x = linen.Dense(self.repr_dim, use_bias=self.use_bias)(x)
            return x
        else: 
            x = linen.Dense(self.layer_sizes[0], use_bias=self.use_bias)(x)
            # Sine activation after first FC layer. 
            if self.use_sine_activation:
                x = jnp.sin(x)
            x = linen.LayerNorm()(x) if (self.use_ln > 0) else x
            x = linen.swish(x) 

            for size in self.layer_sizes[1:]:
                delta = linen.Dense(size, use_bias=self.use_bias)(x)
                delta = linen.LayerNorm()(delta) if (self.use_ln > 0) else delta
                delta = linen.swish(delta)
                delta = linen.Dense(size, use_bias=self.use_bias)(delta)
                delta = linen.LayerNorm()(delta) if (self.use_ln > 0) else delta
                x = linen.swish(delta + x)
            
            x = linen.Dense(self.repr_dim, use_bias=self.use_bias)(x)
            # Outputs: (batch_size, repr_dim) 
            return x
        
# Standard convolutional neural net to process images, 
# then ResNet with layer norms on the CNN output. 
class NeuralNetCNNConvs(linen.Module):
    layer_sizes: Sequence[int]
    use_ln: int # 0: none, 1: LayerNorm, 2: GroupNorm, 3: BatchNorm. In CGL experiments, we do not use any feature norm. 
    repr_dim: int
    net_shape: str # "flatten" or "max_pool" after CNN layers
    coord_scale: float
    linear_network: bool

    """
    CNN network class for CGL experiments. 
    Functions of this class are CNNs that take in x and output phi(x). 

    Args:
        layer_sizes: Width of the hidden layers for ResNet network. 
        use_ln: Normalization type. 0: none, 1: LayerNorm, 2: GroupNorm, 3: BatchNorm. In CGL experiments, we do not use any feature norm. 
        repr_dim: Dimension of the representation (M).
        net_shape: "flatten" or "max_pool" after CNN layers.
        coord_scale: Scale of the coordinate convolution.
        linear_network: Whether or not to use activations after convolutions.
    """

    @linen.compact
    def __call__(self, x):
        def norm_fn(x):
            if self.use_ln == 0:
                return x
            elif self.use_ln == 1:
                return linen.LayerNorm()(x) # ignores spatial structure, so is bad in CNN layers
            elif self.use_ln == 2:
                return linen.GroupNorm()(x)
            elif self.use_ln == 3: # just layer norm on the features
                return x
            else:
                raise ValueError("Invalid use_ln value; choose 0, 1, 2, or 3.")

        # Set initializers for weights and biases. 
        weight_init = linen.initializers.variance_scaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
        bias_init = linen.initializers.constant(0.01)

        # Coordinate convolution (CoordConv) from Liu et al. 2018, 
        # "An intriguing failing of convolutional neural networks and the CoordConv solution".
        # to explicitly add in spatial information to pixel inputs.
        x_dims, y_dims = x.shape[1:3]
        i, j = jnp.meshgrid(jnp.arange(x_dims), jnp.arange(y_dims), indexing='ij')
        i = jnp.broadcast_to(i[None, :, :, None], (x.shape[0], x_dims, y_dims, 1))
        j = jnp.broadcast_to(j[None, :, :, None], (x.shape[0], x_dims, y_dims, 1))

        if x_dims > 1:
            i = ((i / (x_dims - 1)) * 2 - 1) / self.coord_scale
        if y_dims > 1:
            j = ((j / (y_dims - 1)) * 2 - 1) / self.coord_scale

        combined = jnp.concatenate([x, i, j], axis=-1)

        # CNN layers. Padding is set to "SAME" so output and input have same dims. 
        x = linen.Conv(features=self.layer_sizes[0],
                       kernel_size=(3, 3),
                       strides=4,
                       kernel_init = weight_init,
                       bias_init = bias_init,
                       padding='SAME')(combined)
        x = linen.GroupNorm(num_groups = 8)(x) if self.use_ln > 0 else x
        x = linen.swish(x) if not self.linear_network else x

        for features in self.layer_sizes[1:]:
            x = linen.Conv(features=features,
                           kernel_size=(3, 3),
                           strides=4,
                            kernel_init = weight_init,
                            bias_init = bias_init,
                            padding='SAME')(x)
            x = norm_fn(x) # Normalize output features. 
            x = linen.swish(x) if not self.linear_network else x

            # Pool after every CNN layer if specified. Not used in paper. 
            # max_pool is a nonlinear operation.  
            if self.net_shape == "max_pool":
                x = linen.max_pool(x,
                                   window_shape=(2, 2),
                                   strides=(2, 2),
                                   padding='SAME')

        # Flatten or max pool output. 
        if self.net_shape == "flatten":
            x = x.reshape((x.shape[0], -1))
        elif self.net_shape == "max_pool": 
            x = jnp.max(x, axis=(1, 2))
        
        # Final FC w/ LayerNorm and swish activations
        # then projection into representation space. 
        x = linen.Dense(features=self.layer_sizes[-1], kernel_init = weight_init, bias_init = bias_init)(x)
        x = linen.LayerNorm()(x) 
        x = linen.swish(x) if not self.linear_network else x
        x = linen.Dense(features=self.repr_dim, kernel_init = weight_init, bias_init = bias_init)(x)
        return x

# Class for functions that return desired phi, psi representations and A, B matrices.
class ReprFnLinearLocalSubspaces(linen.Module):
    net: linen.Module
    repr_dim: int
    use_ortho_P: bool
    phase_normalize: bool
    normalize_data: str
    mean: jnp.ndarray
    std: jnp.ndarray
    image_normalize: bool

    """
    ReprFnLinearLocalSubspaces class for all experiments. 
    Functions of this class take in x and y ==> output phi(x), phi(y), A, B. 

    Args:
        net: Neural network.
        repr_dim: Dimension of the representation (M).
        use_ortho_P: Whether or not to use an orthogonal P matrix.
        phase_normalize: Whether or not to wrap phases to be within [0, 2pi).
        normalize_data: Whether or not to normalize data with pre-computed mean and std (see training.py).
        mean: Pre-computed mean of the data.
        std: Pre-computed standard deviation of the data.
        image_normalize: Whether or not to normalize uint8 image data so intensity falls within [0, 1].

    Returns:
        phi_x: Representation of x_t.
        phi_y: Representation of x_{t+1}.
        A: Skew-symmetric matrix.
        B: Symmetric matrix.
    """

    @linen.compact
    def __call__(self, x, y):

        ########### Normalizations #########################################################
        # Stack x and y along the *batch dimension*. 
        # Useful for any modifications on y that is dependent on corresponding x.
        # (Not used in paper, but included for completeness.)
        inputs_to_shift = jnp.concatenate([x, y], axis=-1)

        # Normalize phases to be within [0, 2pi).
        if self.phase_normalize:
            inputs_to_shift = inputs_to_shift % (2 * jnp.pi)

        # Recase to orginal shape
        x_input, y_input = jnp.split(inputs_to_shift, 2, axis=-1)

        # Normalize image data so intensity falls within [0, 1].
        if self.image_normalize == True:
            # Need to ensure that x_input is a float (not uint8) before division.
            # Currently, x_inputs are cast to float32s in the contrastive loss function
            # so this step is not necessary, but kept for reproducibility of paper results
            # (i.e. make sure that float precisions are accurate).
            x_input = x_input.astype(jnp.float16) / 255.0 
            y_input = y_input.astype(jnp.float16) / 255.0

        # On-the-fly normalization to save disk space and GPU memory.
        # Note that, in the case of uint8 data, we do **not** want to normalize the entire dataset
        # prior to training / store normalized datasets, as normalization necessarily 
        # converts uint8 to much more costly floats (uint8 ==> float32 is 4x more expensive).

        # Normalizes data with pre-computed mean and std.
        if self.normalize_data != "none":
            x_input = (x_input - self.mean) / (self.std + 1e-8) # add small epsilon to avoid division by zero
            y_input = (y_input - self.mean) / (self.std + 1e-8) # since float16 ==> float32 here, epsilon is fp32 precision
        #########################################################################################

        # Concatenate inputs along *batch dimension* for parallel processing.
        inputs = jnp.concatenate([x_input, y_input], axis=0)

        # Config for param initialization.
        num_subspaces = self.repr_dim // 2
        input_dim = x_input.shape[1]

        # Get phi representations.
        # Psi reps are linear functions of phi reps.
        if self.use_ortho_P: 
            # Learn a bias vector b
            b = self.param(
                    'b_param',
                    linen.initializers.normal(stddev=0.01),
                    (self.repr_dim,)
                )

            # Initialize parameter controlling
            # for of the P matrix to be Ginibri-distributed matrix
            # (i.e. a matrix with i.i.d. entries).
            raw_param = self.param(
                'P_param',
                linen.initializers.normal(stddev=0.01),
                (input_dim, self.repr_dim)
            )

            # Return a (semi)-orthogonal matrix P that is *intialized*
            # as a uniformly-at-random orthogonal matrix, where P is always orthogonal by construction.
            # Algorithm from Mezzadri (2006).
            # Note that uniformly-random orthogonal matrix init. is ****NOT**** guaranteed if one
            # uses a QR decomposition of Ginibri matrix (matrix with i.i.d. entries).

            # QR decomposition
            Q, R = jnp.linalg.qr(raw_param)

            # Fix sign so R has a positive diagonal
            diag_sign = jnp.sign(jnp.diag(R))
            Q = Q * diag_sign  # Broadcasts over rows.
            P = Q

            # Update saved values that are *not learnable*
            P_matrix = self.variable('matrices', 'P', jnp.zeros, (input_dim, self.repr_dim))
            is_initialized = self.has_variable('matrices','P')
            if is_initialized:
                P_matrix.value = P

            # Apply P to x and y to get phi representations.
            x_input_squeeze = jnp.squeeze(x_input, axis=-1)
            y_input_squeeze = jnp.squeeze(y_input, axis=-1)
            phi_x = jnp.dot(x_input_squeeze, P) + b
            phi_y = jnp.dot(y_input_squeeze, P) + b
        else:
            # Get phi representation from NeuralNet. 
            phi = self.net(inputs)
            phi_x, phi_y = jnp.split(phi, 2, axis=0)

        # Initializing A matrix, where A = (L - L.T)/2.
        # Both learn a skew-symmetric matrix A in block diagonal form 
        # (aka, each block takes the form [0, -a; a, 0]).
        if self.use_ortho_P: 
            raw_evals_L = self.param(
                'L_evals_param',
                linen.initializers.normal(stddev=0.01),
                (num_subspaces,)
            )
            L = jnp.zeros((self.repr_dim, self.repr_dim), dtype=jnp.float32)

            # Note that matrix is already in block-diagonal form with [0, -a; a, 0].
            for i in range(num_subspaces):
                block = jnp.zeros((2, 2), dtype=jnp.float32)
                block = block.at[0, 1].set(-raw_evals_L[i])
                block = block.at[1, 0].set(raw_evals_L[i])
                L = L.at[2*i:2*i+2, 2*i:2*i+2].set(block)
        else: 
            L = jnp.zeros((self.repr_dim, self.repr_dim), dtype=jnp.float32)
            for i in range(num_subspaces):
                block = self.param(f'L_block_{i}', linen.initializers.orthogonal(), (2, 2))
                L = L.at[2*i:2*i+2, 2*i:2*i+2].set(block)

            # Setting A to be skew-symmetric.
            L = (L - L.T)/2

        # Initialize B matrix. 
        B = self.param('B', linen.initializers.normal(stddev=0.01), (self.repr_dim, self.repr_dim))
        B = (B + B.T)/2 # B is symmetric

        # Update saved values that are *not learnable* for calling in torch
        A_asym = self.variable('matrices', 'A_asym', jnp.zeros, (self.repr_dim, self.repr_dim))
        B_sym = self.variable('matrices', 'B_sym', jnp.zeros, (self.repr_dim, self.repr_dim))

        is_initialized = self.has_variable('matrices','A_asym')

        # Updates the *non-learnable* values of A and B.
        if is_initialized:
            A_asym.value = (L - L.T)/2
            B_sym.value = (B + B.T)/2

        # Returns representations of x_t and x_{t+1}.
        # Returns A and B matrices.
        # Outputs are passed thru the contrastive loss function in losses.py. 
        return phi_x, phi_y, L, B


# Returns encoder function that maps from x, y ==> phi(x), phi(y), A, B
# Outputs are passed thru the contrastive loss function in losses.py. 
def make_cl_networks(
    config,
):
    # Initialize any additional flags used to create NNs. 
    # Use CNN if specified, with CNN feature counts for each convolutional layer.
    if config.use_cnn:
        hidden_layer_sizes = config.cnn_layer_sizes
    else: 
        hidden_layer_sizes = [config.h_dim] * config.num_layers

    # Initializing neural nets. 

    # CNNs for the CGL case.
    if config.use_cnn: 
        net = NeuralNetCNNConvs(
            layer_sizes=hidden_layer_sizes,
            use_ln=config.use_ln,
            repr_dim=config.repr_dim,
            net_shape=config.net_shape,
            coord_scale = config.coord_scale,
            linear_network = config.linear_network
        )
    else: # NeuralNets with fully-connected layers for all other experiments.
        net = NeuralNet(
            layer_sizes=hidden_layer_sizes,
            use_ln=config.use_ln,
            repr_dim=config.repr_dim,
            linear_network = config.linear_network,
            hidden_size = config.h_dim,
            use_bias = config.use_bias,
            use_sine_activation = config.use_sine_activation
        )

    # Initializing encoder.
    encoder = ReprFnLinearLocalSubspaces(
        net=net,
        repr_dim=config.repr_dim,
        phase_normalize = config.phase_normalize,
        use_ortho_P = config.use_ortho_P,
        normalize_data = config.normalize_dataset,
        mean = config.mean, # For on-the-fly normalization.
        std = config.std, # For on-the-fly normalization.
        image_normalize = config.image_normalize # For on-the-fly normalization (divide by 255.0) of image data so pixel values fall within [0, 1].
    )

    return encoder