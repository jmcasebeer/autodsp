from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

"""
This file contains our custom complex valued GRU and associated complex activations
and initialization schemes. The specifics of the setup are based on this paper
https://openreview.net/attachment?id=H1T2hmZAb&name=pdf.
"""

# split complex activations


def CSigmoid(x):
    return jax.nn.sigmoid(x.real + x.imag)


def Ctanh(x):
    return jnp.tanh(x.real) + 1j * jnp.tanh(x.imag)


def Crelu(x):
    return jax.nn.relu(x.real) + 1.j * jax.nn.relu(x.imag)

def ComplexVarianceScaling(shape, dtype):
    """ Function to initialize compluex valued networks using the scheme described in 
    https://openreview.net/attachment?id=H1T2hmZAb&name=pdf 
    """
    real = hk.initializers.VarianceScaling()(
        shape, dtype=jnp.float32)
    imag = hk.initializers.VarianceScaling()(
        shape, dtype=jnp.float32)

    mag = jnp.sqrt(real**2 + imag**2)
    angle = hk.initializers.RandomUniform(
        minval=-jnp.pi, maxval=jnp.pi)(shape, dtype=jnp.float32)

    return mag * jnp.exp(1j * angle)


def complex_zero_init(shape, dtype):
    return jnp.zeros(shape, dtype=jnp.complex64)

# Complex GRU modified from base kaiku implementation


def add_batch(nest, batch_size: Optional[int]):
    """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
    def broadcast(x): return jnp.broadcast_to(x, (batch_size,) + x.shape)
    return jax.tree_map(broadcast, nest)


def deep_initial_state(batch_size, h_size, stack_size):
    """ Function to make a stack of inital state for a multi-layer GRU.
    """
    return tuple(static_initial_state(batch_size, h_size) for layer in range(stack_size))


def static_initial_state(batch_size, h_size):
    """ Function to make an initial state for a single GRU.
    """
    state = jnp.zeros([h_size], dtype=jnp.complex64)
    if batch_size is not None:
        state = add_batch(state, batch_size)
    return state


class CGRU(hk.RNNCore):
    r"""Gated Recurrent Unit.
    The implementation is based on: https://arxiv.org/pdf/1412.3555v1.pdf with
    biases.
    Given :math:`x_t` and the previous state :math:`h_{t-1}` the core computes
    .. math::
     \begin{array}{ll}
     z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z) \\
     r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r) \\
     a_t &= \tanh(W_{ia} x_t + W_{ha} (r_t \bigodot h_{t-1}) + b_a) \\
     h_t &= (1 - z_t) \bigodot h_{t-1} + z_t \bigodot a_t
     \end{array}
    where :math:`z_t` and :math:`r_t` are reset and update gates.
    The output is equal to the new hidden state, :math:`h_t`.
    Warning: Backwards compatibility of GRU weights is currently unsupported.
    TODO(tycai): Make policy decision/benchmark performance for GRU variants.
    """

    def __init__(
        self,
        hidden_size: int,
        w_i_init: Optional[hk.initializers.Initializer] = None,
        w_h_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.w_i_init = w_i_init or ComplexVarianceScaling
        self.w_h_init = w_h_init or ComplexVarianceScaling
        self.b_init = b_init or complex_zero_init
        self.sig = CSigmoid

    def __call__(self, inputs, state):
        if inputs.ndim not in (1, 2):
            raise ValueError("GRU input must be rank-1 or rank-2.")

        input_size = inputs.shape[-1]
        hidden_size = self.hidden_size
        w_i = hk.get_parameter("w_i", [input_size, 3 * hidden_size], inputs.dtype,
                               init=self.w_i_init)
        w_h = hk.get_parameter("w_h", [hidden_size, 3 * hidden_size], inputs.dtype,
                               init=self.w_h_init)
        b = hk.get_parameter("b", [3 * hidden_size],
                             inputs.dtype, init=self.b_init)
        w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[
                                 2 * hidden_size], axis=1)
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)
        zr_x, a_x = jnp.split(
            gates_x, indices_or_sections=[2 * hidden_size], axis=-1)
        zr_h = jnp.matmul(state, w_h_z)
        zr = zr_x + zr_h + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(self.sig(zr), indices_or_sections=2, axis=-1)

        a_h = jnp.matmul(r * state, w_h_a)
        a = Ctanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a
        return next_state, next_state

    def initial_state(self, batch_size: Optional[int]):
        state = jnp.zeros([self.hidden_size], dtype=jnp.complex64)
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state
