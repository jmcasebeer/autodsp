import jax
from jax import numpy as jnp
from jax.experimental import optimizers

"""
This file contains the LMS/NLMS optimizers used as baselines. They use 
recursive exponentially smoothed estimates for normalizing factors as 
well as gradient clipping.
"""

@optimizers.optimizer
def init_FNLMS(step_size, g=.9, **kwargs):
    eps = 1e-8
    block_size = kwargs['block_size']
    grad_clip_mag = kwargs['grad_clip_mag']

    def init(optimizee_p):
        u = jnp.zeros((block_size, 1))
        P = jnp.zeros((block_size, 1))
        return (optimizee_p, P, u)

    def update(i, grad, state):
        x, P, u = state

        # make the running normalizer
        U = jnp.fft.fft(u, axis=0)
        P = g * P + (1 - g) * (jnp.abs(U)**2)

        # clip the gradients
        grad = jnp.clip(jnp.abs(grad), 0, grad_clip_mag) * \
            jnp.exp(1j * jnp.angle(grad))

        # update step -- special case for the nonlin items
        if grad.shape == (1, 1, 1):
            update = -step_size * grad
        else:
            update = -step_size * grad / (P + eps)

            # clip the update to have zeros
            td_update = jnp.fft.ifft(update, axis=1)
            td_update = jax.ops.index_update(
                td_update, jax.ops.index[:, block_size // 2:, :], 0)
            update = jnp.fft.fft(td_update, axis=1)

        return (x + update, P, u)

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


def init_learnable_fixed_format(x, optimizee_p, rng_input, cfg):

    optimizer_fixed_kwargs = {
        'block_size': cfg['jax_init_kwargs']['block_size'],
        'grad_clip_mag': cfg['jax_init_kwargs']['grad_clip_mag'],
        'is_complex': False,
        'takes_u': True,
    }

    optimizer_learnable_kwargs = {
        'step_size': cfg['jax_init_kwargs']['step_size'],
        'g': cfg['jax_init_kwargs']['g'],
    }

    return optimizer_fixed_kwargs, optimizer_learnable_kwargs
