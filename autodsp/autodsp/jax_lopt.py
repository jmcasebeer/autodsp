import os
import pickle

import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp
from jax import random
from jax.experimental import optimizers

from autodsp.jax_complex_rnn import (CGRU, ComplexVarianceScaling, Crelu,
                                     complex_zero_init, deep_initial_state)

"""
This file contains the learned optimizer module and associated utilities. It uses the
complex valued GRUs defiend in jax_complex_rnn.py.
"""


class Optimizer(hk.Module):
    """ The learned optimizer module itself.
    """
    def __init__(self, h_size=20, p=10.0, mu=.1, grad_features='raw', rnn_depth=2,
                 input_lin_depth=1, output_lin_depth=1, sgd_mode=False):
        super().__init__()
        """
        Learned optimizer initializer

        """
        self.rnn_stack = hk.DeepRNN([CGRU(hidden_size=h_size)] * rnn_depth)

        output_lin = []
        for _ in range(output_lin_depth - 1):
            output_lin.extend([hk.Linear(h_size,
                                         w_init=ComplexVarianceScaling,
                                         b_init=complex_zero_init),
                               Crelu])
        output_lin.extend([hk.Linear(output_size=1,
                                     w_init=ComplexVarianceScaling,
                                     b_init=complex_zero_init)])

        self.output_lin = hk.Sequential(output_lin)

        input_lin = []
        for _ in range(input_lin_depth):
            input_lin.extend([hk.Linear(h_size,
                                        w_init=ComplexVarianceScaling,
                                        b_init=complex_zero_init),
                              Crelu])
        self.input_lin = hk.Sequential(input_lin)

        # p is the feature extraction clip values
        self.p = p
        self.grad_features = grad_features

        # multiply output grad by mu
        self.mu = mu

        # debugging mode is sgd
        self.sgd_mode = sgd_mode

    def __call__(self, input_grad, h):
        # input grad is the gradient
        # h is the hidden state

        if self.grad_features == 'raw':
            grad_features = input_grad

        elif self.grad_features == 'log_clamp':
            mag = jax.lax.clamp(
                jnp.exp(-self.p), jnp.abs(input_grad), jnp.exp(self.p))
            mag = jnp.log(mag + 1e-8) / self.p + 1
            phase = jnp.exp(1.j * jnp.angle(input_grad))
            grad_features = mag * phase

        x = self.input_lin(grad_features)

        if self.rnn_stack is not None:
            # reshape going in and out of the RNN since the RNN expects B x h
            # current data is n_blocks x block_size x h and is flattened
            x = x.reshape((-1, x.shape[-1]))
            x, h = self.rnn_stack(x, h)
            x = x.reshape((input_grad.shape[0], -1, x.shape[-1]))

        output_grad = self.output_lin(x)

        if self.sgd_mode:  # this is debug mode and just makes the model bypass the RNN
            output_grad = input_grad

        return -self.mu * output_grad, h


def save_optimizer(learnable_kwargs, ckpt_save_dir, e):
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    file = open(os.path.join(ckpt_save_dir, f'best_model_{e}.npz'), "wb")
    pickle.dump(learnable_kwargs, file)


def load_optimizer(ckpt_save_dir, e):
    f_name = os.path.join(ckpt_save_dir, f'best_model_{e}.npz')
    file = open(f_name, 'rb')
    return pickle.load(file)

# this inits to the haiku format


def init_haiku_format(rng, optimizee_p, x, kwargs):
    """ Learned optimizer functional initializer to 
    serialize and conver to jax format.
    """
    system = jax.tree_util.tree_leaves(optimizee_p)[0]

    def _opt_fwd(grad, h=None):
        optimizer = Optimizer(**kwargs)
        if h is None:
            size = grad.shape[0] * grad.shape[1]
            h = deep_initial_state(size, kwargs['h_size'], kwargs['rnn_depth'])
        return optimizer(grad, h)

    rng, rng_input = jax.random.split(rng)
    grad = random.normal(rng_input, shape=system.shape, dtype=system.dtype)
    optimizer = hk.without_apply_rng(hk.transform(_opt_fwd))
    optimizer_p = optimizer.init(rng, grad, h=None)

    return optimizer_p, optimizer

# this inits to the jax optimizers for mat


@optimizers.optimizer
def init_jax_opt_format(**kwargs):
    optimizer_p = kwargs['optimizer_p']
    optimizer = kwargs['optimizer']

    init_kwargs = kwargs['init_kwargs']
    block_size = kwargs['block_size']

    def init(optimizee_p):
        n_params = np.prod(optimizee_p.shape)
        h = deep_initial_state(n_params, init_kwargs['h_size'],
                               init_kwargs['rnn_depth'])

        return (optimizee_p, h)

    def update(i, grad, state):
        optimizee_p, h = state

        update, h = optimizer.apply(optimizer_p, grad, h)

        # clip udpate to have zeros
        if update.shape[1] == block_size:
            td_update = jnp.fft.ifft(update, axis=1)
            td_update = jax.ops.index_update(
                td_update, jax.ops.index[:, block_size // 2:, :], 0)
            update = jnp.fft.fft(td_update, axis=1)

        return (optimizee_p + update, h)

    def get_params(state):
        return state[0]

    return init, update, get_params

# this inits to the learnable fixed split format I have been using


def init_learnable_fixed_format(x, optimizee_p, rng, cfg):
    """ Function to split the optimizer module into learnable and
    fixed disjoint dictionaries for easy working with jax.
    """
    optimizer_p, optimizer = cfg['haiku_init'](rng,
                                               optimizee_p,
                                               x[:2*cfg['sys_length']],
                                               cfg['haiku_init_kwargs'])

    optimizer_fixed_kwargs = {'optimizer': optimizer,
                              'init_kwargs': cfg['haiku_init_kwargs'],
                              'block_size': 2 * cfg['sys_length']}

    optimizer_learnable_kwargs = {'optimizer_p': optimizer_p}

    return optimizer_fixed_kwargs, optimizer_learnable_kwargs


def mean_loss(losses):
    """ The optimzier loss function
    """
    return jnp.nanmean(losses)
