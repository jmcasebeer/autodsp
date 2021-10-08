import haiku as hk
import jax
import numpy as np
from jax import numpy as jnp

"""
This file contains the optimizee or adaptive filter which gets fit.
It also contains our loss functions and evaluation metrics. The optimizee
supports both single- and multi- block modes as well as an optional 
parametric nonlinearity.
"""

def train_loss(optimizee_p, optimizee_state, optimizee, u, d):
    """ MSE training loss
    """
    [y, buffer], state = optimizee.apply(optimizee_p, optimizee_state, u)
    d = d.flatten()
    y = y.flatten()

    e = jnp.abs(d - y)**2
    #jnp.mean(e * jnp.conj(e))
    return e.mean(), [y, buffer, state]


def buffer(audio, buffer_length, hop_length):
    """ Convert a signal into a batch of signals
    """

    if audio.shape[0] >= buffer_length:

        num_all_time_frames = audio.shape[0]
        num_segments = np.max(
            [int((num_all_time_frames-buffer_length)/hop_length+1), 1])
        audio_feat = np.zeros((num_segments, buffer_length,))

        left = 0
        for seg_iter in range(num_segments):
            right = left + buffer_length
            audio_feat[seg_iter] = audio[left:right, ]
            left = left + hop_length

    else:
        assert False

    return audio_feat


def compute_erle(echo, residual_echo, sr, buffer_size=1024, hop_size=512):
    """Segmental and scalar echo return loss enhancement (ERLE)

    Arguments:
        echo {np.ndarray} -- [description]
        residual_echo {np.ndarray} -- [description]
        sr {float} -- [description]

    Keyword Arguments:
        buffer_size {int} -- [description] (default: {1024})
        hop_size {int} -- [description] (default: {512})

    Returns:
        np.ndarray, float -- Returns the segmental ERLE and scalar ERLE
    """

    # buffer up the signal
    N = buffer(echo, buffer_size, hop_size)
    R = buffer(residual_echo, buffer_size, hop_size)

    eps = np.finfo(float).eps

    erle_signal = 10*np.log10(eps + np.array(
        [np.mean(N[i, :]**2)/np.mean(eps + R[i, :]**2) for i in range(N.shape[0])]))
    erle_scalar = 10*np.log10(np.mean(echo**2)/np.mean(residual_echo**2))

    return erle_signal, erle_scalar


def make_erle_signal_loss(hop, block, sr, use_target=True):
    """ Function to make a segental ERLE function
    """
    def erle_loss(optimizee_p, optimizee_state, optimizee, u, d, clean, meta, y):
        y = np.array(y).reshape(-1)
        d = np.array(d).reshape(-1) if use_target else clean

        min_len = min(len(y), len(d))

        # cleaned up signal
        e = d[:min_len] - y[:min_len]
        return compute_erle(d, e, sr=sr, buffer_size=block, hop_size=hop)[0]

    return erle_loss


def make_erle_scalar_loss(hop, block, sr, use_target=True):
    """ Function to make a ERLE function which recurns a scalar
    """
    def erle_loss(optimizee_p, optimizee_state, optimizee, u, d, clean, meta, y):
        y = np.array(y).reshape(-1)
        d = np.array(d).reshape(-1) if use_target else clean

        min_len = min(len(y), len(d))

        # cleaned up signal
        e = d[:min_len] - y[:min_len]
        return compute_erle(d, e, sr=sr, buffer_size=block, hop_size=hop)[1]

    return erle_loss


def w_init(shape, dtype):
    """ Custom weight init for a non-aliased filter
    """
    w = jax.random.normal(hk.next_rng_key(), shape, dtype)
    td_w = jnp.fft.ifft(w, axis=1)
    td_w = jax.ops.index_update(td_w, jax.ops.index[:, w.shape[0] // 2:, :], 0)
    w = jnp.fft.fft(td_w, axis=1)
    return w * 1e-10


class ClipSigNonlin(hk.Module):
    """ The parametric nonlinearity used in our paper.
    """
    def __init__(self):
        super().__init__(name="ClipSigNonlin")

    def __call__(self, x):
        xmax = hk.get_parameter("xmax", (1, 1, 1), init=hk.initializers.Constant(
            5+0j), dtype=jnp.complex64).flatten()

        gamma = hk.get_parameter("gamma", (1, 1, 1), init=hk.initializers.Constant(
            5+0j), dtype=jnp.complex64).flatten()
        a1 = hk.get_parameter(
            "a1", (1, 1, 1), init=hk.initializers.Constant(.5+0j), dtype=jnp.complex64).flatten()
        a2 = hk.get_parameter("a2", (1, 1, 1), init=hk.initializers.Constant(
            0+0j), dtype=jnp.complex64).flatten()

        clipped = jnp.abs(xmax.real) * x / \
            (jnp.sqrt(jnp.abs(xmax.real)**2 + jnp.abs(x)**2))

        sig_input = -(a1.real * clipped + a2.real * clipped**2)
        nonlined = gamma.real * (2 / (1 + jnp.exp(sig_input)) - 1)

        return nonlined


class Optimizee(hk.Module):
    """ Block frequency domain adaptive filter with support multiple blocks
    and an optional nonlinearity.
    """
    def __init__(self, sys_length=100, n_blocks=1, has_nonlin=False):
        super().__init__()
        self.sys_length = sys_length
        self.n_blocks = n_blocks
        self.has_nonlin = has_nonlin
        if self.has_nonlin:
            self.nonlin = ClipSigNonlin()

    def get_w(self):
        w = hk.get_parameter("w", [self.n_blocks, 2 * self.sys_length, 1],
                             dtype=jnp.complex64,
                             init=jnp.zeros)

        return w

    def update_buffer(self, u):
        buffer = hk.get_state("buffer",
                              [self.n_blocks, 2 * self.sys_length],
                              dtype=jnp.complex64,
                              init=hk.initializers.RandomNormal(mean=0+0j, stddev=1e-16))
        buffer = jnp.roll(buffer, 1, axis=0)

        if self.has_nonlin:
            u = self.nonlin(u)

        return buffer.at[0, :].set(jnp.fft.fft(u, axis=0))

    def make_pred(self, W, buffer):
        return (W[:, :, 0] * buffer).sum(0)

    def __call__(self, u):
        u = u.flatten()
        W = self.get_w()

        cur_buffer = self.update_buffer(u)
        hk.set_state("buffer", cur_buffer)

        Y = self.make_pred(W, cur_buffer)
        y = jnp.real(jnp.fft.ifft(Y))[self.sys_length:]
        return y, cur_buffer


def optimizee_init(rng, kwargs):
    """ Function to initialize an adaptive filter and serialize it into
    the jax functional format.
    """
    sys_length = kwargs['sys_length']

    def _optimizee_fwd(x):
        optimizee = Optimizee(**kwargs)
        return optimizee(x)

    optimizee = hk.without_apply_rng(hk.transform_with_state(_optimizee_fwd))
    x = jnp.zeros((2 * sys_length, 1))

    optimizee_p, optimizee_state = optimizee.init(rng, x)
    return optimizee_p, optimizee_state, optimizee
