import jax
import numpy as np
from jax import jit
from jax import numpy as jnp
from jax.interpreters import xla
from jax.tree_util import Partial
from tqdm import tqdm

"""
This file contains the core code for training AutoDSP optimizers. The outer and inner
loop here correspond to those described in our 2021 WASPAA paper. The fit_normal
function is used for validation and testing. fit_normal can run both conventional 
and learned optimmizers.
"""

def gradient_mag(old_params, new_params):
    old_params_flat = jax.tree_util.tree_leaves(old_params)
    new_params_flat = jax.tree_util.tree_leaves(new_params)
    cum_grad = 0
    for i in range(len(old_params_flat)):
        cur_grad = (jnp.abs(new_params_flat[i] - old_params_flat[i])**2).sum()
        cum_grad += cur_grad
    return cum_grad


def tree_duplicate(tmap, n_dup):
    """
    Duplicate all elements in a tree

    Inputs:
        tmap                  - Tree map to split acorss objects
        n_dup                 - Int num duplications

    Outputs:
        tmap                  - Tree map split where all leaves have been duplicated
    """
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_dup), tmap)


def freq_trim_array(x, sys_length):
    """[summary]

    Args:
        x ([type]): [description]
        sys_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    if x.shape[1] == 2 * sys_length:
        td_x = jnp.fft.ifft(x, axis=1)
        td_x = jax.ops.index_update(td_x, jax.ops.index[:, sys_length:, :], 0)
        return jnp.fft.fft(td_x, axis=1)
    else:
        return x


def freq_trim_tree(updates, sys_length):
    """[summary]

    Args:
        updates ([type]): [description]
        sys_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    updates = jax.tree_map(lambda x: freq_trim_array(x, sys_length), updates)
    return updates


def run_several_metrics(optimizee_p, optimizee_state, optimizee, x, y, clean, meta, y_hat, metrics):
    """[summary]

    Args:
        optimizee_p ([type]): [description]
        optimizee_state ([type]): [description]
        optimizee ([type]): [description]
        x ([type]): [description]
        y ([type]): [description]
        extra ([type]): [description]
        y_hat ([type]): [description]
        metrics ([type]): [description]

    Returns:
        [type]: [description]
    """
    results = []
    for key in metrics.keys():
        results.append(np.array(metrics[key](optimizee_p,
                                             optimizee_state,
                                             optimizee, x, y,
                                             clean, meta, y_hat)))
    return np.array(results, dtype=object)


def make_inner_loop(optimizer_init_full, fixed_kwargs, optimizer_loss, optimizee, optimizee_loss, hop, sys_length):
    """[summary]

    Args:
        optimizer_init_full ([type]): [description]
        fixed_kwargs ([type]): [description]
        optimizer_loss ([type]): [description]
        optimizee ([type]): [description]
        optimizee_loss ([type]): [description]
        hop ([type]): [description]
        sys_length ([type]): [description]

    Returns:
        [type]: [description]
    """
    optimizee_grad = jax.value_and_grad(optimizee_loss, has_aux=True)
    optimizer_init_partial = Partial(optimizer_init_full, **fixed_kwargs)

    @jit
    def run_inner_loop(learnable_kwargs,
                       opt_state,
                       optimizee_state,
                       x, y):

        _, opt_update, get_params = optimizer_init_partial(**learnable_kwargs)

        @jit
        def step_update_inner(val, i):
            optimizee_state, opt_state = val

            cur_x = jax.lax.dynamic_slice(x, [i, 0], [2 * sys_length, 1])
            cur_y = jax.lax.dynamic_slice(y, [i, 0], [sys_length, 1])

            grad_aux, optimizee_grads = optimizee_grad(get_params(opt_state),
                                                       optimizee_state,
                                                       optimizee,
                                                       cur_x,
                                                       cur_y)

            cur_optimizee_loss, [cur_y_hat, buffer, optimizee_state] = grad_aux

            optimizee_grads = jax.tree_util.tree_map(
                lambda x: jnp.conj(x) * (2 * sys_length)**2, optimizee_grads)

            # save the prediction we used to comptue the grad
            cur_y_hat = jax.lax.dynamic_slice(cur_y_hat, [0], [hop])

            # write in the current u to the state
            if 'takes_u' in fixed_kwargs and fixed_kwargs['takes_u']:
                opt_state[0][0][-1] = cur_x

            # write in the current buffer to the state
            if 'takes_D' in fixed_kwargs and fixed_kwargs['takes_D']:
                opt_state[0][0][-2] = buffer

            # write in the current target to the state
            if 'takes_Y' in fixed_kwargs and fixed_kwargs['takes_Y']:
                opt_state[0][0][-3] = jnp.fft.fft(
                    jnp.pad(cur_y.flatten(), (sys_length, 0)))

            opt_state = opt_update(0, optimizee_grads, opt_state)
            return (optimizee_state, opt_state), (cur_optimizee_loss, cur_y_hat)

        init_val = (optimizee_state, opt_state)
        steps = jnp.arange(0, len(y) - sys_length + 1, hop)
        final_val, [all_losses, y_hat] = jax.lax.scan(
            step_update_inner, init_val, steps)

        y_hat = y_hat.reshape(-1)
        optimizee_state, opt_state = final_val

        # this is the meta loss function
        final_loss = optimizer_loss(all_losses)

        return final_loss, [all_losses, optimizee_state, opt_state, y_hat]

    return run_inner_loop


def make_outer_loop(optimizer_init_full,
                    optimizer_fixed_kwargs,
                    optimizer_loss,
                    meta_opt_update,
                    meta_opt_get_params,
                    optimizee,
                    optimizee_train_loss,
                    sys_length,
                    unroll,
                    hop,
                    grad_clip_mag=5.0,
                    should_train=True):

    optimizer_init_partial = Partial(
        optimizer_init_full, **optimizer_fixed_kwargs)

    @jit
    def run_sequence_parallel(meta_opt_state,
                              optimizee_p,
                              optimizee_state,
                              x, y):

        batch_sz = x.shape[0]
        n_steps = (x.shape[1] - 2 * sys_length) // hop
        n_optimizer_updates = n_steps // unroll

        optimizer_learnable_kwargs = meta_opt_get_params(meta_opt_state)
        opt_init, _, _ = optimizer_init_partial(**optimizer_learnable_kwargs)
        opt_state = opt_init(optimizee_p)

        # init the optimizer and optimizee  across devices
        opt_state = tree_duplicate(opt_state, batch_sz)
        optimizee_state = tree_duplicate(optimizee_state, batch_sz)

        # learned opt gradient function
        mapped_loss = make_inner_loop(optimizer_init_full,
                                      optimizer_fixed_kwargs,
                                      optimizer_loss,
                                      optimizee,
                                      optimizee_train_loss,
                                      hop,
                                      sys_length)

        optimizer_grad_parallel = jax.vmap(
            jax.grad(mapped_loss, has_aux=True), (None, 0, 0, 0, 0))

        @jit
        def run_outer_loop(optimizee_state, opt_state, meta_opt_state):
            def step_update_outer(val, i):
                optimizee_state, opt_state, meta_opt_state = val
                optimizer_learnable_kwargs = meta_opt_get_params(
                    meta_opt_state)

                x_start = [0, i * unroll * hop, 0]
                x_len = [batch_sz, (unroll - 1) * hop + 2 * sys_length, 1]
                cur_x = jax.lax.dynamic_slice(x, x_start, x_len)

                y_start = [0, x_start[1] + sys_length - 1, 0]
                y_len = [batch_sz, x_len[1] - sys_length, 1]
                cur_y = jax.lax.dynamic_slice(y, y_start, y_len)

                grads, aux = optimizer_grad_parallel(optimizer_learnable_kwargs,
                                                     opt_state,
                                                     optimizee_state,
                                                     cur_x, cur_y)

                (cur_seq_losses, optimizee_state, opt_state, _) = aux

                if should_train:
                    grads = jax.tree_util.tree_map(
                        lambda x: jnp.nanmean(x, 0), grads)
                    grads = jax.tree_util.tree_map(
                        lambda x: jnp.conj(x), grads)
                    grads = jax.tree_map(lambda x: jnp.clip(
                        jnp.abs(x), 0, grad_clip_mag) * jnp.exp(1j * jnp.angle(x)), grads)

                    grads = jax.lax.pmean(grads, axis_name='devices')
                    meta_opt_state = meta_opt_update(0, grads, meta_opt_state)

                out_val = (optimizee_state, opt_state, meta_opt_state)

                return out_val, cur_seq_losses.mean(0)

            # scan over the optimizer updates
            init_val = (optimizee_state, opt_state, meta_opt_state)
            steps = jnp.arange(n_optimizer_updates)

            final_val, optimizee_loss_log = jax.lax.scan(
                step_update_outer, init_val, steps)
            optimizee_state, opt_state, meta_opt_state = final_val

            optimizee_loss_log = optimizee_loss_log.reshape(-1)
            return optimizee_loss_log, optimizee_state, meta_opt_state

        return run_outer_loop(optimizee_state, opt_state, meta_opt_state)

    return run_sequence_parallel


def fit_normal(optimizee_init,
               optimizee_init_kwargs,
               optimizee_train_loss,
               test_metrics,
               opt,
               data_gen,
               sys_length,
               hop,
               rng,
               n_tests=5,
               use_all_data=False,
               hide_progress=True,
               **kwargs):
    """[summary]

    Args:
        optimizee_init ([type]): [description]
        optimizee_init_kwargs ([type]): [description]
        optimizee_train_loss ([type]): [description]
        test_metrics ([type]): [description]
        optimizer ([type]): [description]
        data_gen ([type]): [description]
        sys_length ([type]): [description]
        hop ([type]): [description]
        lr ([type]): [description]
        rng ([type]): [description]
        n_tests (int, optional): [description]. Defaults to 5.
        hide_progress (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    all_metadata = []
    all_grad_preds = []
    all_losses = []
    all_metrics = []
    all_params = []
    all_outputs = []

    # this is to deal with the memory leak
    clear_cache_pd = 5

    # could be  made faster with some Partial
    optimizee_grad = jax.value_and_grad(optimizee_train_loss, has_aux=True)

    # Iterate over n_tests random tests
    n_seen = 0
    for (x, y, clean, meta) in tqdm(data_gen, 'tests', disable=hide_progress):
        x, y, clean, meta = jnp.array(x[0]), jnp.array(
            y[0]), jnp.array(clean[0]), jnp.array(meta[0])

        rng, rng_input = jax.random.split(rng)
        optimizee_p, optimizee_state, optimizee = optimizee_init(
            rng_input, optimizee_init_kwargs)

        opt_init, opt_update, get_params = opt(**kwargs)
        opt_state = opt_init(optimizee_p)

        cur_losses = []
        grad_preds = []
        y_hat = jnp.zeros(len(x))

        @jit
        def step_update(cur_x, cur_y, optimizee_state, opt_state):
            optimizee_p = get_params(opt_state)
            aux, grad = optimizee_grad(
                optimizee_p, optimizee_state, optimizee, cur_x, cur_y)
            loss, [cur_y_hat, buffer, optimizee_state] = aux

            grad = jax.tree_util.tree_map(
                lambda x: jnp.conj(x) * (2 * sys_length)**2, grad)

            if 'takes_u' in kwargs and kwargs['takes_u']:
                # first index 0 contains data
                # second index 0 is the batch (always 1 here)
                # third index 0 is the actual state and u is the last element
                opt_state[0][0][-1] = cur_x

            if 'takes_D' in kwargs and kwargs['takes_D']:
                # write in the current buffer to the state
                opt_state[0][0][-2] = buffer

            if 'takes_Y' in kwargs and kwargs['takes_Y']:
                # write in the current buffer to the state
                opt_state[0][0][-3] = jnp.fft.fft(
                    jnp.pad(cur_y.flatten(), (sys_length, 0)))

            opt_state = opt_update(0, grad, opt_state)
            opt_state[0][0][0] = freq_trim_array(
                    opt_state[0][0][0], sys_length)

            return loss, cur_y_hat, optimizee_state, opt_state

        # Iterate over n_steps
        for i in range(0, len(x) - 2 * sys_length, hop):
            x_start = i
            x_len = 2 * sys_length

            y_start = x_start + sys_length - 1
            y_len = sys_length

            cur_x = x[x_start:x_start + x_len]
            cur_y = y[y_start:y_start + y_len]

            loss, cur_y_hat, optimizee_state, new_opt_state = step_update(cur_x, cur_y,
                                                                          optimizee_state,
                                                                          opt_state)

            grad_preds.append(np.array(gradient_mag(
                get_params(opt_state), get_params(new_opt_state))))
            opt_state = new_opt_state

            y_hat = y_hat.at[y_start:y_start + len(cur_y_hat)].set(cur_y_hat)
            cur_losses.append(loss)

        cur_metrics = run_several_metrics(get_params(
            opt_state), optimizee_state, optimizee, x, y, clean, meta, y_hat, test_metrics)

        all_losses.append(cur_losses)
        all_metrics.append(cur_metrics)
        all_params.append(get_params(opt_state))
        all_outputs.append(np.array([x[:len(y_hat)].flatten(),
                                    y[:len(y_hat)].flatten(),
                                    y_hat.flatten(),
                                    clean[:len(y_hat)].flatten()]))
        all_metadata.append(np.array(meta))
        all_grad_preds.append(np.array(grad_preds))

        if n_seen % clear_cache_pd == 0:
            xla._xla_callable.cache_clear()

        # this exits the test loop early for when we dont need to look at everything
        n_seen += 1
        if n_seen > n_tests and not use_all_data:
            break

    return np.array(all_losses), np.array(all_metrics), all_params, np.array(all_outputs), np.array(all_metadata), np.array(all_grad_preds)


def sweep_kwarg_normal(optimizee_train_loss,
                       test_metrics,
                       optimizee_init,
                       optimizee_init_kwargs,
                       optimizer_init,
                       data_gen,
                       sys_length,
                       hop,
                       kwarg_to_sweep,
                       sweep_options,
                       n_tests=20,
                       hide_progress=True,
                       **extra_kwargs):
    """[summary]

    Args:
        optimizee_train_loss ([type]): [description]
        test_metrics ([type]): [description]
        optimizee_init ([type]): [description]
        optimizee_init_kwargs ([type]): [description]
        optimizer_init ([type]): [description]
        data_gen ([type]): [description]
        sys_length ([type]): [description]
        hop ([type]): [description]
        n_tests (int, optional): [description]. Defaults to 20.
        hide_progress (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    best_loss = np.inf
    best_kwarg = sweep_options[0]
    for cur_kwarg in tqdm(sweep_options, f'sweep {kwarg_to_sweep}', disable=hide_progress):
        if not hide_progress:
            print(f' --- Trying {kwarg_to_sweep} {cur_kwarg} ---')
        try:
            rng = jax.random.PRNGKey(0)
            extra_kwargs[kwarg_to_sweep] = cur_kwarg
            all_loss = fit_normal(optimizee_init,
                                  optimizee_init_kwargs,
                                  optimizee_train_loss,
                                  test_metrics,
                                  optimizer_init,
                                  data_gen,
                                  sys_length,
                                  hop,
                                  rng,
                                  n_tests=n_tests,
                                  hide_progress=hide_progress,
                                  **extra_kwargs)[0]
            # switched to median since mean was too variable
            loss = np.median(all_loss)
        except RuntimeError:
            pass

        if loss < best_loss:
            best_loss = loss
            best_kwarg = cur_kwarg
    print(f' --- Best {kwarg_to_sweep} {best_kwarg} ---')
    return best_loss, best_kwarg


def evaluate(optimizee_train_loss,
             test_metrics,
             optimizee_init,
             optimizee_init_kwargs,
             data_gen,
             sys_length,
             hop,
             opts=None,
             opt_names=None,
             n_tests=100,
             use_all_data=False,
             hide_progress=True):
    """[summary]

    Args:
        optimizee_train_loss ([type]): [description]
        test_metrics ([type]): [description]
        optimizee_init ([type]): [description]
        optimizee_init_kwargs ([type]): [description]
        data_gen ([type]): [description]
        sys_length ([type]): [description]
        hop ([type]): [description]
        opts ([type], optional): [description]. Defaults to None.
        opt_names ([type], optional): [description]. Defaults to None.
        n_tests (int, optional): [description]. Defaults to 100.
        hide_progress (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    fit_data = []
    fit_data_metrics = []
    outputs_all = []
    metadata_all = []
    grads_all = []
    for _, (opt, extra_kwargs) in tqdm(enumerate(opts), disable=hide_progress):
        rng = jax.random.PRNGKey(0)
        res, res_metrics, _, outputs, metadata, grads = fit_normal(optimizee_init,
                                                                   optimizee_init_kwargs,
                                                                   optimizee_train_loss,
                                                                   test_metrics,
                                                                   opt,
                                                                   data_gen,
                                                                   sys_length,
                                                                   hop,
                                                                   rng,
                                                                   n_tests=n_tests,
                                                                   use_all_data=use_all_data,
                                                                   hide_progress=hide_progress,
                                                                   **extra_kwargs)
        fit_data.append(res)
        fit_data_metrics.append(res_metrics)
        outputs_all.append(outputs[:25])
        metadata_all.append(metadata)
        grads_all.append(grads)

    return np.array(fit_data), np.array(fit_data_metrics), np.array(outputs_all), np.array(metadata_all), np.array(grads_all)
