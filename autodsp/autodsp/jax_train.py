import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from jax.experimental import optimizers
from jax.interpreters import xla
from tqdm import tqdm

from autodsp import jax_core, jax_fopt, jax_lopt

"""
This file contains the training and validation loop to train a model. It is called from
jax_run.py and has functionality to log to wandb.
"""

def init_wandb(cfg):
    """ Get wandb setup with the run and project name.
    """
    if cfg['wandb']:
        name = f"{cfg['name']}_batch_sz_{cfg['batch_sz']}"

        wandb.init(project="autodsp", name=name)

        print(f' --- {name} ---')
        print(' --- Initialized WANDB --- ')

    else:
        print(' ---  NO WANDB ---')


def init_all_opt(cfg, rng_input):
    """ Initialize a dummy optimizee as well as the optimizer and meta optimizer.
    """
    x, y, _, _ = cfg['train_data_gen']()
    x, y = x[0], y[0]

    # init dummy optimizee to init optimizer with
    rng, rng_input = jax.random.split(rng_input)
    optimizee_p, optimizee_state, optimizee = cfg['optimizee_init'](
        rng_input, cfg['optimizee_init_kwargs'])
    optimizee_tuple = [optimizee, optimizee_p, optimizee_state]

    # init the optimizer
    rng, rng_input = jax.random.split(rng)
    optimizer_fixed_kwargs, optimizer_learnable_kwargs = cfg['learnable_fixed_init'](
        x, optimizee_p, rng_input, cfg)
    optimizer_tuple = [optimizer_fixed_kwargs, optimizer_learnable_kwargs]

    meta_opt_init, meta_opt_update, meta_opt_get_params = optimizers.adam(
        cfg['lr'])
    meta_opt_state = meta_opt_init(optimizer_learnable_kwargs)
    meta_opt_tuple = [meta_opt_update, meta_opt_get_params, meta_opt_state]

    return meta_opt_tuple, optimizer_tuple, optimizee_tuple


def tune_and_test_all(all_opts, all_names, cfg, tune_kwarg=False, use_all_data=False, hide_progress=True):
    """ Function to run baseline optimizers and optionally perform a grid search over a hyperparameter.
    """
    if tune_kwarg:
        print(' --- Tuning Baselines --- ')
        for i, (opt, extra_kwargs) in enumerate(all_opts):
            # only tune the lr for the fixed opts -- maybe we should change this later?
            if 'step_size' in extra_kwargs and 'LOPT' not in all_names[i]:
                print(f' --- Tuning {all_names[i]} ---')
                _, best_lr = jax_core.sweep_kwarg_normal(cfg['optimizee_train_loss'],
                                                         cfg['test_metrics'],
                                                         cfg['optimizee_init'],
                                                         cfg['optimizee_init_kwargs'],
                                                         opt,
                                                         cfg['val_data_gen'],
                                                         sys_length=cfg['sys_length'],
                                                         hop=cfg['hop'],
                                                         kwarg_to_sweep='step_size',
                                                         sweep_options=[
                                                             .5, .25, .1, .05, .025, 0.01, .001],
                                                         n_tests=cfg['n_tune'],
                                                         hide_progress=hide_progress,
                                                         **extra_kwargs)
                all_opts[i][1]['step_size'] = best_lr
        print(' --- Finished Tuning Baselines --- ')

    # additional optimizers -- i.e. the learned optimizers
    test_data, metrics_data, out_data, metadata, grads = jax_core.evaluate(cfg['optimizee_train_loss'],
                                                                           cfg['test_metrics'],
                                                                           cfg['optimizee_init'],
                                                                           cfg['optimizee_init_kwargs'],
                                                                           cfg['test_data_gen'],
                                                                           cfg['sys_length'],
                                                                           cfg['hop'],
                                                                           opts=all_opts,
                                                                           opt_names=all_names,
                                                                           n_tests=cfg['n_tests'],
                                                                           use_all_data=use_all_data,
                                                                           hide_progress=hide_progress)

    return test_data, metrics_data, out_data, all_opts, metadata, grads


def send_to_wandb(test_data, metrics_data, out_data, metadata, names, metrics_names, cfg, e):
    """ Function to log some visualizations for run tracking.
    """
    # metadata -- n_models x n_tests x [ser, noisy, nonlinear]

    if cfg['wandb']:
        # sending the loss values
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        test_data = np.median(np.array(test_data), 1)

        n_lstms = np.array([int('LSTM' in name) for name in names]).sum()
        lstm_cmap = [plt.cm.Wistia(x) for x in np.linspace(0.05, .95, n_lstms)]

        n_other = len(names) - n_lstms
        other_cmap = [plt.cm.tab10(x) for x in np.linspace(0.0, 1.0, n_other)]

        other_idx, lstm_idx = 0, 0
        for i, name in enumerate(names):
            if 'LSTM' in name:
                color = lstm_cmap[lstm_idx]
                style = 'dashed'
                lw = 2
                lstm_idx += 1
            else:
                color = other_cmap[other_idx]
                style = 'solid'
                lw = 1
                other_idx += 1

            ax[0].plot(test_data[i], label=name,
                       color=color, linestyle=style, lw=lw)
            ax[0].legend()

            ax[1].semilogy(test_data[i], label=name,
                           color=color, linestyle=style, lw=lw)
            ax[1].legend()

        ax[0].set_title('Linear Scale')
        ax[1].set_title('Log Scale')
        plt.suptitle(f'Epoch {e}')
        plt.tight_layout()
        wandb.log({"opt_comparison": wandb.Image(plt), 'epoch': e})
        plt.clf()
        plt.close('all')

        # sending the metrics
        # since the metrics may have different lengths, we have different plots
        # the check is to see if a metric has length or not
        for i, name in enumerate(metrics_names):
            if len(metrics_data[0, 0, i].shape) > 0:
                cur_data = metrics_data[:, :, i]
                for j in range(len(cur_data)):
                    mean_data = np.array(
                        [cur_data[j, k] for k in range(cur_data.shape[1])]).mean(0)
                    plt.plot(mean_data, label=f'{names[j]}')
                plt.legend()

            else:
                cur_data = metrics_data[:, :, i].astype(float).T
                x_pos = np.arange(len(metrics_data))
                plt.violinplot(cur_data,
                               x_pos,
                               widths=0.3,
                               showmeans=True,
                               showextrema=True,
                               showmedians=True)

                plt.xticks(x_pos, names)

            plt.title(f'Epoch {e} -- {name}')
            wandb.log(
                {f"Metrics_Comparison_{name}": wandb.Image(plt), 'epoch': e})
            plt.clf()
            plt.close('all')


def train_optimizer(cfg):
    """ The main function. This orchestrates initialization of optimizers and contains the
    train/val loop as well as checkpointing, logging and early stopping.
    """
    init_wandb(cfg)

    rng = jax.random.PRNGKey(0)

    rng, rng_input = jax.random.split(rng)
    meta_opt_tuple, optimizer_tuple, optimizee_tuple = init_all_opt(
        cfg, rng_input)

    meta_opt_update, meta_opt_get_params, meta_opt_state = meta_opt_tuple
    optimizer_fixed_kwargs,  _ = optimizer_tuple
    optimizee, _, _ = optimizee_tuple

    best_loss = -100000000000000000
    epochs_since_new_best = 0
    time_since_halving = cfg['half_lr_wait'] + 1
    cur_lr = cfg['lr']

    OPTS = [(jax_fopt.init_FNLMS, {'step_size': .1, 'g': 0.9, 'block_size': 2 * cfg['sys_length'], 'takes_u': True, 'grad_clip_mag': 2500}),
            (optimizers.rmsprop, {'step_size': .25})]
    OPT_NAMES = ['FNLMS.9', 'RMS']

    metric_names = cfg['test_metrics'].keys()
    train_data_gen = cfg['train_data_gen']

    train_run_seq = jax_core.make_outer_loop(cfg['jax_init'],
                                             optimizer_fixed_kwargs,
                                             cfg['optimizer_loss'],
                                             meta_opt_update,
                                             meta_opt_get_params,
                                             optimizee,
                                             cfg['optimizee_train_loss'],
                                             cfg['sys_length'],
                                             cfg['unroll'],
                                             cfg['hop'],
                                             should_train=True)
    train_run_seq = jax.pmap(train_run_seq, axis_name='devices')
    #train_run_seq = jax.pmap(train_run_seq, in_axes=(None, 0, 0, 0, 0), axis_name='devices')
    meta_opt_state = jax.tree_util.tree_map(
        lambda x: jnp.stack([x] * cfg['n_gpus']), meta_opt_state)

    # Iterate over training + validation
    epoch_pbar = tqdm(range(cfg['n_epochs']))
    for e in epoch_pbar:

        # Train
        #     For each loop, a new optimizee function is randomly created
        #     and optim_it forward steps is taken, updating every #unroll iteration
        #     Sum over optim_it losses
        for i in range(cfg['n_training_per_epoch']):
            rng, rng_input = jax.random.split(rng)
            optimizee_p, optimizee_state, optimizee = cfg['optimizee_init'](
                rng_input, cfg['optimizee_init_kwargs'])

            optimizee_p = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * cfg['n_gpus']), optimizee_p)
            optimizee_state = jax.tree_util.tree_map(
                lambda x: jnp.stack([x] * cfg['n_gpus']), optimizee_state)

            x, y, _, _ = train_data_gen()
            x, y = jnp.array(x), jnp.array(y)

            x = x.reshape((cfg['n_gpus'], cfg['batch_sz'] //
                          cfg['n_gpus'], x.shape[1], x.shape[2]))
            y = y.reshape((cfg['n_gpus'], cfg['batch_sz'] //
                          cfg['n_gpus'], y.shape[1], y.shape[2]))

            seq_loss, _, meta_opt_state = train_run_seq(meta_opt_state,
                                                        optimizee_p,
                                                        optimizee_state,
                                                        x, y)
            seq_loss = np.array(seq_loss).mean()
            epoch_pbar.set_description(f'Loss:{seq_loss:.5f}')
            if cfg['wandb']:
                wandb.log({'train_train_loss': seq_loss, 'epoch': e,
                          'batch': e * cfg['n_training_per_epoch'] + i})

        single_optimizer_learnable_kwargs = jax.tree_map(
            lambda x: x[0], meta_opt_get_params(meta_opt_state))

        # Validation (no training adaptation)
        #     For each loop, a new optimizee function is randomly created
        #     and optim_it forward steps is taken
        #     Sum over optim_it losses
        val_kwargs = {**single_optimizer_learnable_kwargs,
                      **optimizer_fixed_kwargs}
        val_loss, val_metrics = jax_core.fit_normal(cfg['optimizee_init'],
                                                    cfg['optimizee_init_kwargs'],
                                                    cfg['optimizee_train_loss'],
                                                    cfg['test_metrics'],
                                                    cfg['jax_init'],
                                                    cfg['val_data_gen'],
                                                    cfg['sys_length'],
                                                    cfg['hop'],
                                                    rng_input,
                                                    n_tests=cfg['n_val_per_epoch'],
                                                    hide_progress=True,
                                                    **val_kwargs)[:2]

        # Update best loss
        if np.mean(np.vstack(val_metrics[:, 1])) > best_loss:
            best_loss = np.mean(np.vstack(val_metrics[:, 1]))
            epochs_since_new_best = 0
            time_since_halving = 0
            jax_lopt.save_optimizer(
                single_optimizer_learnable_kwargs, cfg['ckpt_save_dir'], e)
        else:
            epochs_since_new_best += 1
            time_since_halving += 1
            if epochs_since_new_best > cfg['early_stop_wait']:
                print(
                    ' --- {} Epochs Since New Best, Stopping ---'.format(cfg['early_stop_wait']))
                jax_lopt.save_optimizer(
                    single_optimizer_learnable_kwargs, cfg['ckpt_save_dir'], e)
                exit()
            elif epochs_since_new_best > cfg['half_lr_wait'] and time_since_halving > cfg['half_lr_wait']:
                print(
                    ' --- {} Epochs Since New Best, Halving lr ---'.format(cfg['half_lr_wait']))
                time_since_halving = 0
                cur_lr = cur_lr / 2
                cfg['lr'] = cur_lr
                rng, rng_input = jax.random.split(rng)
                meta_opt_update, _, _ = init_all_opt(cfg, rng_input)[0]

        if cfg['wandb'] and cfg['n_val_per_epoch'] > 0:
            wandb.log({'val_loss': val_loss.mean(), 'epoch': e})
            for i, name in enumerate(list(metric_names)):
                wandb.log({f'Metrics_{name}': np.mean(
                    np.vstack(val_metrics[:, i])), 'epoch': e})

        if e % cfg['n_epochs_between_baseline'] == 0:
            jax_lopt.save_optimizer(
                single_optimizer_learnable_kwargs, cfg['ckpt_save_dir'], e)

            test_kwargs = {**single_optimizer_learnable_kwargs,
                           **optimizer_fixed_kwargs}
            all_opts = OPTS + [(cfg['jax_init'], test_kwargs)]
            all_names = OPT_NAMES + [f'LOPT-{e}']

            test_data, metrics_data, out_data, tuned_opts, metadata, _ = tune_and_test_all(all_opts,
                                                                                           all_names,
                                                                                           cfg,
                                                                                           tune_kwarg=(e == 0))
            OPTS = tuned_opts[:-1]
            send_to_wandb(test_data, metrics_data, out_data,
                          metadata, all_names, metric_names, cfg, e)

        # deleting the cache every epoch to deal with memory
        xla._xla_callable.cache_clear()
