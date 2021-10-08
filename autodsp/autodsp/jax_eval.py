import os
import pickle as pkl

import jax
import numpy as np
from jax.experimental import optimizers

from autodsp import jax_fopt, jax_lopt, jax_train

"""
This file contains the code to test an optimizer. It is called from
jax_run.py when the --eval flag is specified and has functionality to log to wandb.
"""

def eval_optimizer(cfg):
    if 'is_double_len' in cfg and cfg['is_double_len']:
        print('--- Runing Double Len ---')

    jax_train.init_wandb(cfg)
    rng = jax.random.PRNGKey(0)

    # initing just the optimizer without a metaopt or anything
    x, y, _, _ = cfg['train_data_gen']()
    x, y = x[0], y[0]

    rng, rng_input = jax.random.split(rng)
    optimizee_p, _, _ = cfg['optimizee_init'](
        rng_input, cfg['optimizee_init_kwargs'])

    rng, rng_input = jax.random.split(rng)
    optimizer_fixed_kwargs, _ = cfg['learnable_fixed_init'](
        x, optimizee_p, rng_input, cfg)

    # list of the baseline fixed optimizers
    OPTS = [(jax_fopt.init_FNLMS, {'step_size': .025, 'g': 0.99, 'block_size': 2 * cfg['sys_length'], 'takes_u': True, 'grad_clip_mag': 2500}),
            (jax_fopt.init_FNLMS, {'step_size': .01, 'g': 0.9, 'block_size': 2 *
             cfg['sys_length'], 'takes_u': True, 'grad_clip_mag': 2500}),
            (optimizers.rmsprop, {'step_size': .01})]
    OPT_NAMES = ['FNLMS.99', 'FNLMS.9', 'RMS']

    # add different epochs of the learned one to the list
    for e in cfg['epochs']:
        cur_learnable_kwargs = jax_lopt.load_optimizer(cfg['ckpt_save_dir'], e)
        cur_kwargs = {**optimizer_fixed_kwargs, **cur_learnable_kwargs}

        OPTS = OPTS + [(cfg['jax_init'], cur_kwargs)]
        OPT_NAMES.append(f'LOPT-{e}')

    print(' --- Running Eval --- ')
    test_data, metrics_data, out_data, _, metadata, grads = jax_train.tune_and_test_all(OPTS,
                                                                                        OPT_NAMES,
                                                                                        cfg,
                                                                                        tune_kwarg=True,
                                                                                        use_all_data=True,
                                                                                        hide_progress=False)

    jax_train.send_to_wandb(test_data,
                            metrics_data,
                            out_data,
                            metadata,
                            OPT_NAMES,
                            list(cfg['test_metrics'].keys()),
                            cfg,
                            cfg['epochs'][-1])

    all_test_results = {'test_data': test_data,
                        'metrics_data': metrics_data,
                        'out_data': out_data,
                        'metadata': metadata,
                        'grads': grads,
                        'model_names': OPT_NAMES,
                        'metrics_names': list(cfg['test_metrics'].keys())}

    save_name = str(cfg['epochs'][-1])+'.pkl'
    if 'is_double_len' in cfg and cfg['is_double_len']:
        save_name = 'double_' + save_name

    with open(os.path.join(cfg['ckpt_save_dir'], save_name), 'wb') as f:
        pkl.dump(all_test_results, f, protocol=pkl.HIGHEST_PROTOCOL)

    print(' --- Finished Eval --- ')
    linear, nonlinear = [], []
    for i in range(metrics_data.shape[1]):
        if metadata[0, i, 2]:
            nonlinear.append(metrics_data[:, i, :])
        else:
            linear.append(metrics_data[:, i, :])
    linear = np.array(linear)
    nonlinear = np.array(nonlinear)

    print('Network Performance ERLE')
    for i in range(len(OPT_NAMES)):
        cur_lin, cur_nonlin = linear[:, i, 1], nonlinear[:, i, 1]
        stack = np.hstack((cur_nonlin, cur_lin))
        try:
            print(OPT_NAMES[i])
            print('All:{} +- {} -- NAN:{} +- {}'.format(stack.mean(),
                  stack.std(), np.nanmean(stack), np.nanstd(stack)))
            print('Nonlinear:{} +- {} -- NAN:{} +- {}'.format(cur_nonlin.mean(),
                  cur_nonlin.std(), np.nanmean(cur_nonlin), np.nanstd(cur_nonlin)))
            print('Linear:{} +- {} -- NAN:{} +- {}'.format(cur_lin.mean(),
                  cur_lin.std(), np.nanmean(cur_lin), np.nanstd(cur_lin)))
            print('Num NAN:', np.isnan(stack.astype(float)).sum())
        except:
            print('Failed on OPT: ', OPT_NAMES[i])
