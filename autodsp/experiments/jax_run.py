import os
import argparse

"""
This is the entry point for our code. This file is used to call and run all experiments in 
conjunction with the configruation file called jax_train_config.py. An example train run 
using four GPUs and the first config in jax_train_config.py would be

python jax_run.py --cfg v2_filt_2048_1_hop_1024_lin_1e4_log_24h_10unroll_2deep_earlystop_echo_noise 
                --GPUS 0 1 2 3

Evaluating that same config on the checkpoint from epoch 100 would be

python jax_run.py --cfg v2_filt_2048_1_hop_1024_lin_1e4_log_24h_10unroll_2deep_earlystop_echo_noise 
                --GPUS 0 --eval --epochs 100

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", default="simple_id")
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--GPUS', nargs='+')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument("--epochs", nargs='+')

    args = vars(parser.parse_args())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if args['GPUS'] is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from jax import config
        config.update('jax_platform_name', 'cpu')
        print(f' --- Backend Set for CPU ---')

    else:
        gpus = ','.join(args['GPUS'])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        import jax
        print(f' --- Backend Set for GPUS {gpus} ---')
        print(f' --- {jax.local_device_count()} GPUS Available --- ')

    from jax_train_config import *
    from autodsp import jax_train, jax_eval

    cfg = globals()[args['cfg']]()
    cfg['wandb'] = args['wandb']
    cfg['n_gpus'] = 1 if args['GPUS'] is None else len(args['GPUS'])

    if args['eval']:
        print(' --- EVAL MODE --- ')
        cfg['ckpt_save_dir'] = os.path.join(cfg['ckpt_save_dir'], args['cfg'])
        cfg['name'] = 'EVAL_' + args['cfg']
        cfg['epochs'] = [int(e) for e in args['epochs']]
        jax_eval.eval_optimizer(cfg)

    else:
        print(' --- TRAIN MODE --- ')
        cfg['name'] = args['cfg']
        cfg['ckpt_save_dir'] = os.path.join(cfg['ckpt_save_dir'], args['cfg'])
        jax_train.train_optimizer(cfg)
