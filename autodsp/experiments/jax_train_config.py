from autodsp import jax_block_id, jax_aec_dset, jax_lopt

"""
This file contains the configurations used to run the experiments in our 2021 WASPAA paper.
They are organized so that they correspond to the same groups of abaltions/experiments.
"""


# --- Feature Extraction Ablation ---


def v2_filt_2048_1_hop_1024_lin_1e4_log_24h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length // 2
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 24,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,
        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len, is_iter=False),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1, is_iter=False),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_2048_1_hop_1024_lin_1e4_raw_24h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length // 2
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 24,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'raw',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_2048_1_hop_2048_lin_1e4_log_24h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 24,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_2048_1_hop_2048_lin_1e4_raw_24h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 24,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'raw',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg

# --- Model Overlap Ablation ---


def v2_filt_2048_1_hop_2048_lin_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length
    batch_sz = 44
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_2048_1_hop_1024_lin_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length // 2
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_2048_1_hop_512_lin_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length // 4
    batch_sz = 44
    sr = 8000
    is_double_len = True
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg

# --- 1 / 2 Overlap MDF Ablation ---


def v2_filt_512_4_hop_512_lin_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 512
    n_blocks = 4
    hop = sys_length
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_256_8_hop_256_lin_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 256
    n_blocks = 8
    hop = sys_length
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
            'has_nonlin': False,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg

# --- 3 / 4 Overlap MDF Ablation ---


def v2_filt_820_4_hop_410_lin_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 820
    n_blocks = 4
    hop = sys_length // 2
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
            'has_nonlin': False,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_456_8_hop_228_lin_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 456
    n_blocks = 8
    hop = sys_length // 2
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
            'has_nonlin': False,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg

# --- 3 / 4 Overlap Nonlin Ablation ---


def v2_filt_2048_1_hop_512_new_clip_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 2048
    n_blocks = 1
    hop = sys_length // 4
    batch_sz = 44
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
            'has_nonlin': True,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_820_4_hop_410_new_clip_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 820
    n_blocks = 4
    hop = sys_length // 2
    batch_sz = 64
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
            'has_nonlin': True,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg


def v2_filt_456_8_hop_228_new_clip_1e4_log_48h_10unroll_2deep_earlystop_echo_noise():
    sys_length = 456
    n_blocks = 8
    hop = sys_length // 2
    batch_sz = 58
    sr = 8000
    is_double_len = False
    cfg = {
        'is_double_len': is_double_len,
        'learnable_fixed_init': jax_lopt.init_learnable_fixed_format,
        'jax_init': jax_lopt.init_jax_opt_format,
        'haiku_init': jax_lopt.init_haiku_format,
        'optimizer_loss': jax_lopt.mean_loss,

        'haiku_init_kwargs': {
            'h_size': 48,
            'p': 10.0,
            'mu': 0.01,
            'grad_features': 'log_clamp',
            'rnn_depth': 2,
        },

        'optimizee_init': jax_block_id.optimizee_init,
        'optimizee_init_kwargs': {
            'sys_length': sys_length,
            'n_blocks': n_blocks,
            'has_nonlin': True,
        },

        'optimizee_train_loss': jax_block_id.train_loss,

        'test_metrics': {
            'erle_signal': jax_block_id.make_erle_signal_loss(hop, sys_length, sr, use_target=False),
            'erle_scalar': jax_block_id.make_erle_scalar_loss(hop, sys_length, sr, use_target=False),
        },

        'train_data_gen': jax_aec_dset.get_msft_data_gen(mode='train', sr=sr, batch_sz=batch_sz, num_workers=10),
        'test_data_gen': jax_aec_dset.get_msft_data_gen(mode='test', sr=sr, batch_sz=1, num_workers=1, is_double_len=is_double_len),
        'val_data_gen': jax_aec_dset.get_msft_data_gen(mode='val', sr=sr, batch_sz=1, num_workers=1),
        'sr': sr,
        'sys_length': sys_length,
        'unroll': 10,
        'hop': hop,
        'batch_sz': batch_sz,
        'lr': 1e-4,
        'early_stop_wait': 25,
        'half_lr_wait': 10,
        'n_epochs': 501,
        'n_training_per_epoch': 200,
        'n_val_per_epoch': 100,
        'n_epochs_between_baseline': 50,
        'n_tests': 20,
        'n_tune': 20,
        'ckpt_save_dir': '../ckpts',
    }

    return cfg
