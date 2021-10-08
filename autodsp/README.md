# Code Organization

```
./
├── autodsp
│   ├── __config__.py       # Put datast dir in here
│   ├── __init__.py
│   ├── jax_aec_dset.py     # Dataset and dataloder
│   ├── jax_block_id.py     # Optimizee/MDF filter
│   ├── jax_complex_rnn.py  # Complex valued GRU
│   ├── jax_core.py         # Inner and outer loop
│   ├── jax_eval.py         # Test/eval code
│   ├── jax_fopt.py         # LMS and NLMS 
│   ├── jax_lopt.py         # Learned optimizer 
│   ├── jax_train.py        # Training and validation loop
│   └── version.py
├── experiments
│   ├── __init__.py
│   ├── jax_run.py          # Entry point for training and testing
│   └── jax_train_config.py # All configuration files
├── LICENSE.txt
├── README.md
├── requirements.txt
└── setup.py
```

You can find further explanations at the top of each file.