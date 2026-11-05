# Optimizer and learning-rate schedule

Canonical optimizer:

- AdamW
- betas `(0.9, 0.95)`
- epsilon `1e-8`
- weight decay `0.1` on matrix-like parameters
- global gradient clipping `1.0`
- peak learning rate `1.5e-4`
- warmup: 75 updates in the continuation configuration

The learning-rate schedule is keyed to the global optimizer-update counter. Resume restores optimizer and scaler state; it does not restart warmup.
