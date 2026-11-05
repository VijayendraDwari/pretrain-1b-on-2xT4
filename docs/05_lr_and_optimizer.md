# Optimizer and learning-rate schedule

The current trainer uses AdamW with betas `(0.9, 0.95)`, epsilon `1e-8`, and weight decay `0.1` on matrix-like parameters. The token embedding and one-dimensional normalization weights are not decayed. Gradients are clipped to a global norm of 1.0.

The peak learning rate is `1.5e-4`. It warms up linearly for 75 updates, then follows a cosine schedule that ends at 10% of the peak at the configured token target. The schedule is indexed by the restored update number, so continuation does not restart warmup.

These settings describe the current run. They are not presented as tuned or optimal hyperparameters. The trainer saves the optimizer and GradScaler state with each complete checkpoint.
