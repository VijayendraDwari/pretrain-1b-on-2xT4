# Parameter count

The canonical training script prints and hard-checks the model total:

```text
1,018,126,848 trainable parameters
```

Why publish the exact count? Because “1B” is a scale label; reproducibility requires the actual architecture.

The total is determined by:

- token embedding / output projection;
- 29 Transformer blocks;
- per-block attention projections;
- per-block FFN projections;
- normalization parameters.

The repo uses **1B-parameter** in titles because 1.018B is standardly within the 1B model scale, while retaining the exact number in specifications.
