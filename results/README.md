# Results

The reference 1.018B-parameter dense model has completed its configured pretraining budget on **2×16GB NVIDIA Tesla T4 GPUs**.

## Final state

| Metric | Final value |
|---|---:|
| Optimizer update | **11,445** |
| Model-input tokens | **1,500,119,040** |
| Validation NLL | **3.212374** |
| Validation perplexity | **24.83799** |
| Validation token accuracy | **39.3593%** |
| Fixed validation sample | **131,072 tokens** |
| Context length | **4096** |
| Trainable parameters | **1,018,126,848** |

The trainer ended normally with `[complete] reached training token target`. The final checkpoint is therefore the planned whole-update boundary, not a time-guard stop.

## Files

- [validation_curve.csv](validation_curve.csv): fixed-validation ledger from early training through the final checkpoint.
- [training_curve.csv](training_curve.csv): sampled training telemetry; sampling density varies across historical sessions.
- [session_checkpoints.csv](session_checkpoints.csv): explicit continuation chain for the final 559.55M → 1.50012B-token phase.
- [milestones.md](milestones.md): human-readable milestone summary.
- [validation_nll.svg](validation_nll.svg): README-facing validation-NLL plot.
- [plot_validation.py](plot_validation.py): plot generator.

The validation metrics are next-token measurements on the same fixed 131,072-token sample. They are not downstream benchmark scores.
