# Memory and FSDP

Two T4s provide 16GB VRAM each. A naïve replicated 1.018B-parameter AdamW training job is not a sensible fit because memory includes much more than the raw parameter tensor:

- model parameters;
- gradients;
- first and second Adam moments;
- mixed-precision working copies;
- activations;
- attention/MLP temporaries;
- CUDA/framework overhead.

The canonical run uses PyTorch FSDP with `FULL_SHARD`, `limit_all_gathers=True`, mixed precision, and per-block wrapping.

Observed training allocation in the documented run was about **11.4 GiB/GPU** at 4096 context.

The systems lesson is that model size alone does not determine fit. **Ownership and lifetime of training state do.**
