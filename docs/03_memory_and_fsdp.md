# Memory and FSDP

A billion FP32 parameters would occupy about 4 GB before gradients, AdamW moments, activations, temporary tensors, and framework overhead are counted. Keeping a full training copy on each 16 GB T4 is not the plan used here.

The trainer wraps each Transformer block with PyTorch FSDP `FULL_SHARD`. Model and optimizer state are sharded across two ranks. It also uses FP16 parameters and activations, FP32 gradient reduction, activation checkpointing, and a chunked vocabulary loss. Gradient accumulation gives the target token batch without requiring a larger per-rank microbatch.

In the supplied live-run log, PyTorch reports a peak of **11.4088 GiB allocated per GPU**. This is `torch.cuda.max_memory_allocated()`, not total device usage or reserved memory. Treat it as an observation from this run and runtime, not a guaranteed memory requirement for another PyTorch build.

The useful question is not only how many parameters the model has. It is where the parameters, gradients, optimizer state, and activations live during each part of the step.
