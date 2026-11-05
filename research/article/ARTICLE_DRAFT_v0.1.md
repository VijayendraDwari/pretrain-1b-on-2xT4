# Pretraining a 1B-Parameter Language Model on Two T4 GPUs: An End-to-End Engineering Guide

*Working draft — v0.1*

Modern language-model tutorials tend to live at one of two extremes. At one end are tiny GPT implementations small enough to understand in a notebook. At the other are production pretraining stacks designed for clusters of high-end accelerators.

I wanted to explore the engineering space in between.

The model in this project has **1,018,126,848 trainable parameters**, a **4096-token context**, and a standard decoder-only causal language-model objective. It is initialized from random weights and trained on tokenized FineWeb data. The unusual part is the hardware: **two NVIDIA Tesla T4 GPUs with 16GB of memory each**.

That is not an efficient way to compete with modern pretrained 1B models. It is a useful way to understand what billion-parameter pretraining actually requires.

Once the model no longer fits comfortably as a replicated training process, concepts that look optional in smaller tutorials become unavoidable: parameter sharding, optimizer-state residency, gradient accumulation, exact token accounting, failure-safe checkpoints, and deterministic continuation.

This article documents that path.

## Why 1B?

The point of the scale is educational rather than numerical.

A 100M-parameter Transformer is already a real language model, but modern GPUs can make much of its systems behavior disappear. At roughly one billion parameters, a pair of 16GB T4s forces the training system to become part of the model experiment.

Our configuration is:

| Field | Value |
|---|---:|
| Parameters | 1,018,126,848 |
| Layers | 29 |
| Width | 1536 |
| Attention heads | 24 |
| Head dimension | 64 |
| FFN | 6144 |
| Vocabulary | 128,256 |
| Context | 4096 |

The model uses dense causal attention. This project is intentionally separate from my custom-attention research: the goal here is to make the standard pretraining recipe as transparent as possible.

## The first systems constraint: memory

A billion FP32 parameters alone require roughly four gigabytes. Training requires much more than the parameter tensor itself: gradients, optimizer moments, activations, temporary buffers, and framework overhead all compete for device memory.

Replicating the entire training state on both 16GB T4s is therefore the wrong execution model.

The run uses PyTorch FSDP with `FULL_SHARD`, which shards model state across the two GPUs and materializes parameter groups as needed. Combined with gradient accumulation, this lets the model train at a 4096-token context while keeping observed allocation around the practical T4 envelope.

The important lesson is that “the model has 1B parameters” is not enough information to decide whether it fits. **The state ownership and lifetime model matters.**

## Token arithmetic

The training run is organized around model-input tokens rather than epochs.

With context 4096, two GPUs, and 16 accumulation steps, each optimizer update represents:

```text
4096 × 2 × 16 = 131,072 tokens
```

That number becomes the ledger for the entire run.

At update 616:

```text
616 × 131,072 = 80,740,352 tokens
```

The project target is 1.5 billion input tokens. That is intentionally much smaller than the token budgets used to make modern 1B models competitive. The goal is an end-to-end pretraining demonstration, not a compute-optimal model release.

## A checkpoint failure is part of the story

The first long session reached update 100 normally. Validation NLL had fallen to about 6.90, and training remained stable. Then checkpoint serialization failed.

The failure was not numerical. It was operational: writing another multi-gigabyte distributed checkpoint exhausted the available persistence path.

That distinction matters. A `.partial` checkpoint can look tantalizingly close to usable, but it is not a valid lineage parent.

The continuation design was changed around four rules:

1. only directories with an explicit COMPLETE marker are resumable;
2. partial rank files are ignored;
3. the mounted parent checkpoint remains immutable;
4. each notebook session writes one large continuation checkpoint near its time guard instead of accumulating many overlapping full checkpoints.

The next run resumed from the last certified checkpoint, replayed the lost compute, and eventually produced a clean checkpoint at update 616.

Reproducibility is not just “can the code start?” It is also **can the training state survive interruption without ambiguity?**

## What the model is learning

The validation trajectory already gives a useful picture:

| Training tokens | Validation NLL | Perplexity | Token accuracy |
|---:|---:|---:|---:|
| 262K | 10.7424 | 46,279 | 3.79% |
| 13.1M | 6.8985 | 990.8 | 12.09% |
| 80.7M | 5.4039 | 222.3 | 21.15% |

This is still the early phase of the run. The model is far from a mature 1B language model, but it is clearly moving from random initialization toward a useful predictive distribution.

The next analytical milestones are 100M, 250M, 500M, 1B and 1.5B tokens.

## What comes next

The repository will publish the canonical trainer, data contract, exact architecture, checkpoint/recovery protocol, milestone logs, and a fixed evaluation suite.

The goal is that another researcher can read the project and understand not only how to launch the run, but why every major systems choice exists.

That is the real purpose of this experiment.
