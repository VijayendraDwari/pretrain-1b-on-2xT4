# Pretraining a 1B-Parameter Language Model on Two T4 GPUs: An Engineering Walkthrough

*Working draft. The run is still in progress; this version reflects the log snapshot shared on 5 November 2026.*

I wanted to understand what changes when a GPT-style model gets too large to treat as a small tutorial exercise. A pair of 16 GB T4s is not the hardware I would choose to train a competitive 1B model. It is, however, a useful constraint for learning about sharding, memory, token accounting, and checkpoint recovery.

The model has **1,018,126,848 trainable parameters** and a **4096-token context**. It starts from random weights and is trained with a causal next-token objective on pre-tokenized FineWeb data. I reuse the Llama 3.2 1B tokenizer; I did not train a tokenizer or load pretrained Transformer weights.

## Model and training setup

The network is a 29-layer decoder-only Transformer with width 1536, 24 attention heads, and a 6144-wide GELU feed-forward network. It uses dense causal PyTorch SDPA, RMSNorm, and the half-split RoPE implementation in the trainer. The input embedding weights are tied to the output projection.

The trainer uses FSDP `FULL_SHARD` across two T4 GPUs, FP16 mixed precision, activation checkpointing, and gradient accumulation. Each update processes:

```text
4096 tokens × 2 ranks × 16 microbatches = 131,072 model-input tokens
```

The sampler chooses a deterministic window from one training shard for each sample. It samples with replacement; the token counter is not a count of unique corpus tokens and does not prove a single sequential pass through the data.

## The checkpoint failure

The first long run reached update 100 with normal training and validation behavior. It then failed while saving a large FSDP checkpoint because the persistence path ran out of room. The directory had partial rank files but no completion marker, so it was not safe to resume from.

The current continuation setup leaves the mounted parent checkpoint untouched. A new session writes rank files to temporary names and only marks the checkpoint complete after both ranks finish. That makes the resume decision explicit, though it does not by itself prove the identity of the dataset, tokenizer, or code that produced the checkpoint.

## Where the live run stands

The shared log snapshot shows the continuation restoring the complete checkpoint at update 616, then reaching update 820. The latest training row is update 820, or **107,479,040 input tokens**. The latest validation in that capture is update 800, or **104,857,600 tokens**:

| Update | Input tokens | Validation NLL | Perplexity | Token accuracy |
|---:|---:|---:|---:|---:|
| 2 | 262,144 | 10.74244 | 46,279.01 | 3.79% |
| 100 | 13,107,200 | 6.89855 | 990.83 | 12.09% |
| 616 | 80,740,352 | 5.40386 | 222.26 | 21.15% |
| 700 | 91,750,400 | 5.34038 | 208.59 | 21.51% |
| 800 | 104,857,600 | 5.24585 | 189.78 | 22.05% |

These are validation metrics on the fixed sample used by the trainer. They show that next-token loss is falling on that sample. They do not tell me how the model performs on downstream tasks, and I have not run a benchmark suite yet.

The training job is still underway. Update 820 is an observation from the supplied log, not a saved checkpoint; update 616 is the last complete parent checkpoint documented for this continuation. I will update the results after the next complete save and validation event.

## What I want this tutorial to explain

The useful outcome is a clear account of what worked, what failed, and what another person needs to check before attempting the same setup. That includes the exact token counter, the external data and tokenizer requirements, the PyTorch/Kaggle environment, and the fact that a new streamed dataset build is not guaranteed to match the historical token files byte for byte.

The target is 1.5B model-input tokens. That is enough to study the mechanics of the run, not enough to claim compute-optimal pretraining or a competitive base model. Once training finishes, I can add the final validation result, measured GPU hours, checkpoint details, and any separate downstream evaluations that are actually completed.
