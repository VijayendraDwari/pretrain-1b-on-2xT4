# Pretraining a 1B-Parameter Language Model on Two T4 GPUs: An Engineering Walkthrough

*Working draft. The run is still in progress; this version reflects the completed session ending on 8 November 2026.*

I wanted to understand what changes when a GPT-style model gets too large to treat as a small tutorial exercise. A pair of 16 GB T4s is not the hardware I would choose to train a competitive 1B model. It is, however, a useful constraint for learning about sharding, memory, token accounting, validation, and checkpoint recovery.

The model has **1,018,126,848 trainable parameters** and a **4096-token context**. It starts from random weights and is trained with a causal next-token objective on pre-tokenized FineWeb data. I reuse the Llama 3.2 1B tokenizer; I did not train a tokenizer or load pretrained Transformer weights.

## Model and training setup

The network is a 29-layer decoder-only Transformer with width 1536, 24 attention heads, and a 6144-wide GELU feed-forward network. It uses dense causal PyTorch SDPA, RMSNorm, and the half-split RoPE implementation in the trainer. The input embedding weights are tied to the output projection.

The trainer uses FSDP `FULL_SHARD` across two T4 GPUs, FP16 mixed precision, activation checkpointing, and gradient accumulation. Each optimizer update processes:

```text
4096 tokens × 2 ranks × 16 microbatches = 131,072 model-input tokens
```

The sampler chooses deterministic windows from individual training shards and samples with replacement. The token counter is therefore a model-input ledger, not a count of unique corpus tokens and not a claim of one sequential pass through the dataset.

## The checkpoint failure

The first long run reached update 100 with normal training and validation behavior. It then failed while saving a large FSDP checkpoint because the persistence path ran out of room. The directory had partial rank files but no completion marker, so it was not safe to resume from.

The continuation design now leaves the mounted parent checkpoint untouched. A session writes each rank file transactionally and creates `complete.json` only after both ranks finish. The trainer resumes only from COMPLETE two-rank checkpoints.

That distinction matters because the project tracks three different frontiers: the latest sampled training log, the latest fixed validation event, and the latest resumable checkpoint.

## Where the completed run stands

The reference run has reached its configured whole-update budget and saved the final COMPLETE checkpoint at:

- **update 11,445**
- **1,500,119,040 model-input tokens**

The final fixed validation on the same 131,072-token sample is **NLL 3.212374**, **perplexity 24.83799**, and **token accuracy 39.3593%**. The latest sampled training telemetry is u11440 / 1,499,463,680 tokens at about **1,999.55 input tokens/second**. The rank-0 PyTorch allocator metric remained approximately **11.4088 GiB**.

The first public Hugging Face release remains the immutable **Anvaya-1B `160M`** checkpoint at u1225 / 160,563,200 tokens. The completed training lineage is now substantially farther ahead than the current public release; the recommended later checkpoint series is documented in [../../releases/README.md](../../releases/README.md).

## What this tutorial should explain

The useful outcome is a clear account of what worked, what failed, and what another person needs to check before attempting the same setup. That includes the exact token counter, the external data and tokenizer requirements, the PyTorch/Kaggle environment, FSDP ownership of model and optimizer state, and the crash-safe resume contract.

The target remains **1.5B model-input tokens**. That is enough to study the mechanics and evolution of the run, not enough to claim compute-optimal pretraining or a competitive base model. The public record will continue to separate measured training state from downstream capability claims.
