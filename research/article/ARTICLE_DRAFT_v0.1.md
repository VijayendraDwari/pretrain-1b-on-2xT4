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

## Where the live run stands

The latest completed session resumed from update **4269 / 559,546,368 model-input tokens**, validated that restored state at NLL **3.58536**, crossed **600M model-input tokens**, and stopped safely after writing a new COMPLETE checkpoint at:

- **update 4879**
- **639,500,288 model-input tokens**

The latest sampled training row is update **4870 / 638,320,640 tokens**, at about **2,132.97 input tokens/second**. The rank-0 PyTorch allocator high-water mark remained approximately **11.4088 GiB**.

The latest fixed validation event is update **4800 / 629,145,600 tokens**:

| Update | Input tokens | Validation NLL | Perplexity | Token accuracy |
|---:|---:|---:|---:|---:|
| 1225 | 160,563,200 | 4.75070 | 115.67 | 25.21% |
| 1829 | 239,730,688 | 4.26552 | 71.20 | 30.09% |
| 2436 | 319,291,392 | 3.98411 | 53.74 | 32.38% |
| 3031 | 397,279,232 | 3.81320 | 45.30 | 33.69% |
| 3638 | 476,839,936 | 3.68761 | 39.95 | 34.72% |
| 4269 | 559,546,368 | 3.58536 | 36.07 | 35.71% |
| 4500 | 589,824,000 | 3.55857 | 35.11 | 35.81% |
| 4600 | 602,931,200 | 3.54933 | 34.79 | 35.97% |
| **4800** | **629,145,600** | **3.52762** | **34.04** | **36.17%** |

The fixed validation curve is still improving at the latest recorded point. These are next-token metrics on one fixed 131,072-token validation sample; they are not downstream benchmark results.

The first public Hugging Face release remains the immutable **Anvaya-1B `160M`** checkpoint at update 1225 / 160,563,200 tokens. The training lineage is now much farther ahead than the public weight release. The latest major public milestone candidate is the preserved **u4879 / 639.50M-token** checkpoint.

## What this tutorial should explain

The useful outcome is a clear account of what worked, what failed, and what another person needs to check before attempting the same setup. That includes the exact token counter, the external data and tokenizer requirements, the PyTorch/Kaggle environment, FSDP ownership of model and optimizer state, and the crash-safe resume contract.

The target remains **1.5B model-input tokens**. That is enough to study the mechanics and evolution of the run, not enough to claim compute-optimal pretraining or a competitive base model. The public record will continue to separate measured training state from downstream capability claims.
