# LinkedIn launch post

Can you actually pretrain a billion-parameter language model from scratch on two 16 GB T4 GPUs?

I wanted to find out — not with a toy GPT, and not by hiding the hard parts behind a large training cluster.

So I built a **1,018,126,848-parameter** decoder-only language model and started pretraining it from random initialization on FineWeb-derived tokens using **2× NVIDIA Tesla T4 GPUs**.

The model itself is deliberately conventional:

- 29 Transformer blocks
- width 1536
- 24 attention heads
- 4096-token context
- dense causal PyTorch SDPA
- RMSNorm + RoPE
- standard GELU MLP
- tied input/output embeddings

The interesting part is the training system.

At this scale on 16 GB GPUs, the problems become very real:

- FSDP `FULL_SHARD`
- activation checkpointing
- 131,072 model-input tokens per optimizer update
- FP16 compute with FP32 reductions
- multi-gigabyte distributed checkpoints
- exact continuation across temporary Kaggle sessions
- crash-safe COMPLETE-only resume semantics
- data/tokenizer lineage and reproducibility

The first long run actually failed while writing a checkpoint.

Training itself was healthy. Persistence was not.

That failure became one of the most useful parts of the project: the continuation design now keeps the mounted parent immutable, writes rank files transactionally, and refuses to resume from incomplete state.

The run has now crossed **500M model-input tokens** and reached a clean COMPLETE checkpoint at:

**u4879 / 639,500,288 tokens**

The latest fixed validation is:

**u4800 / 629,145,600 tokens → NLL 3.52762 / PPL 34.04 / token accuracy 36.17%**

For context, the fixed validation trajectory includes:

- 160.56M → NLL 4.7507 / PPL 115.7
- 239.73M → NLL 4.2655 / PPL 71.2
- 319.29M → NLL 3.9841 / PPL 53.74
- 397.28M → NLL 3.8132 / PPL 45.30
- 476.84M → NLL 3.6876 / PPL 39.95
- 559.55M → NLL 3.5854 / PPL 36.07
- 602.93M → NLL 3.5493 / PPL 34.79
- 629.15M → **NLL 3.5276 / PPL 34.04**

The latest completed continuation resumed exactly from u4269 / 559.55M, validated the restored state, crossed 600M, continued without changing the recipe, and stopped at a safe boundary after writing u4879.

The target remains **1.5B model-input tokens**.

This is not a claim that 1.5B tokens is compute-optimal for a 1B model. It is not a speed record, and it is not intended to compete with heavily pretrained production models.

The purpose is different:

**to expose the complete engineering path from random initialization to real billion-parameter pretraining on hardware an individual researcher can actually access.**

The full recipe is public on GitHub:

**VijayendraDwari/pretrain-1b-on-2xT4**

The repository includes the canonical trainer, FineWeb data-preparation contract, PyTorch FSDP setup, Kaggle continuation workflow, checkpoint recovery design, architecture and parameter-count notes, memory accounting, evaluation protocol, raw validation/training curves, and the failures encountered along the way.

The model now also has a public name: **Anvaya-1B**. The first immutable Hugging Face weight release remains the earlier **`160M`** milestone at 160,563,200 tokens; the training lineage itself is now at 639.50M. The preserved u4879 state is the latest major public checkpoint candidate.

Next major training milestones: **750M, 1B, 1.25B and 1.5B model-input tokens**.

If you work on model pretraining, distributed PyTorch, constrained-compute research, or simply want to understand what changes when a GPT stops being a toy, I’d value your feedback.
