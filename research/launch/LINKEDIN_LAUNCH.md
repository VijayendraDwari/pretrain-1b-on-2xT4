# LinkedIn launch post

Can you actually pretrain a billion-parameter language model from scratch on two 16 GB T4 GPUs?

I wanted to find out — not with a toy GPT, and not by hiding the hard parts behind a large training cluster.

So I built a 1,018,126,848-parameter decoder-only language model and started pretraining it from random initialization on FineWeb-derived tokens using 2× NVIDIA Tesla T4 GPUs.

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

- FSDP FULL_SHARD
- activation checkpointing
- 131,072 input tokens per optimizer update
- FP16 compute with FP32 reductions
- multi-gigabyte distributed checkpoints
- exact continuation across temporary Kaggle sessions
- crash-safe COMPLETE-only resume semantics
- data/tokenizer lineage and reproducibility

The first long run actually failed while writing a checkpoint.

Training itself was healthy. Persistence was not.

That failure became one of the most useful parts of the project: the continuation design now keeps the mounted parent immutable, writes rank files transactionally, and refuses to resume from incomplete state.

The run has now passed 140M model-input tokens.

The validation trajectory so far:

- 80.74M tokens → NLL 5.4039 / PPL 222.3
- 91.75M → NLL 5.3404 / PPL 208.6
- 104.86M → NLL 5.2458 / PPL 189.8
- 117.96M → NLL 5.1737 / PPL 176.6
- 131.07M → NLL 5.1285 / PPL 168.8

The target is 1.5B model-input tokens.

This is not a claim that 1.5B tokens is compute-optimal for a 1B model. It is not a speed record, and it is not intended to compete with heavily pretrained production models.

The purpose is different:

**to expose the complete engineering path from random initialization to real billion-parameter pretraining on hardware an individual researcher can actually access.**

I have now made the full recipe public on GitHub:

**VijayendraDwari/pretrain-1b-on-2xT4**

The repository includes the canonical trainer, FineWeb data-preparation contract, PyTorch FSDP setup, Kaggle continuation workflow, checkpoint recovery design, architecture and parameter-count notes, memory accounting, evaluation protocol, validation curves, and the failures encountered along the way.

I’ll keep updating it as the run progresses through 250M, 500M, 1B and finally 1.5B tokens.

If you work on model pretraining, distributed PyTorch, constrained-compute research, or simply want to understand what changes when a GPT stops being a toy, I’d value your feedback.
