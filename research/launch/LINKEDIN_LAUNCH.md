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

The run has now **completed its configured 1.5B-token training budget** and reached the final clean COMPLETE checkpoint at:

**u11445 / 1,500,119,040 model-input tokens**

Final fixed validation on the same 131,072-token sample:

- NLL **3.21237**
- perplexity **24.84**
- token accuracy **39.36%**

The final continuation resumed from u10917 / 1.43091B tokens, saved u11445, validated the exact final state, and stopped normally because the configured target had been reached.

This is not a claim that 1.5B tokens is compute-optimal for a 1B model. It is not a speed record, and it is not intended to compete with heavily pretrained production models.

The purpose is different:

**to expose the complete engineering path from random initialization to real billion-parameter pretraining on hardware an individual researcher can actually access.**

The full recipe is public on GitHub:

**VijayendraDwari/pretrain-1b-on-2xT4**

The repository includes the canonical trainer, FineWeb data-preparation contract, PyTorch FSDP setup, Kaggle continuation workflow, checkpoint recovery design, architecture and parameter-count notes, memory accounting, evaluation protocol, raw validation/training curves, and the failures encountered along the way.

The model also has a public name: **Anvaya-1B**. The selected training trajectory is now public as immutable Hugging Face revisions: **160M, 240M, 320M, 400M, 560M, 800M, 1B, 1.2B, 1.35B, and 1.5B-final**. The default revision is the completed u11445 / 1,500,119,040-token model.

If you work on model pretraining, distributed PyTorch, constrained-compute research, or simply want to understand what changes when a GPT stops being a toy, I’d value your feedback.
