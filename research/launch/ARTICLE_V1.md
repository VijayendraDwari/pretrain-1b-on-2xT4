# Pretraining a 1B-Parameter Language Model on Two T4 GPUs: An Engineering Walkthrough

Modern language-model tutorials tend to live at two extremes.

At one end are beautifully small GPT implementations that can be understood in a notebook. At the other are production pretraining stacks designed around clusters of high-end accelerators.

I wanted to explore the engineering space in between.

What changes when the model is no longer a toy, but the hardware is still something an individual researcher can realistically access?

That question led to this project: a **1,018,126,848-parameter decoder-only language model**, initialized from random weights and pretrained on FineWeb-derived tokens using **two NVIDIA Tesla T4 GPUs with 16 GB of memory each**.

The model uses a 4096-token context. The training system uses PyTorch FSDP `FULL_SHARD`, activation checkpointing, mixed precision, gradient accumulation, deterministic token-window sampling, fixed validation, and crash-safe distributed checkpoint continuation across temporary Kaggle sessions.

The target is **1.5 billion model-input tokens**.

This is not an attempt to produce a compute-optimal 1B foundation model. Modern small language models are trained on vastly larger token budgets and much faster hardware. If the only objective were to obtain a useful 1B model, downloading an existing pretrained model would be the rational choice.

The purpose here is educational and systems-oriented:

> **What does billion-parameter pretraining actually require when the infrastructure can no longer be taken for granted?**

## Why 1B parameters?

A smaller Transformer can teach the architecture very well. But many of the engineering problems that dominate real pretraining remain invisible.

At roughly one billion parameters on two 16 GB GPUs, they stop being optional.

The model has:

| Setting | Value |
|---|---:|
| Trainable parameters | **1,018,126,848** |
| Transformer blocks | 29 |
| Width | 1536 |
| Attention heads | 24 |
| Head dimension | 64 |
| FFN width | 6144 |
| Vocabulary | 128,256 |
| Context | 4096 |

The architecture is intentionally conventional. Each block uses pre-normalized residual branches, dense causal attention through PyTorch SDPA, half-split RoPE on Q and K, and a two-layer GELU feed-forward network.

The token embedding matrix is tied to the output projection.

This separation matters. The goal of this project is not to test a new attention mechanism. It is to make the **training system** inspectable.

## The first constraint: memory

A billion FP32 parameters alone represent roughly four gigabytes of storage.

Training requires substantially more:

- parameters;
- gradients;
- the first AdamW moment;
- the second AdamW moment;
- mixed-precision working copies;
- activations;
- attention and MLP intermediates;
- communication buffers;
- CUDA and framework overhead.

A naïve replicated data-parallel job is therefore the wrong execution model for two T4s.

The run uses PyTorch FSDP with `FULL_SHARD`. Parameters, gradients and optimizer state are distributed across the two ranks, while each wrapped module gathers the full parameters required for its computation.

The measured rank-0 PyTorch allocator high-water mark in the current run is approximately **11.41 GiB**.

That figure is deliberately reported as an allocator measurement, not as total device usage. The repository separately documents the arithmetic estimate for persistent sharded FP32 parameters, gradients and AdamW state.

One lesson became clear very quickly:

> **Model size alone does not determine whether training fits. State ownership and tensor lifetime do.**

## Batch size is really token accounting

Each rank processes one 4096-token sequence per microbatch.

There are two ranks and 16 gradient-accumulation microbatches per optimizer step.

So:

```text
4096 × 2 × 16 = 131,072 model-input tokens per optimizer update
```

That number becomes the ledger for the complete run.

Update 800, for example, corresponds exactly to:

```text
800 × 131,072 = 104,857,600 tokens
```

The configured 1.5B-token target is reached at the next whole-update boundary, so the eventual ledger target is **1,500,119,040 input tokens**.

The word “tokens” here is important. The training sampler draws deterministic 4096-token windows from the token shards with replacement. Therefore 1.5B model-input tokens is not the same thing as claiming that every corpus token was visited exactly once.

## The data contract

The historical corpus used by the run contains:

- 1.485B training tokens;
- 15M validation tokens;
- 24 training shards;
- little-endian uint32 token IDs;
- vocabulary size 128,256.

The source is the FineWeb `CC-MAIN-2025-26` configuration, with the Llama 3.2 1B tokenizer family.

The project publishes a data-preparation script and the historical tokenizer hashes.

It also makes an important distinction: a fresh streamed rebuild follows the documented corpus contract, but is not automatically assumed to be byte-identical to the historical dataset.

Reproducibility is not just a dataset name.

It is dataset identity, tokenizer identity, code identity and checkpoint identity.

## The checkpoint failure that changed the project

The first long training session reached update 100 normally.

Validation loss was falling. The model was healthy.

Then the checkpoint save failed.

The problem was not numerical instability. The persistence path ran out of room while serializing the distributed state.

The directory contained partial rank files. It looked almost like a checkpoint.

But “almost” is not sufficient for a training lineage.

That failure led to a much stricter continuation rule:

1. each rank first writes `rankN.pt.partial`;
2. the rank file is atomically renamed only after serialization completes;
3. both ranks synchronize;
4. rank 0 writes `complete.json`;
5. only a directory containing the expected rank files **and** the completion marker may be used as a parent.

The mounted parent checkpoint remains immutable while the current Kaggle session writes a new state.

If the session dies midway through the save, the incomplete child is ignored and the previous parent remains a valid rollback point.

This turned out to be one of the most educational parts of the experiment.

A training recipe is not reproducible merely because it can start.

It must also define what happens when the machine disappears.

## The learning curve so far

The current run has reached a **clean COMPLETE checkpoint at 160,563,200 model-input tokens**.

The fixed validation measurements currently recorded are:

| Update | Input tokens | Validation NLL | Perplexity | Token accuracy |
|---:|---:|---:|---:|---:|
| 2 | 262,144 | 10.7424 | 46,279 | 3.79% |
| 100 | 13.11M | 6.8985 | 990.8 | 12.09% |
| 616 | 80.74M | 5.4039 | 222.3 | 21.15% |
| 700 | 91.75M | 5.3404 | 208.6 | 21.51% |
| 800 | 104.86M | 5.2458 | 189.8 | 22.05% |
| 900 | 117.96M | 5.1737 | 176.6 | 22.52% |
| 1000 | 131.07M | 5.1285 | 168.8 | 22.94% |
| 1100 | 144.18M | 4.9181 | 136.7 | 23.95% |
| 1200 | 157.29M | **4.7922** | **120.6** | **24.85%** |

The important observation is not a single batch loss. It is the direction of the fixed validation curve. The latest captured session then passed checkpoint preflight, wrote a COMPLETE checkpoint at update 1225, and exited at a safe boundary.

At the same time, these numbers are still only next-token metrics on the project's fixed validation sample. They are **not** downstream benchmark results and do not establish general model capability.

Keeping those claims separate is part of the project.

## Why two T4s?

The T4 is not an ideal pretraining GPU.

That is precisely why the constraint is useful.

A faster accelerator can make bad systems assumptions less visible. On two 16 GB devices, decisions around FSDP wrapping, mixed precision, activation checkpointing, optimizer state, vocabulary-loss memory and persistence become impossible to ignore.

The result is not a throughput record. Current training throughput is roughly **2.1K model-input tokens per second**.

The value is that every trade-off is visible enough to study.

## What “from scratch” means

The Transformer weights begin from random initialization.

No pretrained Transformer weights are loaded.

The tokenizer is reused, and the corpus is prepared before training. That is documented explicitly because “from scratch” can otherwise hide several different meanings.

The claim here is therefore narrow:

> **the language model itself is pretrained from random parameters using a causal next-token objective.**

## Why publish before 1.5B?

Because the systems recipe already exists.

Waiting until the final checkpoint would hide the actual process that makes the project valuable: continuation, failures, fixes, validation history and the changing training state.

The public repository therefore separates three frontiers:

- the latest observed training update;
- the latest validation event;
- the latest COMPLETE resumable checkpoint.

Those are not always the same thing.

When the final target is reached, the repository can add the final checkpoint information, final validation result, total compute accounting and a versioned downstream evaluation suite.

But the engineering path is useful now.

## What the repository contains

The public project includes:

- the canonical 1.018B dual-T4 trainer;
- Kaggle launcher and notebook;
- FineWeb corpus builder and dataset contract;
- tokenizer provenance and hashes;
- architecture and exact parameter arithmetic;
- detailed FSDP and memory notes;
- optimizer and learning-rate schedule;
- crash-safe checkpoint protocol;
- restart semantics across Kaggle sessions;
- fixed validation protocol;
- raw milestone CSVs and plots;
- failure notes and limitations.

The repository is:

**VijayendraDwari/pretrain-1b-on-2xT4**

## What this experiment can—and cannot—claim

It demonstrates that a real 1.018B-parameter dense language model can be continuously pretrained at 4K context on two 16 GB T4s using a carefully designed distributed training and persistence system.

It does **not** demonstrate:

- compute-optimal 1B training;
- competitive performance against modern production small language models;
- a new Transformer architecture;
- a speed advantage;
- useful downstream capability before those evaluations are actually run.

Those limitations are not footnotes. They define the experiment.

## What comes next

The training target remains 1.5B model-input tokens.

I plan to keep the public record updated at major stages:

- 250M tokens;
- 500M tokens;
- 1B tokens;
- 1.5B tokens.

Once the final training state exists, I can publish a fixed downstream evaluation, final systems accounting and—if appropriate—the model checkpoint itself.

For now, the most useful artifact is already public:

**the full path from random initialization to a working billion-parameter pretraining system on constrained hardware.**
