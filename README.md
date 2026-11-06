# Pretrain a 1B-Parameter Language Model From Scratch on 2× T4 GPUs



This repository documents a real pretraining experiment: a 1,018,126,848-parameter dense decoder-only Transformer, randomly initialized and trained on FineWeb-derived tokens using two 16GB NVIDIA Tesla T4 GPUs. The focus is the engineering around running a model at this size in a constrained, temporary environment: FSDP sharding, memory and token accounting, validation, checkpointing, and continuation across Kaggle sessions.

This is an educational case study, not a new training algorithm or architecture, a compute-optimal recipe, or a claim that the resulting model is competitive. The tokenizer is reused, not trained here. The 1.5B input-token target is a training-budget milestone; it does not mean every corpus token is seen once or establish downstream capability. The repository separates recorded measurements from assumptions and work still to be done.

## Live run status

The latest run snapshot supplied for this repository update is from **6 November 2026**; it is not a real-time status feed.

| Evidence | Latest recorded state |
|---|---|
| Training log | Update **2430** · **318,504,960** model-input tokens · **2,112.84** input tokens/second at that sampled update |
| PyTorch memory metric | Rank-0 allocator high-water mark **11.4088 GiB** through the captured continuation; see [memory and FSDP details](docs/03_memory_and_fsdp.md) |
| Validation | Update **2400** · **314,572,800** model-input tokens · NLL **3.99674** · perplexity **54.42** · token accuracy **32.09%** on **131,072** validation tokens |
| Last complete checkpoint | Update **2436** · **319,291,392** model-input tokens |

The latest sampled training observation in the newly supplied continuation log is update **2430**. The session then passed checkpoint preflight, wrote a COMPLETE checkpoint at update **2436**, and exited at a safe boundary. The latest fixed validation is update **2400**, with NLL **3.99674**, perplexity **54.42**, and token accuracy **32.09%** on 131,072 validation tokens. The continuous training lineage has therefore reached **319,291,392 model-input tokens** in a resumable state. The run has crossed both the **250M** and **300M** analytical milestones.

The live trainer copy supplied with this run and the repository trainer `train/train_1b_dual_t4.py` match byte-for-byte (SHA-256: `55a57185b100a8f63f1e3b77c5fd80726b952c8abe49b0a06f9aa5af5a83f440`).

## What I am training

The model starts with random weights and uses a causal next-token objective on tokenized FineWeb data. The tokenizer is reused; it was not trained as part of this project.

| Setting | Value |
|---|---:|
| Trainable parameters | **1,018,126,848** |
| Transformer blocks | 29 |
| Width | 1536 |
| Attention heads | 24 × 64 dimensions |
| FFN width | 6144 |
| Vocabulary | 128,256 |
| Context | 4096 tokens |
| Attention | Dense causal PyTorch SDPA |
| Normalization / position | RMSNorm / half-split RoPE |

The input and output embeddings share weights. The feed-forward block is a standard two-layer GELU MLP; it is not a gated SwiGLU block. See [the architecture notes](docs/01_architecture.md) and [parameter-count derivation](docs/02_parameter_count.md).

![Diagram of the model architecture](docs/assets/architecture.png)

## Hardware and batch size

The run uses two 16 GB NVIDIA Tesla T4 GPUs with PyTorch FSDP `FULL_SHARD`. Each rank processes one 4096-token sequence per microbatch. With 16 accumulation steps, that is:

```text
2 GPUs × 4096 tokens × 16 microbatches = 131,072 input tokens per update
```

The supplied log records a **rank-0 PyTorch allocator high-water mark of 11.4088 GiB** and about **2.1K input tokens/second** through the captured continuation. The memory value is not total device usage or free memory, and it is not a measurement from both ranks. The exact FSDP settings and a derived state-memory estimate are in [memory and FSDP](docs/03_memory_and_fsdp.md).

## Validation so far

Validation uses the same 131,072-token sample for each reported event. The table shows the update and actual number of model-input tokens processed at that point; it does not imply that every token in the training corpus has been seen exactly once.

| Update | Input tokens | Validation NLL | Perplexity | Token accuracy |
|---:|---:|---:|---:|---:|
| 2 | 262,144 | 10.74244 | 46,279.01 | 3.79% |
| 100 | 13,107,200 | 6.89855 | 990.83 | 12.09% |
| 616 | 80,740,352 | 5.40386 | 222.26 | 21.15% |
| 700 | 91,750,400 | 5.34038 | 208.59 | 21.51% |
| 800 | 104,857,600 | 5.24585 | 189.78 | 22.05% |
| 900 | 117,964,800 | 5.17369 | 176.57 | 22.52% |
| 1000 | 131,072,000 | 5.12850 | 168.76 | 22.94% |
| 1100 | 144,179,200 | 4.91807 | 136.74 | 23.95% |
| 1200 | 157,286,400 | 4.79223 | 120.57 | 24.85% |
| 1225 | **160,563,200** | **4.75070** | **115.67** | **25.21%** |
| 1300 | 170,393,600 | 4.64783 | 104.36 | 26.20% |
| 1400 | 183,500,800 | 4.54832 | 94.47 | 27.31% |
| 1500 | 196,608,000 | 4.48414 | 88.60 | 28.09% |
| 1600 | 209,715,200 | 4.40366 | 81.75 | 28.82% |
| 1700 | 222,822,400 | 4.33161 | 76.07 | 29.57% |
| 1800 | 235,929,600 | 4.27305 | 71.74 | 30.00% |
| 1829 | 239,730,688 | 4.26552 | 71.20 | 30.09% |
| 1900 | 249,036,800 | 4.23285 | 68.91 | 30.42% |
| 2000 | 262,144,000 | 4.18993 | 66.02 | 30.73% |
| 2100 | 275,251,200 | 4.12064 | 61.60 | 31.24% |
| 2200 | 288,358,400 | 4.07375 | 58.78 | 31.45% |
| 2300 | 301,465,600 | 4.02982 | 56.25 | 31.86% |
| **2400** | **314,572,800** | **3.99674** | **54.42** | **32.09%** |

![Validation negative log-likelihood by training tokens](results/validation_nll.svg)

The fixed validation curve has continued to improve through 314.57M model-input tokens, but these are next-token metrics on one validation sample. They do not establish downstream capability. A fixed downstream benchmark suite is still pending. The continuous training run is now at a clean **319.29M-token** checkpoint, while the **Anvaya-1B 160M** model-only FP16 checkpoint remains the current public weight release on Hugging Face at **[Vijayendra/Anvaya-1B](https://huggingface.co/Vijayendra/Anvaya-1B)** under the immutable `160M` revision.

## Public checkpoint generation verification

The public `160M` Hugging Face artifact has been loaded independently on an NVIDIA L4 and verified end to end. The released package reconstructs **1,018,126,848 parameters**, returns finite logits with shape `[1, 1, 128256]`, and generates recognizable English continuations.

At this early checkpoint, the model already shows local grammar and phrase structure, while factual associations and semantic stability remain weak. For example, open-ended prompts can produce sentence-like prose, but factual prompts such as `The capital of France is` are still unreliable. This is exactly why selected intermediate checkpoints are useful: they let the project observe the emergence of language behavior during pretraining rather than presenting only a final checkpoint.

The current public weights remain the immutable **160M** milestone even though the training lineage itself has advanced to **319.29M tokens**.

## Early behavioral comparison with Pythia-1B

I also ran a frozen generation suite against **EleutherAI/pythia-1b**, whose **1,011,781,632 parameters** make it an unusually close scale reference for Anvaya's **1,018,126,848** parameters.

The first comparison brackets Anvaya's current token exposure:

| Model | Approx. tokens | Optimizer updates |
|---|---:|---:|
| Pythia-1B step64 | 134.2M | 64 |
| **Anvaya-1B 160M** | **160.56M** | **1,225** |
| Pythia-1B step128 | 268.4M | 128 |

Under the same prompt suite, Anvaya at 160.56M tokens is already substantially more sentence-like than the tested Pythia step64 and step128 checkpoints. That is an interesting behavioral observation, **not an architecture-superiority claim**: Pythia uses a much larger global batch and its early checkpoints are still near the beginning of a 1% warmup over 143,000 iterations, while Anvaya uses a 75-update warmup and many more optimizer updates per comparable token exposure.

The full comparison, example generations, training-schedule caveats, and the proposed distinction between **token-matched** and **update-matched** comparisons are documented in **[Anvaya-1B vs Pythia-1B at early training](research/anvaya_vs_pythia_early_training.md)**.

## How the run continues across Kaggle sessions

Kaggle sessions are temporary, and a full FSDP checkpoint is large. The trainer resumes only from a two-rank checkpoint with a `complete.json` marker. It writes rank files to temporary names, renames them when serialization finishes, and creates the marker after both ranks synchronize. A mounted parent checkpoint remains untouched while the next session writes its output.

The first long run failed during a checkpoint save after update 100. That was a storage failure, not a training-loss failure. The current continuation design came out of that experience. More detail is in [checkpointing](docs/06_checkpointing.md) and [failure notes](docs/08_failure_modes.md).

## Try the recipe

The intended environment is a Kaggle notebook with **GPU T4 ×2** and the FineWeb-D1 token dataset attached. The corpus is external to this repository, and the Meta tokenizer may require accepting its license and authenticating with Hugging Face.

From a copy of the repository in `/kaggle/working`, run:

```bash
python train/launch_kaggle.py
```

The launcher reads `train/config_1b_t4.json`. For continuation, attach the output containing a complete checkpoint. With `resume_from` empty, the trainer searches mounted Kaggle inputs for a compatible checkpoint. The detailed setup and data contract are in [`data/prepare_fineweb.md`](data/prepare_fineweb.md) and [`docs/07_resume_across_sessions.md`](docs/07_resume_across_sessions.md).

A new stream rebuild follows the documented dataset contract; it is not guaranteed to reproduce the historical token files byte for byte. Exact continuation requires the original token shards and tokenizer files.

## Token budget and sampling

The nominal target is **1.5 billion model-input tokens**. The trainer samples deterministic 4096-token windows from individual training shards. It samples with replacement, so “1.5B tokens” is an update ledger, not a promise of one exact pass over all 1.485B corpus tokens. With whole updates, the run’s configured target is 11,445 updates, or **1,500,119,040** input tokens.

## What “from scratch” means here

The Transformer parameters are randomly initialized. No pretrained Transformer weights are loaded. I reuse the Llama 3.2 1B tokenizer, and the training corpus is prepared in advance. The tokenizer’s provenance and historical hashes are documented separately.

## Related projects

This project borrows the step-by-step spirit of [build-nanogpt](https://github.com/karpathy/build-nanogpt), but tackles the systems issues that appear when the model no longer fits as a replicated job. It is not a performance comparison with [TinyLlama](https://github.com/jzhang38/TinyLlama), which trained at a much larger token and hardware scale. [LitGPT](https://github.com/Lightning-AI/litgpt) is a useful reference for runnable training instructions; [OLMo](https://allenai.org/olmo) is a much broader example of an open model lifecycle.

## License

The MIT license in this repository covers the original code and documentation here. FineWeb and the tokenizer have their own terms; see [the source notes](research/SOURCES.md) before redistributing data or derived artifacts.
