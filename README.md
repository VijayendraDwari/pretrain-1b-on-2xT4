# Pretrain a 1B-Parameter Language Model From Scratch on 2× T4 GPUs

I wanted to see how far I could take a randomly initialized, billion-parameter decoder-only model on two modest GPUs, and to write down the parts that small tutorials usually skip: sharding, token accounting, validation, and surviving a notebook session ending halfway through a run.

This is a training experiment and an engineering tutorial. It is not a claim that the resulting model is competitive, or that two T4s are an efficient way to get a useful 1B model.

## Run status

**Training is in progress.** The latest log snapshot I have added here is from 5 November 2026. It shows the continuation at update **820** (**107,479,040 input tokens**). The most recent validation in that snapshot is update **800** (**104,857,600 input tokens**): NLL **5.24585**, perplexity **189.78**, and token accuracy **22.05%**.

Update 820 is a training-log observation, not a saved checkpoint. The last complete parent checkpoint used for this continuation was update 616. The run writes a new checkpoint at a safe session boundary; I’ll update this page when that happens and when the next validation results are available.

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

The captured training logs report about **11.41 GiB** peak allocated memory per GPU and roughly **2.1K input tokens/second** at the current setup. These are measurements from this run, not a hardware benchmark.

## Validation so far

Validation uses the same 131,072-token sample for each reported event. The table shows the update and actual number of model-input tokens processed at that point; it does not imply that every token in the training corpus has been seen exactly once.

| Update | Input tokens | Validation NLL | Perplexity | Token accuracy |
|---:|---:|---:|---:|---:|
| 2 | 262,144 | 10.74244 | 46,279.01 | 3.79% |
| 100 | 13,107,200 | 6.89855 | 990.83 | 12.09% |
| 616 | 80,740,352 | 5.40386 | 222.26 | 21.15% |
| 700 | 91,750,400 | 5.34038 | 208.59 | 21.51% |
| 800 | 104,857,600 | 5.24585 | 189.78 | 22.05% |

![Validation negative log-likelihood by training tokens](results/validation_nll.png)

The curve is moving in the expected direction, but these are next-token metrics on one validation sample. They do not establish downstream capability. I have not yet run a fixed benchmark suite or released model weights.

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
