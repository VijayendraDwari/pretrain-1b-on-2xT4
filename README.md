# Pretrain a 1B-Parameter Language Model From Scratch on 2× T4 GPUs

**1.018B parameters · 4K context · FineWeb · PyTorch FSDP · 2×16GB Tesla T4 · target 1.5B tokens**

> **Status:** active pretraining project. The training lineage has passed 80.7M tokens and is continuing toward 1.5B. Final checkpoint and final benchmark table will be added when ready.

This repository is an educational, end-to-end recipe for taking a randomly initialized billion-parameter decoder-only Transformer through real web-corpus pretraining on two modest GPUs.

It is not a claim that two T4s are the fastest way to train a 1B model. They are not. The point is to make the systems mechanics visible.

## Why this project

Small GPT tutorials make model architecture wonderfully understandable, but can hide the systems problems that appear at larger scales. Production LLM projects solve those problems on clusters that most individual researchers cannot reproduce.

At ~1B parameters on 2×16GB T4s, the important engineering becomes unavoidable:

- parameter and optimizer-state sharding;
- gradient accumulation;
- exact token accounting;
- mixed precision;
- validation during pretraining;
- multi-gigabyte distributed checkpoints;
- crash-safe resume;
- continuation across ephemeral Kaggle sessions.

## Model

| field | value |
|---|---:|
| exact trainable parameters | **1,018,126,848** |
| layers | 29 |
| width | 1536 |
| attention heads | 24 |
| head dimension | 64 |
| FFN | 6144 |
| vocabulary | 128,256 |
| context | 4096 |
| attention | dense causal SDPA |
| objective | next-token causal LM |
| initialization | random |

## Hardware and distributed system

```text
2 × NVIDIA Tesla T4
16 GB VRAM / GPU
PyTorch FSDP FULL_SHARD
gradient accumulation = 16
global tokens/update = 131,072
```

Observed allocation during the documented run is about **11.4 GiB/GPU**, with steady training throughput around **2.1K input tokens/sec**.

## Current learning curve

| tokens | validation NLL | perplexity | token accuracy |
|---:|---:|---:|---:|
| 262K | 10.7424 | 46,279 | 3.79% |
| 13.1M | 6.8985 | 990.8 | 12.09% |
| 80.7M | 5.4039 | 222.3 | 21.15% |

Target: **1.5B input tokens**.

These numbers measure pretraining progress. They are not claims of competitive downstream performance.

## Quick start on Kaggle

1. Select **GPU T4 ×2**.
2. Attach FineWeb-D1, or prepare a compatible corpus with `data/prepare_fineweb.py`.
3. If continuing, attach the previous notebook output containing a COMPLETE checkpoint.
4. Copy this repository into the notebook environment.
5. Run:

```bash
python train/launch_kaggle.py
```

The trainer auto-discovers the highest exact-lineage COMPLETE checkpoint under `/kaggle/input` unless `resume_from` is explicitly configured.

A correct resume looks like:

```text
[resume] .../checkpoints/update_00000616 update=616 tokens=80,740,352
[continuation policy] immutable mounted COMPLETE parent; ...
```

## Crash-safe checkpoints

The first long run exposed a real failure mode: training was healthy at update 100, but a periodic multi-gigabyte FSDP save exhausted the persistence path.

The fix became part of the public recipe:

```text
mounted immutable COMPLETE parent
            ↓
         exact resume
            ↓
       train session
            ↓
 one full save near time guard
            ↓
  atomic COMPLETE marker
```

Partial rank files are never resumable.

See [`docs/06_checkpointing.md`](docs/06_checkpointing.md) and [`docs/08_failure_modes.md`](docs/08_failure_modes.md).

## Repository map

```text
train/     canonical trainer, launcher, configuration
data/      FineWeb builder + dataset/tokenizer contracts
eval/      fixed LM evaluation path and protocol
docs/      architecture and systems explanations
results/   milestone tables and raw curves
notebooks/ Kaggle launch notebook
releases/  checkpoint/release ledger
research/  prior-art notes and references
```

## Reproduce the dataset contract

Historical D1:

```text
HuggingFaceFW/fineweb @ v1.4.0
CC-MAIN-2025-26
1.485B train tokens
15M validation tokens
24 train shards
uint32 token IDs
vocab 128,256
```

See [`data/prepare_fineweb.md`](data/prepare_fineweb.md).

## Milestones

- [x] random initialization
- [x] distributed qualification checkpoint
- [x] crash-safe continuation design
- [x] 80.7M-token COMPLETE checkpoint
- [ ] 100M analytical milestone
- [ ] 250M
- [ ] 500M
- [ ] 1B
- [ ] **1.5B final target**
- [ ] final checkpoint release
- [ ] final fixed benchmark report

## What “from scratch” means here

No pretrained Transformer weights are loaded. The model starts from random parameters and is optimized using a causal next-token objective on tokenized FineWeb data.

The tokenizer is reused rather than trained from scratch; its provenance and exact historical hashes are documented.

## What this project does **not** claim

A 1B model trained on 1.5B tokens is undertrained relative to modern small-LLM production recipes.

If your goal is simply to obtain a capable 1B model, downloading an existing pretrained model is much more sensible.

If your goal is to understand how billion-parameter pretraining actually works—including the failures—this repository is for you.

## Prior art

This project is inspired by the openness and educational clarity of:

- Andrej Karpathy — [nanoGPT](https://github.com/karpathy/nanoGPT) and [build-nanogpt](https://github.com/karpathy/build-nanogpt)
- Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu — [TinyLlama](https://github.com/jzhang38/TinyLlama)
- Allen Institute for AI — [OLMo](https://allenai.org/olmo)
- Lightning AI — [LitGPT](https://github.com/Lightning-AI/litgpt)

See [`research/prior_art.md`](research/prior_art.md) for positioning notes.

## Author

**Vijayendra Dwari**

## License

MIT for the original code and documentation in this repository. Upstream datasets and tokenizers retain their own licenses and terms.
