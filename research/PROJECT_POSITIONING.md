# How I describe this project

## Working title

**Pretrain a 1B-Parameter Language Model From Scratch on 2× T4 GPUs**

For a longer article, I am using **Pretraining a 1B-Parameter Language Model From Scratch on Two T4 GPUs: An Engineering Walkthrough**.

## Canonical public description

**A reproducible recipe for pretraining a 1B-parameter language model from scratch on 2×16GB NVIDIA T4 GPUs using FineWeb-derived data, PyTorch FSDP, 4K context, fixed validation, and crash-safe Kaggle continuation.**

Use “from scratch” only with the explicit qualification that the **Transformer weights start from random initialization while the tokenizer is reused**. Prefer “LLM pretraining recipe” or “language-model pretraining recipe” when describing the repository to a general technical audience; the project is an engineering walkthrough rather than a new pretraining algorithm.

## The project in one paragraph

I am training a 1,018,126,848-parameter decoder-only Transformer from random initialization on two 16 GB T4 GPUs. The repository documents the complete systems path closely enough that another reader can inspect or reproduce the recipe: FineWeb-derived token preparation, model sizing, PyTorch FSDP `FULL_SHARD`, gradient accumulation, activation checkpointing, optimizer and learning-rate configuration, fixed validation, distributed checkpointing, and continuation across Kaggle sessions. The tokenizer is reused, and the training data is prepared before the run.

## What I am not claiming

This is not a compute-optimal recipe, a speed record, or a competitive 1B model. The target of 1.5B input tokens is intentionally small compared with serious production pretraining. I am not claiming that 1B-parameter pretraining itself is novel.

The project is useful if the reader wants to understand the system around a training loop: what state must fit in memory, how the token counter is defined, what happens when a large checkpoint save fails, and how a later session resumes.

## The scale and the comparison

The model is large enough that data, memory, and persistence choices show up in everyday training. It is still small enough that the full path fits in a single repository and a tutorial can explain it without assuming access to a large cluster.

That makes this a different kind of project from TinyLlama, which trained a 1.1B model on a much larger token budget and a multi-GPU A100 setup. It is also narrower than OLMo's open model lifecycle. I use build-nanogpt as a reference for explaining code step by step, and LitGPT as a reference for clear runnable instructions. The links and notes are in [prior art](prior_art.md).

## How I will keep the claims current

The public status should track the latest log snapshot, not imply that a training-log update is a saved checkpoint. I will report the update number, model-input token count, validation event, and whether the state is actually resumable. If a metric or a claim cannot be traced to a run log or artifact, I will leave it out or label it as unverified.

The current validation curve measures next-token prediction on the fixed validation sample. It is not evidence of useful behavior on downstream tasks. I will only add such claims if a fixed benchmark or generation evaluation is added and documented.
