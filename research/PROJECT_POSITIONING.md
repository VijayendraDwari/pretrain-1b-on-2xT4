# Project Positioning

## Working identity

**Repository:** `pretrain-1b-on-2xT4`

**README title:**  
# Pretrain a 1B-Parameter Language Model From Scratch on 2× T4 GPUs

**Subtitle:**  
*A reproducible 1.018B-parameter, 4K-context PyTorch/FSDP pretraining recipe on FineWeb — from random initialization to 1.5B training tokens.*

**Article title:**  
# Pretraining a 1B-Parameter Language Model on Two T4 GPUs: An End-to-End Engineering Guide

## What the project is

An educational, reproducible demonstration of the full pretraining path for a real billion-parameter decoder-only language model on modest hardware.

The project explains:

- architecture sizing and exact parameter counting;
- tokenized web-data ingestion;
- causal language-model training from random initialization;
- PyTorch FSDP `FULL_SHARD` on 2×16GB Tesla T4 GPUs;
- gradient accumulation and global token-batch arithmetic;
- learning-rate scheduling;
- validation during pretraining;
- crash-safe sharded checkpointing;
- exact continuation across notebook sessions;
- training progress from initialization toward 1.5B input tokens.

## What the project is not

It is **not** a claim that 1.5B training tokens is compute-optimal for a 1B model.

It is **not** a claim that two T4 GPUs are the fastest or cheapest way to obtain a useful 1B model.

It is **not** intended to compete with heavily pretrained production 1B models.

The purpose is to make billion-parameter pretraining mechanics understandable and reproducible.

## Core educational thesis

> Small GPT tutorials teach model architecture. Production foundation-model projects hide much of the training system behind large clusters. This project explores the middle ground: a model large enough that distributed systems engineering becomes unavoidable, but still small enough for an individual researcher to understand end to end.

## Why “1B-parameter” is appropriate

The exact model contains **1,018,126,848 trainable parameters**. In standard model naming practice, this is naturally described as a **1B-parameter model**. Use the exact count prominently in the subtitle/specification table.

“Billion-parameter-class” is useful in prose when discussing the scale category, but is weaker as a searchable title.

## Differentiator

The novelty is not the existence of a 1B Transformer. The differentiating combination is:

- 1.018B real decoder-only model;
- random initialization;
- 4096-token context;
- real FineWeb pretraining data;
- only 2× Tesla T4 16GB GPUs;
- FSDP FULL_SHARD;
- checkpoint/resume across constrained Kaggle sessions;
- a complete educational explanation of the systems mechanics.

## Recommended tone

Engineering-focused, transparent, and non-hyped.

Prefer:
> “Here is the complete path I used to make billion-parameter pretraining work on modest hardware.”

Avoid:
> “I trained a competitive 1B LLM on two cheap GPUs.”
