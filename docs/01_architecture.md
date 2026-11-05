# Architecture

The educational model is a standard decoder-only Transformer with dense causal attention.

| property | value |
|---|---:|
| trainable parameters | **1,018,126,848** |
| layers | 29 |
| model width | 1536 |
| attention heads | 24 |
| head dimension | 64 |
| FFN dimension | 6144 |
| vocabulary | 128,256 |
| context | 4096 |

The implementation uses RMSNorm, rotary position embeddings (RoPE), causal PyTorch SDPA, a gated/standard feed-forward stack as defined in the canonical trainer, and a causal next-token loss.

This repository deliberately keeps the dense educational line separate from custom-attention research.
