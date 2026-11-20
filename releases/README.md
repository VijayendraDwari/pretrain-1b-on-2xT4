# Checkpoint releases

The reference training run is complete at **u11445 / 1,500,119,040 model-input tokens**. Raw two-rank FSDP checkpoints are training artifacts; public Hugging Face releases should be model-only exports with exact provenance back to a COMPLETE training checkpoint.

The existing public release remains **Anvaya-1B `160M`**, exported from **u1225 / 160,563,200 tokens**. The later checkpoints below are trained and preserved but should be marked public only after each export has been reconstructed and independently load-tested.

## Recommended public series

| HF revision | Source checkpoint | Exact input tokens | Status / role |
|---|---:|---:|---|
| `160M` | u1225 | 160,563,200 | **Public** — early-language checkpoint |
| `240M` | u1829 | 239,730,688 | Recommended — early emergence |
| `320M` | u2436 | 319,291,392 | Recommended — early/mid curve |
| `400M` | u3031 | 397,279,232 | Recommended — near-400M milestone |
| `560M` | u4269 | 559,546,368 | Recommended — post-500M milestone |
| `800M` | u6042 | 791,937,024 | Recommended — mid-run checkpoint |
| `1B` | u7879 | 1,032,716,288 | Recommended — first clean checkpoint beyond 1B |
| `1.27B` | u9712 | 1,272,971,264 | Recommended — late-training checkpoint |
| `1.43B` | u10917 | 1,430,913,024 | Optional — near-final comparison |
| `1.5B-final` | **u11445** | **1,500,119,040** | **Recommended final base checkpoint** |

Friendly revision names are labels; every model card must state the exact update and token ledger.

## Minimum export contract

Each public revision should include:

1. model-only weights, preferably safetensors;
2. exact architecture/configuration;
3. tokenizer files and tokenizer identity;
4. source update and exact model-input-token count;
5. fixed-validation NLL, perplexity, token accuracy and validation-token count;
6. export checksum(s);
7. a short note that the model was pretrained from random Transformer initialization on FineWeb-derived data;
8. a generation smoke test showing finite logits and successful autoregressive generation.

Do not publish raw FSDP rank checkpoints as the user-facing model package. Preserve them separately for training reproducibility.
