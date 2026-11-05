# Batch and token accounting

Canonical arithmetic:

```text
context length       4096
world size           2
gradient accumulation 16

4096 × 2 × 16 = 131,072 input tokens / optimizer update
```

This repository treats model-input tokens as the primary training ledger.

Examples:

```text
update 2   =      262,144 tokens
update 100 =   13,107,200 tokens
update 616 =   80,740,352 tokens
update 763 =  100,007,936 tokens
```

The final target is 1.5B input tokens. Because 1.5B is not divisible by 131,072, the trainer targets the corresponding whole-update boundary and reports exact realized tokens.
