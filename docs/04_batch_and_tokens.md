# Batch size and token accounting

The trainer processes one 4096-token sequence on each of two ranks per microbatch. It accumulates gradients over 16 microbatches before taking an optimizer step:

```text
4096 tokens × 2 ranks × 16 microbatches = 131,072 input tokens/update
```

The run counts model-input tokens. At update `u`, the ledger is `u × 131,072`. For example, update 800 corresponds to 104,857,600 input tokens.

The target is 1.5 billion input tokens. Since the trainer completes whole optimizer updates, `ceil(1,500,000,000 / 131,072)` gives 11,445 updates, or 1,500,119,040 input tokens. The final run will therefore pass the nominal target by 119,040 tokens.

This is not a sequential epoch counter. `Tokens.batch()` chooses a shard and offset deterministically from the update, microbatch, rank, and seed. Windows are sampled with replacement and do not cross shard boundaries. A token budget says how much input the model processed; it does not prove that every corpus token was visited exactly once.
