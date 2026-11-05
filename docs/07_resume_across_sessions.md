# Continue the run in another Kaggle session

The run is split across notebook sessions because the GPU session is temporary. The basic handoff is:

1. Attach the FineWeb-D1 token dataset.
2. Attach the previous notebook output that contains a complete checkpoint.
3. Start the trainer with `resume_from` empty, or point it to a checkpoint explicitly.
4. Let the trainer validate the restored state, then continue from its saved update number.
5. Keep the new output in the writable session directory until a new complete checkpoint is available.

With automatic discovery, the trainer scans mounted Kaggle inputs and chooses the highest update whose metadata matches the configured model dimensions, context, accumulation, seed, target, and world size. It requires both rank files and a `complete.json` marker. The selected rank restores its own model shard, optimizer state, GradScaler state, and update counter.

The next data windows are derived from the update number, microbatch, rank, and seed. This preserves the logical token/update ledger across a restart. It is not a promise of bit-for-bit identical GPU math, and the current checkpoint fingerprint does not include hashes for the corpus, tokenizer, or trainer source. For the current identity checks, see the trainer's `_checkpoint_config_expected()` function.
