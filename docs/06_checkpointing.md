# Crash-safe checkpointing

A checkpoint is resumable **only** if its directory contains:

```text
update_XXXXXXXX/
├── rank0.pt
├── rank1.pt
└── complete.json
```

Each rank first writes `rankN.pt.partial` and atomically renames it only after serialization completes. Rank 0 writes `complete.json` only after both workers synchronize.

A directory containing only `.partial` files is explicitly non-resumable.

The current session keeps one newly completed checkpoint. The mounted parent is immutable and therefore remains the rollback point until the new COMPLETE marker exists.
