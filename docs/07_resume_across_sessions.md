# Resume across Kaggle sessions

The Kaggle workflow is intentionally session-oriented:

```text
mount FineWeb-D1
mount previous notebook output
        ↓
auto-discover highest exact-lineage COMPLETE checkpoint
        ↓
restore model + optimizer + scaler + update counter
        ↓
validate immediately
        ↓
train until time guard
        ↓
write one new COMPLETE checkpoint
```

The auto-discovery logic checks the checkpoint configuration and ignores partial directories.

This makes ephemeral notebook sessions part of the reproducible training design rather than an operational afterthought.
