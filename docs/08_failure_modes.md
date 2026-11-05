# Failure modes and what they taught us

## 1. Periodic full-checkpoint storage failure

The initial long run reached update 100 with healthy training, then failed while serializing another multi-gigabyte FSDP checkpoint. The directory contained `.partial` rank files but no `complete.json`.

**Lesson:** model stability and persistence stability are different problems.

**Fix:** immutable mounted parent + one full save near the session time guard + COMPLETE-only resume.

## 2. Partial checkpoints

Never infer that “most of the bytes exist” means training state is valid.

**Rule:** no COMPLETE marker, no resume.

## 3. Duplicate console lines

In Kaggle, distributed subprocess output can appear duplicated even when only one two-worker training job is active.

**Rule:** verify global update/token progression and launcher ownership instead of treating duplicated text alone as evidence of duplicated training.

## 4. Exact token ledger

A GradScaler overflow would mean the optimizer update did not happen even though data microbatches were consumed.

**Rule:** the canonical trainer fails closed rather than silently advancing the token/update ledger after an overflow.
