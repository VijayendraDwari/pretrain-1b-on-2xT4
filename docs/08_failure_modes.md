# Problems encountered so far

## A checkpoint save ran out of room

The first long run reached update 100 with normal training and validation behavior, then failed while writing a large FSDP checkpoint. Rank files were left as partial files; there was no `complete.json` marker.

The run could not safely resume from that directory. The continuation setup now keeps the mounted parent read-only and writes one new checkpoint near the session time guard. A checkpoint is eligible for resume only after both rank files and the marker are present.

## Kaggle output sometimes repeats console lines

The captured notebook output contains duplicate copies of several rank-zero messages. The update numbers and token counts still move forward once per step in the supplied log. When checking a run, compare the update ledger and checkpoint state rather than counting repeated display lines.

## The token ledger depends on a real optimizer update

If GradScaler reports an overflow, the optimizer step may be skipped even though the microbatches have been read. The trainer stops on overflow instead of advancing the update and token counters as though an update had happened.

These safeguards make interruption handling clearer. They do not replace external verification of checkpoint bytes, data identity, or the exact runtime environment; those checks are not yet part of the published lineage.
