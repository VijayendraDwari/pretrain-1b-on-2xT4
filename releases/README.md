# Checkpoint releases

There is no public model checkpoint in this repository yet. The update-616 checkpoint was used as the parent for the current continuation, but it has not been packaged here for download. The shared run log now shows training through update 1070; that log entry is not itself a resumable checkpoint.

| Stage | Current record | Public checkpoint |
|---|---|---|
| Update 616 / 80.74M tokens | Complete parent used for continuation | Not released |
| Update 800 / 104.86M tokens | First validation after crossing 100M | Not released |
| Update 900 / 117.96M tokens | Validation NLL 5.17369 | Not released |
| Update 1000 / 131.07M tokens | Validation NLL 5.12850 | Not released |
| Update 1070 / 140.25M tokens | Latest shared training-log snapshot | Not a checkpoint |
| 250M, 500M, 1B, 1.5B | Planned milestones | Pending |

If I publish a checkpoint, I will include the update number and token count, validation metrics, file size and checksums, model configuration, tokenizer identity, and usage notes. A training log alone will not be labeled as a checkpoint release.
