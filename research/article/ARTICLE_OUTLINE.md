# Article outline

**Working title:** Pretraining a 1B-Parameter Language Model on Two T4 GPUs: An Engineering Walkthrough

The article should read like a record of one real training run, not a claim that this is the best way to train a 1B model. Keep the distinction clear between what the run has demonstrated and what remains untested.

## Story and sections

1. **Why try this?** Start with the gap between small tutorials and large-cluster training. Explain why this project is useful to follow even though two T4s are not an efficient route to a competitive model.
2. **What “from scratch” means here.** Random Transformer initialization, reused tokenizer, pre-tokenized data, and no pretrained weights.
3. **The model.** Walk through the architecture diagram, dimensions, tied embeddings, and exact parameter count.
4. **Fitting the run on two T4s.** Explain FSDP `FULL_SHARD`, activation checkpointing, FP16, and what the memory log measures.
5. **Batch size and tokens.** Derive 131,072 input tokens per update. Explain that the sampler draws deterministic windows with replacement, rather than making one guaranteed pass through the corpus.
6. **The data contract.** Describe the FineWeb source, tokenizer identity, shard format, and what a fresh rebuild cannot guarantee.
7. **The training step.** Follow a batch through forward pass, chunked loss, accumulated gradients, clipping, AdamW, and validation.
8. **The checkpoint failure.** Use the update-100 storage failure to explain partial files, the completion marker, and the immutable mounted parent.
9. **The live curve.** Plot only logged validation points. Identify the checkpoint/update and token count for each point; don't interpolate claims or treat NLL as a downstream score.
10. **Session continuation.** Show how the next Kaggle session finds and restores a completed two-rank state.
11. **What this run can and cannot show.** Separate a working systems path from model capability and compute-optimal training.
12. **Reproduction and open questions.** Give the exact environment, dependencies, data requirements, and limitations of the current lineage checks.
13. **Update after the target.** Add final NLL, throughput, GPU hours, checkpoint size, and a versioned downstream evaluation only if those measurements are completed.

## Current data for the draft

The shared live log shows training at update 1070 (140,247,040 input tokens). The latest validation in that snapshot is update 1000 (131,072,000 tokens): NLL 5.12850, perplexity 168.76, and token accuracy 22.94%. The 100M analytical milestone has been crossed. The last complete parent checkpoint documented for the continuation is still update 616. Keep the training-log frontier, validation frontier, and resumable-checkpoint frontier separate.
