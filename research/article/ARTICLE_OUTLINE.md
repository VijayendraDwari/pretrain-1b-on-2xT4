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
9. **The live curve.** Plot only logged validation points. Show the continued NLL decline through u4800 / 629.15M tokens and keep the latest validation separate from the later u4879 checkpoint frontier.
10. **Session continuation.** Show how a new Kaggle session restores the highest compatible COMPLETE two-rank state and preserves the logical token/update ledger.
11. **Continuation beyond 600M.** Use the u4269 → u4879 continuation as the newest end-to-end example: exact restore, validation, uninterrupted continuation across 600M, transactional save, safe boundary.
12. **Public checkpoints and Anvaya-1B.** Explain why the public HF revision remains `160M` while the private/resumable training lineage has reached 639.50M, and why u4879 is the latest major public-milestone candidate.
13. **What this run can and cannot show.** Separate a working systems path and improving next-token validation from downstream capability and compute-optimal training.
14. **Reproduction and open questions.** Give the exact environment, dependencies, data requirements, identity checks, and remaining reproducibility gaps.
15. **Update after the target.** Add final NLL, throughput, GPU hours, checkpoint size, and a versioned downstream evaluation only if those measurements are completed.

## Current data for the draft

As of the completed **8 November 2026** session:

- latest sampled training row: **u4870 / 638,320,640 tokens**
- latest fixed validation: **u4800 / 629,145,600 tokens**
  - NLL **3.527621**
  - perplexity **34.042887**
  - token accuracy **36.1664%**
  - validation sample **131,072 tokens**
- latest COMPLETE resumable checkpoint: **u4879 / 639,500,288 tokens**
- public Hugging Face weight revision: **Anvaya-1B `160M`** at u1225 / 160,563,200 tokens
- next major public checkpoint candidate: **u4879 / ~640M**
- target: **1.5B model-input tokens**, with the final whole-update ledger at 1,500,119,040 tokens

Keep the training-log frontier, validation frontier, resumable-checkpoint frontier, and public-weight frontier separate.
