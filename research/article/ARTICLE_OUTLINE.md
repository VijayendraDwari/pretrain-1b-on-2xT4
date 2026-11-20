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
9. **The completed curve.** Plot only logged validation points through u11445 / 1.50012B tokens and show the final NLL of 3.21237.
10. **Session continuation.** Show how each Kaggle session restores the highest compatible COMPLETE two-rank state and preserves the logical token/update ledger.
11. **The final continuation chain.** Use the u4269 → u11445 sequence as the end-to-end example of immutable parents, validation on resume, transactional saves, and completion at the configured target.
12. **Public checkpoints and Anvaya-1B.** Explain why the public HF revision can lag the private/resumable training lineage and document the recommended completed checkpoint series.
13. **What this run can and cannot show.** Separate a working systems path and improving next-token validation from downstream capability and compute-optimal training.
14. **Reproduction and open questions.** Give the exact environment, dependencies, data requirements, identity checks, and remaining reproducibility gaps.
15. **Post-training evaluation.** Add downstream benchmark results only when run under a versioned evaluation protocol.

## Current data for the draft

As of the completed **20 November 2026** run:

- final COMPLETE checkpoint: **u11445 / 1,500,119,040 tokens**
- final fixed validation:
  - NLL **3.212374**
  - perplexity **24.83799**
  - token accuracy **39.3593%**
  - validation sample **131,072 tokens**
- latest sampled training telemetry: **u11440 / 1,499,463,680 tokens**
- public Hugging Face weight revision currently available: **Anvaya-1B `160M`** at u1225 / 160,563,200 tokens
- recommended later release points include **u6042 (~800M), u7879 (~1B), u9712 (~1.27B), and u11445 (1.5B-final)**

Keep the training-log frontier, validation frontier, resumable-checkpoint frontier, and public-weight frontier separate.
