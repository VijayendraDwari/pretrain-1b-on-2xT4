# Validation protocol

The current evaluation is a small, fixed next-token check during pretraining. It is useful for tracking the run; it is not a downstream benchmark.

At each validation event, the trainer evaluates 16 deterministic 4096-token windows on each of two ranks. That gives 131,072 validation tokens with the default settings. It reports mean causal cross-entropy (NLL), perplexity, and next-token top-1 accuracy. The perplexity code computes `exp(min(NLL, 30))` to cap the exponent; the NLL values reported so far are below that cap, so their perplexities equal `exp(NLL)`. The same validation shard and window IDs are used as the run progresses.

The number of validation steps can be changed with the `VAL_STEPS` environment variable. To compare numbers, keep that value, the validation data, and the code version fixed. The validation CSV records the measured token count and update for each row. These measurements come from the supplied run log; the corresponding checkpoint files and raw JSONL logs are not included in this repository.

A downstream evaluation suite may be useful later, but it should be added as a separate, versioned evaluation rather than mixed into this pretraining curve. No downstream capability claim is made from the current metrics.
