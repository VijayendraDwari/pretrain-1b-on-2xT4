# Fixed evaluation protocol

The first evaluation layer is intentionally simple and stable:

1. use the same frozen 15M-token D1 validation corpus;
2. evaluate a fixed 131,072 validation tokens per reported checkpoint;
3. report full-sequence causal cross-entropy / NLL;
4. report perplexity as `exp(NLL)`;
5. report next-token top-1 accuracy;
6. never tune on the validation windows used for the milestone curve.

Named analytical milestones:

- 100M
- 250M
- 500M
- 1B
- 1.5B input tokens

The active training session is not required to stop exactly at these milestones. If a physical checkpoint is not present at an exact milestone, the nearest explicit validation event is reported with its actual token count.

A downstream benchmark suite can be added later, but it must be versioned and then held fixed across subsequent milestones.
