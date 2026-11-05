# Article Architecture

## Working title

**Pretraining a 1B-Parameter Language Model on Two T4 GPUs: An End-to-End Engineering Guide**

## Opening promise

This article is not about obtaining the strongest 1B model for the least money. It is about understanding every major engineering step between random initialization and a real billion-parameter pretraining run.

## Narrative arc

### 1. Why train a 1B model yourself?

Explain the educational gap:
- tiny GPT tutorials fit easily and hide scaling problems;
- production LLM training assumes infrastructure unavailable to most individuals;
- ~1B parameters is large enough that distributed systems become unavoidable.

### 2. What exactly counts as “from scratch”?

Define:
- random parameter initialization;
- tokenizer/data already prepared is acceptable if documented;
- no pretrained Transformer weights;
- causal LM objective;
- optimizer and LR schedule begin from step zero.

### 3. The model

Walk through:
- 29 layers;
- 1536 width;
- 24 heads;
- 64 head dimension;
- 6144 FFN;
- 128,256 vocabulary;
- 4K context;
- exact 1,018,126,848 parameter total.

Include parameter-count derivation.

### 4. Why two T4s are interesting

Not optimal hardware—constrained hardware.

Discuss:
- 16GB/device;
- why naïve replication fails;
- why FSDP FULL_SHARD changes the problem.

### 5. Batch and token arithmetic

Derive:
- 4K context;
- world size 2;
- accumulation 16;
- 131,072 tokens/update.

Explain why token accounting is more useful than “epochs” for web-scale pretraining.

### 6. The data path

FineWeb token shards:
- train corpus;
- fixed validation shard;
- deterministic/random contiguous sequence loading;
- why data contracts matter.

### 7. The actual training loop

Readable walkthrough:
- get batch;
- forward;
- causal CE;
- accumulated backward;
- grad clipping;
- AdamW;
- LR schedule;
- validation;
- checkpoint.

### 8. The checkpoint failure

Tell the real story:
- update 100 training was healthy;
- checkpoint serialization failed;
- partial checkpoint appeared;
- why resuming it would corrupt lineage;
- immutable-parent + COMPLETE-marker redesign.

### 9. What the learning curve looks like

Show milestone table/plots.

Current data:
- 262K: NLL 10.742;
- 13.1M: NLL 6.899;
- 80.7M: NLL 5.404.

### 10. Throughput versus accessibility

Discuss ~2.1K tokens/sec honestly.

Contrast with A100-class training without treating throughput as the goal.

### 11. What 1.5B tokens means—and does not mean

Explain undertraining.

Separate:
- systems demonstration;
- language-model learning;
- competitive foundation-model capability.

### 12. Reproduce it

Give exact repository workflow.

### 13. Lessons

Candidate lessons:
1. exact accounting beats intuition;
2. memory topology matters more than nominal model size;
3. checkpointing is part of training, not an afterthought;
4. token lineage matters;
5. constrained hardware is a useful teacher.

### 14. Final result

Fill after 1.5B run:
- final NLL/PPL;
- final throughput;
- total GPU hours;
- checkpoint size;
- evaluation suite;
- sample generations if meaningful.
