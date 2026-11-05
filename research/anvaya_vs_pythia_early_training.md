# Anvaya-1B vs Pythia-1B at early training

This note records an early behavioral comparison between the public **Anvaya-1B 160M** checkpoint and two early checkpoints from **EleutherAI/pythia-1b**.

The result is visually striking, but it needs careful interpretation.

At roughly comparable early token exposure, Anvaya-1B produces substantially more sentence-like English than the two Pythia checkpoints tested here. This is **not** evidence that the Anvaya architecture is superior to Pythia. The models differ in architecture, tokenizer, corpus, context length, global batch size, learning-rate schedule, and number of optimizer updates.

The useful question exposed by the comparison is:

> **At the same parameter scale and broadly similar token exposure, how much do optimizer-update density and schedule position affect the emergence of usable language behavior?**

## Models compared

| Model | Parameters | Checkpoint | Approx. model-input tokens | Optimizer updates |
|---|---:|---:|---:|---:|
| **Anvaya-1B** | **1,018,126,848** | **160M / update 1225** | **160,563,200** | **1,225** |
| Pythia-1B | 1,011,781,632 | step64 | 134,217,728 | 64 |
| Pythia-1B | 1,011,781,632 | step128 | 268,435,456 | 128 |

Pythia-1B is an unusually close parameter-scale reference: about **1.012B parameters** versus Anvaya's **1.018B**.

The early token exposures bracket Anvaya:

~~~text
Pythia step64   ~134.2M tokens
Anvaya-1B        160.56M tokens
Pythia step128  ~268.4M tokens
~~~

But the update counts are radically different.

Anvaya processes **131,072 model-input tokens per optimizer update**:

~~~text
2 GPUs × 4096 tokens × 16 accumulation steps
= 131,072 model-input tokens/update
~~~

Pythia-1B's published configuration uses 64 GPUs, micro-batch size 16 and sequence length 2048:

~~~text
64 × 16 × 2048
= 2,097,152 tokens/update
~~~

So by broadly similar token exposure, Anvaya has received roughly **19× more optimizer updates than Pythia step64**.

## Frozen prompt suite

The same prompts were used for all three checkpoints under three decoding regimes:

- greedy;
- temperature 0.2, top-k 10;
- temperature 0.8, top-k 50.

The prompt set included:

~~~text
The capital of France is
The largest planet in the solar system is
Machine learning is
Once upon a time, there was
Artificial intelligence will change the world by
In a quiet village beside the mountains,
The scientist looked at the experiment and realized
India is a country with
1, 2, 3, 4,
Monday, Tuesday, Wednesday,
A cat is an animal. A dog is an
Paris is in France. Tokyo is in
The opposite of hot is
~~~

This is a qualitative behavioral probe, not a benchmark score.

## What Anvaya-1B looks like at 160.56M tokens

Anvaya already produces locally coherent English syntax, although factual knowledge and semantic stability remain weak.

For the greedy prompt:

~~~text
Machine learning is
~~~

Anvaya continues:

~~~text
Machine learning is a good way to get a good idea of what you are doing.
You can also use a lot of different methods to
~~~

For:

~~~text
Artificial intelligence will change the world by
~~~

Anvaya produces:

~~~text
Artificial intelligence will change the world by creating a more sustainable
and sustainable future. The world is a new generation of people who are
looking for a more sustainable future
~~~

With sampled decoding:

~~~text
In a quiet village beside the mountains,
~~~

becomes:

~~~text
In a quiet village beside the mountains, the main house is the perfect place
for a comfortable escape.
~~~

These outputs are repetitive and frequently wrong, but they are unmistakably sentence-like. At this checkpoint, **language structure has clearly emerged while reliable factual knowledge has not**.

## Pythia-1B at step64

At approximately 134.2M tokens, the tested Pythia checkpoint is much more degenerate under the same suite.

For:

~~~text
Machine learning is
~~~

the greedy continuation is essentially:

~~~text
Machine learning is a good.
~~~

followed by repeated whitespace tokens.

For:

~~~text
India is a country with
~~~

the continuation collapses into:

~~~text
India is a country with the same the same the the the the the...
~~~

And the sequence probe:

~~~text
1, 2, 3, 4,
~~~

continues:

~~~text
1, 2, 3, 4, and, and, and, and, and...
~~~

Sampled generations frequently contain repeated function words, blank regions, braces or markup-like fragments.

## Pythia-1B at step128

At approximately 268.4M tokens, Pythia remains highly repetitive in this frozen suite.

Examples include:

~~~text
Machine learning is a is a is a is a is a is a...
~~~

and:

~~~text
Once upon a time, there was a a a a a a a a a...
~~~

The pattern prompt:

~~~text
1, 2, 3, 4,
~~~

continues with repeated 1 tokens.

So on this particular behavioral probe, **Anvaya-1B at 160.56M tokens is substantially more language-like than Pythia-1B at either step64 or step128**.

That is the observation. The explanation is not yet established.

## Why this is not an architecture claim

The most important confound is the optimization schedule.

Pythia-1B's published configuration specifies:

~~~yaml
train-iters: 143000
warmup: 0.01
extra-save-iters: [0,1,2,4,8,16,32,64,128,256,512]
~~~

One percent of 143,000 iterations is approximately **1,430 warmup steps**.

Therefore:

~~~text
Pythia step64   ≈ 4.5% through warmup
Pythia step128  ≈ 9.0% through warmup
~~~

Both checkpoints are still extremely early in the learning-rate ramp.

Anvaya's warmup is only **75 optimizer updates**. At update 1225, Anvaya is long past warmup.

The token-matched comparison therefore also compares:

- different global batch sizes;
- radically different optimizer-update counts;
- different positions in the LR schedule;
- different datasets;
- different tokenizers;
- different Transformer designs.

It would be incorrect to conclude that **Anvaya's architecture learns faster than Pythia**.

The defensible conclusion is narrower:

> **At roughly matched early token exposure and nearly identical parameter scale, the published Anvaya-1B 160M checkpoint shows much stronger local language behavior than Pythia-1B step64 and step128 under this frozen generation suite. The two training recipes reach that token exposure with radically different optimizer-update density and schedule position.**

## Architecture differences

| Property | Anvaya-1B | Pythia-1B |
|---|---:|---:|
| Parameters | 1.018B | 1.012B |
| Layers | 29 | 16 |
| Width | 1536 | 2048 |
| Heads | 24 | 8 |
| FFN width | 6144 | 8192 |
| Context | 4096 | 2048 |
| Vocabulary | 128,256 | 50,304 |
| Embeddings | tied | untied |
| Corpus | FineWeb-derived | The Pile |

Pythia is therefore a useful **reference model**, not a controlled baseline.

## Two comparison axes going forward

### Token-matched

Compare models after similar amounts of model-input data.

~~~text
Anvaya 160M
vs
Pythia ~134M / ~268M
~~~

This asks:

> What behavior has emerged per unit of training data?

### Update-matched

Anvaya's current public checkpoint is update **1225**.

A Pythia checkpoint around step1024 is much closer in optimizer-update count, although it has consumed roughly **2.15B tokens** because of its much larger global batch.

This asks:

> What behavior has emerged after a similar number of parameter-update events?

The distinction between these two views may be more educational than a single "tokens trained" number.

## Current interpretation

At 160.56M model-input tokens, Anvaya appears to have crossed an early qualitative threshold:

> **Language structure is clearly present; factual knowledge and semantic stability are not yet reliable.**

The next useful experiment is not to declare a winner. It is to repeat the same frozen suite at:

- Anvaya 250M;
- Anvaya 500M;
- Anvaya 1B;
- Anvaya 1.5B;

and compare against both token-matched and update-matched public references.

That turns a one-off generation demo into a longitudinal study of **how language behavior emerges during pretraining**.

## Sources

- Anvaya-1B model: https://huggingface.co/Vijayendra/Anvaya-1B
- Anvaya training repository: https://github.com/VijayendraDwari/pretrain-1b-on-2xT4
- Pythia-1B model: https://huggingface.co/EleutherAI/pythia-1b
- Pythia-1B training configuration: https://github.com/EleutherAI/pythia/blob/main/models/1B/pythia-1b.yml
