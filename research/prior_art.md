# Related projects

I use these projects as references for different parts of the tutorial. They are not direct performance baselines for this run.

- **[build-nanogpt](https://github.com/karpathy/build-nanogpt)** is the closest reference for teaching by walking through the code. Its README explains that the commits were kept step by step so readers can follow the build. I want this repository to be similarly inspectable, while focusing on the extra memory and checkpoint work needed at roughly 1B parameters.
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** is a compact training codebase with practical commands and measured examples. It is useful to study for concise, runnable documentation. It is not the same project: its README describes different models, data, and hardware.
- **[TinyLlama](https://github.com/jzhang38/TinyLlama)** shows what a 1.1B pretraining project looks like at a much larger scale. Its README documents a 3T-token target, 16 A100-40G GPUs, and intermediate checkpoints. Those results are not a fair comparison for a 1.5B-token run on two T4s.
- **[LitGPT](https://github.com/Lightning-AI/litgpt)** provides practical setup instructions for pretraining and resuming runs, with commands, configuration, and notes about data choices. It is a general framework, while this repository follows one fixed model and one constrained-hardware run.
- **[OLMo](https://allenai.org/olmo)** is an example of a much broader open model lifecycle, including data, training code, weights, and evaluation resources. This repository does not yet offer that level of openness.

The narrow aim here is to make one training path understandable: a 1.018B dense decoder, 4K context, FineWeb-derived tokens, two 16GB T4s, FSDP `FULL_SHARD`, and continuation across temporary Kaggle sessions.

I am not claiming the first 1B model, an open 1B recipe, compute-optimal training, a speed advantage, or competitive final capability.
