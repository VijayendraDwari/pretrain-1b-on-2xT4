# Prior-art notes

This repository does not claim that 1B-parameter pretraining itself is novel.

Useful reference projects:

- **nanoGPT / build-nanogpt — Andrej Karpathy**  
  Educational GPT training with unusually readable code and strong emphasis on understanding the complete loop.

- **TinyLlama — Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu**  
  A genuinely pretrained open ~1.1B model at much larger token/compute scale.

- **OLMo — Allen Institute for AI**  
  A model family/project built around openness across data, training, checkpoints, and evaluation.

- **LitGPT — Lightning AI**  
  A reusable optimized training framework supporting many model families and scales.

The narrower contribution here is the complete **1.018B + 4K context + FineWeb + 2×16GB T4 + FSDP + crash-safe continuation** educational path.

The project does not claim:
- first 1B model from scratch;
- first open 1B recipe;
- compute-optimal 1B pretraining;
- competitive 1B final capability;
- throughput superiority.
