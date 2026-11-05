# Dataset layout expected by the trainer

The trainer reads pre-tokenized IDs from a mounted directory:

```text
<root>/
├── corpus_meta.json
├── tokenizer/
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── tokens/
    ├── train/
    │   ├── train_00000.bin
    │   └── ...
    └── validation/
        └── validation_00000.bin
```

Each `.bin` file is a contiguous stream of little-endian `uint32` token IDs. The canonical dataset has 1,485,000,000 training tokens in 24 shards and 15,000,000 validation tokens in one shard. The vocabulary has 128,256 entries; IDs must be in the range `[0, 128256)`.

The current loader memory-maps each shard. For every sample it chooses one shard and one offset, then returns a 4096-token input window and its one-token-shifted labels. The choice is deterministic from the sample ID and seed. Samples do not cross shard boundaries, and the run does not walk the corpus sequentially.

The trainer currently checks that the token directories exist and that `corpus_meta.json` reports the expected vocabulary size and `uint32` dtype. It does not verify the token-shard hashes or tokenizer hashes at startup. Keep the data manifest with the run if you need to identify the exact corpus used.
