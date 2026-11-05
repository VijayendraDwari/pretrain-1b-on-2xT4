# Dataset contract

The trainer expects:

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

Binary token files are contiguous little-endian `uint32` token IDs.

Canonical D1 counts:

| split | tokens |
|---|---:|
| train | 1,485,000,000 |
| validation | 15,000,000 |

Vocabulary size is 128,256. Every token must satisfy `0 <= token_id < 128256`.

The training loader chooses deterministic 4096-token windows from the concatenated training shards. Validation uses the fixed D1 validation corpus throughout the project.
