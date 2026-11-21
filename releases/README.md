# Checkpoint releases

The Anvaya-1B reference run is complete at **u11445 / 1,500,119,040 model-input tokens**, and the selected model-only checkpoints are now public as immutable revisions in the single Hugging Face repository:

**https://huggingface.co/Vijayendra/Anvaya-1B**

`main` points to the final `1.5B-final` export.

## Public training lineage

| HF revision | Source checkpoint | Exact input tokens | Validation NLL | PPL | Token accuracy |
|---|---:|---:|---:|---:|---:|
| `160M` | u1225 | 160,563,200 | 4.75070 | 115.67 | 25.21% |
| `240M` | u1829 | 239,730,688 | 4.26552 | 71.20 | 30.09% |
| `320M` | u2436 | 319,291,392 | 3.98411 | 53.74 | 32.38% |
| `400M` | u3031 | 397,279,232 | 3.81320 | 45.30 | 33.69% |
| `560M` | u4269 | 559,546,368 | 3.58536 | 36.07 | 35.71% |
| `800M` | u6042 | 791,937,024 | 3.41344 | 30.37 | 37.31% |
| `1B` | u7879 | 1,032,716,288 | 3.30334 | 27.20 | 38.32% |
| `1.2B` | u9090 | 1,191,444,480 | 3.25586 | 25.94 | 38.92% |
| `1.35B` | u10282 | 1,347,682,304 | 3.22915 | 25.26 | 39.13% |
| **`1.5B-final`** | **u11445** | **1,500,119,040** | **3.21237** | **24.84** | **39.36%** |

Friendly revision names are labels; exact update and token ledgers are preserved in each revision's `config.json`, `source_complete.json`, and `export_manifest.json`.

The raw two-rank FSDP checkpoints remain training artifacts. Public releases contain model-only FP16 safetensors plus the exact tokenizer/config/reference model code, rather than optimizer or GradScaler state.

## Export contract used for every public revision

Each released revision contains:

1. model-only FP16 safetensors;
2. exact architecture/configuration;
3. tokenizer files and tokenizer identity;
4. source update and exact model-input-token count;
5. fixed-validation NLL, perplexity, token accuracy and validation-token count;
6. SHA-256 file hashes in the export manifest;
7. source COMPLETE checkpoint metadata;
8. a generation/inference smoke test in the export pipeline.

This layout makes the repository both a usable final model artifact and a longitudinal record of how a billion-parameter model changes during pretraining.
