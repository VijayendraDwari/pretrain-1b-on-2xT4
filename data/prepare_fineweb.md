# Preparing FineWeb-D1 token files

The training run uses token files prepared from the FineWeb `CC-MAIN-2025-26` configuration. The historical dataset contract is:

- dataset: `HuggingFaceFW/fineweb`, revision `v1.4.0`;
- split: `train`;
- tokenizer family: `meta-llama/Llama-3.2-1B`;
- vocabulary: 128,256; EOS token: 128001;
- 1,485,000,000 training tokens and 15,000,000 validation tokens;
- 24 training shards, plus one validation shard;
- little-endian `uint32` token IDs.

The Kaggle dataset used by the run is mounted at:

```text
/kaggle/input/datasets/vijayendrad/fineweb-d1/
  sca_fineweb_2025_26_clean_llama32_1p5b/
```

The `sca_` directory name comes from the earlier research project that prepared the corpus. This repository's model uses ordinary dense causal attention.

## Build a compatible corpus

The public builder streams the named FineWeb revision, tokenizes documents, appends EOS, writes the requested token counts, and records file hashes in `corpus_meta.json`:

```bash
python data/prepare_fineweb.py --output ./fineweb_d1_1p5b
```

Access to the Meta tokenizer may require accepting its license and logging in to Hugging Face. To use local tokenizer files instead:

```bash
python data/prepare_fineweb.py \
  --tokenizer /path/to/tokenizer \
  --output ./fineweb_d1_1p5b
```

The builder checks vocabulary size and EOS ID. Those checks do not prove that local tokenizer files are byte-identical to the historical ones. Their recorded historical SHA-256 values are:

```text
tokenizer.json
6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b

tokenizer_config.json
a42a345ade5eba1c27f3a9beca0635200f9e652db3a3b87cc4c459aa890d9c33
```

## What a rebuild does and does not reproduce

A new streamed build follows the documented counts, file format, tokenizer family, and split procedure. It is not promised to recreate the historical token files byte for byte. Dataset streaming, library versions, source order, and tokenizer files can affect the resulting bytes. For exact continuation, use the historical token shards and tokenizer files, and retain their checksum manifest with the checkpoint.
