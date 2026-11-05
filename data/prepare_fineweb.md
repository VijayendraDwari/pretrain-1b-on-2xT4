# FineWeb-D1 preparation

The historical training corpus used by this repository is a certified token dataset with this contract:

- source: `HuggingFaceFW/fineweb`
- revision: `v1.4.0`
- config: `CC-MAIN-2025-26`
- split: `train`
- tokenizer family: `meta-llama/Llama-3.2-1B`
- vocabulary: 128,256
- EOS token: 128001
- train: **1,485,000,000 tokens**
- validation: **15,000,000 tokens**
- 24 train shards
- one validation shard
- little-endian `uint32` token IDs

The historical Kaggle corpus is mounted in our training notebooks as:

```text
/kaggle/input/datasets/vijayendrad/fineweb-d1/
  sca_fineweb_2025_26_clean_llama32_1p5b/
```

The legacy `sca_` directory prefix reflects the research project in which the corpus was originally prepared. The dense model itself uses ordinary dense causal attention.

## Exact historical tokenizer identity

Historical certified hashes:

```text
tokenizer.json
6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b

tokenizer_config.json
a42a345ade5eba1c27f3a9beca0635200f9e652db3a3b87cc4c459aa890d9c33
```

For exact continuation of our published lineage, use the historical corpus/tokenizer and verify its checksums.

For an independent reproduction, run:

```bash
python data/prepare_fineweb.py --output ./fineweb_d1_1p5b
```

Access to the Meta Llama tokenizer repository may require accepting its license and authenticating with Hugging Face. You can instead pass a local tokenizer directory:

```bash
python data/prepare_fineweb.py   --tokenizer /path/to/the/tokenizer   --output ./fineweb_d1_1p5b
```

## Reproduction caveat

The public builder reproduces the **documented corpus contract**. It does not claim that a future streamed rebuild will be byte-identical to the historical D1 dataset if the upstream data stream, library behavior, or tokenizer files change.

For research lineage, dataset identity is a hash/checksum question—not merely a dataset-name question.
