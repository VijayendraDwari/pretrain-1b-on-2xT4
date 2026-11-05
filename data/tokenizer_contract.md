# Tokenizer used by the run

The corpus was tokenized with the Llama 3.2 1B tokenizer family. The model has a 128,256-row vocabulary, and the EOS token ID is 128001.

The historical tokenizer files were recorded with these SHA-256 values:

- `tokenizer.json`: `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`
- `tokenizer_config.json`: `a42a345ade5eba1c27f3a9beca0635200f9e652db3a3b87cc4c459aa890d9c33`

Matching the vocabulary size and EOS ID is not enough to prove that two tokenizer files are identical. The current trainer consumes token IDs and does not load or hash the tokenizer files at startup. Keep the tokenizer files and their checksums alongside the dataset manifest when preserving a run's lineage.
