# Tokenizer contract

The historical corpus uses the tokenizer family from `meta-llama/Llama-3.2-1B`.

Canonical properties:

- vocabulary size: **128,256**
- EOS: **128001**
- historical `tokenizer.json` SHA-256: `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b`
- historical `tokenizer_config.json` SHA-256: `a42a345ade5eba1c27f3a9beca0635200f9e652db3a3b87cc4c459aa890d9c33`

Exact checkpoint continuation should use the historical tokenizer identity. Independent reproductions may rebuild the corpus with a compatible tokenizer, but should report their tokenizer hashes.
