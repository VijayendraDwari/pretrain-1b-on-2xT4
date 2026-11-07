# Checkpoint releases

The training lineage is ahead of the public model release. The current Hugging Face public checkpoint is **Anvaya-1B `160M`**, exported from update **1225 / 160,563,200 model-input tokens** as model-only FP16 safetensors. The raw resumable FSDP checkpoints remain training artifacts rather than GitHub downloads.

The latest documented COMPLETE training checkpoint is update **3638 / 476,839,936 tokens**. Its nearest measured validation is update **3600 / 471,859,200 tokens**: NLL **3.69297**, perplexity **40.16**, token accuracy **34.653%**.

## Public checkpoint series

The public series should favor scientifically useful snapshots rather than every session boundary. Each Hugging Face milestone should be an immutable named revision and should contain model-only weights plus the exact configuration, tokenizer identity, validation record, export checksum, and provenance back to the resumable training checkpoint.

| HF revision | Source checkpoint | Exact model-input tokens | Status / rationale |
|---|---:|---:|---|
| `160M` | u1225 | 160,563,200 | **Public now** — first early-language checkpoint |
| `240M` | u1829 | 239,730,688 | Candidate — useful early-emergence comparison |
| `320M` | u2436 | 319,291,392 | Candidate — next preserved clean checkpoint |
| `400M` | u3031 | 397,279,232 | Candidate — preserved clean checkpoint near 400M |
| `500M` | first clean checkpoint at/after 500M | ≥500M | **Next major public milestone** |
| `750M` | first clean checkpoint at/after 750M | ≥750M | Planned |
| `1B` | first clean checkpoint at/after 1B | ≥1B | Planned |
| `1.25B` | first clean checkpoint at/after 1.25B | ≥1.25B | Planned |
| `1.5B` | final whole-update checkpoint | 1,500,119,040 | Planned final base checkpoint |

Update **3638 / 476.84M** is an important resumable parent but is **not** proposed as a separate public revision because the training lineage is already close to 500M. Preserve it privately for continuation and provenance; publish the first clean ≥500M checkpoint instead.

The friendly revision names are labels, not rounded claims about exact token exposure. Every release must state its exact update and token ledger explicitly.
