# Checkpoint releases

The training lineage is ahead of the public model release. The current Hugging Face public checkpoint is **Anvaya-1B `160M`**, exported from update **1225 / 160,563,200 model-input tokens** as model-only FP16 safetensors. The raw resumable FSDP checkpoints remain training artifacts rather than GitHub downloads.

The latest documented COMPLETE training checkpoint is update **4879 / 639,500,288 tokens**. The latest measured validation is update **4800 / 629,145,600 tokens**: NLL **3.52762**, perplexity **34.04**, token accuracy **36.166%**. The 600M model-input-token milestone has now been crossed cleanly.

## Public checkpoint series

The public series favors scientifically useful snapshots rather than every session boundary. Each Hugging Face milestone should be an immutable named revision and should contain model-only weights plus the exact configuration, tokenizer identity, validation record, export checksum, and provenance back to the resumable training checkpoint.

| HF revision | Source checkpoint | Exact model-input tokens | Status / rationale |
|---|---:|---:|---|
| `160M` | u1225 | 160,563,200 | **Public now** — first early-language checkpoint |
| `240M` | u1829 | 239,730,688 | Candidate — useful early-emergence comparison |
| `320M` | u2436 | 319,291,392 | Candidate — preserved clean checkpoint |
| `400M` | u3031 | 397,279,232 | Candidate — preserved clean checkpoint near 400M |
| `560M` | u4269 | 559,546,368 | Preserved post-500M checkpoint |
| `640M` | u4879 | 639,500,288 | **Latest major public milestone candidate** — clean checkpoint after crossing 600M |
| `750M` | first clean checkpoint at/after 750M | ≥750M | Planned |
| `1B` | first clean checkpoint at/after 1B | ≥1B | Planned |
| `1.25B` | first clean checkpoint at/after 1.25B | ≥1.25B | Planned |
| `1.5B` | final whole-update checkpoint | 1,500,119,040 | Planned final base checkpoint |

Update **4269 / 559.55M** remains an important preserved post-500M checkpoint with clean resume validation (NLL **3.58536**). The newer u4879 checkpoint now provides a clearer post-600M milestone for any next public export.

The friendly revision names are labels, not rounded claims about exact token exposure. Every release must state its exact update and token ledger explicitly.
