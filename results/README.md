# Results files

`validation_curve.csv` holds the validation events reported so far. Each row records the update, input-token count, NLL, perplexity, and top-1 accuracy. `training_curve.csv` contains selected training-log rows. It is not the complete JSONL log; there are gaps between the early rows and the continuation rows.

The continuation entries now extend through sampled training update **2430**. Duplicate console copies are kept once per update. The latest supplied session then saved a COMPLETE checkpoint at update **2436** (**319,291,392 model-input tokens**) and ended at a safe boundary; update 2436 is therefore the latest documented resumable state. The validation CSV now extends through update **2400** / **314,572,800 tokens**, where validation NLL is **3.99674**, perplexity **54.42**, and token accuracy **32.092%**.

To rebuild the chart from the validation CSV, run:

```bash
python results/plot_validation.py
```

The script requires `matplotlib` and can regenerate a local PNG from the CSV. The repository tracks `results/validation_nll.svg` as the README-facing plot so the checked-in graph stays synchronized with the recorded validation rows.
