# Results files

`validation_curve.csv` is the canonical validation-event ledger for this project. Each row records update, cumulative model-input tokens, NLL, perplexity, and top-1 token accuracy on the fixed **131,072-token** validation sample.

The ledger now extends through update **4200** / **550,502,400 tokens**, where validation NLL is **3.59542**, perplexity **36.43**, and token accuracy **35.509%**. The latest COMPLETE resumable checkpoint is later: update **4269** / **559,546,368 tokens**. The same completed session crossed 500M model-input tokens cleanly.

`training_curve.csv` contains selected training-log rows and now extends through sampled update **4260** / **558,366,720 tokens**. It remains a sampled ledger, not the complete JSONL history.

To rebuild the chart from the validation CSV, run:

```bash
python results/plot_validation.py
```

The repository tracks `results/validation_nll.svg` as the README-facing plot so the checked-in graph stays synchronized with the recorded validation rows.
