# Results files

`validation_curve.csv` is the canonical validation-event ledger for this project. Each row records update, cumulative model-input tokens, NLL, perplexity, and top-1 token accuracy on the fixed **131,072-token** validation sample.

The ledger now extends through update **3600** / **471,859,200 tokens**, where validation NLL is **3.69297**, perplexity **40.16**, and token accuracy **34.653%**. The latest COMPLETE resumable checkpoint is later: update **3638** / **476,839,936 tokens**. There is deliberately no fabricated validation score for update 3638.

`training_curve.csv` contains selected historical training-log rows rather than every optimizer update; it should not be treated as the canonical validation ledger.

To rebuild the chart from the validation CSV, run:

```bash
python results/plot_validation.py
```

The repository tracks `results/validation_nll.svg` as the README-facing plot so the checked-in graph stays synchronized with the recorded validation rows.
