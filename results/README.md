# Results files

`validation_curve.csv` holds the validation events reported so far. Each row records the update, input-token count, NLL, perplexity, and top-1 accuracy. `training_curve.csv` contains selected training-log rows. It is not the complete JSONL log; there are gaps between the early rows and the continuation rows.

The continuation entries now extend through sampled training update **1820**. Duplicate console copies are kept once per update. The latest supplied session then saved a COMPLETE checkpoint at update **1829** (**239,730,688 model-input tokens**) and ended at a safe boundary; update 1829 is therefore the latest documented resumable state. The validation CSV now extends through update **1800** / **235,929,600 tokens**.

To rebuild the chart from the validation CSV, run:

```bash
python results/plot_validation.py
```

The script requires `matplotlib`. It writes `results/validation_nll.png`. The repository also carries `results/validation_nll.svg` as the lightweight README-facing plot.
