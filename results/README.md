# Results files

`validation_curve.csv` holds the validation events reported so far. Each row records the update, input-token count, NLL, perplexity, and top-1 accuracy. `training_curve.csv` contains selected training-log rows. It is not the complete JSONL log; there are gaps between the early rows and the continuation rows.

The continuation entries through update 1220 were copied from the notebook output shared for this update. Duplicate console copies were kept once per update. The latest sampled training row is update 1220. The session then saved a COMPLETE checkpoint at update 1225 and ended at a safe boundary; update 1225 is therefore the latest documented resumable state.

To rebuild the chart from the validation CSV, run:

```bash
python results/plot_validation.py
```

The script requires `matplotlib`. It writes `results/validation_nll.png`. The repository also carries `results/validation_nll.svg` as the lightweight README-facing plot.
