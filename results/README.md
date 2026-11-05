# Results files

`validation_curve.csv` holds the validation events reported so far. Each row records the update, input-token count, NLL, perplexity, and top-1 accuracy. `training_curve.csv` contains selected training-log rows. It is not the complete JSONL log; there are gaps between the early rows and the continuation rows.

The continuation entries through update 820 were copied from the notebook output shared for this update. Duplicate console copies were kept once per update. The latest training row is not a checkpoint. The last complete parent checkpoint documented for this run is update 616.

To rebuild the chart from the validation CSV, run:

```bash
python results/plot_validation.py
```

The script requires `matplotlib`. It writes `results/validation_nll.png`.
