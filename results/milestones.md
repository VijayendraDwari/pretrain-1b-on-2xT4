# Training milestones

This file records the completed reference run for the 1,018,126,848-parameter dense decoder-only model trained at 4096 context on 2×16GB Tesla T4 GPUs.

## Completion

The configured budget was reached at **update 11,445 / 1,500,119,040 model-input tokens**. Final fixed validation was **NLL 3.212374**, **perplexity 24.83799**, and **token accuracy 39.3593%**. The trainer terminated normally after saving the complete checkpoint and reporting that the token target had been reached.

## Selected validation milestones

| Update | Input tokens | NLL | PPL | Token accuracy |
|---:|---:|---:|---:|---:|
| 2 | 262,144 | 10.74244 | 46,279.01 | 3.79% |
| 1,225 | 160,563,200 | 4.75070 | 115.67 | 25.21% |
| 1,829 | 239,730,688 | 4.26552 | 71.20 | 30.09% |
| 2,436 | 319,291,392 | 3.98411 | 53.74 | 32.38% |
| 3,031 | 397,279,232 | 3.81320 | 45.30 | 33.69% |
| 4,269 | 559,546,368 | 3.58536 | 36.07 | 35.71% |
| 6,042 | 791,937,024 | 3.41344 | 30.37 | 37.31% |
| 7,879 | 1,032,716,288 | 3.30334 | 27.20 | 38.32% |
| 9,712 | 1,272,971,264 | 3.24208 | 25.59 | 39.05% |
| 10,917 | 1,430,913,024 | 3.21850 | 24.99 | 39.25% |
| **11,445** | **1,500,119,040** | **3.21237** | **24.84** | **39.36%** |

The complete fixed-validation ledger is in [validation_curve.csv](validation_curve.csv).

## Final continuation chain

The last phase was executed as a sequence of immutable-parent Kaggle continuations:

| Resume checkpoint | Saved checkpoint |
|---:|---:|
| u4269 / 559,546,368 | u4879 / 639,500,288 |
| u4879 / 639,500,288 | u5444 / 713,555,968 |
| u5444 / 713,555,968 | u6042 / 791,937,024 |
| u6042 / 791,937,024 | u6663 / 873,332,736 |
| u6663 / 873,332,736 | u7267 / 952,500,224 |
| u7267 / 952,500,224 | u7879 / 1,032,716,288 |
| u7879 / 1,032,716,288 | u8459 / 1,108,738,048 |
| u8459 / 1,108,738,048 | u9090 / 1,191,444,480 |
| u9090 / 1,191,444,480 | u9712 / 1,272,971,264 |
| u9712 / 1,272,971,264 | u10282 / 1,347,682,304 |
| u10282 / 1,347,682,304 | u10917 / 1,430,913,024 |
| u10917 / 1,430,913,024 | **u11445 / 1,500,119,040** |

Machine-readable provenance is in [session_checkpoints.csv](session_checkpoints.csv).

## What the curve shows

From u4269 to the final checkpoint, fixed-validation NLL improved from **3.58536 to 3.21237** and perplexity from **36.07 to 24.84**. Token accuracy improved from **35.71% to 39.36%**. Training remained numerically stable through the final continuation, while the cosine schedule annealed toward the configured minimum learning rate.
