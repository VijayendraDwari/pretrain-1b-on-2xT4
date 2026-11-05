# Training milestones

| update | input tokens | validation NLL | perplexity | token accuracy | status |
|---:|---:|---:|---:|---:|---|
| 2 | 262,144 | 10.74244 | 46,279.01 | 3.794% | certified initial checkpoint |
| 100 | 13,107,200 | 6.89855 | 990.83 | 12.092% | replayed learning milestone |
| 616 | 80,740,352 | 5.40386 | 222.26 | 21.155% | canonical continuation checkpoint |
| 763 | 100,007,936 | pending | pending | pending | analytical milestone |
| — | 250M | pending | pending | pending | planned |
| — | 500M | pending | pending | pending | planned |
| — | 1B | pending | pending | pending | planned |
| — | 1.5B | pending | pending | pending | final target |

The active job may pass milestone token counts without stopping. All published metrics must report the checkpoint/update and actual token count used.
