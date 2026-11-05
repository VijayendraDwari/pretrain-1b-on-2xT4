# Training milestones

The table records the update and token count used for each validation result. Training can pass an analytical milestone between validation events, so report the measured update rather than rounding it to the milestone label.

| Update | Model-input tokens | Validation NLL | Perplexity | Token accuracy | Note |
|---:|---:|---:|---:|---:|---|
| 2 | 262,144 | 10.74244 | 46,279.01 | 3.794% | Initial validation |
| 100 | 13,107,200 | 6.89855 | 990.83 | 12.092% | Earlier run log |
| 616 | 80,740,352 | 5.40386 | 222.26 | 21.155% | Complete parent checkpoint |
| 700 | 91,750,400 | 5.34038 | 208.59 | 21.506% | Continuation validation |
| 800 | 104,857,600 | 5.24585 | 189.78 | 22.049% | First validation after passing 100M |
| 900 | 117,964,800 | 5.17369 | 176.57 | 22.522% | Continuation validation |
| 1000 | 131,072,000 | 5.12850 | 168.76 | 22.940% | Latest captured validation |

The shared continuation log reaches update 1070 (140,247,040 tokens). That is a training observation, not a completed checkpoint. The last complete parent checkpoint documented here is update 616.

Next analytical milestones are 250M, 500M, 1B, and 1.5B input tokens. The target is reached at a whole-update boundary, so the final token count will be 1,500,119,040 rather than exactly 1.5B. The CSV contains the reported training and validation rows; it is not a complete raw run log.
