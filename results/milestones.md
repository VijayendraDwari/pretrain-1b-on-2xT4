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
| 1000 | 131,072,000 | 5.12850 | 168.76 | 22.940% | Continuation validation |
| 1100 | 144,179,200 | 4.91807 | 136.74 | 23.952% | Continuation validation |
| 1200 | 157,286,400 | 4.79223 | 120.57 | 24.852% | Continuation validation |
| 1225 | 160,563,200 | 4.75070 | 115.67 | 25.205% | Clean validation immediately after exact resume |
| 1300 | 170,393,600 | 4.64783 | 104.36 | 26.204% | Continuation validation |
| 1400 | 183,500,800 | 4.54832 | 94.47 | 27.312% | Continuation validation |
| 1500 | 196,608,000 | 4.48414 | 88.60 | 28.094% | Continuation validation |
| 1600 | 209,715,200 | 4.40366 | 81.75 | 28.819% | Continuation validation |
| 1700 | 222,822,400 | 4.33161 | 76.07 | 29.567% | Continuation validation |
| 1800 | 235,929,600 | 4.27305 | 71.74 | 29.997% | Latest captured validation |

The latest supplied continuation reaches sampled training update **1820** (238,551,040 tokens), then passes checkpoint preflight and saves a COMPLETE checkpoint at update **1829** (**239,730,688 tokens**) before a safe session end. The latest fixed validation is update **1800**.

The next analytical milestone is **250M tokens**, followed by 500M, 1B, and 1.5B input tokens. The target is reached at a whole-update boundary, so the final token count will be 1,500,119,040 rather than exactly 1.5B. The CSV contains the reported training and validation rows; it is not a complete raw run log.
