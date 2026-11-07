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
| 1225 | 160,563,200 | 4.75070 | 115.67 | 25.205% | Current public HF checkpoint (`160M`) |
| 1300 | 170,393,600 | 4.64783 | 104.36 | 26.204% | Continuation validation |
| 1400 | 183,500,800 | 4.54832 | 94.47 | 27.312% | Continuation validation |
| 1500 | 196,608,000 | 4.48414 | 88.60 | 28.094% | Continuation validation |
| 1600 | 209,715,200 | 4.40366 | 81.75 | 28.819% | Continuation validation |
| 1700 | 222,822,400 | 4.33161 | 76.07 | 29.567% | Continuation validation |
| 1800 | 235,929,600 | 4.27305 | 71.74 | 29.997% | Continuation validation |
| 1829 | 239,730,688 | 4.26552 | 71.20 | 30.089% | Clean resume validation; preserved complete checkpoint |
| 1900 | 249,036,800 | 4.23285 | 68.91 | 30.415% | Validation immediately before crossing 250M |
| 2000 | 262,144,000 | 4.18993 | 66.02 | 30.727% | First validation after crossing 250M |
| 2100 | 275,251,200 | 4.12064 | 61.60 | 31.237% | Continuation validation |
| 2200 | 288,358,400 | 4.07375 | 58.78 | 31.454% | Continuation validation |
| 2300 | 301,465,600 | 4.02982 | 56.25 | 31.857% | First validation above 300M |
| 2400 | 314,572,800 | 3.99674 | 54.42 | 32.092% | Continuation validation |
| 2436 | 319,291,392 | 3.98411 | 53.74 | 32.375% | Clean resume validation; preserved complete checkpoint |
| 2500 | 327,680,000 | 3.95959 | 52.44 | 32.594% | Continuation validation |
| 2600 | 340,787,200 | 3.92907 | 50.86 | 32.736% | Continuation validation |
| 2700 | 353,894,400 | 3.90170 | 49.49 | 33.068% | Continuation validation |
| 2800 | 367,001,600 | 3.88005 | 48.43 | 33.199% | Continuation validation |
| 2900 | 380,108,800 | 3.85094 | 47.04 | 33.354% | Continuation validation |
| 3000 | 393,216,000 | 3.83001 | 46.06 | 33.566% | Continuation validation |
| 3031 | 397,279,232 | 3.81320 | 45.30 | 33.694% | Clean resume validation; preserved complete checkpoint |
| 3100 | 406,323,200 | 3.79637 | 44.54 | 33.816% | Continuation validation |
| 3200 | 419,430,400 | 3.76876 | 43.33 | 34.036% | Continuation validation |
| 3300 | 432,537,600 | 3.74860 | 42.46 | 34.296% | Continuation validation |
| 3400 | 445,644,800 | 3.72747 | 41.57 | 34.421% | Continuation validation |
| 3500 | 458,752,000 | 3.71039 | 40.87 | 34.553% | Continuation validation |
| 3600 | 471,859,200 | 3.69297 | 40.16 | 34.653% | Latest captured validation |

The latest supplied completed session reaches training update **3630** (**475,791,360 tokens**), then passes checkpoint preflight and saves a COMPLETE checkpoint at update **3638** (**476,839,936 tokens**) before a safe session end. The latest fixed validation remains update **3600** / **471,859,200 tokens** with NLL **3.69297**, perplexity **40.16**, and token accuracy **34.653%**.

The **250M**, **300M**, and **400M** token levels have been crossed in the continuous training lineage, and the run is now close to the **500M** analytical milestone. The target is reached at a whole-update boundary, so the configured final token count is **1,500,119,040** rather than exactly 1.5B. The validation CSV contains every canonical validation event currently documented; it is not a complete raw training log.
