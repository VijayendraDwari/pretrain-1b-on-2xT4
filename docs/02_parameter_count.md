# Parameter count

The trainer prints the parameter count at startup and checks that it is above one billion. For the current dimensions, the exact count is **1,018,126,848**.

The input embedding has `128,256 × 1,536 = 197,001,216` parameters. Its weights are tied to the output projection, so they are counted once.

Each Transformer block has four attention matrices (`Q`, `K`, `V`, and output), two feed-forward matrices, and two RMSNorm scale vectors:

```text
attention: 4 × 1,536 × 1,536       =  9,437,184
FFN:       2 × 1,536 × 6,144       = 18,874,368
RMSNorm:   2 × 1,536                =      3,072
one block                            28,314,624
29 blocks                           821,124,096
```

The final RMSNorm adds 1,536 parameters. There are no linear biases in this implementation.

```text
197,001,216 embedding
+821,124,096 Transformer blocks
+      1,536 final RMSNorm
=1,018,126,848 trainable parameters
```

The formulas follow the current model code. They should be recalculated if the vocabulary, width, depth, FFN, or parameter sharing changes.
