# Memory budget and FSDP configuration

The supplied run log reports one allocator metric: **11.408836 GiB peak allocated on rank 0**. The trainer records `torch.cuda.max_memory_allocated() / 2**30` in its rank-0 training row and does not reset the peak counter during the run. This is rank 0's PyTorch allocator high-water mark, not total T4 device usage, reserved memory, free memory, or a measurement from both GPUs. The log does not contain a profiler breakdown by tensor category.

## Persistent training-state estimate

The following is an arithmetic estimate for evenly sharded FP32 parameters, FP32 gradients, and the two FP32 AdamW moment tensors. It is **not** an allocator measurement. It assumes the original parameters and optimizer state are held in FP32, consistent with the model being constructed in FP32 and the configured FSDP mixed-precision policy casting parameters for computation. The original sharded parameter is counted once; no extra master-weight copy is added.

The model has 1,018,126,848 parameters and two FSDP ranks. A full FP32 copy is 4 × parameter-count bytes. The idealized per-rank shard is therefore:

| State | Calculation per rank | Estimated GiB |
|---|---:|---:|
| Original parameter shard | 1,018,126,848 × 4 bytes ÷ 2 | 1.8964 |
| Gradient shard, reduced in FP32 | 1,018,126,848 × 4 bytes ÷ 2 | 1.8964 |
| AdamW first moment (`exp_avg`) | 1,018,126,848 × 4 bytes ÷ 2 | 1.8964 |
| AdamW second moment (`exp_avg_sq`) | 1,018,126,848 × 4 bytes ÷ 2 | 1.8964 |
| **Estimated persistent tensor subtotal** | **16 bytes per parameter ÷ 2 ranks** | **7.5856** |

AdamW creates its moment tensors lazily at the first optimizer step. The scalar step counters and GradScaler state are small relative to these tensors. This estimate describes the persistent, sharded state, not the total peak during forward, backward, optimizer updates, validation, or checkpointing.

The measured rank-0 high-water mark is about **3.8232 GiB above** that idealized subtotal. The supplied logs do not let us assign this difference precisely. It may include transient all-gathered parameters, checkpoint-boundary activations, attention/MLP intermediates, loss/logit tensors, CUDA kernel workspaces, and FSDP/NCCL-related allocations. PyTorch allocator statistics also do not include all memory visible to device-level tools such as `nvidia-smi`; do not treat 11.4088 GiB as a total-device-usage figure or infer the remaining VRAM from it.

## FSDP and precision settings used

The trainer launches two local workers with `torchrun --standalone --max_restarts=0 --nproc_per_node=2` and initializes the `nccl` process group. It wraps the 29 Transformer `Block` modules individually using `transformer_auto_wrap_policy`; the embedding and final normalization remain parameters owned by the root FSDP module.

The FSDP wrapper in the trainer is configured as follows:

```python
MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float16,
)

FSDP(
    model,
    auto_wrap_policy=partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    ),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp,
    device_id=device,
    sync_module_states=True,
    use_orig_params=True,
    limit_all_gathers=True,
)
```

The trainer does not set `keep_low_precision_grads`, so FSDP uses its default `False` and upcasts reduced gradients to full precision after backward. This is why the estimate above includes FP32 gradient shards. See the [PyTorch FSDP mixed-precision API](https://docs.pytorch.org/docs/2.14/fsdp.html) for the dtype semantics.

`FULL_SHARD` shards parameters, gradients, and optimizer states across the two ranks. During computation, FSDP gathers the parameters needed by an FSDP unit in FP16 and releases them according to the unit lifecycle. Since the blocks are wrapped separately, a Transformer block has 28,314,624 parameters, or about **54.01 MiB** when gathered in FP16. The root-owned token embedding contains 197,001,216 parameters, or **375.75 MiB** as a full FP16 tensor. These are tensor-size calculations for transient full-parameter representations; they are not additive to the measured peak as if every tensor were simultaneously resident. `limit_all_gathers=True` constrains FSDP's rate of issuing all-gathers; it does not eliminate the temporary gathered parameters.

The model uses one sequence of 4096 tokens per rank per microbatch and 16 gradient-accumulation steps. That gives 131,072 model-input tokens per optimizer update across both GPUs. Accumulation reduces activation needs per microbatch; it does not divide the persistent parameter, gradient, or optimizer-state footprint by 16.

Activation checkpointing is applied **after** FSDP wrapping to each `Block`, using `CheckpointImpl.NO_REENTRANT` with `preserve_rng_state=False`. A single FP16 hidden-state tensor with shape `[1, 4096, 1536]` is 12 MiB; 29 such block-boundary tensors are about **348 MiB** before other retained tensors and temporary workspaces. Checkpointing saves block inputs and recomputes block internals in backward. The exact live activation peak depends on PyTorch's autograd behavior and the SDPA kernel selected at runtime.

The trainer calls `torch.nn.functional.scaled_dot_product_attention` with causal masking and zero dropout; it does not explicitly force a FlashAttention backend. It computes the full-vocabulary loss in 128-position chunks. Each chunk has 128 × 128,256 logits: about **31.31 MiB** in FP16 and **62.63 MiB** for the FP32 logits passed to cross-entropy. These are per-chunk tensor sizes, not a claim that total loss memory is limited to one chunk: the current code accumulates chunk-loss graphs and calls backward after the forward pass.

Other settings that affect memory and stability are FP16 autocast, `torch.amp.GradScaler` with initial scale 128 and growth interval 1000, global gradient clipping at 1.0, and AdamW with `foreach=False`. The latter avoids the extra temporary tensor lists used by foreach implementations. The optimizer uses betas `(0.9, 0.95)` and epsilon `1e-8`; the two moment tensors are included in the persistent-state estimate above.

## Checkpoint storage is not GPU memory

During a save, the trainer creates sharded model and optimizer state dictionaries with `offload_to_cpu=True`. Before saving, rank 0 checks that `/kaggle/working` has at least **11 GiB of free disk space** for the two-rank checkpoint and partial-write headroom. This is a storage preflight, not a VRAM requirement. Rank files are written as `.partial`, renamed after serialization, and considered resumable only after rank 0 writes `complete.json`.

## What would make the breakdown truly measured

To replace the estimate with a category-by-category budget, collect a profiler or explicit per-stage snapshots on both ranks after the optimizer state is initialized. Record PyTorch allocated and reserved peaks, device-level usage, optimizer-state tensor dtypes and bytes, and the selected SDPA backend. The existing captured log is sufficient to report the rank-0 allocator high-water mark, but not to attribute every GiB of that peak to a specific category.
