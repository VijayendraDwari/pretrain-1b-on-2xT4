#!/usr/bin/env python3
"""Evaluate a COMPLETE sharded checkpoint on the fixed FineWeb-D1 validation set.

Run:
  python eval/evaluate_lm.py --checkpoint /path/to/update_00000616 --data-root /path/to/d1

The script launches exactly two workers because checkpoints are saved as per-rank FSDP state.
"""
from __future__ import annotations
import argparse, functools, os, subprocess, sys
from datetime import timedelta
from pathlib import Path

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--data-root',required=True)
    return p.parse_args()

def main():
    args=parse()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'train'))
    if 'LOCAL_RANK' not in os.environ:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--max_restarts=0',
             '--nproc_per_node=2',str(Path(__file__).resolve()),
             '--checkpoint',args.checkpoint,'--data-root',args.data_root]
        raise SystemExit(subprocess.run(cmd,env=os.environ.copy()).returncode)

    os.environ['RESUME_FROM']=str(Path(args.checkpoint).resolve())
    os.environ['FINEWEB_ROOT']=str(Path(args.data_root).resolve())
    import torch
    import torch.distributed as dist
    import train_1b_dual_t4 as t
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl

    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); local=int(os.environ['LOCAL_RANK'])
    if world != 2: raise RuntimeError('Evaluation requires the two-rank checkpoint topology')
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(hours=1),device_id=device)

    data=t.contract(); val=t.Tokens(data/'tokens/validation','validation_*.bin')
    bare=t.LM().to(device)
    mp=MixedPrecision(param_dtype=torch.float16,reduce_dtype=torch.float32,buffer_dtype=torch.float16)
    fsdp=FSDP(bare,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={t.Block}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,mixed_precision=mp,device_id=device,
        sync_module_states=True,use_orig_params=True,limit_all_gathers=True)
    apply_activation_checkpointing(fsdp,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper,checkpoint_impl=CheckpointImpl.NO_REENTRANT,preserve_rng_state=False),
        check_fn=lambda m:isinstance(m,t.Block))
    decay=[]; no_decay=[]
    for n,p in fsdp.named_parameters():
        (decay if p.ndim>=2 and 'tok_emb' not in n else no_decay).append(p)
    opt=torch.optim.AdamW([{'params':decay,'weight_decay':.1},{'params':no_decay,'weight_decay':0.}],
                          lr=t.LR,betas=(.9,.95),eps=1e-8,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000)
    update=t.resume(fsdp,opt,scaler,rank)
    t.validate(fsdp,val,device,rank,update)
    dist.barrier(); dist.destroy_process_group()

if __name__=='__main__':
    main()
