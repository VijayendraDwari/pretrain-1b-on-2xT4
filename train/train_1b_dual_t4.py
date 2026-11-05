#!/usr/bin/env python3
"""Educational dense 1.018B GPT continuation on Kaggle 2xT4.
Launch from notebook: !python /kaggle/working/train_dense_1b_kaggle.py
Dependencies: Kaggle PyTorch with torch.distributed.fsdp (FSDP1).
Same FineWeb-D1 uint32 token shards and tokenizer IDs as densecontrol-con3.

This is the crash-safe CONTINUATION trainer for the educational dense-1B lineage.
Resume ONLY from a COMPLETE two-rank checkpoint. Partial checkpoint directories are ignored.
Each Kaggle session writes at most one large full FSDP checkpoint near the time guard.
The mounted parent remains the immutable rollback checkpoint, avoiding duplicate multi-GB
checkpoints in /kaggle/working.
"""
import os,sys,json,math,time,random,shutil,signal,hashlib,socket,subprocess,warnings,fcntl
from pathlib import Path
from datetime import timedelta
from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType,
  ShardedStateDictConfig, ShardedOptimStateDictConfig, FullStateDictConfig)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (apply_activation_checkpointing,checkpoint_wrapper,CheckpointImpl)

ROOT=Path(os.environ.get('RUN_ROOT','/kaggle/working/educational_dense_1b_continue_v3'))
DATA=Path(os.environ.get('FINEWEB_ROOT','/kaggle/input/datasets/vijayendrad/fineweb-d1/sca_fineweb_2025_26_clean_llama32_1p5b'))
RESUME_FROM=os.environ.get('RESUME_FROM','')
SEED=int(os.environ.get('SEED','71001'))
VOCAB=128256; WIDTH=1536; DEPTH=29; HEADS=24; HD=64; FF=6144
CTX=int(os.environ.get('CONTEXT','4096')); GA=int(os.environ.get('GRAD_ACCUM','16'))
TARGET=int(os.environ.get('TARGET_TOKENS','1500000000'))
LR=float(os.environ.get('PEAK_LR','0.00015')); MIN_LR_RATIO=0.1; WARMUP=int(os.environ.get('WARMUP_UPDATES','75'))
SAVE_INTERVAL=int(os.environ.get('SAVE_EVERY','0')); VAL_INTERVAL=int(os.environ.get('VAL_EVERY','100'))
VAL_STEPS=int(os.environ.get('VAL_STEPS','16')); CHUNK=int(os.environ.get('LOSS_CHUNK','128'))
SESSION_LIMIT=int(os.environ.get('SESSION_LIMIT_SEC','39600')); SAVE_MARGIN=int(os.environ.get('SAVE_MARGIN_SEC','1800'))
QUALIFY_UPDATES=int(os.environ.get('QUALIFY_UPDATES','0'))  # 0=normal; >0 run this many updates then save/stop
WORLD=2; MICRO_TOKENS=WORLD*CTX; TOKENS_PER_UPDATE=MICRO_TOKENS*GA
TOTAL_UPDATES=math.ceil(TARGET/TOKENS_PER_UPDATE)
assert CTX>=256 and CTX%128==0 and GA>=1
STOP=False

# Suppress *only* known FSDP1 deprecation noise in this Kaggle/PyTorch version.
# Real errors and other warnings remain visible.
warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*FSDP\.state_dict_type\(\).*')
warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*FullyShardedDataParallel\.sharded_optim_state_dict.*')
warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*ShardedTensor.*')
warnings.filterwarnings('ignore', category=FutureWarning, message=r'.*DTensor instead.*')
warnings.filterwarnings('ignore', category=UserWarning, message=r'.*_get_pg_default_device.*')
warnings.filterwarnings('ignore', category=UserWarning, message=r'.*barrier\(\): using the device under current context.*')

def stop_handler(*_):
 global STOP
 STOP=True
signal.signal(signal.SIGTERM,stop_handler)

def emit(o):
 if dist.get_rank()==0: print(o,flush=True)

def append_log(path,obj):
 if dist.get_rank()==0:
  path.parent.mkdir(parents=True,exist_ok=True)
  with open(path,'a') as f: f.write(json.dumps(obj,sort_keys=True)+'\n')

def rms(x,w,eps=1e-5):
 t=x.float(); return (t*torch.rsqrt(t.square().mean(-1,keepdim=True)+eps)*w.float()).to(x.dtype)

class Norm(nn.Module):
 def __init__(self):
  super().__init__();self.weight=nn.Parameter(torch.ones(WIDTH))
 def forward(self,x):return rms(x,self.weight)

class Attention(nn.Module):
 def __init__(self):
  super().__init__();self.qkv=nn.Linear(WIDTH,3*WIDTH,bias=False);self.out=nn.Linear(WIDTH,WIDTH,bias=False)
 def forward(self,x):
  b,t,c=x.shape
  q,k,v=self.qkv(x).view(b,t,3,HEADS,HD).unbind(2)
  q=q.transpose(1,2);k=k.transpose(1,2);v=v.transpose(1,2)
  # half-split RoPE, as in the parent notebook
  pos=torch.arange(t,device=x.device,dtype=torch.float32)
  inv=10000.0**(-torch.arange(HD//2,device=x.device,dtype=torch.float32)/(HD//2))
  phase=pos[:,None]*inv[None,:]; cs=phase.cos()[None,None];sn=phase.sin()[None,None]
  def rot(y):
   a,b=y.float().chunk(2,dim=-1)
   return torch.cat((a*cs-b*sn,a*sn+b*cs),dim=-1).to(y.dtype)
  q=rot(q);k=rot(k)
  y=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.0)
  return self.out(y.transpose(1,2).contiguous().view(b,t,c))

class Block(nn.Module):
 def __init__(self):
  super().__init__();self.norm1=Norm();self.attn=Attention();self.norm2=Norm()
  self.fc=nn.Linear(WIDTH,FF,bias=False);self.proj=nn.Linear(FF,WIDTH,bias=False)
 def forward(self,x):
  x=(x.float()+self.attn(self.norm1(x)).float()).to(x.dtype)
  y=self.proj(F.gelu(self.fc(self.norm2(x)),approximate='tanh'))
  return (x.float()+y.float()).to(x.dtype)

class LM(nn.Module):
 def __init__(self):
  super().__init__();self.tok_emb=nn.Embedding(VOCAB,WIDTH)
  self.blocks=nn.ModuleList(Block() for _ in range(DEPTH));self.norm=Norm()
  self.apply(self.init_weights)
  for block in self.blocks:
   nn.init.normal_(block.attn.out.weight,std=.02/math.sqrt(2*DEPTH))
   nn.init.normal_(block.proj.weight,std=.02/math.sqrt(2*DEPTH))
 @staticmethod
 def init_weights(m):
  if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=.02)
 def forward(self,x,y):
  z=self.tok_emb(x)
  for b in self.blocks:z=b(z)
  z=self.norm(z)
  # Explicit chunked full-vocabulary loss: avoids B*T*128256 logits in memory
  losses=[];correct=torch.zeros((),dtype=torch.float32,device=x.device)
  for a in range(0,x.numel(),CHUNK):
   h=z.reshape(-1,WIDTH)[a:a+CHUNK]
   labels=y.reshape(-1)[a:a+CHUNK]
   logits=F.linear(h,self.tok_emb.weight)
   losses.append(F.cross_entropy(logits.float(),labels,reduction='sum'))
   with torch.no_grad():correct+= (logits.argmax(-1)==labels).sum()
  return torch.stack(losses).sum()/y.numel(),correct/y.numel()

class Tokens:
 def __init__(self,path,glob):
  files=sorted(path.glob(glob));assert files,f'Missing token files in {path}'
  self.arr=[np.memmap(p,dtype='<u4',mode='r') for p in files]
  self.total=sum(map(len,self.arr));self.files=[str(p) for p in files]
  assert min(map(len,self.arr))>CTX+1
 def batch(self,sid,seed):
  z=(int(seed)*6364136223846793005+int(sid)*1442695040888963407)&((1<<64)-1)
  a=self.arr[z%len(self.arr)]
  z2=((z>>17)^(z*0x9E3779B97F4A7C15))&((1<<64)-1)
  off=z2%(len(a)-(CTX+1))
  v=np.asarray(a[off:off+CTX+1],dtype=np.int64)
  return torch.from_numpy(v[:-1].copy())[None],torch.from_numpy(v[1:].copy())[None]

def contract():
 if not (DATA/'tokens/train').is_dir():
  found=list(Path('/kaggle/input').glob('**/sca_fineweb_2025_26_clean_llama32_1p5b'))
  if len(found)==1:return found[0]
  raise FileNotFoundError('Mount vijayendrad/fineweb-d1 or set FINEWEB_ROOT')
 return DATA

def lr_at(u):
 if u<WARMUP:return LR*(u+1)/max(WARMUP,1)
 p=min(1.,max(0.,(u-WARMUP)/max(1,TOTAL_UPDATES-WARMUP)))
 return LR*(MIN_LR_RATIO+(1-MIN_LR_RATIO)*.5*(1+math.cos(math.pi*p)))


def _checkpoint_config_expected():
 return {'depth':DEPTH,'width':WIDTH,'heads':HEADS,'vocab':VOCAB,'ctx':CTX,
         'ga':GA,'seed':SEED,'target':TARGET,'world':WORLD}

def _valid_complete_checkpoint_dir(p):
 try:
  if not (p/'complete.json').is_file(): return None
  if not all((p/f'rank{i}.pt').is_file() for i in range(WORLD)): return None
  m=json.loads((p/'complete.json').read_text())
  if not bool(m.get('complete',False)): return None
  if m.get('config') != _checkpoint_config_expected(): return None
  return int(m['update']),p
 except Exception:
  return None

def discover_resume_source():
 # Explicit path wins.
 if RESUME_FROM:
  p=Path(RESUME_FROM)
  if not p.exists(): raise FileNotFoundError(f'RESUME_FROM does not exist: {p}')
  return p

 # Current output root is considered first only if it already contains COMPLETE state.
 candidates=[]
 roots=[ROOT/'checkpoints']
 for r in roots:
  if r.exists():
   if (r/'complete.json').exists():
    v=_valid_complete_checkpoint_dir(r)
    if v:candidates.append(v)
   for p in r.glob('update_*'):
    v=_valid_complete_checkpoint_dir(p)
    if v:candidates.append(v)

 # Mounted Kaggle outputs/datasets: only COMPLETE checkpoints matching the exact lineage contract.
 kaggle_input=Path('/kaggle/input')
 if kaggle_input.exists():
  for meta in kaggle_input.glob('**/complete.json'):
   p=meta.parent
   v=_valid_complete_checkpoint_dir(p)
   if v:candidates.append(v)

 if not candidates:
  return None
 candidates.sort(key=lambda z:z[0])
 best=candidates[-1][1]
 return best

def checkpoint_bytes(p):
 return sum(x.stat().st_size for x in p.glob('rank*.pt') if x.is_file())

def checkpoint_preflight(rank):
 # One full two-rank sharded checkpoint is large. Refuse to start the save if
 # working storage cannot safely hold it. The mounted parent is read-only rollback.
 ok=True; msg=''
 if rank==0:
  free=shutil.disk_usage('/kaggle/working').free
  # Empirical qualification checkpoint was several GiB per rank.
  # 11 GiB leaves modest metadata/partial-write headroom.
  need=11*1024**3
  ok=free>=need
  msg=f'free_working_gib={free/1024**3:.2f} required_gib={need/1024**3:.2f}'
  emit('[checkpoint preflight] '+msg)
 flag=torch.tensor([1 if ok else 0],device=torch.device('cuda',int(os.environ['LOCAL_RANK'])),dtype=torch.int32)
 dist.broadcast(flag,src=0)
 if int(flag.item()) != 1:
  raise RuntimeError('Insufficient /kaggle/working free space for full two-rank checkpoint: '+msg)

def save(fsdp,opt,scaler,rank,u,root,reason):
 # One crash-safe sharded checkpoint per Kaggle continuation session.
 # Mounted parent remains immutable rollback; partial new directories are never resumable.
 checkpoint_preflight(rank)
 ck=root/'checkpoints'/f'update_{u:08d}'
 if rank==0 and ck.exists() and not (ck/'complete.json').exists():
  shutil.rmtree(ck)
 dist.barrier()
 ck.mkdir(parents=True,exist_ok=True)
 mc=ShardedStateDictConfig(offload_to_cpu=True);oc=ShardedOptimStateDictConfig(offload_to_cpu=True)
 with FSDP.state_dict_type(fsdp,StateDictType.SHARDED_STATE_DICT,mc,oc):
  sd=fsdp.state_dict(); od=FSDP.optim_state_dict(fsdp,opt)
 rec={'model':sd,'optimizer':od,'scaler':scaler.state_dict(),'update':u,'tokens':u*TOKENS_PER_UPDATE,
      'config':{'depth':DEPTH,'width':WIDTH,'heads':HEADS,'vocab':VOCAB,'ctx':CTX,'ga':GA,'seed':SEED,'target':TARGET,'world':WORLD},'reason':reason}
 tmp=ck/f'rank{rank}.pt.partial';torch.save(rec,tmp);os.replace(tmp,ck/f'rank{rank}.pt')
 dist.barrier()
 if rank==0:
  meta={'update':u,'tokens':u*TOKENS_PER_UPDATE,'complete':True,'reason':reason,'path':str(ck),
        'config':rec['config'],'saved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
  temp=ck/'complete.json.partial';temp.write_text(json.dumps(meta,indent=2));os.replace(temp,ck/'complete.json')
  emit(f'[saved] {ck}')
 dist.barrier()
 # Current session keeps only its newly completed checkpoint.
 # Deletion happens AFTER the new COMPLETE marker. The mounted parent remains untouched.
 if rank==0:
  for old in sorted((root/'checkpoints').glob('update_*')):
   if old != ck and (old/'complete.json').exists():
    shutil.rmtree(old)
 dist.barrier()
 del sd,od,rec

def resume(fsdp,opt,scaler,rank):
 source=discover_resume_source()
 if source is None:
  return 0

 # source may be an exact update directory or a checkpoints directory.
 if (source/'complete.json').exists():
  choices=[source]
 else:
  choices=list(source.glob('update_*')) if source.exists() else []

 valid=[]
 for p in choices:
  v=_valid_complete_checkpoint_dir(p)
  if v: valid.append(v)
 if not valid:
  raise RuntimeError(f'No COMPLETE exact-lineage checkpoint found under {source}')

 u,p=max(valid,key=lambda z:z[0])
 rec=torch.load(p/f'rank{rank}.pt',map_location='cpu',weights_only=False)
 expected=_checkpoint_config_expected()
 if rec['config']!=expected:
  raise RuntimeError('Checkpoint config differs; exact resume refused: '+str(rec['config']))

 mc=ShardedStateDictConfig(offload_to_cpu=True)
 oc=ShardedOptimStateDictConfig(offload_to_cpu=True)
 with FSDP.state_dict_type(fsdp,StateDictType.SHARDED_STATE_DICT,mc,oc):
  fsdp.load_state_dict(rec['model'],strict=True)
  opt.load_state_dict(FSDP.optim_state_dict_to_load(fsdp,opt,rec['optimizer']))
 scaler.load_state_dict(rec['scaler'])
 dist.barrier()
 emit(f'[resume] {p} update={u} tokens={u*TOKENS_PER_UPDATE:,}')
 return u

@torch.no_grad()
def validate(fsdp,val,device,rank,u):
 fsdp.eval(); ls=0.;ac=0.
 for j in range(VAL_STEPS):
  x,y=val.batch(900_000_000+j*WORLD+rank,SEED+77)
  with torch.autocast('cuda',dtype=torch.float16):
   loss,acc=fsdp(x.to(device),y.to(device))
  ls+=float(loss);ac+=float(acc)
 # all ranks take same number of calls, then average
 vals=torch.tensor([ls,ac],device=device,dtype=torch.float64)
 dist.all_reduce(vals)
 nll=float(vals[0])/(VAL_STEPS*WORLD)
 out={'update':u,'tokens':u*TOKENS_PER_UPDATE,'nll':nll,'ppl':math.exp(min(nll,30)),
  'token_accuracy':float(vals[1])/(VAL_STEPS*WORLD),'val_tokens':VAL_STEPS*WORLD*CTX}
 append_log(ROOT/'logs/validation.jsonl',out);emit('[validation] '+str(out));fsdp.train()

def worker():
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
 assert world==WORLD, 'Requires exactly two GPUs'
 torch.cuda.set_device(local);device=torch.device('cuda',local)
 dist.init_process_group('nccl',timeout=timedelta(hours=2),device_id=device)
 if rank != 0:
  # Rank 1 diagnostics are preserved in a separate file, not dumped on notebook output.
  ROOT.mkdir(parents=True,exist_ok=True)
  ranklog=open(ROOT/f'worker_rank{rank}.log','a',buffering=1)
  sys.stdout=ranklog
  sys.stderr=ranklog
 torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED);torch.cuda.manual_seed_all(SEED)
 ROOT.mkdir(parents=True,exist_ok=True)
 data=contract();train=Tokens(data/'tokens/train','train_*.bin');val=Tokens(data/'tokens/validation','validation_*.bin')
 info=json.loads((data/'corpus_meta.json').read_text())
 assert int(info['vocab_size'])==VOCAB and info['token_dtype'].lower()=='uint32',info
 emit(f'[dataset] {data} train_tokens={train.total:,} validation_tokens={val.total:,}')
 emit(f'[GPU] {torch.cuda.get_device_name(local)} rank={rank} context={CTX} world={world}')
 bare=LM();params=sum(p.numel() for p in bare.parameters());emit(f'[parameters] {params:,}')
 assert params>=1_000_000_000,params
 # Wrap FSDP blocks first; checkpoint wrappers are installed afterward (matching proven notebook).
 mp=MixedPrecision(param_dtype=torch.float16,reduce_dtype=torch.float32,buffer_dtype=torch.float16)
 fsdp=FSDP(bare,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Block}),
     sharding_strategy=ShardingStrategy.FULL_SHARD,mixed_precision=mp,device_id=device,
     sync_module_states=True,use_orig_params=True,limit_all_gathers=True)
 apply_activation_checkpointing(fsdp,checkpoint_wrapper_fn=partial(checkpoint_wrapper,checkpoint_impl=CheckpointImpl.NO_REENTRANT,preserve_rng_state=False),check_fn=lambda m:isinstance(m,Block))
 decay=[];no_decay=[]
 for n,p in fsdp.named_parameters():
  (decay if p.ndim>=2 and 'tok_emb' not in n else no_decay).append(p)
 opt=torch.optim.AdamW([{'params':decay,'weight_decay':.1},{'params':no_decay,'weight_decay':0.}],
   lr=LR,betas=(.9,.95),eps=1e-8,foreach=False)
 scaler=torch.amp.GradScaler('cuda',init_scale=128.,growth_interval=1000)
 u=resume(fsdp,opt,scaler,rank)
 if u>0:
  emit('[continuation policy] immutable mounted COMPLETE parent; no periodic full checkpoint unless SAVE_EVERY>0; one time-guard checkpoint')
 if u==0:
  manifest={'model_params':params,'dataset':str(data),'context':CTX,'world':WORLD,'ga':GA,
  'tokens_per_update':TOKENS_PER_UPDATE,'target_tokens':TARGET,'total_updates':TOTAL_UPDATES,
  'attention':'torch SDPA causal','new_random_init':True,'checkpoint_policy':'one_full_checkpoint_per_session','run_start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
  if rank==0:(ROOT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
  emit('[fresh start] '+str(manifest))
 validate(fsdp,val,device,rank,u)
 session=time.time();last_saved=u;fsdp.train();session_start_update=u
 while u<TOTAL_UPDATES:
  if STOP or time.time()-session>SESSION_LIMIT-SAVE_MARGIN:
   if u!=last_saved:save(fsdp,opt,scaler,rank,u,ROOT,'time_guard')
   emit('[session end] safe boundary');break
  start=time.time();opt.zero_grad(set_to_none=True)
  lr=lr_at(u)
  for group in opt.param_groups:group['lr']=lr
  sum_loss=0.
  for g in range(GA):
   sid=(u*GA+g)*WORLD+rank
   x,y=train.batch(sid,SEED+11)
   x=x.to(device);y=y.to(device)
   with torch.autocast('cuda',dtype=torch.float16):loss,acc=fsdp(x,y)
   if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite loss at update {u}')
   scaler.scale(loss/GA).backward();sum_loss+=float(loss.detach())
  scaler.unscale_(opt);norm=fsdp.clip_grad_norm_(1.)
  if not bool(torch.isfinite(norm)):raise FloatingPointError(f'Nonfinite gradient at {u}')
  prev=scaler.get_scale();scaler.step(opt);scaler.update()
  if scaler.get_scale()<prev:
   # IMPORTANT: no optimizer update, and deterministic data microbatches must not be skipped.
   raise FloatingPointError('GradScaler overflow: exact token/update ledger would drift; aborting safely')
  torch.cuda.synchronize();u+=1
  dt=time.time()-start
  if rank==0:
   row={'update':u,'tokens':u*TOKENS_PER_UPDATE,'loss_rank0':sum_loss/GA,'grad_norm':float(norm),
      'lr':lr,'step_sec':dt,'tokens_per_sec':TOKENS_PER_UPDATE/dt,'allocated_gib':torch.cuda.max_memory_allocated()/(1024**3)}
   if u%10==0 or u==1:emit('[train] '+str(row))
  if rank==0 and (u%10==0 or u==1):append_log(ROOT/'logs/train.jsonl',row)
  if u%VAL_INTERVAL==0:validate(fsdp,val,device,rank,u)
  if SAVE_INTERVAL>0 and u%SAVE_INTERVAL==0:
   save(fsdp,opt,scaler,rank,u,ROOT,'periodic');last_saved=u
  if QUALIFY_UPDATES>0 and u-session_start_update>=QUALIFY_UPDATES:
   if u!=last_saved:save(fsdp,opt,scaler,rank,u,ROOT,'qualification');last_saved=u
   emit('[qualification complete] saved resumable state; rerun with QUALIFY_UPDATES=0');break
 if u>=TOTAL_UPDATES:
  if u!=last_saved:save(fsdp,opt,scaler,rank,u,ROOT,'target')
  validate(fsdp,val,device,rank,u);emit('[complete] reached training token target')
 dist.barrier();dist.destroy_process_group()

def main():
    # The notebook invokes *one* top-level Python process. It launches torchrun
    # *once*. An OS lock prevents overlapping notebook cells from launching a
    # second independent job with another rank 0 on the same GPUs.
    if 'LOCAL_RANK' in os.environ:
        return worker()
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Select Kaggle GPU T4 x2 before launching')
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / '.launcher.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A training launcher already owns this RUN_ROOT. Do not execute the training cell twice.')
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = '1'  # prevent torchrun's duplicated startup warnings
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONWARNINGS'] = 'ignore::FutureWarning'  # checkpoint FSDP1 compatibility noise
        args = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--max_restarts=0', '--nproc_per_node=2', str(Path(__file__).resolve())]
        print('[launcher] one torchrun process; 2 workers; rank-zero-only console logging', flush=True)
        result = subprocess.run(args, env=env)
        raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()

