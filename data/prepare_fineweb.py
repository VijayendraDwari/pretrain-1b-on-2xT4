#!/usr/bin/env python3
"""Build a FineWeb token corpus matching the public D1 contract used by this project.

Historical D1 contract:
  source: HuggingFaceFW/fineweb @ v1.4.0
  config: CC-MAIN-2025-26
  split: train
  tokenizer family: meta-llama/Llama-3.2-1B
  vocab: 128256
  EOS: 128001
  train tokens: 1,485,000,000
  validation tokens: 15,000,000
  train shards: 24
  token dtype: little-endian uint32

IMPORTANT:
This is a reproducibility builder for the documented contract, not a claim that
a fresh 2026+ streaming run will be byte-identical to the historical D1 corpus.
Use the published historical tokenizer hashes / dataset checksums when exact
lineage identity is required.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np

SOURCE_DATASET = "HuggingFaceFW/fineweb"
SOURCE_REVISION = "v1.4.0"
SOURCE_CONFIG = "CC-MAIN-2025-26"
SOURCE_SPLIT = "train"
VOCAB = 128256
EOS = 128001
TRAIN_TOKENS = 1_485_000_000
VAL_TOKENS = 15_000_000
N_TRAIN_SHARDS = 24
VAL_MODULUS = 100
VAL_BUCKET = 0

def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def atomic_json(p: Path, obj):
    tmp=p.with_suffix(p.suffix+".partial")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True))
    os.replace(tmp,p)

def stable_bucket(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8","ignore")).digest()[:8],"big") % VAL_MODULUS

def load_tokenizer(source: str):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(source, use_fast=True)
    if len(tok) != VOCAB:
        raise RuntimeError(f"Tokenizer vocab mismatch: {len(tok)} != {VOCAB}")
    if tok.eos_token_id != EOS:
        raise RuntimeError(f"Tokenizer EOS mismatch: {tok.eos_token_id} != {EOS}")
    return tok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output", default="./fineweb_d1_1p5b")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B",
                    help="HF tokenizer repo or local tokenizer directory. HF access may be gated.")
    ap.add_argument("--train-tokens", type=int, default=TRAIN_TOKENS)
    ap.add_argument("--validation-tokens", type=int, default=VAL_TOKENS)
    ap.add_argument("--train-shards", type=int, default=N_TRAIN_SHARDS)
    args=ap.parse_args()

    from datasets import load_dataset
    out=Path(args.output); (out/"tokens/train").mkdir(parents=True,exist_ok=True)
    (out/"tokens/validation").mkdir(parents=True,exist_ok=True)
    (out/"tokenizer").mkdir(parents=True,exist_ok=True)

    tok=load_tokenizer(args.tokenizer)
    tok.save_pretrained(out/"tokenizer")

    # Streaming keeps local disk requirements bounded.
    ds=load_dataset(SOURCE_DATASET, SOURCE_CONFIG, split=SOURCE_SPLIT,
                    streaming=True, revision=SOURCE_REVISION)

    shard_targets=[args.train_tokens//args.train_shards]*args.train_shards
    for i in range(args.train_tokens % args.train_shards):
        shard_targets[i]+=1

    train_ix=0; train_written=0; val_written=0; docs=0
    train_file=None; train_in_shard=0
    val_path=out/"tokens/validation/validation_00000.bin"
    val_f=val_path.open("wb")

    def open_train(ix):
        return (out/f"tokens/train/train_{ix:05d}.bin").open("wb")

    train_file=open_train(0)
    try:
        for row in ds:
            if train_written>=args.train_tokens and val_written>=args.validation_tokens:
                break
            text=(row.get("text") or "").strip()
            if not text: continue
            docs+=1
            ids=tok.encode(text, add_special_tokens=False)
            if not ids: continue
            ids.append(EOS)
            arr=np.asarray(ids,dtype="<u4")
            if int(arr.max(initial=0)) >= VOCAB:
                raise RuntimeError("token id >= vocab")

            is_val = stable_bucket(text)==VAL_BUCKET and val_written<args.validation_tokens
            if is_val:
                take=min(len(arr), args.validation_tokens-val_written)
                arr[:take].tofile(val_f); val_written+=take
                continue

            if train_written>=args.train_tokens:
                continue
            pos=0
            while pos<len(arr) and train_written<args.train_tokens:
                target=shard_targets[train_ix]
                room=target-train_in_shard
                take=min(room,len(arr)-pos,args.train_tokens-train_written)
                arr[pos:pos+take].tofile(train_file)
                pos+=take; train_written+=take; train_in_shard+=take
                if train_in_shard==target:
                    train_file.flush(); os.fsync(train_file.fileno()); train_file.close()
                    train_ix+=1; train_in_shard=0
                    if train_ix<args.train_shards:
                        train_file=open_train(train_ix)
            if docs % 10000 == 0:
                print(f"[build] docs={docs:,} train={train_written:,}/{args.train_tokens:,} val={val_written:,}/{args.validation_tokens:,}",flush=True)
    finally:
        if train_file and not train_file.closed:
            train_file.flush(); train_file.close()
        val_f.flush(); val_f.close()

    if train_written != args.train_tokens or val_written != args.validation_tokens:
        raise RuntimeError(f"Incomplete build train={train_written} val={val_written}")

    artifacts={}
    for p in sorted((out/"tokens").rglob("*.bin")):
        artifacts[str(p.relative_to(out))]={"bytes":p.stat().st_size,"sha256":sha256_file(p),
                                          "tokens":p.stat().st_size//4}
    meta={
        "status":"COMPLETE_AND_VERIFIED",
        "source":{"dataset":SOURCE_DATASET,"revision":SOURCE_REVISION,"config":SOURCE_CONFIG,"split":SOURCE_SPLIT},
        "tokenizer_source":args.tokenizer,
        "vocab":VOCAB,"vocab_size":VOCAB,"eos":EOS,"token_dtype":"uint32",
        "train_tokens":train_written,"validation_tokens":val_written,
        "num_train_shards":args.train_shards,"validation_hash_modulus":VAL_MODULUS,
        "validation_hash_bucket":VAL_BUCKET,"documents_seen":docs,"artifacts":artifacts
    }
    atomic_json(out/"corpus_meta.json",meta)
    atomic_json(out/"_SUCCESS.json",meta)
    print(json.dumps(meta,indent=2))

if __name__=="__main__":
    main()
