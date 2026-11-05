#!/usr/bin/env python3
"""Launch the canonical 1.018B dual-T4 trainer.

Typical Kaggle use:
    python train/launch_kaggle.py

The trainer itself launches exactly two torchrun workers.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_1b_dual_t4.py"
CONFIG = Path(os.environ.get("DENSE1B_CONFIG", HERE / "config_1b_t4.json"))

def main() -> int:
    cfg = json.loads(CONFIG.read_text())
    env = os.environ.copy()
    mapping = {
        "RUN_ROOT": cfg["run_root"],
        "FINEWEB_ROOT": cfg["fineweb_root"],
        "CONTEXT": str(cfg["context"]),
        "GRAD_ACCUM": str(cfg["grad_accum"]),
        "TARGET_TOKENS": str(cfg["target_tokens"]),
        "PEAK_LR": str(cfg["peak_lr"]),
        "WARMUP_UPDATES": str(cfg["warmup_updates"]),
        "VAL_EVERY": str(cfg["val_every"]),
        "SAVE_EVERY": str(cfg["save_every"]),
        "SESSION_LIMIT_SEC": str(cfg["session_limit_sec"]),
        "SAVE_MARGIN_SEC": str(cfg["save_margin_sec"]),
        "QUALIFY_UPDATES": str(cfg.get("qualify_updates", 0)),
    }
    if cfg.get("resume_from"):
        mapping["RESUME_FROM"] = cfg["resume_from"]
    env.update(mapping)
    print("[launch config]", json.dumps(mapping, indent=2), flush=True)
    return subprocess.run([sys.executable, "-u", str(TRAINER)], env=env).returncode

if __name__ == "__main__":
    raise SystemExit(main())
