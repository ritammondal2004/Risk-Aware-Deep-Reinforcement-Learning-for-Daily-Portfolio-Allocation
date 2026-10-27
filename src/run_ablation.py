"""One entry point for M1-M4: python -m src.run_ablation --model M1 --seed 0 --fold 1 [--smoke | --steps N]"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import CFG, MODEL_SPECS, config_hash, get_paths, with_lambda
from .data import build_folds
from .environment import EnvParams, PortfolioEnv
from .evaluation import compute_metrics, run_policy
from .features import build_dataset, standardize
from .models import ActorCritic
from .ppo import PPO
from .utils import get_device, set_global_seed

# The ONLY model-specific switch outside the env: which encoder reads the observation window.
ENCODER = {"M1": "mlp", "M2": "lstm", "M3": "lstm", "M4": "lstm"}


# Maps CFG.ppo's exact field names to the names ppo.py uses (identical for all four models).
def ppo_params(cfg):
    q = cfg.ppo
    return SimpleNamespace(
        lr=q.learning_rate, adam_betas=q.adam_betas, adam_eps=q.adam_eps, gamma=q.gamma,
        gae_lambda=q.gae_lambda, clip=q.clip_range, value_coef=q.value_coef,
        entropy_coef=q.entropy_coef, max_grad_norm=q.max_grad_norm, rollout_steps=q.rollout_len,
        epochs=q.n_epochs, minibatch=q.minibatch_size, normalize_reward=q.normalize_reward,
        total_steps=q.total_env_steps)


# Checks that models.py can honour the architecture in the config, so no config field is silently ignored.
def check_arch(m):
    assert m.lstm_layers == 1, "models.py implements a single LSTM layer"
    assert tuple(m.mlp_hidden) == (m.lstm_hidden, m.lstm_hidden) and m.repr_dim == m.lstm_hidden, \
        "models.py uses one width for the MLP (2 layers) and the LSTM; config values must agree"


# One folder name per run; metrics/logs go under outputs/metrics, the checkpoint under outputs/checkpoints.
def run_dirs(model, fold, seed, tag):
    paths = get_paths()
    name = f"{model}_fold{fold}_seed{seed}{tag}"
    m, c = Path(paths["metrics"]) / name, Path(paths["checkpoints"]) / name
    m.mkdir(parents=True, exist_ok=True)
    c.mkdir(parents=True, exist_ok=True)
    return m, c

# Returns cfg with the given PPO fields overridden (same override applies to every model you run).
def with_ppo(cfg, **kw):
    kw = {k: v for k, v in kw.items() if v is not None}
    return replace(cfg, ppo=replace(cfg.ppo, **kw)) if kw else cfg
      
# Trains one (model, fold, seed) run, evaluates greedily on val and test, saves everything.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=1, help="1-based fold number")
    ap.add_argument("--lam", type=float, default=None, help="override lambda_turnover (M4 tuning only)")
    ap.add_argument("--smoke", action="store_true", help="2 PPO updates only, to test the pipeline")
    ap.add_argument("--steps", type=int, default=None, help="override total training steps (short test runs)")
    ap.add_argument("--raw-inputs", action="store_true", help="skip standardization (A/B check only)")
    ap.add_argument("--lr", type=float, default=None, help="override PPO learning rate (stabilization sweep)")
    ap.add_argument("--epochs", type=int, default=None, help="override PPO epochs per update")
    ap.add_argument("--target-kl", type=float, default=None, help="enable KL early stopping")
    a = ap.parse_args()  

    cfg = with_lambda(CFG, a.lam) if a.lam is not None else CFG   
    cfg = with_ppo(cfg, learning_rate=a.lr, n_epochs=a.epochs, target_kl=a.target_kl) 

    check_arch(cfg.model)
    device = get_device()
    ds, _ = build_dataset()
    fold = build_folds(ds.dates)[a.fold - 1]
    X = ds.X if a.raw_inputs else standardize(ds.X, *fold["train"])   # train-only stats, same for all models
    params = EnvParams.from_cfg(a.model, cfg)                  # reward mode and lam differ per model HERE
    p = ppo_params(cfg)
    steps = a.steps or (2 * p.rollout_steps if a.smoke else p.total_steps)

    tr = PortfolioEnv(X, ds.ret_next, *fold["train"], params)
    set_global_seed(a.seed)                                    # seed right before model creation -> same init
    model = ActorCritic(ENCODER[a.model], tr.obs_shape[1], tr.action_dim,
                        hidden=cfg.model.lstm_hidden,
                        alpha_init=cfg.model.alpha_init, log_alpha_min=cfg.model.log_alpha_min,
                        log_alpha_max=cfg.model.log_alpha_max).to(device)
    print(f"{a.model} | encoder={ENCODER[a.model]} reward={params.reward} lam={params.lam} | "
          f"fold {a.fold} train steps {fold['train'][1] - fold['train'][0]} | device {device} | train {steps} steps"
          f" | inputs {'raw' if a.raw_inputs else 'standardized'}")

    log = PPO(model, tr, p, device, a.seed).train(steps)

    tag = "_smoke" if a.smoke else (f"_s{a.steps}" if a.steps else "")
    tag += f"_lam{a.lam}" if a.lam is not None else ""
    tag += "_raw" if a.raw_inputs else "" 
    tag += f"_lr{a.lr}" if a.lr is not None else ""
    tag += f"_ep{a.epochs}" if a.epochs is not None else ""
    tag += f"_kl{a.target_kl}" if a.target_kl is not None else "" 
    
    out, ckpt = run_dirs(a.model, a.fold, a.seed, tag)
    results = {"model": a.model, "seed": a.seed, "fold": a.fold, "train_steps": steps,
               "reward": params.reward, "lam": params.lam, "config_hash": config_hash(cfg)} 

    for split in ("val", "test"):                              # test is reported, never used for selection
        env = PortfolioEnv(X, ds.ret_next, *fold[split], params)
        rec = run_policy(model, env, device)
        results[split] = compute_metrics(rec["r_net"], rec["turnover"], rec["cost"], params.v0)
        np.savez(out / f"daily_{split}.npz", dates=np.asarray(ds.dates)[rec["index"].astype(int)].astype("datetime64[D]"),
                 **{k: rec[k] for k in ("r_net", "turnover", "cost", "v")})
    
    torch.save({"model": model.state_dict(), "name": a.model, "seed": a.seed}, ckpt / "model.pt")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "train_log.json").write_text(json.dumps(log))
    print(json.dumps({s: results[s] for s in ("val", "test")}, indent=2))
    print("saved to", out, "| checkpoint", ckpt)


if __name__ == "__main__":
    main()