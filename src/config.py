import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace 
from pathlib import Path
from typing import Tuple

import pandas as pd

PROV_PAPER, PROV_STANDARD, PROV_OURS = "PAPER", "STANDARD", "OURS"

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("BTP1_HOME", ROOT))   # set BTP1_HOME before import to redirect data/outputs (e.g. to Drive)


def get_paths(create=True):
    p = {
        "raw": HOME / "data" / "raw",
        "processed": HOME / "data" / "processed",
        "metrics": HOME / "outputs" / "metrics",
        "logs": HOME / "outputs" / "logs",
        "checkpoints": HOME / "outputs" / "checkpoints",
        "plots": HOME / "outputs" / "plots",
    }
    if create:
        for d in p.values():
            d.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class DataConfig:
    tickers: tuple = (
        "RELIANCE.NS",   # Energy / Conglomerate
        "TCS.NS",        # Information Technology
        "HDFCBANK.NS",   # Banking
        "ICICIBANK.NS",  # Banking
        "ITC.NS",        # FMCG / Consumer
        "HINDUNILVR.NS", # FMCG / Consumer
        "LT.NS",         # Industrials / Infrastructure
        "SUNPHARMA.NS",  # Healthcare / Pharma
        "MARUTI.NS",     # Automobile  
        "BHARTIARTL.NS"  # Telecom
    ) 
    download_start: str = "2009-01-01"    # warm-up for EMA-200
    sample_start: str = "2011-01-01"
    sample_end: str = "2025-12-31"


@dataclass(frozen=True)
class IndicatorConfig:
    ema_fast: int = 50          # Wang & Liu 2025, p.8
    ema_slow: int = 200
    rsi_period: int = 14
    macd_fast: int = 12 
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_num_std: float = 2.0
    adx_period: int = 14        # STANDARD: Wilder (1978)
    volume_sma_period: int = 20 # OURS 


@dataclass(frozen=True)
class StateConfig:
    n_features_per_asset: int = 13
    window: int = 30            # Zou et al. 2023, Table 2


@dataclass(frozen=True)
class ModelConfig:
    repr_dim: int = 512
    lstm_hidden: int = 512      # Zou et al. 2023, Table 3
    lstm_layers: int = 1
    mlp_hidden: Tuple[int, ...] = (512, 512)
    action_threshold: float = 0.01   # tau, OURS


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99             # Zou et al. 2023, Table 1
    clip_range: float = 0.2 
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    rollout_len: int = 128
    gae_lambda: float = 0.95
    n_epochs: int = 10
    minibatch_size: int = 64
    total_env_steps: int = 300_000


@dataclass(frozen=True)
class RewardConfig:
    dsr_eta: float = 0.01
    dsr_epsilon: float = 1e-8
    lambda_turnover: float = 0.1


@dataclass(frozen=True)
class CostConfig:
    c_trans: float = 0.001           # Zou et al. 2023, p.5
    initial_value: float = 1_000_000.0
    trading_days_per_year: int = 252


@dataclass(frozen=True)
class SplitConfig:
    # OURS. Rolling-origin walk-forward with a FIXED-LENGTH training window

    # fold k: train 6y -> val 1y -> test 2y, origin steps forward 2y.
    #   1: train 2011-2016 | val 2017 | test 2018-2019
    #   2: train 2013-2018 | val 2019 | test 2020-2021
    #   3: train 2015-2020 | val 2021 | test 2022-2023
    #   4: train 2017-2022 | val 2023 | test 2024-2025
    # 2025 is left completely untouched as a final never-looked-at holdout. 
    first_train_start: str = "2011-01-01"
    train_years: int = 6
    val_years: int = 1
    test_years: int = 2
    step_years: int = 2
    n_folds: int = 4
    # fold k: train 6y -> val 1y -> test 2y, origin steps 2y; test windows tile 2018-2025


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    ind: IndicatorConfig = field(default_factory=IndicatorConfig)
    state: StateConfig = field(default_factory=StateConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    seeds: Tuple[int, ...] = (0, 1, 2)
    master_seed: int = 20251008

    @property  
    def n_assets(self):
        return len(self.data.tickers)

    @property
    def feature_dim(self):          # 13N
        return self.n_assets * self.state.n_features_per_asset

    @property
    def state_dim(self):            # 14N + 1
        return self.feature_dim + self.n_assets + 1

    @property
    def action_dim(self):           # N assets + cash
        return self.n_assets + 1


CFG = Config()

# The only thing allowed to differ between models.
MODEL_SPECS = {
    "M1": {"encoder": "mlp",  "reward": "net_return", "turnover_penalty": False},
    "M2": {"encoder": "lstm", "reward": "net_return", "turnover_penalty": False},
    "M3": {"encoder": "lstm", "reward": "dsr",   "turnover_penalty": False},
    "M4": {"encoder": "lstm", "reward": "dsr",      "turnover_penalty": True},
}

def with_lambda(cfg, lam):
    """Copy of cfg with a different lambda_turnover (for the M4 sweep). 
    Changes config_hash, so runs stay traceable."""
    return replace(cfg, reward=replace(cfg.reward, lambda_turnover=float(lam)))


def assert_controlled_ablation(specs=MODEL_SPECS):
    order = ["M1", "M2", "M3", "M4"]
    for a, b in zip(order, order[1:]):
        diff = sorted(k for k in specs[a] if specs[a][k] != specs[b][k])
        if len(diff) != 1:
            raise AssertionError(f"{a} -> {b} changes {len(diff)} mechanisms {diff}; expected exactly 1")
    return True


def config_json(cfg=CFG):
    return json.dumps(asdict(cfg), sort_keys=True, default=str)


def config_hash(cfg=CFG):
    return hashlib.sha256(config_json(cfg).encode()).hexdigest()[:16]


def provenance_df(cfg=CFG):
    rows = [
        ("universe", f"{cfg.n_assets} NSE large-caps", PROV_OURS, "No source uses a 10-stock universe; survivorship bias must be disclosed"),
        ("sample period", f"{cfg.data.sample_start}..{cfg.data.sample_end}", PROV_OURS, "No canonical split exists in the sources"),
        ("features 1-3", "log ret, C/O-1, H/L-1", PROV_OURS, "Not in any source paper"),
        ("feature 4", "log(V_t / SMA_20(V))", PROV_OURS, "Not defined in any source"),
        ("EMA recursion, alpha=2/(n+1)", "-", PROV_PAPER, "Wang & Liu 2025, p.8"),
        ("EMA 50/200, RSI 14, MACD 12/26/9, BB 20/2", "-", PROV_PAPER, "Wang & Liu 2025, p.8"),
        ("ATR", "SMA_14(TR) / C_t", PROV_PAPER, "Wang & Liu 2025, p.8; /C_t scaling is OURS"),
        ("MACD, signal scaling", "/ C_t", PROV_OURS, "Cross-asset comparability"),
        ("RSI smoothing", "Wilder recursive", PROV_STANDARD, "Paper gives formula, not the averaging"),
        ("Bollinger %B, bandwidth, std ddof=0", "-", PROV_STANDARD, "Paper defines bands only"),
        ("ADX 14", "Wilder", PROV_STANDARD, "Wilder (1978); absent from sources"),
        ("window W", cfg.state.window, PROV_PAPER, "Zou et al. 2023, Table 2 p.9"),
        ("LSTM hidden H", cfg.model.lstm_hidden, PROV_PAPER, "Zou et al. 2023, Table 3 p.9"),
        ("MLP hidden (M1)", str(cfg.model.mlp_hidden), PROV_OURS, "Non-temporal counterpart"),
        ("Dirichlet policy alpha=exp(m)", "-", PROV_PAPER, "Yang, Park & Lee 2022, Sec 3.3, Eqs 13-16"),
        ("tau", cfg.model.action_threshold, PROV_OURS, "Not in Yang et al. Eqs 13-16"),
        ("PPO clip / GAE", "-", PROV_PAPER, "Schulman 2017 / 2015"),
        ("gamma, clip, lr, coefs, rollout", "0.99/0.2/3e-4/0.5,0.01/128", PROV_PAPER, "Zou et al. 2023, Table 1 p.8"),
        ("gae_lambda, n_epochs", "0.95 / 10", PROV_STANDARD, "Common PPO defaults"),
        ("total_env_steps", cfg.ppo.total_env_steps, PROV_OURS, "Same budget for M1-M4"),
        ("DSR", "Eqs 6-7", PROV_PAPER, "Millea 2021, pp.8-9"),
        ("dsr_eta, dsr_epsilon", f"{cfg.reward.dsr_eta}, {cfg.reward.dsr_epsilon}", PROV_OURS, "Millea gives eta symbolically; no epsilon"),
        ("lambda_turnover", cfg.reward.lambda_turnover, PROV_OURS, "Squared penalty is our design"),
        ("c_trans, initial value", f"{cfg.cost.c_trans}, {cfg.cost.initial_value:.0f}", PROV_PAPER, "Zou et al. 2023, p.5"),
        ("trading days/year", cfg.cost.trading_days_per_year, PROV_STANDARD, "-"),
        ("walk-forward folds", "4 x (6y/1y/2y), step 2y", PROV_OURS, "Chronological, no random splits"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "class", "source"])


if __name__ == "__main__":
    assert_controlled_ablation()
    assert CFG.state_dim == 14 * CFG.n_assets + 1
    print("config hash", config_hash())
    print(f"N={CFG.n_assets} | 13N={CFG.feature_dim} | state 14N+1={CFG.state_dim} | action N+1={CFG.action_dim}")
    print("runs:", 4 * CFG.split.n_folds * len(CFG.seeds))
    print(provenance_df()["class"].value_counts().to_dict())
    print("home:", HOME)