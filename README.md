# Risk-Aware Deep Reinforcement Learning for Daily Portfolio Allocation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![B.Tech Project-I](https://img.shields.io/badge/IIT%20Kharagpur-B.Tech%20Project--I-navy.svg)](https://www.iitkgp.ac.in/)

 ### [**read the paper here**](https://www.researchgate.net/publication/417808837_Risk-Aware_Deep_Reinforcement_Learning_for_Daily_Portfolio_Allocation)

> **Academic Evaluation Repository**  
> **Candidate:** Ritam Mondal (Roll No. 23IM30018)  
> **Supervisor:** Dr. Abhishek Sharma  
> **Department:** Department of Industrial and Systems Engineering, Indian Institute of Technology Kharagpur  
> **Presentation Slide Deck:** [`thesis/BTP_PPT.pptx`]

---



## Abstract

Daily equity portfolio allocation is fundamentally a sequential decision-making problem governed by a competing trilemma: compounding capital growth (return), mitigating downside tail risk and drawdowns (risk), and curtailing rebalancing transaction fees (friction). While modern deep reinforcement learning (DRL) literature proposes numerous architectural and algorithmic enhancements, they are conventionally evaluated in disparate experimental settings—conflating their true individual contributions.  

This project implements a **rigorous single-factor controlled ablation study** within a continuous-action Proximal Policy Optimization (PPO) framework to systematically isolate three core mechanisms:
1. **Temporal Sequence Memory:** Stateless Multi-Layer Perceptron (MLP) vs. 30-day Long Short-Term Memory (LSTM) encoder ($M_1 \to M_2$).
2. **Risk-Aware Objective Formulation:** Raw net portfolio return vs. Differential Sharpe Ratio (DSR) ($M_2 \to M_3$).
3. **Turnover Regularization:** Unconstrained rebalancing vs. quadratic turnover penalty ($M_3 \to M_4$).

The models are formulated as Partially Observable Markov Decision Processes (POMDPs) over a universe of 10 liquid National Stock Exchange (NSE) large-cap equities plus cash ($s_t \in \mathbb{R}^{141}$) using a continuous Dirichlet policy on the unit simplex with realistic 0.1% proportional transaction friction. Rigorous evaluation is conducted across a 4-fold chronological walk-forward protocol spanning 2018–2025 over 3 random seeds (48 runs of 300,000 steps each) against a passive $1/N$ Equal-Weight benchmark.

---


## Repository Structure

```text
BTP1_repo/
├── src/                               # Core Python source package
│   ├── config.py                      # Global parameters, tickers, hyperparameters, paths
│   ├── data.py                        # YFinance data downloader, calendar aligner, fold builder
│   ├── features.py                    # 13 technical features per equity, fold-isolated standardizer
│   ├── environment.py                 # Simplex POMDP portfolio environment with 0.1% cost & pruning
│   ├── models.py                      # Shared MLP/LSTM encoders, Dirichlet Actor head, Critic head
│   ├── ppo.py                         # Rollout buffer, GAE, clipped surrogate loss, Adam trainer
│   ├── evaluation.py                  # Financial metrics (CAGR, Sharpe, Sortino, Drawdown, Turnover)
│   ├── run_ablation.py                # Single run CLI runner (train, val-checkpoint, test eval)
│   ├── run_grid.py                    # Automated multi-process/resume-safe 48-run grid executor
│   ├── summarize.py                   # Aggregator collecting all metrics.json into pandas table
│   ├── report.py                      # Paired statistical analysis, bootstrap CIs, and plot generator
│   └── utils.py                       # Device detection, global seeds, data integrity hashes
├── tests/                             # Pytest test suite
│   ├── test_config.py                 # Configuration and path validation
│   ├── test_environment.py            # Portfolio balance dynamics, cost deductions, simplex logic
│   ├── test_features.py               # Feature calculations and zero-lookahead alignment
│   ├── test_ppo.py                    # GAE calculation, rollout buffer, and gradient step test
│   ├── test_evaluation.py             # Metric calculations under constant/boundary return series
│   └── test_report.py                 # Summary table and report verification
├── data/                              # Cached market data (generated on first run)
│   ├── raw/                           # Raw OHLCV Parquet files per ticker
│   └── processed/                     # Aligned dataset and fold splits
├── outputs/                           # Experimental outputs
│   ├── checkpoints/                   # Saved PyTorch model weights (.pt)
│   ├── metrics/                       # Run metadata and JSON performance metrics
│   └── plots/                         # Generated comparison figures and curves
├── ppt/                               # Academic Presentation
│   └── BTP1_Risk_Aware_DRL_Portfolio.pptx # 13-slide academic defense slide deck
├── thesis/                            # LaTeX thesis report and official figures
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git tracking rules
└── README.md                          # Repository documentation
```

---

## Step-by-Step Reproduction & Training Guide

### 1. Environment Setup

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
# Clone the repository
git clone https://github.com/ritammondal2004/Risk-Aware-Deep-Reinforcement-Learning-for-Daily-Portfolio-Allocation.git
cd BTP1_repo

# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify your PyTorch and GPU configuration:

```bash
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### 2. Verify Pipeline Integrity (Unit Tests)

Run the automated test suite to ensure the environment accounting, feature alignment, and PPO gradient updates function correctly:

```bash
pytest tests/ -v
```

---

### 3. Pipeline Sanity Check (Smoke Test)

Run a fast 2-update smoke test to verify data alignment, feature calculation, network initialization, and checkpointing:

```bash
python -m src.run_ablation --model M1 --fold 1 --seed 0 --smoke
python -m src.run_ablation --model M2 --fold 1 --seed 0 --smoke
```

---

### 4. Training a Single Experiment

To train a specific model on a chosen fold and random seed with the standard 300,000-step budget:

```bash
# Train M1 (Stateless MLP + Net Return) on Fold 1, Seed 0
python -m src.run_ablation --model M1 --fold 1 --seed 0

# Train M2 (LSTM + Net Return) on Fold 2, Seed 1
python -m src.run_ablation --model M2 --fold 2 --seed 1

# Train M3 (LSTM + Differential Sharpe Ratio) on Fold 3, Seed 0
python -m src.run_ablation --model M3 --fold 3 --seed 0

# Train M4 (LSTM + DSR + Turnover Penalty lambda=0.1) on Fold 4, Seed 2
python -m src.run_ablation --model M4 --fold 4 --seed 2 --lam 0.1
```

*Key Training Arguments:*
- `--model {M1,M2,M3,M4}`: Target ablation model.
- `--fold {1,2,3,4}`: Walk-forward fold (1: 2018–19, 2: 2020–21, 3: 2022–23, 4: 2024–25).
- `--seed {0,1,2}`: Random seed for environment sampling and network initialization.
- `--eval-every 25000`: Frequency of validation Sharpe evaluations (saves best checkpoint).
- `--lam 0.1`: Turnover penalty coefficient ($\lambda_{\text{turn}}$, M4 only).

---

### 5. Running the Complete 48-Run Ablation Grid

The automated runner executes all 48 configurations ($4 \text{ models} \times 4 \text{ folds} \times 3 \text{ seeds}$), skipping completed runs automatically:

```bash
# Execute the complete ablation grid
python -m src.run_grid --models M1 M2 M3 M4 --folds 1 2 3 4 --seeds 0 1 2

# Optional: Parallelize across multiple terminals/machines using job partitioning:
# Worker 1 of 2:
python -m src.run_grid --part 1/2
# Worker 2 of 2:
python -m src.run_grid --part 2/2
```

---

### 6. Results Summarization & Plot Generation

Once training runs are complete, aggregate all metrics and generate analytical figures:

```bash
# Display summary table of out-of-sample test results across all runs
python -m src.summarize --split test

# Generate comprehensive comparison tables, bootstrap CIs, and figures
python -m src.report
```

Output tables and charts will be saved in `outputs/plots/` and `outputs/metrics/`.

---
