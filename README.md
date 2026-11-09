# Risk-Aware Deep Reinforcement Learning for Daily Portfolio Allocation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![B.Tech Project-I](https://img.shields.io/badge/IIT%20Kharagpur-B.Tech%20Project--I-navy.svg)](https://www.iitkgp.ac.in/)

> **Academic Evaluation Repository**  
> **Candidate:** Ritam Mondal (Roll No. 23IM30018)  
> **Supervisor:** Dr. Abhishek Sharma  
> **Department:** Department of Industrial and Systems Engineering, Indian Institute of Technology Kharagpur  
> **Presentation Slide Deck:** [`ppt/BTP1_Risk_Aware_DRL_Portfolio.pptx`](ppt/BTP1_Risk_Aware_DRL_Portfolio.pptx)

---

## Abstract

Daily equity portfolio allocation is fundamentally a sequential decision-making problem governed by a competing trilemma: compounding capital growth (return), mitigating downside tail risk and drawdowns (risk), and curtailing rebalancing transaction fees (friction). While modern deep reinforcement learning (DRL) literature proposes numerous architectural and algorithmic enhancements, they are conventionally evaluated in disparate experimental settings—conflating their true individual contributions.

This project implements a **rigorous single-factor controlled ablation study** within a continuous-action Proximal Policy Optimization (PPO) framework to systematically isolate three core mechanisms:
1. **Temporal Sequence Memory:** Stateless Multi-Layer Perceptron (MLP) vs. 30-day Long Short-Term Memory (LSTM) encoder ($M_1 \to M_2$).
2. **Risk-Aware Objective Formulation:** Raw net portfolio return vs. Differential Sharpe Ratio (DSR) ($M_2 \to M_3$).
3. **Turnover Regularization:** Unconstrained rebalancing vs. quadratic turnover penalty ($M_3 \to M_4$).

The models are formulated as Partially Observable Markov Decision Processes (POMDPs) over a universe of 10 liquid National Stock Exchange (NSE) large-cap equities plus cash ($s_t \in \mathbb{R}^{141}$) using a continuous Dirichlet policy on the unit simplex with realistic 0.1% proportional transaction friction. Rigorous evaluation is conducted across a 4-fold chronological walk-forward protocol spanning 2018–2025 over 3 random seeds (48 runs of 300,000 steps each) against a passive $1/N$ Equal-Weight benchmark.

---

## Controlled Four-Model Ablation Design

Each model transition modifies exactly **one** isolated algorithmic component while holding the asset universe, indicators, costs, PPO hyperparameters, training budget, and walk-forward folds strictly identical:

| Model | Encoder Architecture | Reward Function ($r_t$) | Turnover Penalty | Algorithmic Mechanism Isolated |
| :---: | :---: | :---: | :---: | :--- |
| **M1** | 2-Layer MLP (512 units) | Net Return ($R_{\text{net}, t}$) | None | Stateless baseline |
| **M2** | 1-Layer LSTM (512 units, 30 days) | Net Return ($R_{\text{net}, t}$) | None | **M1 $\to$ M2:** Adds 30-day temporal sequence memory |
| **M3** | 1-Layer LSTM (512 units, 30 days) | Differential Sharpe ($D_t$) | None | **M2 $\to$ M3:** Adds online risk-aware reward |
| **M4** | 1-Layer LSTM (512 units, 30 days) | Differential Sharpe ($D_t$) | Quadratic ($\lambda = 0.1$) | **M3 $\to$ M4:** Adds turnover friction regularization |

---

## Empirical Results Summary

Averaged out-of-sample metrics across 4 chronological walk-forward folds (8 test years: 2018–2025) and 3 seeds (12 paired evaluations per model):

| Model | Mean 2-Year Return | Annualized Sharpe | Max Drawdown | Total Cost (% of $V_0$) | Daily Turnover |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M1 (MLP)** | 20.62% | 0.71 | −19.50% | 9.68% | 0.186 |
| **M2 (LSTM)** | 28.23% | 0.92 | −18.52% | 4.27% | 0.080 |
| **M3 (DSR)** | 26.93% | 0.92 | −19.95% | 4.14% | 0.079 |
| **M4 (Turnover Penalty)** | 20.97% | 0.92 | **−16.04%** | **0.97%** | **0.019** |
| **Equal-Weight (Benchmark)** | **36.48%** | **1.13** | −18.91% | **0.61%** | — |

### Key Findings & Answers to Research Questions:
- **RQ1 (Temporal Memory): YES.** Replacing the stateless MLP with an LSTM encoder provides substantial, statistically robust gains: Sharpe ratio improves by $+0.201$ (95% CI $[+0.156, +0.246]$, better in 4/4 folds), return increases by $+7.61$ pp, and daily turnover and costs are cut by more than half.
- **RQ2 (Risk-Aware Reward): INCONCLUSIVE.** The Differential Sharpe Ratio yielded no detectable overall improvement over net return ($\Delta \text{Sharpe} = +0.003$, 95% CI $[-0.295, +0.222]$ straddles zero).
- **RQ3 (Turnover Regularization): YES, WITH A RETURN TRADE-OFF.** Quadratic turnover regularization successfully reduces turnover by 76% ($0.079 \to 0.019$) and trading costs to $0.97\%$ while securing the best maximum drawdown ($-16.04\%$), but the conservative positioning caps cumulative return ($20.97\%$).
- **RQ4 (Benchmark Competitiveness): NO.** Passive Equal-Weight outperformed all RL agents. Primary factors: (1) full equity exposure during the strong 2018–2025 Indian bull run; (2) cash drag from RL agents holding zero-yield cash allocations (8.7%–32.4% avg); and (3) high organic growth among the 10 large-cap blue-chip equities.

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
git clone https://github.com/your-username/BTP1_repo.git
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

## Methodological & Scientific Protocols

To ensure rigorous reproducibility and eliminate common financial RL pitfalls:
1. **Strict Chronological Walk-Forward:** No random shuffling or future lookahead. Training (6y) $\to$ Validation (1y) $\to$ Test (2y).
2. **Zero Information Leakage:** Technical indicators use backward-looking windows only. Feature scaling parameters ($\mu, \sigma$) are fitted strictly on each fold's training split.
3. **Realistic Market Frictions:** Proportional transaction cost of 0.1% deducted on every rebalance.
4. **Simplex Constraint via Dirichlet Policy:** Raw action $a_t \sim \operatorname{Dirichlet}(\alpha_t)$ naturally satisfies non-negativity and unit sum.
5. **Decoupled Environment Pruning:** Positions $<1\%$ are filtered and renormalized in the environment to prevent micro-trading fees; this does not alter policy log-likelihoods.
6. **Objective Checkpointing:** Models are checkpointed based strictly on validation Sharpe ratio, never on test performance.

---

## Academic References

1. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).**  
   *Proximal Policy Optimization Algorithms.*  
   arXiv:1707.06347. [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

2. **Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).**  
   *High-Dimensional Continuous Control Using Generalized Advantage Estimation.*  
   arXiv:1506.02438. [https://arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)

3. **Zou, J., Lou, J., Wang, B., & Liu, S. (2024).**  
   *A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks.*  
   *Expert Systems with Applications*, 242, 122801. [https://doi.org/10.1016/j.eswa.2023.122801](https://doi.org/10.1016/j.eswa.2023.122801)

4. **Millea, A. (2021).**  
   *Deep Reinforcement Learning for Trading—A Critical Survey.*  
   *Data*, 6(11), 119. [https://doi.org/10.3390/data6110119](https://doi.org/10.3390/data6110119)

5. **Moody, J., & Wu, L. (1997 / 2001).**  
   *Learning to Trade via Direct Reinforcement.*  
   *IEEE Transactions on Neural Networks*, 12(4), 875–889. [https://doi.org/10.1109/72.935097](https://doi.org/10.1109/72.935097)

6. **Liu, X.-Y., Xia, Z., Yang, H., Gao, J., Zha, D., Zhu, M., Wang, C. D., Wang, Z., & Guo, J. (2024).**  
   *Dynamic Datasets and Market Environments for Financial Reinforcement Learning.*  
   *Machine Learning*, 113, 2795–2839. [https://doi.org/10.1007/s10994-023-06511-w](https://doi.org/10.1007/s10994-023-06511-w)

7. **Wang, X., & Liu, L. (2025).**  
   *Risk-Sensitive Deep Reinforcement Learning for Portfolio Optimization.*  
   *Journal of Risk and Financial Management*, 18(7), 347. [https://doi.org/10.3390/jrfm18070347](https://doi.org/10.3390/jrfm18070347)

8. **Yang, H., Park, H., & Lee, K. (2022).**  
   *A Selective Portfolio Management Algorithm with Off-Policy Reinforcement Learning Using Dirichlet Distribution.*  
   *Axioms*, 11(12), 664. [https://doi.org/10.3390/axioms11120664](https://doi.org/10.3390/axioms11120664)
