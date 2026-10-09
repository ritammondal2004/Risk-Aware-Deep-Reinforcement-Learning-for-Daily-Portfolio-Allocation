# BTP-1: Risk-Aware Deep Reinforcement Learning for Daily Portfolio Allocation

Controlled ablation of temporal memory, risk-aware reward, and turnover regularization.

## Ablation

- **M1:** PPO (MLP) + net portfolio return
- **M2:** LSTM-PPO + net portfolio return
- **M3:** LSTM-PPO + Differential Sharpe Ratio (DSR)
- **M4:** LSTM-PPO + DSR - turnover regularization

All models use the same universe, features, portfolio environment, transaction-cost model, walk-forward folds, training budget, and evaluation protocol. Only the intended mechanism changes between ablation stages.

## Repository structure

```text
BTP1_repo/
├── src/
│   ├── __init__.py          (empty)
│   ├── config.py            Cells 2 + provenance
│   ├── utils.py             Cell 1 (seeds, hashes, device)
│   ├── data.py              Cells 3, 4, 6 (download, align, folds)
│   ├── features.py          Cell 5                       [next]
│   ├── environment.py       accounting + env             [next]
│   ├── models.py            MLP/LSTM encoder + Dirichlet policy
│   ├── ppo.py               rollout, GAE, clipped loss, training loop
│   ├── evaluation.py        metrics
│   └── run_ablation.py      --model --seed --fold
├── tests/                   test_config.py, test_features.py, test_environment.py
├── requirements.txt
├── .gitignore
└── README.md
```

## VS Code setup

Open this folder directly in VS Code. Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Check PyTorch:

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Run

```powershell
python src/run_ablation.py --model M1 --seed 42 --fold 1
python src/run_ablation.py --model M2 --seed 42 --fold 1
python src/run_ablation.py --model M3 --seed 42 --fold 1
python src/run_ablation.py --model M4 --seed 42 --fold 1
```

Do not compare final test performance until the walk-forward protocol and training/validation selection rules are fixed.

## Scientific rules

1. No random train/test split.
2. Feature scaling statistics must be fitted on training data only.
3. Never use future returns in the state.
4. Transaction costs are deducted once in portfolio value dynamics.
5. PPO stores the raw Dirichlet action and its log-probability; thresholding is an environment/action transformation.
6. The same threshold is used for M1-M4.
7. DSR is used only in M3/M4.
8. Turnover regularization is used only in M4.
9. Use identical seeds/folds/training budgets when comparing models.
10. Keep test data untouched until final evaluation.
