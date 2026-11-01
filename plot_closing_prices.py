"""
Usage:
    python plot_closing_prices.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── locate the raw parquet ─
RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
parquet_files = list(RAW_DIR.glob("raw_ohlcv_*.parquet"))

if not parquet_files:
    raise FileNotFoundError(f"No raw_ohlcv_*.parquet found in {RAW_DIR}")

parquet_path = parquet_files[0]
print(f"Loading: {parquet_path}\n")

# ── load & inspect ─────
panel = pd.read_parquet(parquet_path)
panel["date"] = pd.to_datetime(panel["date"])

print("=" * 70)
print("PARQUET DATASET — FIRST 20 ROWS")
print("=" * 70)
print(panel.head(20).to_string(index=False))
print(f"\nShape : {panel.shape}")
print(f"Columns: {list(panel.columns)}")
print(f"Tickers: {sorted(panel['ticker'].unique())}")
print(f"Date range: {panel['date'].min().date()}  to  {panel['date'].max().date()}")
print("=" * 70)

# ── pivot to wide-form for plotting ─────────────────────────────────
close_wide = panel.pivot(index="date", columns="ticker", values="Close").sort_index()

# ── plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8))

for ticker in close_wide.columns:
    ax.plot(close_wide.index, close_wide[ticker], linewidth=1.2, label=ticker)

ax.set_title("Closing Prices — All 10 Stocks", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Adjusted Close (₹)", fontsize=12)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.autofmt_xdate()
fig.tight_layout()

# save to outputs/plots
out_dir = Path(__file__).resolve().parent / "outputs" / "plots"
out_dir.mkdir(parents=True, exist_ok=True)
save_path = out_dir / "closing_prices_all.png"
fig.savefig(save_path, dpi=150)
print(f"\nPlot saved -> {save_path}")

plt.show()
