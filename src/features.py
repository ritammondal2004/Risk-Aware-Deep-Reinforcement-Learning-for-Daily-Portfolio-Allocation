
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CFG
from .utils import sha256_frame

FEATURE_NAMES = ["log_ret", "c_over_o", "h_over_l", "log_vol_rel", "c_ema50", "c_ema200",
                 "rsi", "macd", "macd_sig", "atr", "bb_pctb", "bb_bw", "adx"]
assert len(FEATURE_NAMES) == 13

                    
def _ema(s, n): 
    return s.ewm(span=n, adjust=False, min_periods=n).mean()          # alpha = 2/(n+1)
       

def _wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l-pc).abs()], axis=1).max(axis=1, skipna=False)  


def asset_features(o, h, l, c, v, ic=CFG.ind):  
    f = pd.DataFrame(index=c.index)
    f["log_ret"] = np.log(c / c.shift(1))
    f["c_over_o"] = c / o - 1
    f["h_over_l"] = h / l - 1
    vv = v.clip(lower=1)
    f["log_vol_rel"] = np.log(vv / vv.rolling(ic.volume_sma_period).mean())
    f["c_ema50"] = c / _ema(c, ic.ema_fast) - 1
    f["c_ema200"] = c / _ema(c, ic.ema_slow) - 1

    d = c.diff()
    g, ls = _wilder(d.clip(lower=0), ic.rsi_period), _wilder((-d).clip(lower=0), ic.rsi_period)
    tot = g + ls                                   # RSI/100 = G/(G+L) = 1 - 1/(1+G/L)
    f["rsi"] = (g / tot).where(tot > 0, 0.5).where(g.notna())

    macd = _ema(c, ic.macd_fast) - _ema(c, ic.macd_slow)
    f["macd"] = macd / c
    f["macd_sig"] = _ema(macd, ic.macd_signal) / c

    tr = _true_range(h, l, c)
    f["atr"] = tr.rolling(ic.atr_period).mean() / c

    mb = c.rolling(ic.bb_period).mean()
    sd = c.rolling(ic.bb_period).std(ddof=0)
    ub, lb = mb + ic.bb_num_std * sd, mb - ic.bb_num_std * sd
    f["bb_pctb"] = (c - lb) / (ub - lb)
    f["bb_bw"] = (ub - lb) / mb

    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index).where(up.notna())
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index).where(up.notna())
    atr_w = _wilder(tr, ic.adx_period)
    pdi, mdi = 100 * _wilder(pdm, ic.adx_period) / atr_w, 100 * _wilder(mdm, ic.adx_period) / atr_w
    den = pdi + mdi
    dx = (100 * (pdi - mdi).abs() / den).where(den > 0, 0.0).where(pdi.notna())
    f["adx"] = _wilder(dx, ic.adx_period) / 100
    return f[FEATURE_NAMES]


def build_feature_tensor(O, H, L, C, V, ic=CFG.ind):
    """(T, N, 13), asset order = column order of C."""
    return np.stack([asset_features(O[t], H[t], L[t], C[t], V[t], ic).values for t in C.columns], axis=1)


def check_causality(O, H, L, C, V, ic=CFG.ind, n=6, seed=0, min_idx=250):
    """Features at row i must not change when everything after i is removed."""
    full = build_feature_tensor(O, H, L, C, V, ic)
    rng = np.random.default_rng(seed)
    for i in rng.choice(np.arange(min_idx, len(C)), size=n, replace=False):
        part = build_feature_tensor(O.iloc[:i + 1], H.iloc[:i + 1], L.iloc[:i + 1], C.iloc[:i + 1], V.iloc[:i + 1], ic)
        if not np.allclose(part[-1], full[i], equal_nan=True, atol=1e-12):
            raise AssertionError(f"look-ahead detected at row {i} ({C.index[i].date()})")
    return True


@dataclass
class Dataset:
    X: np.ndarray            # (T, 13N), asset-major: [f_1, ..., f_N]
    ret_next: np.ndarray     # (T, N), simple return close_t -> close_{t+1}  (R_{i,t})
    close: np.ndarray        # (T, N)
    dates: pd.DatetimeIndex  # decision dates
    tickers: list
    hash: str


def build_dataset(cfg=CFG, aligned=None):
    if aligned is None:
        from .data import load_aligned
        aligned, _, _ = load_aligned(cfg) 

    O, H, L, C, V = (aligned[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    feat = build_feature_tensor(O, H, L, C, V, cfg.ind)
    dates = C.index 
    in_win = (dates >= pd.Timestamp(cfg.data.sample_start)) & (dates <= pd.Timestamp(cfg.data.sample_end))
    dec = in_win & np.r_[np.ones(len(dates) - 1, bool), False]      # last day has no next return

    assert np.isfinite(feat[dec]).all(), "non-finite features inside the sample window"
    X = feat[dec].reshape(dec.sum(), -1)
   
    assert X.shape[1] == cfg.feature_dim
    ret_next = (C.shift(-1) / C - 1).values[dec]
   
    assert np.isfinite(ret_next).all()
    ds = Dataset(X=X, ret_next=ret_next, close=C.values[dec], dates=dates[dec], tickers=list(C.columns),
                 hash=sha256_frame(pd.DataFrame(X, index=dates[dec])))
    return ds, (O, H, L, C, V)
        

if __name__ == "__main__":
    ds, ohlcv = build_dataset()
    check_causality(*ohlcv, min_idx=int(np.searchsorted(ohlcv[3].index, pd.Timestamp(CFG.data.sample_start))))
    print(f"causality check passed | X {ds.X.shape} | ret_next {ds.ret_next.shape} | "
          f"{ds.dates[0].date()} .. {ds.dates[-1].date()} | sha {ds.hash[:16]}")
    st = pd.DataFrame(ds.X.reshape(-1, 13), columns=FEATURE_NAMES).describe().T[["min", "mean", "std", "max"]]
    print(st.to_string())