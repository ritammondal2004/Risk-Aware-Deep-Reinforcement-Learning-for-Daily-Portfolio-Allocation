import json
import time

import numpy as np
import pandas as pd
import yfinance as yf

from .config import CFG, config_hash, get_paths
from .utils import env_info, sha256_frame, sha256_text

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def universe_tag(cfg=CFG):
    return sha256_text("|".join(cfg.data.tickers) + cfg.data.download_start + cfg.data.sample_end)[:12]


def _extract_ticker_frame(raw, ticker):
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        frame = None
        for lvl in range(raw.columns.nlevels):
            if ticker in set(raw.columns.get_level_values(lvl)):
                frame = raw.xs(ticker, axis=1, level=lvl)
                break
        if frame is None:
            return pd.DataFrame()
        df = frame.copy()
    else:
        df = raw.copy()
    if not all(c in df.columns for c in OHLCV_COLS):
        return pd.DataFrame()
    df = df.loc[:, OHLCV_COLS].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(how="all")


def download_ohlcv(tickers, start, end_inclusive, max_retries=3, pause=3.0, min_rows=250):
    """auto_adjust=True: split and dividend adjusted OHLCV."""
    end_excl = (pd.Timestamp(end_inclusive) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames, pending = {}, list(tickers)
    for attempt in range(1, max_retries + 1):
        if not pending:
            break
        raw = yf.download(pending, start=start, end=end_excl, auto_adjust=True, actions=False,
                          progress=False, group_by="ticker", threads=True)
        still = []
        for t in pending:
            df = _extract_ticker_frame(raw, t)
            if df.empty or len(df) < min_rows:
                still.append(t)
            else:
                frames[t] = df
        pending = still
        if pending and attempt < max_retries:
            print(f"  [retry {attempt}] {pending}")
            time.sleep(pause * attempt)
    return frames, pending


def frames_to_panel(frames):
    parts = []
    for t in sorted(frames):
        d = frames[t].copy()
        d.index.name = "date"
        d = d.reset_index()
        d["ticker"] = t
        parts.append(d)
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    return panel[["date", "ticker"] + OHLCV_COLS]


def load_or_download_raw(cfg=CFG, force=False):
    path = get_paths()["raw"] / f"raw_ohlcv_{universe_tag(cfg)}.parquet"
    if path.exists() and not force:
        panel = pd.read_parquet(path)
        panel["date"] = pd.to_datetime(panel["date"])
        return panel, "cache"
    frames, failed = download_ohlcv(cfg.data.tickers, cfg.data.download_start, cfg.data.sample_end)
    if failed:
        raise RuntimeError(f"Download failed for {failed}. Ticker may be delisted/renamed/demerged. "
                           "Do NOT continue with a partial universe.")
    panel = frames_to_panel(frames)
    panel.to_parquet(path, index=False)
    return panel, "download"


def align_panel(panel, cfg=CFG, max_end_gap_days=5):
    """Long panel -> dict of aligned (date x ticker) frames on the common trading calendar."""
    tickers = list(cfg.data.tickers)
    wide = {f: panel.pivot(index="date", columns="ticker", values=f)[tickers].sort_index() for f in OHLCV_COLS}
    close = wide["Close"]
    firsts = pd.to_datetime(close.apply(lambda s: s.first_valid_index()))
    lasts = pd.to_datetime(close.apply(lambda s: s.last_valid_index()))
    if (lasts.max() - lasts.min()).days > max_end_gap_days:
        raise RuntimeError("End dates differ across tickers (renamed/delisted/demerged?):\n" + str(lasts))
    if firsts.max() > pd.Timestamp(cfg.data.sample_start) - pd.Timedelta(days=300):
        raise RuntimeError("Not enough history for EMA-200 warm-up:\n" + str(firsts))

    ok = np.logical_and.reduce([wide[f].notna().values for f in OHLCV_COLS]).all(axis=1)
    common = close.index[ok]
    dropped = close.index.difference(common)
    out = {f: wide[f].loc[common].astype(float) for f in OHLCV_COLS}
    O, H, L, C, V = (out[f] for f in OHLCV_COLS)

    assert not C.isna().any().any(), "NaN close after alignment"
    for name, d in (("Open", O), ("High", H), ("Low", L), ("Close", C)):
        assert (d > 0).all().all(), f"non-positive {name}"

    tol = 1e-6
    lr = np.log(C / C.shift(1))
    report = pd.DataFrame({
        "first": firsts, "last": lasts,
        "bad_high": (H.values < np.maximum.reduce([O.values, C.values, L.values]) * (1 - tol)).sum(axis=0),
        "bad_low": (L.values > np.minimum.reduce([O.values, C.values, H.values]) * (1 + tol)).sum(axis=0),
        "zero_vol": (V <= 0).sum().values,
        "|logret|>20%": (lr.abs() > 0.20).sum().values,
    }, index=tickers)
    extreme = {t: [str(d.date()) for d in lr.index[lr[t].abs() > 0.20]] for t in tickers}
    info = {"n_union": len(close), "n_common": len(common), "n_dropped": len(dropped),
            "report": report, "extreme_dates": {t: v for t, v in extreme.items() if v}}
    return out, info


def load_aligned(cfg=CFG, force=False, save=True):
    """raw -> aligned. Returns (dict of O/H/L/C/V frames, manifest dict)."""
    panel, src = load_or_download_raw(cfg, force=force)
    aligned, info = align_panel(panel, cfg)
    raw_hash = sha256_frame(panel)
    align_hash = sha256_frame(pd.concat(aligned, axis=1))
    manifest = {"raw_source": src, "raw_hash": raw_hash, "align_hash": align_hash,
                "universe_tag": universe_tag(cfg), "config_hash": config_hash(cfg),
                "n_common_days": info["n_common"], "n_dropped_days": info["n_dropped"], "env": env_info()}
    if save:
        p = get_paths()["processed"]
        pd.concat(aligned, axis=1).to_parquet(p / f"aligned_{universe_tag(cfg)}.parquet")
        (p / f"data_manifest_{universe_tag(cfg)}.json").write_text(json.dumps(manifest, indent=2, default=str))
    return aligned, manifest, info


def build_folds(dates, sp=CFG.split):
    """Rolling-origin walk-forward folds over decision dates. Segments are [start, end) index pairs."""
    dates = pd.DatetimeIndex(dates)
    folds = []
    for k in range(sp.n_folds):
        o = pd.Timestamp(sp.first_train_start) + pd.DateOffset(years=k * sp.step_years)
        b = [o + pd.DateOffset(years=y) for y in
             (0, sp.train_years, sp.train_years + sp.val_years, sp.train_years + sp.val_years + sp.test_years)]
        seg = {}
        for name, a, z in zip(("train", "val", "test"), b[:-1], b[1:]):
            idx = np.where((dates >= a) & (dates < z))[0]
            seg[name] = (int(idx[0]), int(idx[-1]) + 1) if len(idx) else None
        folds.append({"fold": k + 1, "bounds": [str(x.date()) for x in b], **seg})
    validate_folds(folds, len(dates))
    return folds


def validate_folds(folds, n_days):
    for f in folds:
        assert all(f[s] is not None for s in ("train", "val", "test")), f"empty segment in fold {f['fold']}"
        assert f["train"][1] == f["val"][0] and f["val"][1] == f["test"][0], "segments not contiguous"
    for a, b in zip(folds, folds[1:]):
        assert a["test"][1] <= b["test"][0], "test windows overlap across folds"
    assert folds[-1]["test"][1] <= n_days


if __name__ == "__main__":
    aligned, manifest, info = load_aligned()
    print(f"source={manifest['raw_source']} | common days {info['n_common']} | dropped {info['n_dropped']}")
    print(info["report"].to_string())
    if info["extreme_dates"]:
        print("extreme-return dates (check for unadjusted splits):", info["extreme_dates"])
    print("raw sha", manifest["raw_hash"][:16], "| aligned sha", manifest["align_hash"][:16])
    days = aligned["Close"].index
    days = days[(days >= CFG.data.sample_start) & (days <= CFG.data.sample_end)]
    for f in build_folds(days[:-1]):
        print(f["fold"], f["bounds"])