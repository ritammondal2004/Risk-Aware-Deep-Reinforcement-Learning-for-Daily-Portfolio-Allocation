import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, asset_features, build_feature_tensor, check_causality


def synth(T=700, N=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=T)
    cols = list("ABC")[:N]
    c = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), 0)), index=idx, columns=cols)
    o = c * (1 + rng.normal(0, 0.003, (T, N)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, (T, N))))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, (T, N))))
    v = pd.DataFrame(rng.integers(100_000, 1_000_000, (T, N)).astype(float), index=idx, columns=cols)
    return o, h, l, c, v


def test_shape_and_names():
    o, h, l, c, v = synth()
    assert list(asset_features(o["A"], h["A"], l["A"], c["A"], v["A"]).columns) == FEATURE_NAMES
    assert build_feature_tensor(o, h, l, c, v).shape == (700, 3, 13)


def test_features_are_causal():
    assert check_causality(*synth(), n=8, min_idx=250)

       
def test_ranges_after_warmup():
    f = build_feature_tensor(*synth())[250:]
    assert np.isfinite(f).all()
    for name in ("rsi", "adx", "bb_bw"):
        k = FEATURE_NAMES.index(name)
        assert f[..., k].min() >= 0
    assert f[..., FEATURE_NAMES.index("rsi")].max() <= 1 and f[..., FEATURE_NAMES.index("adx")].max() <= 1