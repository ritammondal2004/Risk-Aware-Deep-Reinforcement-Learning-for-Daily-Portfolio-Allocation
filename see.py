import pandas as pd
df = pd.read_parquet(r"data\raw\raw_ohlcv_c98da520e393.parquet")
print(df.head())
