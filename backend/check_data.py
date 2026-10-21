import pandas as pd
import numpy as np

from pathlib import Path
root = Path(__file__).resolve().parents[1]
candidates = [root / 'data' / 'flood.csv', Path(r'C:/Users/PRISHA/Downloads/flood.csv')]
data_path = next(p for p in candidates if p.exists())
df = pd.read_csv(data_path)
FEATURES = [c for c in df.columns if c not in ['FloodProbability']]

# Check if FloodProbability is a simple sum
row_sum = df[FEATURES].sum(axis=1)
# Compare with FloodProbability
print("Feature row sum sample:", row_sum.head(5).tolist())
print("FloodProbability sample:", df['FloodProbability'].head(5).tolist())

# Check correlation between sum and probability
print("Corr(sum, prob):", round(float(df[FEATURES].sum(axis=1).corr(df['FloodProbability'])), 4))

# Check unique FloodProbability values count
print("Unique FloodProb values:", df['FloodProbability'].nunique())
print("Range:", df['FloodProbability'].min(), "-", df['FloodProbability'].max())
