import pandas as pd

df = pd.read_csv('./trajectories-0400-0415.csv')
df = df[:40000]
df.to_csv('./data40k.csv')