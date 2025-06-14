import pandas as pd
df = pd.read_parquet("../result/clean_featured.parquet")
print(df.head())