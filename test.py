import pandas as pd
df = pd.read_parquet("Cleaned_Mar24_alarms.parquet")
print(df.columns.tolist())

