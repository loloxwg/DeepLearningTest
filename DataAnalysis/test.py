

import pandas as pd
file_path="./grades.csv"

df = pd.read_csv(file_path)
print(df.describe())
print(df.head())
print(df.info())