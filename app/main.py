import pandas as pd

df = pd.read_json("data/data.json")
print("DataFrame")
print(df)
print("\nInfo:")
print(df.info())