import pickle
import pandas as pd

df = pd.read_pickle("/data/users/mpilligua/hackupc-2024/data/df.pkl")
print(df.head())
#iterate over the img_link column and remote the \n at the end
df["img_link"] = df["img_link"].apply(lambda x: x[:-1])
print(df.head())