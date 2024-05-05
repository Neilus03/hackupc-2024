import pickle
import pandas as pd
import sys

sys.path.append("/data/users/mpilligua/hackupc-2024/web_app/")
from utils import ourName2TheirName

df = pd.read_pickle("/data/users/mpilligua/hackupc-2024/data/df2.pkl")
print(df.head())

# show how many values there are in img_name
print(df["img_name"].value_counts())

txt2dict = ourName2TheirName()
df["img_link"] = df.apply(lambda x: txt2dict.get(f"{x['folder']}/{x['img_name']}.jpg"), axis=1)

# remove all rows that have any None in the img_link column then disolve the group
df = df.dropna(subset=["img_link"])

df.to_pickle("/data/users/mpilligua/hackupc-2024/data/df2.pkl")