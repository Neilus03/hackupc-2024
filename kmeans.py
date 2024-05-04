# We have embeddings for each image, now we can use kmeans to cluster the images
""" 
The structure of the embeddings is the following:
    - garment1
        - image1.npy
        - image2.npy
        ...
    - garment2
        - image1.npy
        - image2.npy
        ...    
"""
import numpy as np
import os
from collections import Counter

# for each garment
embeddings = []
folders = []
img_name = []
for root, dir, files in os.walk("/data/users/mpilligua/hackathon/embeddings"):
    if len(files) == 0:
        continue
    
    # folders.extend([root.split("/")[-1] for file in files])
    # img_name.extend([file.split(".")[0] for file in files])
    # embeddings.extend([np.load(os.path.join(root, file)) for file in files])
    
    folders.append(root.split("/")[-1])
    img_name.append(files[0].split(".")[0])
    embeddings.append(np.load(os.path.join(root, files[0])))
    

import pandas as pd
df = pd.DataFrame()
df["folder"] = folders
df["img_name"] = img_name
df["embeddings"] = embeddings
print(df.head())

from sklearn.cluster import KMeans

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)

df["cluster"] = kmeans.labels_

# remove the embeddings
df = df.drop("embeddings", axis=1)

# save the clusters
df.to_csv("/data/users/mpilligua/hackathon/clusters.csv", index=False)