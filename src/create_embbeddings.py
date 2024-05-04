import sys
#sys.path.append("fashion-clip/")
from fashion_clip.fashion_clip import FashionCLIP
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression


# create a list with all the images from /data/users/mpilligua/hackathon/images they have the structure
"""
    - garment1
        - image1.jpg
        - image2.jpg
        ...
    - garment2
        - image1.jpg
        - image2.jpg
        ...    
"""
import os
    
list_images = []
for i, (root, dir, files) in enumerate(os.walk("/data/users/mpilligua/hackathon/images")):
    # print(root, dir, files)
    for file in files:
        list_images.append(os.path.join(root, file))
        
    # if i > 10: 
    #     break
        
print(len(list_images))

fclip = FashionCLIP('fashion-clip')
embeddings = fclip.encode_images(list_images, batch_size=32)


for img, emb in zip(list_images, embeddings):
    folder, file = img.split("/")[-2:]
    file = file.split(".")[0]
    
    os.makedirs(f"/data/users/mpilligua/hackathon/embeddings/{folder}", exist_ok=True)
    np.save(f"/data/users/mpilligua/hackathon/embeddings/{folder}/{file}.npy", emb)
    