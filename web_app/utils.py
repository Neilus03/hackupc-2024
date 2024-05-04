#   folder img_name                                         embeddings  cluster
# 0  18131        1  [-0.3950624, 1.0118845, -0.2195734, -0.6599766...  0
# 1  31090        1  [-0.05779321, 0.88969994, 0.68793654, -0.15897...  1
# 2  30819        1  [-0.1456738, -0.15988454, -0.52963406, -0.4746...  10
# 3  27784        1  [0.11842453, 1.0050128, -0.019765982, -0.19121...  2
# 4  23851        1  [0.03181005, 1.5079372, -0.035510685, -0.48048...  43

import numpy as np
import os
from collections import Counter
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
import os

def get_closest_in_same_cluster(df, folder, file, n=1):
    """
    Get the closest n images in the same cluster
    """
    
    row = df[df["folder"] == folder & df["file"] == file]
    cluster = row["cluster"].values[0]
    emb = row["embeddings"].values[0]
    
    itemsInSameCluster = df[df["cluster"] == cluster]
    itemsInSameCluster["dist"] = itemsInSameCluster["embeddings"].apply(lambda x: np.linalg.norm(x - emb))
    itemsInSameCluster = itemsInSameCluster.sort_values("dist")
    return itemsInSameCluster.head(n)


def get_closest_from_embbedings(df, emb, n=1, kmeans=None):
    """
    Get the closest n images in the same cluster
    """
    if kmeans is not None:
        temp = pd.DataFrame()
        temp["embeddings"] = [emb]
        cluster = kmeans.predict(temp["embeddings"].iloc[0].tolist())[0]
        df = df[df["cluster"] == cluster]
    
    # compute the cosine similarity between emb and each row in df
    df["dist"] = df["embeddings"].apply(lambda x: np.dot(x.reshape(1, 512), emb.T)/(np.linalg.norm(x) * np.linalg.norm(emb)))
    df = df.sort_values("dist", ascending=False)
    
    txt2dict = ourName2TheirName()
    df["img_link"] = df.apply(lambda x: txt2dict.get(f"{x['folder']}/{x['img_name']}.jpg"), axis=1)
    return df.head(n)


def get_random_from_other_cluster(df, folder, file): 
    """
    Get a random image from another cluster
    """
    row = df[df["folder"] == folder & df["file"] == file]
    cluster = row["cluster"].values[0]
    
    itemsInOtherCluster = df[df["cluster"] != cluster]
    return itemsInOtherCluster.sample(1)

def embed_text(text, fclip):
    """
    Embed a text using FashionCLIP
    """
    if text is str:
        text = [text]
    
    return fclip.encode_text(text, batch_size=32)

def get_embeddings_df(folder = "/data/users/mpilligua/hackathon/embeddings"):
    # for each garment
    embeddings = []
    folders = []
    img_name = []
    for root, dir, files in os.walk(folder):
        if len(files) == 0:
            continue

        folders.append(root.split("/")[-1])
        img_name.append(files[0].split(".")[0])
        embeddings.append(np.load(os.path.join(root, files[0])))
        
    df = pd.DataFrame()
    df["folder"] = folders
    df["img_name"] = img_name
    df["embeddings"] = embeddings
    return df 

def compute_KMeans(df, n_clusters=50):
    """
    Compute KMeans clustering
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df["cluster"] = kmeans.fit(df["embeddings"].values.tolist()).labels_
    return df, kmeans

def convert_to_wav(input_file, output_file):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Check if the input file is already in WAV format
    if input_file.lower().endswith('.wav'):
        print("Input file is already in WAV format.")
        # return

    # Convert to WAV format
    audio.export(output_file, format="wav")
    print("Conversion complete.")
    
    #return output_file
    return output_file


def ourName2TheirName():
    """
    Convert our name (40316/3.jpg) to their name (https://static.zara.net/photos///2024/V/0/3/p/4428/664/500/2/w/2048/4428664500_3_1_1.jpg?ts=1709724616829)
    """
    with open("/data/users/mpilligua/hackupc-2024/data/filenames.txt") as f:
        lines = f.readlines()
    
    txt2dict = {}
    for line in tqdm(lines):
        try:
            txt2dict[line.split(": ")[0]] = line.split(": ")[1]
        except:
            print(line)
    return txt2dict