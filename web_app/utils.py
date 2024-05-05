import numpy as np
import os
from collections import Counter
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
import os
from sklearn.cluster import KMeans

def get_closest_in_same_cluster(df, link, img_shown):
    """
    Get the closest n images in the same cluster
    """
    row = df[df["img_link"] == link]
    cluster = row["cluster"].values[0]
    emb = row["embeddings"].values[0]
    
    itemsInSameCluster = df[df["cluster"] == cluster]
    itemsInSameCluster = itemsInSameCluster[~itemsInSameCluster["img_link"].isin(img_shown)]
    itemsInSameCluster["dist"] = itemsInSameCluster["embeddings"].apply(lambda x: np.linalg.norm(x - emb))
    itemsInSameCluster = itemsInSameCluster.sort_values("dist")
    return itemsInSameCluster.iloc[0]["img_link"]

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
    if n == 0:
        return df
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
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df["cluster"] = kmeans.fit(df["embeddings"].values.tolist()).labels_
    return df, kmeans

def convert_to_wav(input_file, output_file):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Check if the input file is already in WAV format
    # if input_file.lower().endswith('.wav'):
    #     print("Input file is already in WAV format.")
        # return

    # Convert to WAV format
    audio.export(output_file, format="wav")
    # print("Conversion complete.")
    
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
            txt2dict[line.split(": ")[0]] = line.split(": ")[1][:-1]
        except:
            pass # falla a 175 linies
    return txt2dict

def extract3angleimages(link):

    dictio=ourName2TheirName()
    # Detect key from the dictionary
    detected_key = None
    for key, value in dictio.items():
        if link == value:
            detected_key = key 
            break
    
    # Extract the 3 images from the same folder
    extracted_images = []
    for i in range(1,4):
        extracted_images.append(dictio[detected_key.split("/")[0] + "/" + str(i) + ".jpg"])
    return extracted_images

def get1imgfromEachCluster(df):
    """
    Get 1 image from each cluster
    returns a dict with the cluster number and a list of images
    """
    
    clusters = df["cluster"].unique()
    images = {}
    for cluster in clusters:
        images[cluster] = df[df["cluster"] == cluster]["img_link"].tolist()[:10]
    return images
    
def log_img(filename):
    with open(filename, "a") as f:
        f.write(filename + "\n")

def read_img(filename, img):
    with open(filename, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        if line[:-1] == img:
            return True
    
    return False
    
def read_shown_imgs(filename, img):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    shownImages = []
    for line in lines:
        shownImages.append(line[:-1])
    
    return shownImages
    
def like(df, link, ours2theirs):
    return "https://static.zara.net/photos///2024/V/0/3/p/1165/774/250/2/w/2048/1165774250_6_2_1.jpg?ts=1713252261014"
    link = getFirstImageFromLink(df, link, ours2theirs)
    shownImages = read_shown_imgs("shownImages.txt", link) # REMEMBER: to initialize the txt empty 
    return get_closest_in_same_cluster(df, link, shownImages)
    
    
def dislike(df, link, ours2theirs = None):
    return "https://static.zara.net/photos///2024/W/1/1/p/1222/210/102/2/w/2048/1222210102_6_2_1.jpg?ts=1702973294840"
    link = getFirstImageFromLink(df, link, ours2theirs)
    row = df[df["img_link"] == link]
    cluster = row["cluster"].values[0]
    
    img_shown = read_shown_imgs("shownImages.txt", link) # REMEMBER: to initialize the txt empty 
    itemsInSameCluster = df[df["cluster"] == cluster]
    itemsInSameCluster = itemsInSameCluster[~itemsInSameCluster["img_link"].isin(img_shown)]
    return itemsInSameCluster.sample(1)["img_link"].values[0]


def getFirstImageFromLink(df, link, ours2theirs = None):
    row = df[df["img_link"] == link]
    folder = row["folder"].values[0]
    
    return ours2theirs.get(f"{folder}/1.jpg")
    
    
    
    
    
if __name__ == "__main__":
    import pickle
    import pandas as pd
    
    df = pd.read_pickle("/data/users/mpilligua/hackupc-2024/data/df2.pkl")
    
    with open("/data/users/mpilligua/hackupc-2024/data/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    
    ulr = "https://static.zara.net/photos///2024/V/0/1/p/3186/227/300/2/w/2400/3186227300_3_1_1.jpg?ts=1713343397071"
    
    row = df[df["img_link"] == ulr]
    print(row)
    cluster = row["cluster"].values[0]
    
    df = df[df["cluster"] == cluster]
    
    initial_embedding = row["embeddings"].values[0]
    
    closest = get_closest_from_embbedings(df, initial_embedding, n=0, kmeans=kmeans)

    closest.to_pickle("/data/users/mpilligua/hackupc-2024/data/closest.pkl")
    
    # print(df.head())
    # d = get1imgfromEachCluster(df)    
    
    # print(extract3angleimages("https://static.zara.net/photos///2024/V/0/3/p/4428/664/500/2/w/2048/4428664500_3_1_1.jpg?ts=1709724616829"))