

from speech_to_text import mp3_to_text
from utils import *
import numpy as np
import os
from collections import Counter
import pandas as pd
from fashion_clip.fashion_clip import FashionCLIP

def retrieve(wav_file, df, kmeans, fclip):

    print("Retrieving images...")
    audio = convert_to_wav(wav_file, f"{wav_file[:-4]}_converted.wav")

    #get the description of the garment to retrieve via mp3
    description = mp3_to_text(audio)
    print(f"Description: {description}")
    
    embedding_text = embed_text([description], fclip)

    closest = get_closest_from_embbedings(df, embedding_text, n=10, kmeans=kmeans)

    return closest

if __name__ == "__main__":
    import pickle
    import pandas as pd
    # Save the df and kmeans 
    # df = get_embeddings_df()
    # df, kmeans = compute_KMeans(df)
    # df.to_pickle("/data/users/mpilligua/hackathon/df.pkl")
    # with open("/data/users/mpilligua/hackathon/kmeans.pkl", "wb") as f:
    #     pickle.dump(kmeans, f)
    
    # read the model and df from the pickle files
    with open("/data/users/mpilligua/hackupc-2024/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
        
    df = pd.read_pickle("/data/users/mpilligua/hackupc-2024/df.pkl")
    
    
    print(df.head())
    # fclip = FashionCLIP('fashion-clip')
    # embedding_text = embed_text(["I want a orange cap"], fclip)
    
    # df = pd.DataFrame()
    # df["embeddings"] = [embedding_text]
    # kmeans.predict(df["embeddings"].iloc[0].tolist())[0]