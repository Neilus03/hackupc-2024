

from speech_to_text import mp3_to_text
from utils import *

import numpy as np
import os
from collections import Counter
import pandas as pd

from fashion_clip.fashion_clip import FashionCLIP

#get the description of the garment to retrieve via mp3
description = mp3_to_text("Rev.mp3")


fclip = FashionCLIP('fashion-clip')

df = get_embeddings_df()
embedding_text = embed_text([description], fclip)

closest = get_closest_from_embbedings(df, embedding_text, n=10)

print(closest)