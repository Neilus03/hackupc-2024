from PIL import Image
import requests
from io import BytesIO
import pandas as pd

data = pd.read_csv('.\inditextech_hackupc_challenge_images.csv')

'''for i, link in enumerate(data['IMAGE_VERSION_1']):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    img.show()
    if i == 2:
        break'''

