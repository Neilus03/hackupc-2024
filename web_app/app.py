from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import os
from retrieval import retrieve
from fashion_clip.fashion_clip import FashionCLIP
import pickle
from utils import *
from yolo import detect_hand_gesture_webcam

app = Flask(__name__)

fclip = FashionCLIP('fashion-clip')
df = pd.read_pickle("/data/users/mpilligua/hackupc-2024/data/df2.pkl")
kmeans = pickle.load(open("/data/users/mpilligua/hackupc-2024/data/kmeans.pkl", "rb"))
data = pd.read_csv('/data/users/mpilligua/hackupc-2024/data/inditextech_hackupc_challenge_images.csv')

# Expanded database of images sampling randomly from the original dataset
CLOTHING_IMAGES = get1imgfromEachCluster(df)

'''{
    't-shirts': data['IMAGE_VERSION_1'][:5].tolist(),
    'jeans': data['IMAGE_VERSION_1'][5:10].tolist(),
    'dresses': data['IMAGE_VERSION_1'][10:15].tolist(),
    'jackets': data['IMAGE_VERSION_1'][15:20].tolist(),
    'shorts': data['IMAGE_VERSION_1'][20:25].tolist(),
}'''

# Ensure the audio directory exists
audio_directory = 'web_app/audio/'
os.makedirs(audio_directory, exist_ok=True)

pd.set_option('display.max_colwidth', None)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/category/all')
def all_categories():
    print("Number of categories:", len(CLOTHING_IMAGES))
    # Pass all clothing types and their first image
    initial_images = {k: v[0] for k, v in CLOTHING_IMAGES.items()}
    print("Initial images:", initial_images)  # Debug to see what's being passed
    return render_template('category.html', categories=initial_images, all_images=CLOTHING_IMAGES)

@app.route('/view_catalog/<category>')
def view_catalog(category):
    image_urls = CLOTHING_IMAGES.get(category, [])  # Fetch images by category number
    print
    return render_template('view_catalog.html', category=category, image_urls=image_urls)


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' in request.files:
        audio = request.files['audio']
        print("Audio File Received:", audio.filename)  # Debugging: log filename
        # try:
            # Define the path to save the audio
        filename = os.path.join(audio_directory, 'received_audio2.wav')
        audio.save(filename)
        
        print("Audio Saved Successfully in", filename)  # Confirm audio is saved
        retrieved_images = retrieve(filename, df, kmeans, fclip) # Retrieve images
        # print(retrieved_images.head())

        image_urls = retrieved_images['img_link'].tolist()  # Convert URL column to a list
        # print("Image URLs:", image_urls)  # Log the image URLs
        return {"images": image_urls}  # Redirect to display images
        
        # except Exception as e:
        #     print("Error processing audio file:", e)  # Log errors
        #     return jsonify({"error": "Failed to process the audio file: " + str(e)}), 500
    else:
        print("No audio file uploaded")  # Log if no file is detected
        return jsonify({"error": "No audio file uploaded"}), 400

@app.route('/display')
def display():
    image_urls = request.args.get('image_urls').split(",")
    if image_urls:
        image_urls = image_urls  # Convert the comma-separated string back to a list
        print("Displaying images...")  # Log the images to display
        return render_template('display.html', image_urls=image_urls)
    else:
        return "No images to display", 404

@app.route('/view_image/<path:image_url>')
def view_image(image_url):
    image_urls = extract3angleimages(image_url)  # This function should return a list of three URLs
    
    return render_template('view_image.html', image_urls=image_urls)


@app.route('/run_script', methods=['POST'])
def run_script():
    # Code to execute your script here
    # For example:
    import subprocess
    subprocess.run(['python', '/data/users/mpilligua/hackupc-2024/web_app/yolo.py'])
    return 'Script executed successfully!', 200


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    
    # Assuming your YOLO integration is ready and function named 'detect_gesture' exists
    gesture = detect_hand_gesture_webcam()  # This should return 'Thumbs up' or 'Thumbs down'

    url = "https://static.zara.net/photos///2024/V/0/3/p/5767/521/712/2/w/2048/5767521712_6_1_1.jpg?ts=1707751045954"

    if gesture == 'Thumbs up':
        like(df, url)  # Your function to handle a "like"
        return jsonify({'action': 'like'})
    elif gesture == 'Thumbs down':
        dislike(df, url)  # Your function to handle a "dislike"
        return jsonify({'action': 'dislike'})
    print("No action taken, gesture:", gesture)
    return jsonify({'action': 'none'})

if __name__ == '__main__':
    app.run(debug=False, port=7020)
