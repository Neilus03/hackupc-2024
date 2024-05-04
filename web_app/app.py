from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

data = pd.read_csv('.\inditextech_hackupc_challenge_images.csv')

# Expanded database of images sampling randomly from the original dataset
CLOTHING_IMAGES = {
    't-shirts': data['IMAGE_VERSION_1'][:5].tolist(),
    'jeans': data['IMAGE_VERSION_1'][5:10].tolist(),
    'dresses': data['IMAGE_VERSION_1'][10:15].tolist(),
    'jackets': data['IMAGE_VERSION_1'][15:20].tolist(),
    'shorts': data['IMAGE_VERSION_1'][20:25].tolist(),
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/category/all')
def all_categories():
    # Pass all clothing types and their first image
    initial_images = {k: v[0] for k, v in CLOTHING_IMAGES.items()}
    return render_template('category.html', categories=initial_images, all_images=CLOTHING_IMAGES)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
