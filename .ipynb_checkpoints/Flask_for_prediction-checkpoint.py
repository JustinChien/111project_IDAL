from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('D:\\111project\\github\\041-111project\\data\\models\\DenseNet-BC-121-32-no-top.h5')

# Define a function to preprocess input images
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define a route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    image = request.files['image']

    # Preprocess the image
    image = preprocess_image(image)

    # Make a prediction using the trained model
    prediction = model.predict(image)

    # Convert the prediction to a human-readable label
    if prediction[0][0] > 0.5:
        label = 'malignant'
    else:
        label = 'benign'

    # Return the prediction result as JSON
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run()
