from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
from werkzeug.utils import secure_filename
from difflib import get_close_matches

app = Flask(__name__)

# === Load model and setup dictionary
model = load_model('handwritten_character_recog_model.h5')
words = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',
         10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',
         18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

# For autocorrect
dictionary = ["ABHINAV", "ALEX", "JULIA", "ELISE", "DANIEL", "JOHN", "TOM", "KEN", "KAREN", "NORA"]

# === Home route
@app.route('/')
def home():
    return "✅ API running! Use POST /predict with an image file."

# === Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Preprocess image
    img_resized = cv2.resize(img, (28, 28))
    _, img_thresh = cv2.threshold(img_resized, 100, 255, cv2.THRESH_BINARY_INV)
    final_image = np.reshape(img_thresh, (1, 28, 28, 1)) / 255.0

    # Predict
    prediction = model.predict(final_image)
    predicted_label = words[np.argmax(prediction)]

    # Attempt autocorrect (simulate full word recognition)
    raw_word = predicted_label  # In future, this will be a full string
    closest = get_close_matches(raw_word, dictionary, n=1)
    auto = closest[0] if closest else None

    return jsonify({
        'prediction': predicted_label,
        'autocorrected': auto
    })

# === Run server
if __name__ == '__main__':
    app.run(debug=True)
