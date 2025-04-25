from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from werkzeug.utils import secure_filename
from difflib import get_close_matches

app = Flask(__name__)

# === Load Model (disable optimizer load to avoid warnings)
try:
    model = load_model('handwritten_character_recog_model.h5', compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# === Character Mapping
words = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# === Sample dictionary (for autocorrect)
dictionary = ["ABHINAV", "ALEX", "JULIA", "ELISE", "DANIEL", "JOHN", "TOM", "KEN", "KAREN", "NORA"]

@app.route('/')
def index():
    return "📝 Voxscribe API is running. Use POST /predict with an image."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No filename provided'}), 400

    try:
        image_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({'error': 'Unable to decode image'}), 400

        # Preprocess: resize, threshold, normalize
        img_resized = cv2.resize(img, (28, 28))
        _, img_thresh = cv2.threshold(img_resized, 100, 255, cv2.THRESH_BINARY_INV)
        final_input = np.reshape(img_thresh, (1, 28, 28, 1)) / 255.0

        # Predict
        preds = model.predict(final_input)
        predicted_label = words[np.argmax(preds)]

        # Optional autocorrect (later use full strings)
        raw_output = predicted_label
        suggestion = get_close_matches(raw_output, dictionary, n=1)
        autocorrected = suggestion[0] if suggestion else None

        return jsonify({
            'prediction': raw_output,
            'autocorrected': autocorrected
        })

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
