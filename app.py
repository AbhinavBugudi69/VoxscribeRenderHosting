from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from difflib import get_close_matches
import tensorflow as tf

app = Flask(__name__)

# === Load the compressed TFLite model
interpreter = tf.lite.Interpreter(model_path="handwritten_model_quantized.tflite")
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Character mapping
words = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Sample dictionary for autocorrection
dictionary = ["ABHINAV", "JULIA", "ELISE", "DANIEL", "KAREN", "TOM", "NORA", "ALEX", "JOHN", "EMMA"]

# === Home route
@app.route("/")
def home():
    return "✅ Voxscribe TFLite API running! POST an image to /predict"

# === Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400

    # === Preprocess image
    img_resized = cv2.resize(img, (28, 28))
    _, img_thresh = cv2.threshold(img_resized, 100, 255, cv2.THRESH_BINARY_INV)
    input_image = img_thresh.astype(np.float32) / 255.0
    input_image = np.expand_dims(input_image, axis=(0, -1))  # (1, 28, 28, 1)

    # === Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_index = np.argmax(output_data)
    predicted_letter = words[predicted_index]

    # === Autocorrect if needed (you can modify logic later)
    raw_word = predicted_letter
    closest = get_close_matches(raw_word, dictionary, n=1)
    auto = closest[0] if closest else None

    return jsonify({
        "prediction": predicted_letter,
        "autocorrected": auto
    })

# === Run app (for local testing only — Render uses gunicorn)
if __name__ == "__main__":
    app.run(debug=True)
