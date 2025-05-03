import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# ----------------------
# Setup
# ----------------------
app = Flask(__name__)
interpreter = tf.lite.Interpreter(model_path="voxscribe_emnist_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------
# Characters Set
# ----------------------
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# ----------------------
# Preprocess (mimics JS)
# ----------------------
def preprocess_js_style(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find bounding box of the digit/letter
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    roi = gray[y:y+h, x:x+w]

    target_dim = 28
    edge_size = 2
    resize_dim = target_dim - edge_size * 2

    # Padding to make it square
    h, w = roi.shape
    pad_vertically = w > h
    pad_size = (max(h, w) - min(h, w)) // 2

    if pad_vertically:
        padded = np.pad(roi, ((pad_size, pad_size), (0, 0)), constant_values=255)
    else:
        padded = np.pad(roi, ((0, 0), (pad_size, pad_size)), constant_values=255)

    # Resize to 24x24 then pad with 2px white to get 28x28
    resized = cv2.resize(padded, (resize_dim, resize_dim))
    final = np.pad(resized, ((edge_size, edge_size), (edge_size, edge_size)), constant_values=255)

    # Normalize and invert like in JS
    normalized = 1.0 - (final.astype(np.float32) / 255.0)
    return np.expand_dims(normalized, axis=(0, -1))  # shape: (1,28,28,1)

# ----------------------
# Prediction
# ----------------------
def predict_image(img):
    input_tensor = preprocess_js_style(img)
    if input_tensor is None:
        return "?"

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    idx = np.argmax(output)
    return characters[idx]

# ----------------------
# API Endpoints
# ----------------------
@app.route("/")
def home():
    return "✅ VoxScribe Character Recognition API running."

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = predict_image(img)
    return jsonify({"prediction": result})

# ----------------------
# Run Server
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
