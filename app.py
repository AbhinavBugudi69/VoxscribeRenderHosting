import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# -----------------------------------
# Setup
# -----------------------------------
app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="emnistCNN.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------------
# Preprocess for EMNIST
# -----------------------------------
def preprocess_for_model(gray_img):
    target_dim = 28
    edge_size = 2
    resize_dim = target_dim - edge_size * 2

    h, w = gray_img.shape
    pad_vertically = w > h
    pad_size = (max(h, w) - min(h, w)) // 2

    if pad_vertically:
        pad = ((pad_size, pad_size), (0, 0))
    else:
        pad = ((0, 0), (pad_size, pad_size))

    padded = np.pad(gray_img, pad, mode='constant', constant_values=255)
    resized = cv2.resize(padded, (resize_dim, resize_dim))
    final = np.pad(resized, ((edge_size, edge_size), (edge_size, edge_size)), mode='constant', constant_values=255)

    final = 1.0 - (final.astype('float32') / 255.0)
    final = np.expand_dims(final, axis=(0, -1))  # (1,28,28,1)
    return final

# -----------------------------------
# Predict a Single Letter
# -----------------------------------
def predict_single_letter(img):
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    input_tensor = preprocess_for_model(gray)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    idx = int(np.argmax(output))
    return characters[idx]

# -----------------------------------
# Routes
# -----------------------------------
@app.route("/", methods=["GET"])
def home():
    return "✅ VoxScribe Character API is running."

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    try:
        result = predict_single_letter(img)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
