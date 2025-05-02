# Student ID : W1947458
# Student Name : Abhinava Sai Bugudi
# Supervisor : Dr. Dimitris Dracopoulos
# Module : 6COSC023W.Y Computer Science Final Project


import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

# Loading a model
model = tf.lite.Interpreter(model_path="voxscribe_emnist_model.tflite")
model.allocate_tensors()

# Preparing a model
input_info = model.get_input_details()
output_info = model.get_output_details()

# Preparing the input image
def prep_image(gray):
    size = 28
    pad = 2
    resize_to = size - pad * 2

    h, w = gray.shape
    vertical = w > h
    gap = (max(h, w) - min(h, w)) // 2

    if vertical:
        padded = np.pad(gray, ((gap, gap), (0, 0)), constant_values=255)
    else:
        padded = np.pad(gray, ((0, 0), (gap, gap)), constant_values=255)

    resized = cv2.resize(padded, (resize_to, resize_to))
    final = np.pad(resized, ((pad, pad), (pad, pad)), constant_values=255)

    final = 1.0 - (final.astype('float32') / 255.0)
    return np.expand_dims(final, axis=(0, -1))

# Processing the prediction
def guess_char(img):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tensor = prep_image(gray)

    model.set_tensor(input_info[0]['index'], tensor)
    model.invoke()
    out = model.get_tensor(output_info[0]['index'])

    return chars[int(np.argmax(out))]

# Hosting the API
@app.route("/", methods=["GET"])
def status():
    return "VoxScribe API is live"

# Sending out the prediction
@app.route("/predict", methods=["POST"])
def handle_prediction():
    if "image" not in request.files:
        return jsonify({"error": "No image found. Try again"}), 400

    raw = request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    try:
        answer = guess_char(img)
        return jsonify({"prediction": answer})
    except Exception as err:
        return jsonify({"error": str(err)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)