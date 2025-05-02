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
    # Resize to 28x28 first
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert: EMNIST has white text (0) on black bg (255)
    inverted = cv2.bitwise_not(resized)

    # Normalize: Scale to [0,1]
    normalized = inverted.astype('float32') / 255.0

    # Reshape to model input shape [1, 28, 28, 1]
    return np.expand_dims(normalized, axis=(0, -1))


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