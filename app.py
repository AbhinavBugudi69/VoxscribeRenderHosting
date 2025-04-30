import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load your EMNIST-based TFLite model
interpreter = tf.lite.Interpreter(model_path="emnistCNN.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map index to character
CHARACTER_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Match the JavaScript frontend's preprocessing
def preprocess_for_model(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = cv2.bitwise_not(resized)
    normalized = inverted.astype(np.float32) / 255.0
    return normalized.reshape(1, 28, 28, 1)

def predict_image(img):
    input_tensor = preprocess_for_model(img)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return CHARACTER_MAP[np.argmax(output)]

@app.route("/", methods=["GET"])
def home():
    return "✅ Handwriting Recognition API (Single Character) is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    prediction = predict_image(img)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
