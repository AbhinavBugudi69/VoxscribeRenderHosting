from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="voxscribe_emnist_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# EMNIST labels
alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    img_array = np.array(image).astype(np.float32)
    img_array = 1.0 - (img_array / 255.0)
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream)
    input_tensor = preprocess_image(image)

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    prediction_index = np.argmax(output_data)
    predicted_char = alphabet[prediction_index]

    return jsonify({"prediction": predicted_char})

if __name__ == "__main__":
    app.run(debug=True)
