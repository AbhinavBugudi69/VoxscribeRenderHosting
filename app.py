# Student ID : W1947458
# Student Name : Abhinava Sai Bugudi
# Module : 6COSC023W.Y Computer Science Final Project

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from textblob import TextBlob

app = Flask(__name__)

model = tf.lite.Interpreter(model_path="voxscribe_emnist_model.tflite")
model.allocate_tensors()

input_info = model.get_input_details()
output_info = model.get_output_details()

def prep(gray):
    s = 28
    pad = 2
    r = s - pad * 2
    h, w = gray.shape
    gap = (max(h, w) - min(h, w)) // 2
    p = ((gap, gap), (0, 0)) if w > h else ((0, 0), (gap, gap))
    padded = np.pad(gray, p, constant_values=255)
    resized = cv2.resize(padded, (r, r))
    out = np.pad(resized, ((pad, pad), (pad, pad)), constant_values=255)
    return np.expand_dims(1.0 - out.astype("float32") / 255.0, axis=(0, -1))

def segment(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > 100]
    return sorted(b, key=lambda x: (x[1], x[0]))

def predict(img):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    boxes = segment(img)
    text = ""
    prev = None
    line_y = None

    for (x, y, w, h) in boxes:
        c = img[y:y + h, x:x + w]
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        t = prep(g)

        model.set_tensor(input_info[0]['index'], t)
        model.invoke()
        out = model.get_tensor(output_info[0]['index'])
        ch = chars[int(np.argmax(out))]

        if prev is not None:
            gap = x - (prev[0] + prev[2])
            if gap > w * 1.5:
                text += " "
            if abs(y - line_y) > 30:
                text += "\n"

        text += ch
        prev = (x, y, w, h)
        line_y = y

    return str(TextBlob(text).correct())

@app.route("/", methods=["GET"])
def status():
    return "VoxScribe Sentence API live."

@app.route("/predict", methods=["POST"])
def handle():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    raw = request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    try:
        result = predict(img)
        return jsonify({"prediction": result})
    except Exception as err:
        return jsonify({"error": str(err)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
