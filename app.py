import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify

app = Flask(__name__)

model = tf.lite.Interpreter(model_path="voxscribe_emnist_model.tflite")
model.allocate_tensors()
inp = model.get_input_details()
out = model.get_output_details()

def prep(gray):
    sz, p = 28, 2
    rsz = sz - p * 2
    h, w = gray.shape
    pad = ((max(h, w) - min(h, w)) // 2,)
    pad = ((pad[0], pad[0]), (0, 0)) if w > h else ((0, 0), (pad[0], pad[0]))
    padded = np.pad(gray, pad, constant_values=255)
    img = cv2.resize(padded, (rsz, rsz))
    img = np.pad(img, ((p, p), (p, p)), constant_values=255)
    img = 1.0 - (img.astype("float32") / 255.0)
    return np.expand_dims(img, axis=(0, -1))

def segment(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    c, _ = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(i) for i in c if cv2.contourArea(i) > 50]
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes

def predict(img):
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    bits = segment(img)
    res = ""
    for x, y, w, h in bits:
        crop = img[y:y+h, x:x+w]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        t = prep(g)
        model.set_tensor(inp[0]['index'], t)
        model.invoke()
        r = model.get_tensor(out[0]['index'])
        res += chars[np.argmax(r)]
    return res

@app.route("/", methods=["GET"])
def ping():
    return "VoxScribe sentence recognizer live"

@app.route("/predict", methods=["POST"])
def go():
    if "image" not in request.files:
        return jsonify({"error": "Image missing"}), 400
    try:
        raw = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        text = predict(img)
        return jsonify({"prediction": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
