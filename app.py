import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from textblob import TextBlob

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
# Preprocessing (same as before)
# -----------------------------------
def preprocess_for_model(cropped_img):
    target_dim = 28
    edge_size = 2
    resize_dim = target_dim - edge_size * 2

    h, w = cropped_img.shape
    pad_vertically = w > h
    pad_size = (max(h, w) - min(h, w)) // 2

    if pad_vertically:
        pad = ((pad_size, pad_size), (0, 0))
    else:
        pad = ((0, 0), (pad_size, pad_size))

    padded = np.pad(cropped_img, pad, mode='constant', constant_values=255)
    resized = cv2.resize(padded, (resize_dim, resize_dim))
    final = np.pad(resized, ((edge_size, edge_size), (edge_size, edge_size)), mode='constant', constant_values=255)

    final = 1.0 - (final.astype('float32') / 255.0)
    final = np.expand_dims(final, axis=(0, -1))  # (1,28,28,1)
    return final

# -----------------------------------
# Segment letters
# -----------------------------------
def segment_letters(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:
            boxes.append((x, y, w, h))

    return boxes

# -----------------------------------
# Sort letters properly
# -----------------------------------
def sort_boxes(boxes):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[1])

    lines = []
    current_line = [boxes[0]]

    for box in boxes[1:]:
        if abs(box[1] - current_line[-1][1]) < 20:
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    lines.append(current_line)

    for line in lines:
        line.sort(key=lambda b: b[0])

    sorted_boxes = [box for line in lines for box in line]
    return sorted_boxes

# -----------------------------------
# Prediction
# -----------------------------------
def predict_image(img):
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    boxes = segment_letters(img)
    boxes = sort_boxes(boxes)

    predicted_text = ""
    prev_box = None
    current_line_y = None

    for (x, y, w, h) in boxes:
        cropped = img[y:y+h, x:x+w]
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        input_tensor = preprocess_for_model(cropped_gray)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        idx = np.argmax(output)
        char = characters[idx]

        if prev_box is not None:
            gap = x - (prev_box[0] + prev_box[2])

            if gap > w * 1.5:
                predicted_text += " "

            if abs(y - current_line_y) > 30:
                predicted_text += "\n"

        predicted_text += char
        prev_box = (x, y, w, h)
        current_line_y = y

    # Auto-correct words using TextBlob
    corrected = str(TextBlob(predicted_text).correct())

    return corrected

# -----------------------------------
# Routes
# -----------------------------------
@app.route("/", methods=["GET"])
def home():
    return "Handwriting Recognition API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = predict_image(img)

    return jsonify({"prediction": result})

# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
