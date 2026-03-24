from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import onnxruntime as ort
import os

app = Flask(__name__)

# Load ONNX model (LIGHTWEIGHT)
session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name

class_names = [
    "mouse_bite",
    "open_circuit",
    "short_circuit",
    "spur",
    "spurious_copper"
]

@app.route("/")
def home():
    return "ONNX AOI Server Running"

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    original = img.copy()
    img = cv2.resize(img, (640, 640))

    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    outputs = session.run(None, {input_name: img})[0]
    outputs = np.squeeze(outputs)

    if outputs.shape[0] == 84:
        outputs = outputs.T

    h, w, _ = original.shape
    labels = []

    for row in outputs:
        obj_conf = row[4]
        if obj_conf < 0.3:
            continue

        class_scores = 1 / (1 + np.exp(-row[5:]))
        class_id = np.argmax(class_scores)

        if class_id >= len(class_names):
            continue

        labels.append(class_names[class_id])

    return jsonify({
        "labels": list(set(labels))
    })

# Render port fix
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)