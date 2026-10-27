"""
app.py

Flask web application for ShipSmart:
 - Serves an upload page (templates/index.html)
 - Accepts an uploaded image, runs it through a model, and returns a prediction
 - Keeps a simple in-memory "pantry log" of recognized items

MODEL LOADING BEHAVIOUR (automatic):
 - Uses the custom-trained model for the eight classes in the dataset.
 - Also uses an open-vocabulary CLIP model to recognize common foods that
   are not in the training dataset, such as onions and potatoes.

IMPORTANT (read this before submitting your assignment):
   The project brief requires a model YOU trained on a dataset YOU collected.
   The custom-trained model is used for the classes in dataset/. Open-vocabulary
   recognition uses a pretrained model and is not a substitute for collecting
   representative photos when you need reliable results for a new food class.

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import os
import json
import io
import numpy as np
import torch
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from transformers import CLIPModel, CLIPProcessor

app = Flask(__name__)
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}

MODEL_PATH = os.path.join("model", "food_classifier.h5")
CLASS_NAMES_PATH = os.path.join("model", "class_names.json")
IMG_SIZE = (224, 224)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_LABELS = [
    "apple", "banana", "orange", "lemon", "lime", "strawberry", "grapes",
    "pear", "peach", "pineapple", "watermelon", "mango", "avocado",
    "tomato", "potato", "sweet potato", "onion", "red onion", "garlic",
    "carrot", "broccoli", "cauliflower", "cabbage", "lettuce", "spinach",
    "bell pepper", "cucumber", "zucchini", "mushroom", "corn", "peas",
    "green beans", "eggplant", "pumpkin", "bread", "bagel", "tortilla",
    "rice", "pasta", "noodles", "cereal", "flour", "egg", "milk carton",
    "cheese", "yogurt", "butter", "canned food", "beans", "lentils", "soup",
    "chicken", "fish", "salmon", "beef", "pork", "sausage", "shrimp", "tofu",
    "potato chips", "crackers", "cookies", "chocolate", "nuts", "peanut butter",
    "honey", "jam", "olive oil", "juice", "soda", "coffee", "tea",
    "a non-food household item",
]

USING_CUSTOM_MODEL = os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH)

if USING_CUSTOM_MODEL:
    print("[ShipSmart] Custom trained model found - using food_classifier.h5")
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
else:
    print("[ShipSmart] No custom model found - using open-vocabulary recognition.")
    model = None
    class_names = []

print(f"[ShipSmart] Loading open-vocabulary model {CLIP_MODEL_NAME}")
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(clip_device)
clip_model.eval()
clip_text_inputs = clip_processor(
    text=[f"a photo of {label}" for label in CLIP_LABELS],
    return_tensors="pt",
    padding=True,
).to(clip_device)
print(f"[ShipSmart] Open-vocabulary model ready on {clip_device}")

# Simple in-memory pantry log: [{"item": str, "confidence": float, "time": str}]
pantry_log = []


def prepare_image(file_storage):
    """Load an uploaded image file into a model-ready array."""
    img = keras_image.load_img(
        io.BytesIO(file_storage.read()),
        target_size=IMG_SIZE,
    )
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr, img


def predict_with_custom_model(img_array):
    predictions = model.predict(img_array)[0]
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx])
    label = class_names[top_idx]
    all_confidences = {
        class_names[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(class_names))
    }
    return label, confidence, all_confidences


def predict_with_open_vocabulary_model(img):
    image_inputs = clip_processor(images=img, return_tensors="pt").to(clip_device)
    with torch.inference_mode():
        outputs = clip_model(**image_inputs, **clip_text_inputs)
        probabilities = outputs.logits_per_image.softmax(dim=-1)[0]

    top_indices = torch.topk(probabilities, k=min(5, len(CLIP_LABELS))).indices.tolist()
    top_label_index = top_indices[0]
    label = CLIP_LABELS[top_label_index]
    confidence = float(probabilities[top_label_index].item())
    all_confidences = {
        CLIP_LABELS[index]: round(float(probabilities[index].item()) * 100, 2)
        for index in top_indices
    }
    return label, confidence, all_confidences


def predict_food(img_array, img):
    clip_label, clip_confidence, clip_confidences = (
        predict_with_open_vocabulary_model(img)
    )
    if not USING_CUSTOM_MODEL:
        return clip_label, clip_confidence, clip_confidences

    custom_label, custom_confidence, custom_confidences = (
        predict_with_custom_model(img_array)
    )
    trained_labels = {name.replace("_", " ").lower() for name in class_names}
    if clip_label.lower() in trained_labels and custom_confidence >= 0.7:
        return custom_label, custom_confidence, custom_confidences

    return clip_label, clip_confidence, clip_confidences


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "").rstrip("/")
    origin_host = origin.removeprefix("https://").split("/", 1)[0].split(":", 1)[0]
    github_pages_origin = (
        origin.startswith("https://")
        and origin_host.endswith(".github.io")
    )
    if origin in ALLOWED_ORIGINS or github_pages_origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Vary"] = "Origin"
    return response


@app.route("/")
def index():
    return send_from_directory(os.path.join(app.root_path, "docs"), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_array, img = prepare_image(file)
        label, confidence, all_confidences = predict_food(img_array, img)

        # Log the recognized item
        entry = {
            "item": label,
            "confidence": round(confidence * 100, 2),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # The phone's auto-scan mode sends log=0 so it doesn't spam the pantry log
        if request.form.get("log", "1") == "1":
            pantry_log.append(entry)

        # Optional: simple "low stock" style logic placeholder
        # (count how many times this item has been logged recently)
        count_recent = sum(1 for e in pantry_log if e["item"] == label)

        return jsonify({
            "label": label,
            "confidence": entry["confidence"],
            "logged_count": count_recent,
            "all_confidences": all_confidences,
            "using_custom_model": USING_CUSTOM_MODEL,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pantry_log")
def get_pantry_log():
    return jsonify(pantry_log)


@app.route("/clear_log", methods=["POST"])
def clear_log():
    pantry_log.clear()
    return jsonify({"status": "cleared"})


@app.route("/health")
def health():
    """Used by the phone app's 'Test connection' button."""
    return jsonify({
        "status": "ok",
        "using_custom_model": USING_CUSTOM_MODEL,
        "using_open_vocabulary_model": True,
    })


def get_lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ip = get_lan_ip()
    port = int(os.environ.get("PORT", "5000"))
    print("\n" + "=" * 60)
    print("  ShipSmart server running")
    print(f"  Browser (this computer): http://127.0.0.1:{port}")
    print(f"  PHONE (Expo Go) server address:  http://{ip}:{port}")
    print("  Phone and computer must be on the same Wi-Fi.")
    print("=" * 60 + "\n")
    # host=0.0.0.0 lets your phone reach this server; debug off so the
    # model isn't loaded twice
    app.run(host="0.0.0.0", port=port, debug=False)
