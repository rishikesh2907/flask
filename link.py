from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # allow requests from your frontend

# Load model
model = load_model("my_model.h5")

# Define class names (order must match dataset class order)
class_names = [
    'Banni', 'Gir', 'Jaffrabadi', 'Jersey',
    'Kankrej', 'Murrah', 'Nagpuri',
    'Sahiwal', 'Tharparkar', 'Toda'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['image']

        # Preprocess image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))  # match model input size
        x = np.array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        # Run prediction
        preds = model.predict(x)[0]

        # Top-3 predictions with rank (apply your custom accuracy adjustments)
        top_indices = preds.argsort()[-3:][::-1]
        top_predictions = []
        for rank, idx in enumerate(top_indices, start=1):
            confidence = round(float(preds[idx] * 100), 2)

            if rank == 1:
                # For top-1: just flip
                adjusted_conf = round(100 - confidence, 2)
            else:
                # For top-2 and top-3: flip and subtract 20
                adjusted_conf = round(max(0, confidence ), 2)

            top_predictions.append({
                "breed": class_names[idx],
                "confidence": adjusted_conf,
                "rank": rank
            })

        # Response
        return jsonify({
            "predicted_breed": top_predictions[0]["breed"],
            "confidence": top_predictions[0]["confidence"],  # adjusted value
            "top_predictions": top_predictions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
