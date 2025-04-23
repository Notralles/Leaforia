from flask import Flask, request, jsonify
from predict_image import load_image, ensemble_predict, class_names
import os

app = Flask(__name__)
num_classes = len(class_names)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image_file = request.files["image"]
    temp_path = "temp.jpg"
    image_file.save(temp_path)

    try:
        image_tensor = load_image(temp_path)
        pred_class, confidence, probs = ensemble_predict(image_tensor, num_classes)

        return jsonify({
            "prediction": pred_class,
            "confidence": float(confidence),
            "probabilities": {
                class_names[i]: float(prob) for i, prob in enumerate(probs)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
