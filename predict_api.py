from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_image import load_image, ensemble_predict, class_names, load_mobilenetv2, load_efficientnet_b0, load_squeezenet, load_shufflenet, load_mnasnet
import os

# Flask uygulamasını başlat
app = Flask(__name__)
CORS(app)

num_classes = len(class_names)

# Modelleri baştan yükle ve bellekte tut
mobilenet_model = load_mobilenetv2(num_classes)
efficientnet_model = load_efficientnet_b0(num_classes)
mnasnet_model = load_mnasnet(num_classes)
shufflenet_model = load_shufflenet(num_classes)
squeezenet_model = load_squeezenet(num_classes)

models = [mobilenet_model, efficientnet_model, mnasnet_model, shufflenet_model, squeezenet_model]

@app.route("/predict", methods=["POST"])
def predict():
    # Eğer gelen istekle birlikte 'image' parametresi gelmemişse hata dönüyoruz
    if "image" not in request.files:
        return jsonify({"error": "No image file found"}), 400

    image_file = request.files["image"]
    temp_path = "temp.jpg"  # Geçici olarak resim dosyasını kaydedeceğiz

    # Resim dosyasını kaydediyoruz
    image_file.save(temp_path)

    try:
        # Resmi yükleyip modele uygun hale getiriyoruz
        image_tensor = load_image(temp_path)
        # Modeli çalıştırıyoruz ve sonuçları alıyoruz
        pred_class, confidence, probs = ensemble_predict(image_tensor, models)

        # Tahmin sonuçlarını JSON formatında geri döndürüyoruz
        return jsonify({
            "prediction": pred_class,
            "confidence": float(confidence),
            "probabilities": {
                class_names[i]: float(prob) for i, prob in enumerate(probs)
            }
        })

    except Exception as e:
        # Hata durumunda error mesajı dönüyoruz
        return jsonify({"error": str(e)}), 500

    finally:
        # Geçici dosyayı silip temizliyoruz
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Flask uygulamasını başlatıyoruz
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
