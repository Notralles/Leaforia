import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    mobilenet_v2,
    efficientnet_b0,
    squeezenet1_0,
    shufflenet_v2_x1_0,
    mnasnet1_0
)
from PIL import Image
import numpy as np
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy']

input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = input_transform(image).unsqueeze(0)
    return image.to(device)

def load_mobilenetv2(num_classes):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv2.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("efficientnetb0.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_squeezenet(num_classes):
    model = squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    model.load_state_dict(torch.load("squeezenet.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("shufflenet.pth", map_location=device))
    model.eval()
    return model.to(device)

def load_mnasnet(num_classes):
    model = mnasnet1_0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("mnasnet.pth", map_location=device))
    model.eval()
    return model.to(device)

def ensemble_predict(image_tensor, models):
    probs_list = []

    with torch.no_grad():
        for model in models:
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            probs_list.append(probs.cpu().numpy())

    ensemble_probs = np.mean(probs_list, axis=0)
    pred_idx = np.argmax(ensemble_probs)
    pred_class = class_names[pred_idx]
    confidence = ensemble_probs[0][pred_idx]

    return pred_class, confidence, ensemble_probs[0]

# Test amaçlı çalıştırma
if __name__ == "__main__":
    image_path = "path/test_image.JPG"
    num_classes = len(class_names)

    # Test için geçici dosya kontrolü
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
    else:
        image_tensor = load_image(image_path)
        start_time = time.time()

        # Modelleri yükle
        mobilenet_model = load_mobilenetv2(num_classes)
        efficientnet_model = load_efficientnet_b0(num_classes)
        mnasnet_model = load_mnasnet(num_classes)
        shufflenet_model = load_shufflenet(num_classes)
        squeezenet_model = load_squeezenet(num_classes)

        models = [mobilenet_model, efficientnet_model, mnasnet_model, shufflenet_model, squeezenet_model]

        pred_class, confidence, class_probs = ensemble_predict(image_tensor, models)
        end_time = time.time()

        print(f"Predicted Class: {pred_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Inference Time: {end_time - start_time:.4f} seconds")
