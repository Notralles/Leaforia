import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    squeezenet1_0, SqueezeNet1_0_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
)
from PIL import Image
import numpy as np
import os
import time
import warnings

# Suppress FutureWarnings from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (make sure this matches the training classes)
class_names = ['Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy']


# Image preprocessing
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = input_transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Load MobileNetV3 Small model
def load_mobilenetv3(num_classes):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv3_model.pth", map_location=device))
    model.eval()
    return model.to(device)

# Load MobileNetV2 model
def load_mobilenetv2(num_classes):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv2_model.pth", map_location=device))
    model.eval()
    return model.to(device)

# Load EfficientNetB0 model
def load_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("efficientnetb0_model.pth", map_location=device))
    model.eval()
    return model.to(device)

# Load SqueezeNet model
def load_squeezenet(num_classes):
    model = squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes
    model.load_state_dict(torch.load("squeezenet1.0_model.pth", map_location=device))
    model.eval()
    return model.to(device)

# Load ShuffleNet model
def load_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("shufflenetv2_model.pth", map_location=device))
    model.eval()
    return model.to(device)

# Ensemble prediction using soft voting
def ensemble_predict(image_tensor, num_classes):
    models = [
        load_mobilenetv3(num_classes),
        load_mobilenetv2(num_classes),
        load_efficientnet_b0(num_classes),
        load_squeezenet(num_classes),
        load_shufflenet(num_classes)
    ]

    probs_list = []

    with torch.no_grad():
        for model in models:
            output = model(image_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            probs_list.append(probs.cpu().numpy())

    # Average the probabilities
    ensemble_probs = np.mean(probs_list, axis=0)
    pred_idx = np.argmax(ensemble_probs)
    pred_class = class_names[pred_idx]
    confidence = ensemble_probs[0][pred_idx]

    return pred_class, confidence, ensemble_probs[0]

# ---------- Main Execution ---------- #
if __name__ == "__main__":
    image_path = "path/test_image.JPG"
    num_classes = len(class_names)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
    else:
        image_tensor = load_image(image_path)

        start_time = time.time()
        pred_class, confidence, class_probs = ensemble_predict(image_tensor, num_classes)
        end_time = time.time()

        inference_time = end_time - start_time

        print(f"Predicted Class: {pred_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Inference Time: {inference_time:.4f} seconds")
