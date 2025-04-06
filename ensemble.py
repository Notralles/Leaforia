import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm  # ConvNeXt için
import os

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset yolu (sadece test için)
dataset_path = "C:/Users/krono/PycharmProjects/LeaforiaTrue/dataset/test"

# Veri dönüşümü
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
test_dataset = datasets.ImageFolder(dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(test_dataset.classes)

# MobileNetV2
mobilenet = models.mobilenet_v2(weights=None)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, num_classes)
mobilenet.load_state_dict(torch.load("mobilenetv2.pth", map_location=device))
mobilenet.to(device)
mobilenet.eval()

# EfficientNet-B0
efficientnet = models.efficientnet_b0(weights=None)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.load_state_dict(torch.load("efficientnet.pth", map_location=device))
efficientnet.to(device)
efficientnet.eval()

# ConvNeXt Tiny (timm ile)
convnext = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
convnext.load_state_dict(torch.load("convnext.pth", map_location=device))
convnext.to(device)
convnext.eval()

# Ensemble test
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        out1 = torch.softmax(mobilenet(images), dim=1)
        out2 = torch.softmax(efficientnet(images), dim=1)
        out3 = torch.softmax(convnext(images), dim=1)

        ensemble_output = (out1 + out2 + out3) / 3
        _, predicted = torch.max(ensemble_output, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"\n📊 Ensemble Model (MobileNet + EfficientNet + ConvNeXt) Accuracy: {accuracy:.2f}%")
