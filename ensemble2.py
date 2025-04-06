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

# Ensemble Modeli Tanımlama
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model3):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

    def forward(self, x):
        out1 = torch.softmax(self.model1(x), dim=1)
        out2 = torch.softmax(self.model2(x), dim=1)
        out3 = torch.softmax(self.model3(x), dim=1)

        # Ensemble prediction: soft voting (average of softmax outputs)
        ensemble_output = (out1 + out2 + out3) / 3
        return ensemble_output

# Ensemble Modelini Oluşturma
ensemble_model = EnsembleModel(mobilenet, efficientnet, convnext)
ensemble_model.to(device)
ensemble_model.eval()

# Ensemble Modelini Kaydetme
torch.save(ensemble_model.state_dict(), "ensemble_model.pth")
print("Ensemble model saved successfully!")

# Kaydedilen Ensemble modelini yüklemek
ensemble_model_loaded = EnsembleModel(mobilenet, efficientnet, convnext)
ensemble_model_loaded.load_state_dict(torch.load("ensemble_model.pth"))
ensemble_model_loaded.to(device)
ensemble_model_loaded.eval()

# Test işlemi
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Ensemble modelinin test edilmesi
        ensemble_output = ensemble_model_loaded(images)
        _, predicted = torch.max(ensemble_output, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"\n📊 Loaded Ensemble Model Accuracy: {accuracy:.2f}%")
