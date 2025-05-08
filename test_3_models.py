import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test verisi yolu
test_dir = r"C:\dataset\test"

# Dönüşüm
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Test dataset ve dataloader
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
num_classes = len(test_dataset.classes)

# Model yükleme fonksiyonları
def load_mobilenetv2(path):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

def load_efficientnetb0(path):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

def load_mnasnet(path):
    model = models.mnasnet1_0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

def load_shufflenet(path):
    model = models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

def load_squeezenet(path):
    model = models.squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device)

# Değerlendirme fonksiyonu
def evaluate_model(model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

# Model yolları ve isimleri
models_info = {
    "MobileNetV2": ("mobilenetv2.pth", load_mobilenetv2),
    "EfficientNetB0": ("efficientnetb0.pth", load_efficientnetb0),
    "MNASNet": ("mnasnet.pth", load_mnasnet),
    "ShuffleNetV2": ("shufflenet_best.pth", load_shufflenet),
    "SqueezeNet": ("squeezenet_best.pth", load_squeezenet)
}

# Sonuçlar
results = []

for name, (path, loader_fn) in models_info.items():
    if not os.path.exists(path):
        print(f"Model dosyası bulunamadı: {path}")
        continue
    model = loader_fn(path)
    y_true, y_pred = evaluate_model(model)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    })

# DataFrame tablosu
df = pd.DataFrame(results)
print(df.to_string(index=False, float_format="%.4f"))
