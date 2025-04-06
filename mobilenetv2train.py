import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sınıf sayısı
num_classes = 4  # kendi sınıf sayını yaz

# Modeli yeniden oluştur ve eğitilmiş ağırlıkları yükle
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("mobilenetv2.pth", map_location=device))
model = model.to(device)
model.eval()

# Test verisi için transform (senin eğitimde kullandığınla uyumlu olmalı!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Test veri kümesi ve DataLoader
test_dir = 'dataset/test'  # Test verisinin yolu
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Etiket isimleri (klasör adları)
class_names = test_dataset.classes

# Tahmin ve gerçek etiket listeleri
all_preds = []
all_labels = []

# Test döngüsü
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy ve classification report
acc = accuracy_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

print(f"Accuracy: {acc:.4f}")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 🎨 Bar Plot: Precision, Recall, F1-Score
metrics = ['precision', 'recall', 'f1-score']
plot_data = {metric: [report[cls][metric] for cls in class_names] for metric in metrics}

x = np.arange(len(class_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, plot_data[metric], width, label=metric.capitalize())

ax.set_ylabel('Skor')
ax.set_title('Her sınıf için Precision / Recall / F1-Score')
ax.set_xticks(x + width)
ax.set_xticklabels(class_names)
ax.legend()
plt.tight_layout()
plt.show()
