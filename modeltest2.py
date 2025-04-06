import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

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

# Ensemble Modelini Yükleme (LeaforiaNet)
ensemble_model = torch.load("leaforia_net.pth")
ensemble_model.to(device)
ensemble_model.eval()

# Test işlemi: Tahminler ve Gerçek Etiketler
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Ensemble modelinin tahminleri
        outputs = ensemble_model(images)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrikleri hesaplama
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

# Sonuçları yazdırma
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Sonuçları plotlama
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]

# Bar chart
plt.figure(figsize=(8, 5))
plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
plt.title("Model Evaluation Metrics (LeaforiaNet)")
plt.xlabel("Metrics")
plt.ylabel("Scores")
plt.ylim(0, 1)

# Label'lar
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')

plt.show()
