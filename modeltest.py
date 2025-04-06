import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from torchvision import datasets, transforms
import torch.nn.functional as F

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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Ensemble Model Yükle
ensemble_model = torch.load("ensemble_model.pth", map_location=device)
ensemble_model.to(device)

# Test için metrikler
correct = 0
total = 0

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Ensemble model tahminlerini al
        ensemble_output = ensemble_model(images)
        _, predicted = torch.max(ensemble_output, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Metriklerin hesaplanması
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

# Sonuçları pandas DataFrame olarak hazırlama
metrics = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Score': [accuracy, precision, recall, f1]
}

df_metrics = pd.DataFrame(metrics)

# Sonuçları yazdırma
print("\nEnsemble Model Metrics:")
print(df_metrics)

# Plotting metrics
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [accuracy, precision, recall, f1]

# Plot
plt.figure(figsize=(8, 6))
plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange', 'red'])
plt.title('Ensemble Model Evaluation Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.show()
