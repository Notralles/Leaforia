import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# === 1. VERİYİ YÜKLE ===
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# test klasörünü kendi test veri klasörün ile değiştir
test_dataset = datasets.ImageFolder("dataset/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class_names = test_dataset.classes

# === 2. MODELİ YÜKLE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("mobilenetv2.pth")  # kendi model path’ini gir
model.eval()
model = model.to(device)

# === 3. TEST VE TAHMİNLER ===
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# === 4. METRİKLERİ HESAPLA ===
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average=None, labels=range(len(class_names)))

print(f"Accuracy: {accuracy:.4f}")
for idx, class_name in enumerate(class_names):
    print(f"{class_name}: Precision={precision[idx]:.2f}, Recall={recall[idx]:.2f}, F1-Score={f1[idx]:.2f}")

# === 5. METRİKLERİ PLOTLA ===
x = np.arange(len(class_names))
plt.figure(figsize=(12, 6))
plt.bar(x - 0.2, precision, width=0.2, label='Precision', color='skyblue')
plt.bar(x, recall, width=0.2, label='Recall', color='lightgreen')
plt.bar(x + 0.2, f1, width=0.2, label='F1-score', color='salmon')
plt.xticks(x, class_names, rotation=45)
plt.ylabel('Score')
plt.title('Precision, Recall, F1-score per Class')
plt.legend()
plt.tight_layout()
plt.show()

# === 6. CONFUSION MATRIX ===
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
