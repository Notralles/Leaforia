import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score

# 📌 Cihaz seçimi (GPU varsa onu kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 Dataset yolu
dataset_path = "C:/Users/krono/PycharmProjects/LeaforiaTrue/dataset"

# 📦 Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 📂 Eğitim ve test datasetleri
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=transform)

# 🔄 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 📌 Sınıf sayısını al
num_classes = len(train_dataset.classes)

# 🧠 ResNet18 modelini yükle (pretrained=True ile başlatılır)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Output sınıf sayısına göre ayarla
model = model.to(device)

# 🎯 Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 🔁 Eğitim fonksiyonu
def train_model(epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.2f}%")

    # 📥 Modeli kaydet
    torch.save(model.state_dict(), "resnet18_leaforia.pth")
    print("✅ Model kaydedildi: resnet18_leaforia.pth")

# 🔍 Test fonksiyonu
def test_model():
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# 🔁 Çalıştırma
if __name__ == "__main__":
    train_model(epochs=10)
    test_model()
