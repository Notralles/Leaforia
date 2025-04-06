import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm  # GhostNet için
import os
import numpy as np
from sklearn.metrics import accuracy_score

# 📌 Cihazı ayarla (CUDA varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Veri seti yolu
dataset_path = "C:/Users/krono/PycharmProjects/LeaforiaTrue/dataset"

# 📌 Dönüşümler (Data Augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 📌 Train ve Test veri setlerini yükle
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=transform)

# 📌 DataLoader (num_workers=0 hatayı önler)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# 📌 Sınıf sayısını al
num_classes = len(train_dataset.classes)

# 📌 MODELLERİ TANIMLA (GhostNet + MobileNetV2)
class LeaforiaNet(nn.Module):
    def __init__(self, num_classes):
        super(LeaforiaNet, self).__init__()

        # 📌 GhostNet (Timm ile)
        self.ghostnet = timm.create_model("ghostnet_100", pretrained=True, num_classes=num_classes)

        # 📌 MobileNetV2 (Önceden eğittiğin modeli yükleyeceğiz!)
        self.mobilenet = models.mobilenet_v2(weights=None)  # Pretrained yüklemiyoruz, kendi modelini alacağız
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        out1 = self.ghostnet(x)
        out2 = self.mobilenet(x)
        return (out1 + out2) / 2  # Soft Voting

# 📌 Modeli oluştur
model = LeaforiaNet(num_classes).to(device)

# 📌 Önceden eğitilmiş MobileNetV2'yi yükle
mobilenet_path = "mobilenetv2.pth"  # ✅ Kendi kaydettiğin model dosyası
if os.path.exists(mobilenet_path):
    model.mobilenet.load_state_dict(torch.load(mobilenet_path, map_location=device))
    print("✅ MobileNetV2 modeli yüklendi!")

# 📌 Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 📌 Modeli eğitme fonksiyonu
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

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {train_acc:.2f}%")

    # 📌 Modeli kaydet
    torch.save(model.state_dict(), "LeaforiaNet.pth")
    print("✅ Model kaydedildi: LeaforiaNet.pth")

# 📌 Test fonksiyonu (Accuracy hesaplar)
def test_model():
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%")

# 📌 Eğer bu dosya çalıştırılıyorsa
if __name__ == "__main__":
    train_model(epochs=10)
    test_model()
