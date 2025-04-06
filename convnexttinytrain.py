import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
from sklearn.metrics import accuracy_score

# 📌 Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 Dataset yolu
dataset_path = "C:/Users/krono/PycharmProjects/LeaforiaTrue/dataset"

# 🔁 Dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# 📂 Datasetleri yükle
train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

num_classes = len(train_dataset.classes)

# 🔍 ConvNeXt modelini yükle
model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
model = model.to(device)

# 🔧 Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# 🚂 Eğitim fonksiyonu
def train_model(epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0

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

    # 💾 Modeli kaydet
    torch.save(model.state_dict(), "convnext.pth")
    print("✅ Model kaydedildi: convnext.pth")

# 🧪 Test fonksiyonu
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

# 🚀 Çalıştır
if __name__ == "__main__":
    train_model(epochs=10)
    test_model()
