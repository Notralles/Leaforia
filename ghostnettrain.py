import os
import torch
import timm  # GhostNet burada!
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Cihazı belirle (GPU varsa kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset yolu
dataset_path = os.path.join(os.path.expanduser("~"), "PycharmProjects", "LeaforiaTrue", "dataset")

# Veri dönüşümleri
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Train veri setini yükle
full_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)

# %70 Train, %15 Validation, %15 Test olarak bölme
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Veri yükleyicileri oluştur
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# GhostNet Modelini Yükle
model = timm.create_model('ghostnet_100', pretrained=True)  # %100 ölçekli GhostNet
num_ftrs = model.classifier.in_features  # Modelin çıkış özelliklerini al

# Son katmanı sınıf sayısına göre değiştir
num_classes = len(full_dataset.classes)  # Kaç sınıf olduğunu al
model.classifier = nn.Linear(num_ftrs, num_classes)  # Yeni çıkış katmanı ekle

# Modeli cihaza gönder
model = model.to(device)

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Modeli eğitme fonksiyonu
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        val_loss, val_accuracy = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss/len(train_loader):.4f} - Train Acc: {train_accuracy:.2f}% - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.2f}%")

# Modeli değerlendirme fonksiyonu
def evaluate_model(model, data_loader):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return loss / len(data_loader), accuracy

# Modeli eğit
train_model(model, train_loader, val_loader, epochs=10)

# Modeli kaydet
model_path = os.path.join(os.path.expanduser("~"), "PycharmProjects", "LeaforiaTrue", "ghostnet_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model başarıyla kaydedildi: {model_path}")

# Test veri seti üzerinde modelin doğruluğunu hesapla
test_loss, test_accuracy = evaluate_model(model, test_loader)
print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
