import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

# Cihazı belirle (GPU varsa CUDA, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset yolu (bu yolu kendi dataset yolunuza göre ayarlayın)
dataset_path = "C:/Users/krono/PycharmProjects/LeaforiaTrue/dataset"

# Veri dönüşümleri (train ve validation için aynı dönüşümleri kullanıyoruz)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # MobileNet için standart normalizasyon
])

# Dataseti yükleme
full_train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=transform)

# Train veri setini %70 train, %30 validation olarak böl
train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# DataLoader oluşturma
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Kontrol için veri kümesi boyutlarını yazdırma
print(f"Train Set: {len(train_dataset)} görüntü")
print(f"Validation Set: {len(val_dataset)} görüntü")
print(f"Test Set: {len(test_dataset)} görüntü")

# MobileNetV2 modelini yükleme
model = models.mobilenet_v2(pretrained=True)

# Modeli yeniden yapılandırma (son katmanı değiştiriyoruz)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(full_train_dataset.classes))  # Kendi sınıf sayınıza göre ayarlayın

# Modeli doğru cihaza (GPU/CPU) taşıma
model = model.to(device)

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Modeli eğitme fonksiyonu
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Modeli eğitim moduna al
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Optimizer sıfırlama
            optimizer.zero_grad()

            # İleriye doğru geçiş
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Geriye doğru geçiş (backpropagation)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Modeli doğrulama moduna al
        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels).item()
                total_preds += labels.size(0)

        epoch_acc = correct_preds / total_preds
        print(f"Validation Accuracy: {epoch_acc:.4f}")

        # En iyi modeli kaydetme
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


# Modeli eğitme
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Modeli kaydetme
torch.save(model.state_dict(), 'mobilenetv2.pth')
print("Model başarıyla kaydedildi!")


# Test doğruluğunu hesaplama
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")


# Test verisi üzerinde değerlendirme
evaluate_model(model, test_loader)
