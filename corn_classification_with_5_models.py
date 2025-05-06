import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri yolları
dataset_dir = r"C:\dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

# Dönüştürmeler
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
class_names = ['c0', 'c1', 'c2', 'c3']

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model hazırlayıcılar
def prepare_model(name, num_classes):
    if name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    elif name == "shufflenet":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mnasnet":
        model = models.mnasnet1_0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)


# Eğitim fonksiyonu (early stopping dahil)
def train_model(model, name, optimizer, criterion, train_loader, val_loader, num_epochs=30, patience=10):
    best_model = deepcopy(model.state_dict())
    best_loss = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        print(f"{name} - Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"{name} - Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), f"{name}_model.pth")
    print(f"{name} saved to {name}_model.pth")

    # Loss grafiği
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"{name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{name}_loss.png")
    plt.show()

    return model


# Değerlendirme
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# Confusion matrix
def plot_confusion(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14})
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()


# Metrik hesapla
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


# Ana blok
if __name__ == "__main__":
    model_names = ["squeezenet", "shufflenet", "mnasnet", "mobilenetv2", "efficientnet"]
    trained_models = []
    probs_list = []
    y_true = None

    criterion = nn.CrossEntropyLoss()

    for name in model_names:
        model = prepare_model(name, num_classes=4)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        model = train_model(model, name, optimizer, criterion, train_loader, val_loader)
        y_true, y_pred, probs = evaluate_model(model, test_loader)
        metrics = compute_metrics(y_true, y_pred)
        print(f"\n{name.upper()} Metrics:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"{k}: {v:.4f}")
        plot_confusion(metrics["confusion_matrix"], f"{name.upper()} Confusion Matrix")
        trained_models.append(model)
        probs_list.append(probs)

    # Ensemble
    ensemble_probs = np.mean(np.array(probs_list), axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_metrics = compute_metrics(y_true, ensemble_preds)
    print("\nENSEMBLE MODEL METRİKLERİ:")
    for k, v in ensemble_metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
    plot_confusion(ensemble_metrics["confusion_matrix"], "Ensemble Confusion Matrix")
