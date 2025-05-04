import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import (
    squeezenet1_0, mobilenet_v2, efficientnet_b0,
    densenet121, shufflenet_v2_x1_0
)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# === Reproducibility ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Paths ===
dataset_dir = r"C:\dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Datasets & Loaders ===
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

dataloaders = {"train": train_loader, "val": val_loader}

# === Model Initialization Functions ===
def init_squeezenet(num_classes):
    model = squeezenet1_0(weights=None)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    return model.to(device)

def init_mobilenetv2(num_classes):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)

def init_efficientnetb0(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model.to(device)

def init_densenet121(num_classes):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model.to(device)

def init_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

# === Evaluation Function ===
def evaluate_model(model, dataloader, class_names, model_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("\n--- Test Set Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

# === Training Function ===
def train_model(model, dataloaders, criterion, optimizer, num_epochs, model_name, class_names):
    best_val_acc = 0.0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - {model_name}")
        print("-" * 30)

        # === Training ===
        model.train()
        train_loss, train_corrects = 0.0, 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels)

        epoch_train_loss = train_loss / len(dataloaders['train'].dataset)
        epoch_train_acc = train_corrects.double() / len(dataloaders['train'].dataset)
        train_losses.append(epoch_train_loss)

        # === Validation ===
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)

        epoch_val_loss = val_loss / len(dataloaders['val'].dataset)
        epoch_val_acc = val_corrects.double() / len(dataloaders['val'].dataset)
        val_losses.append(epoch_val_loss)

        print(f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print(f">>> Best model saved with acc: {best_val_acc:.4f}")

    # === Loss Plot ===
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
    plt.title(f'Loss per Epoch: {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_loss.png")
    plt.close()

    # === Load Best Model and Evaluate ===
    print(f"\n>>> Loading best model for final evaluation: {model_name}")
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    evaluate_model(model, test_loader, class_names, model_name)

# === Main ===
if __name__ == "__main__":
    num_classes = 4
    num_epochs = 30
    class_names = train_dataset.classes
    criterion = nn.CrossEntropyLoss()

    model_fns = {
        "squeezenet1.0": init_squeezenet,
        "mobilenetv2": init_mobilenetv2,
        "efficientnetb0": init_efficientnetb0,
        "densenet121": init_densenet121,
        "shufflenetv2": init_shufflenet
    }

    for name, init_fn in model_fns.items():
        print(f"\n=== Training {name} ===")
        model = init_fn(num_classes)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_model(model, dataloaders, criterion, optimizer, num_epochs, name, class_names)
        torch.cuda.empty_cache()
