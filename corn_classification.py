import os
import random
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset paths
dataset_dir = r"C:\dataset"  # dataset klasörü burada olacak
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

# Data transformations
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

# Dataset
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# DataLoader'lar
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model preparation functions
def prepare_mobilenet_v3(num_classes):
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

def prepare_mobilenet_v2(num_classes):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def prepare_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# Training function
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    return model

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# Ensemble predictions
def ensemble_predictions(model_probs_list, weights=None):
    if weights is None:
        weights = [1/len(model_probs_list)] * len(model_probs_list)
    ensemble_probs = np.zeros_like(model_probs_list[0])
    for i, probs in enumerate(model_probs_list):
        ensemble_probs += weights[i] * probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, ensemble_probs

# Confusion matrix visualization
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Metric computation
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

# Print metrics
def print_metrics(name, metrics):
    print(f"\n{name} Performance Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

# MAIN BLOK - BURASI OLAYIN KALBİ
if __name__ == "__main__":
    # Model oluştur
    mobilenetv3_model = prepare_mobilenet_v3(num_classes=len(train_dataset.classes))
    mobilenetv2_model = prepare_mobilenet_v2(num_classes=len(train_dataset.classes))
    efficientnetb0_model = prepare_efficientnet_b0(num_classes=len(train_dataset.classes))

    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    mobilenetv3_optimizer = optim.Adam(mobilenetv3_model.parameters(), lr=0.0001)
    mobilenetv2_optimizer = optim.Adam(mobilenetv2_model.parameters(), lr=0.0001)
    efficientnetb0_optimizer = optim.Adam(efficientnetb0_model.parameters(), lr=0.0001)

    # Eğitim
    print("Training MobileNetV3...")
    mobilenetv3_model = train_model(mobilenetv3_model, train_loader, criterion, mobilenetv3_optimizer)

    print("\nTraining MobileNetV2...")
    mobilenetv2_model = train_model(mobilenetv2_model, train_loader, criterion, mobilenetv2_optimizer)

    print("\nTraining EfficientNetB0...")
    efficientnetb0_model = train_model(efficientnetb0_model, train_loader, criterion, efficientnetb0_optimizer)

    # Değerlendirme
    print("\nEvaluating models...")
    y_true, mobilenetv3_preds, mobilenetv3_probs = evaluate_model(mobilenetv3_model, test_loader)
    _, mobilenetv2_preds, mobilenetv2_probs = evaluate_model(mobilenetv2_model, test_loader)
    _, efficientnetb0_preds, efficientnetb0_probs = evaluate_model(efficientnetb0_model, test_loader)

    # Ensemble
    model_probs = [mobilenetv3_probs, mobilenetv2_probs, efficientnetb0_probs]
    ensemble_preds, ensemble_probs = ensemble_predictions(model_probs)

    # Metrikler
    mobilenetv3_metrics = compute_metrics(y_true, mobilenetv3_preds)
    mobilenetv2_metrics = compute_metrics(y_true, mobilenetv2_preds)
    efficientnetb0_metrics = compute_metrics(y_true, efficientnetb0_preds)
    ensemble_metrics = compute_metrics(y_true, ensemble_preds)

    # Yazdır
    print_metrics("MobileNetV3", mobilenetv3_metrics)
    print_metrics("MobileNetV2", mobilenetv2_metrics)
    print_metrics("EfficientNetB0", efficientnetb0_metrics)
    print_metrics("Ensemble", ensemble_metrics)

    # Confusion Matrix çiz
    plot_confusion_matrix(mobilenetv3_metrics["confusion_matrix"], train_dataset.classes, "MobileNetV3 Confusion Matrix")
    plot_confusion_matrix(mobilenetv2_metrics["confusion_matrix"], train_dataset.classes, "MobileNetV2 Confusion Matrix")
    plot_confusion_matrix(efficientnetb0_metrics["confusion_matrix"], train_dataset.classes, "EfficientNetB0 Confusion Matrix")
    plot_confusion_matrix(ensemble_metrics["confusion_matrix"], train_dataset.classes, "Ensemble Model Confusion Matrix")

    # Modelleri kaydet
    torch.save(mobilenetv3_model.state_dict(), "mobilenetv3_model.pth")
    torch.save(mobilenetv2_model.state_dict(), "mobilenetv2_model.pth")
    torch.save(efficientnetb0_model.state_dict(), "efficientnetb0_model.pth")
    print("\nModels saved successfully.")
