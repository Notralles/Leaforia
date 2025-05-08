import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# Klasör yolları
dataset_dir = r"C:\dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

# Dönüşümler
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset ve DataLoader
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model hazırlayıcılar
def prepare_mobilenetv2(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def prepare_efficientnetb0(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def prepare_mnasnet(num_classes):
    model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# Eğitim fonksiyonu (Early stopping'li)
def train_model(model, model_name, criterion, optimizer, num_epochs=30, patience=5):
    model.to(device)
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

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

        # Validation loss
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"{model_name} | Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Loss grafiği
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_loss.png")
    plt.close()

    return model

# Değerlendirme
def evaluate_model(model):
    model.eval()
    preds, probs, labels_all = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            softmax_probs = nn.functional.softmax(outputs, dim=1)
            _, pred_classes = torch.max(softmax_probs, 1)
            preds.extend(pred_classes.cpu().numpy())
            probs.extend(softmax_probs.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return np.array(labels_all), np.array(preds), np.array(probs)

# Ensemble
def ensemble_predictions(prob_list):
    avg_probs = np.mean(prob_list, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds

# Confusion matrix
def plot_conf_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=["c0", "c1", "c2", "c3"],
                yticklabels=["c0", "c1", "c2", "c3"])
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# Metrikler
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "cm": confusion_matrix(y_true, y_pred)
    }

# Ana blok
if __name__ == "__main__":
    num_classes = len(train_dataset.classes)

    models_dict = {
        "MobileNetV2": prepare_mobilenetv2(num_classes),
        "EfficientNetB0": prepare_efficientnetb0(num_classes),
        "MNASNet": prepare_mnasnet(num_classes)
    }

    probs_list = []
    y_true_ref = None

    for name, model in models_dict.items():
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        model = train_model(model, name, criterion, optimizer)
        y_true, y_pred, probs = evaluate_model(model)
        if y_true_ref is None:
            y_true_ref = y_true
        metrics = compute_metrics(y_true, y_pred)
        print(f"\n{name} Metrics:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        plot_conf_matrix(metrics["cm"], f"{name} Confusion Matrix")
        torch.save(model.state_dict(), f"{name.lower()}.pth")
        probs_list.append(probs)

    # Ensemble sonuç
    ensemble_preds = ensemble_predictions(probs_list)
    ensemble_metrics = compute_metrics(y_true_ref, ensemble_preds)
    print(f"\nEnsemble Metrics:")
    print(f"Accuracy:  {ensemble_metrics['accuracy']:.4f}")
    print(f"Precision: {ensemble_metrics['precision']:.4f}")
    print(f"Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"F1 Score:  {ensemble_metrics['f1']:.4f}")
    plot_conf_matrix(ensemble_metrics["cm"], "Ensemble Confusion Matrix")
