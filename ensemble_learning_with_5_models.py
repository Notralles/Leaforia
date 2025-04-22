import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v2, MobileNet_V2_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    squeezenet1_0, SqueezeNet1_0_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
)
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_dir = r"C:\dataset"
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "val")
test_dir = os.path.join(dataset_dir, "test")

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

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model preparation functions
def prepare_mobilenet_v3(num_classes):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv3_model.pth"))
    return model.to(device).eval()

def prepare_mobilenet_v2(num_classes):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("mobilenetv2_model.pth"))
    return model.to(device).eval()

def prepare_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load("efficientnetb0_model.pth"))
    return model.to(device).eval()

def prepare_squeezenet(num_classes):
    model = squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    return model

def prepare_shufflenet(num_classes):
    model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

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

def ensemble_predictions(model_probs_list, weights=None):
    if weights is None:
        weights = [1/len(model_probs_list)] * len(model_probs_list)
    ensemble_probs = np.zeros_like(model_probs_list[0])
    for i, probs in enumerate(model_probs_list):
        ensemble_probs += weights[i] * probs
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    return ensemble_preds, ensemble_probs

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def print_metrics(name, metrics):
    print(f"\n{name} Performance Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")

def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    num_classes = len(train_dataset.classes)

    mobilenetv3_model = prepare_mobilenet_v3(num_classes)
    mobilenetv2_model = prepare_mobilenet_v2(num_classes)
    efficientnetb0_model = prepare_efficientnet_b0(num_classes)

    squeezenet_model = prepare_squeezenet(num_classes)
    shufflenet_model = prepare_shufflenet(num_classes)

    criterion = nn.CrossEntropyLoss()
    squeezenet_optimizer = optim.Adam(squeezenet_model.parameters(), lr=0.0001)
    shufflenet_optimizer = optim.Adam(shufflenet_model.parameters(), lr=0.0001)

    print("Training SqueezeNet...")
    squeezenet_model = train_model(squeezenet_model, train_loader, criterion, squeezenet_optimizer)

    print("Training ShuffleNet...")
    shufflenet_model = train_model(shufflenet_model, train_loader, criterion, shufflenet_optimizer)

    print("\nEvaluating models...")
    y_true, mobv3_preds, mobv3_probs = evaluate_model(mobilenetv3_model, test_loader)
    _, mobv2_preds, mobv2_probs = evaluate_model(mobilenetv2_model, test_loader)
    _, eff_preds, eff_probs = evaluate_model(efficientnetb0_model, test_loader)
    _, squeeze_preds, squeeze_probs = evaluate_model(squeezenet_model, test_loader)
    _, shuffle_preds, shuffle_probs = evaluate_model(shufflenet_model, test_loader)

    all_probs = [mobv3_probs, mobv2_probs, eff_probs, squeeze_probs, shuffle_probs]
    ensemble_preds, ensemble_probs = ensemble_predictions(all_probs)

    metrics = {
        "MobileNetV3": compute_metrics(y_true, mobv3_preds),
        "MobileNetV2": compute_metrics(y_true, mobv2_preds),
        "EfficientNetB0": compute_metrics(y_true, eff_preds),
        "SqueezeNet": compute_metrics(y_true, squeeze_preds),
        "ShuffleNet": compute_metrics(y_true, shuffle_preds),
        "Ensemble": compute_metrics(y_true, ensemble_preds)
    }

    for name, met in metrics.items():
        print_metrics(name, met)
        plot_confusion_matrix(met["confusion_matrix"], train_dataset.classes, f"{name} Confusion Matrix")

    torch.save(squeezenet_model.state_dict(), "squeezenet_model.pth")
    torch.save(shufflenet_model.state_dict(), "shufflenet_model.pth")
    print("\nModels saved successfully.")
